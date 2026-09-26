from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from high_fidelity_schema_study.four_category.common import digest, read_json, seal, identity, write_new
from high_fidelity_schema_study.four_category.offline import make_fixture, mock_transport
from high_fidelity_schema_study.four_category.paper import (build_index, verify_index, categorized_input, verify_categorized,
    classification_errors, extraction_errors, category_counts, fact_view, verify_paper)
from high_fidelity_schema_study.four_category.tasks import load_task, render_task
from high_fidelity_schema_study.four_category.workflow import plan_jobs, run_batch, replay_run, evaluation_split, experiment_errors
from high_fidelity_schema_study.four_category.packet import build_packet, verify_packet


@pytest.fixture
def setup(tmp_path):
    sources = tmp_path / "sources"
    config, corpus = make_fixture(sources)
    tasks = {k: load_task(k) for k in ("classification", "extraction")}
    paper_input = read_json(sources / "paper_input.json")
    response = mock_transport({"model": "fixture", "messages": render_task(tasks["classification"], paper_input)})
    index = build_index(response["text"], paper_input, {"run_id": "fixture", "profile_sha256": "a"*64, "task_sha256": tasks["classification"]["task_sha256"], "request_sha256": "b"*64})
    return sources, config, corpus, tasks, paper_input, index


def test_full_paper_index_replay_and_no_dataset_leak(setup):
    sources, config, corpus, tasks, paper_input, index = setup
    value = categorized_input(paper_input, index)
    assert value["paper_input"] == paper_input
    messages = render_task(tasks["extraction"], paper_input, index)
    assert "DATASET_SECRET_CANARY" not in json.dumps(messages)
    changed = copy.deepcopy(value)
    changed["paper_input"]["pages"][0]["text_regions"][0]["text"] += " invented"
    changed = seal(changed, "categorized_input_sha256")
    assert verify_categorized(changed, paper_input, index)
    changed_index = copy.deepcopy(index)
    changed_index["entries"][0]["categories"] = ["syntax"]
    changed_index = seal(changed_index, "index_sha256")
    assert "classification_raw_derivation_mismatch" in verify_index(changed_index, paper_input)


def test_classification_incomplete_duplicate_wrong_span_and_uncertain(setup):
    _, _, _, _, paper_input, index = setup
    payload = json.loads(index["raw_response"])
    bad = copy.deepcopy(payload); bad["entries"] = []
    assert "classification_unit_set_mismatch" in classification_errors(bad, paper_input)
    bad = copy.deepcopy(payload); bad["entries"].append(bad["entries"][0])
    assert "duplicate_classification_unit" in classification_errors(bad, paper_input)
    bad = copy.deepcopy(payload); bad["entries"][0]["evidence_spans"][0]["quote"] = "invented"
    assert classification_errors(bad, paper_input)
    for entry in payload["entries"]:
        entry.update(state="uncertain", categories=[], evidence_spans=[], rationale="Test fixture ambiguity")
    assert classification_errors(payload, paper_input) == []


def test_assertion_categories_and_malformed_evidence(setup):
    _, _, _, tasks, paper_input, index = setup
    response = mock_transport({"model": "fixture", "messages": render_task(tasks["extraction"], paper_input, index)})
    payload = json.loads(response["text"])
    assert extraction_errors(payload, paper_input) == []
    payload["claims"][0]["category_annotations"][0]["categories"] += ["syntax"]
    counts = category_counts(fact_view(payload))
    assert counts["unique_facts"] == 1 and counts["categories"]["structure"] == counts["categories"]["syntax"] == 1
    bad = copy.deepcopy(payload); bad["claims"][0]["evidence"].append(42)
    assert extraction_errors(bad, paper_input)
    bad = copy.deepcopy(payload); bad["claims"][0]["datatype"] = None
    assert extraction_errors(bad, paper_input)
    bad = copy.deepcopy(payload); bad["claims"][0]["category_annotations"][0]["target_pointer"] = "/datatype"
    assert extraction_errors(bad, paper_input)


def test_batch_resume_swap_and_packet_relocation(setup, tmp_path):
    sources, config, corpus, tasks, paper_input, index = setup
    output = tmp_path / "runs"
    transports = {p["profile_id"]: mock_transport for p in config["profiles"]}
    first = run_batch(config, corpus, source_root=sources, output=output, transports=transports)
    assert first["all_slots_accounted"] and first["real_model_calls"] == 0
    assert first["backend_attempts_this_invocation"] == 11
    assert all(s["status"] == "success" for s in first["slots"])
    resumed = run_batch(config, corpus, source_root=sources, output=output, transports=transports)
    assert resumed["backend_attempts_this_invocation"] == 0
    changed = copy.deepcopy(config); changed["profiles"][0]["model_id"] = "NEW-SYNTHETIC-MODEL"
    swapped = run_batch(changed, corpus, source_root=sources, output=output, transports=transports)
    assert swapped["backend_attempts_this_invocation"] == 3
    before = read_json(output / first["slots"][1]["record_paths"][-1])
    after = read_json(output / swapped["slots"][1]["record_paths"][-1])
    assert before["messages"] == after["messages"] and before["task"] == after["task"]
    assert before["profile_sha256"] != after["profile_sha256"] and before["run_id"] != after["run_id"]
    packet = tmp_path / "packet"
    result = build_packet(config, corpus, first, run_root=output, source_root=sources, output=packet)
    assert result["status"] == "pass", result
    assert result["accuracy"] is None
    shutil.make_archive(str(tmp_path / "packet"), "zip", packet)
    moved = tmp_path / "moved"; moved.mkdir()
    shutil.unpack_archive(str(tmp_path / "packet.zip"), moved)
    new_sources = tmp_path / "new_sources"; shutil.copytree(sources, new_sources)
    assert verify_packet(moved, source_root=new_sources)["status"] == "pass"


def test_invalid_output_and_timeout_do_not_abort_batch(setup, tmp_path):
    sources, config, corpus, tasks, paper_input, index = setup
    transports = {p["profile_id"]: mock_transport for p in config["profiles"]}
    transports[config["roles"]["locals"][0]] = lambda request: {"text": '{"claims": [42]}', "model": request["model"], "finish_reason": "stop"}
    tries = []
    def transient(request):
        tries.append(1)
        if len(tries) == 1:
            raise TimeoutError("synthetic temporary transport fault")
        return mock_transport(request)
    transports[config["roles"]["locals"][1]] = transient
    batch = run_batch(config, corpus, source_root=sources, output=tmp_path / "run", transports=transports)
    assert batch["all_slots_accounted"]
    assert sum(s["status"] == "contract_invalid" for s in batch["slots"]) == 3
    assert batch["backend_attempts_this_invocation"] == 12
    assert any(len(s["record_paths"]) == 2 for s in batch["slots"])


def test_thousand_paper_matrix_and_many_to_many_reuse(setup):
    sources, config, corpus, tasks, paper_input, index = setup
    larger = copy.deepcopy(corpus)
    larger["papers"], larger["matches"] = [], []
    for i in range(1000):
        p = copy.deepcopy(corpus["papers"][0]); p["paper_id"] = f"SYNTHETIC_P{i:04d}"
        p["source"]["source_identity"]["document_id"] = p["paper_id"]
        p["source"] = seal(p["source"], "source_sha256")
        larger["papers"].append(p)
        larger["matches"].append({"match_id": f"m{i}", "paper_id": p["paper_id"], "dataset_id": corpus["datasets"][0]["dataset_id"], "linkage": {"status": "candidate", "evidence": ["synthetic planning only"]}})
    plan = plan_jobs(config, larger)
    assert len(plan["jobs"]) == 11000
    assert len({j["job_id"] for j in plan["jobs"]}) == 11000
    # A second dataset paired with the same paper changes packet membership, not its model input/jobs.
    paired = copy.deepcopy(corpus)
    d = copy.deepcopy(paired["datasets"][0]); d["dataset_id"] += "-another"
    paired["datasets"].append(d)
    paired["matches"].append({**copy.deepcopy(paired["matches"][0]), "match_id": "m2", "dataset_id": d["dataset_id"]})
    assert plan_jobs(config, paired)["jobs"] == plan_jobs(config, corpus)["jobs"]
    split = evaluation_split(paired)
    assert split["status"] == "insufficient_independent_groups" and split["human_labels"] is None


def test_context_overflow_explicit_no_calls(setup, tmp_path):
    sources, config, corpus, tasks, paper_input, index = setup
    calls = []
    def transport(request):
        calls.append(1); return mock_transport(request)
    counter = lambda p, m: {"input_tokens": p["context_window"], "exact": True, "counter_id": "synthetic-test"}
    batch = run_batch(config, corpus, source_root=sources, output=tmp_path / "run", transports={p["profile_id"]: transport for p in config["profiles"]}, token_counter=counter)
    assert not calls and batch["all_slots_accounted"]
    assert batch["slots"][0]["status"] == "invalid_request"
    assert all(s["status"] == "blocked_dependency" for s in batch["slots"][1:])


def _replace(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _reseal_packet(packet):
    manifest = read_json(packet / "manifest.json")
    manifest["files"] = [identity(p, packet) for p in sorted(packet.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    _replace(packet / "manifest.json", manifest)


@pytest.mark.parametrize("target", ["input", "pdf", "index", "dataset", "raw_response", "wire_request", "run_id", "slot_paper", "drop_index", "sampling", "task"])
def test_forged_self_hash_cannot_bypass_relations(setup, tmp_path, target):
    sources, config, corpus, tasks, paper_input, index = setup
    run_root = tmp_path / "runs"
    batch = run_batch(config, corpus, source_root=sources, output=run_root, transports={p["profile_id"]: mock_transport for p in config["profiles"]})
    packet = tmp_path / "packet"
    assert build_packet(config, corpus, batch, run_root=run_root, source_root=sources, output=packet)["status"] == "pass"
    if target in {"input", "pdf"}:
        manifest = read_json(packet / "corpus.json")
        source = manifest["papers"][0]["source"]
        item = source["artifacts"][target]
        path = sources / item["path"]
        if target == "input":
            value = read_json(path)
            value["pages"][0]["text_regions"][0]["text"] += " invented content"
            _replace(path, seal(value, "model_input_sha256"))
        else:
            with path.open("ab") as stream:
                stream.write(b"\n% invented revision\n")
        source["artifacts"][target] = identity(path, sources)
        manifest["papers"][0]["source"] = seal(source, "source_sha256")
        _replace(packet / "corpus.json", manifest)
    elif target == "index":
        path = packet / next(iter(batch["index_paths"].values()))
        value = read_json(path); value["entries"][0]["categories"] = ["syntax"]
        _replace(path, seal(value, "index_sha256"))
    elif target == "dataset":
        path = packet / next(iter(read_json(packet / "manifest.json")["datasets"].values()))
        value = read_json(path); value["facts"][0]["value"] = "invented"
        _replace(path, seal(value, "bundle_sha256"))
    elif target in {"raw_response", "wire_request", "run_id"}:
        path = packet / batch["slots"][1]["record_paths"][-1]
        value = read_json(path)
        if target == "raw_response":
            value["backend_result"]["raw_response"]["text"] = '{}'
        elif target == "wire_request":
            value["backend_result"]["request"]["messages"][-1]["content"] += " invented"
        else:
            value["run_id"] = "forged-run-id"
        _replace(path, seal(value, "record_sha256"))
    elif target == "slot_paper":
        value = read_json(packet / "batch.json"); value["slots"][0]["paper_id"] = "FORGED_PAPER"
        _replace(packet / "batch.json", seal(value, "batch_sha256"))
    elif target == "drop_index":
        value = read_json(packet / "batch.json"); value["index_paths"] = {}
        for slot in value["slots"][1:]:
            slot.update(status="blocked_dependency", record_paths=[])
            slot.pop("selected_run_id", None)
        _replace(packet / "batch.json", seal(value, "batch_sha256"))
    elif target == "sampling":
        config["dataset_parser"]["sample_limit"] = 999
        _replace(packet / "experiment.json", config)
        plan = plan_jobs(config, corpus)
        _replace(packet / "plan.json", plan)
        batch["plan_sha256"] = plan["plan_sha256"]
        _replace(packet / "batch.json", seal(batch, "batch_sha256"))
    else:
        path = packet / "tasks/extraction.json"
        task = read_json(path); task["system_prompt"] += " Return anything."
        _replace(path, seal(task, "task_sha256"))
    _reseal_packet(packet)
    result = verify_packet(packet, source_root=sources)
    assert result["byte_integrity"] == "pass", result
    assert result["derivation_validity"] == "fail", result


def test_malformed_configs_and_manifest_are_explicit(setup, tmp_path):
    sources, config, corpus, tasks, paper_input, index = setup
    bad = copy.deepcopy(config); bad["roles"] = None
    assert experiment_errors(bad)
    bad = copy.deepcopy(config); bad["classification"].pop("parameters")
    assert experiment_errors(bad)
    bad = copy.deepcopy(config); bad["execution"]["retry_statuses"] = ["contract_invalid"]
    assert experiment_errors(bad)
    result = verify_packet(tmp_path / "missing_packet", source_root=sources)
    assert result["byte_integrity"] == "fail" and result["derivation_validity"] == "not_checked"


def test_thousand_matchsets_partial_resume_and_failure_isolation(setup, tmp_path):
    sources, config, corpus, tasks, paper_input, index = setup
    # Many distinct hypothetical dataset bindings share a single real synthetic paper input.
    # This is a scheduler test; these planning-only dataset references are not released as evidence.
    larger = copy.deepcopy(corpus); larger["datasets"] = []; larger["matches"] = []
    for i in range(1000):
        dataset = {**corpus["datasets"][0], "dataset_id": f"planning-only-d{i}"}
        larger["datasets"].append(dataset)
        larger["matches"].append({**corpus["matches"][0], "match_id": f"m{i}", "dataset_id": dataset["dataset_id"]})
    assert len(plan_jobs(config, larger)["jobs"]) == 11
    transports = {p["profile_id"]: mock_transport for p in config["profiles"]}
    transports[config["roles"]["locals"][0]] = lambda r: {"text": "invalid json", "model": r["model"], "finish_reason": "stop"}
    first = run_batch(config, larger, source_root=sources, output=tmp_path / "run", transports=transports, max_jobs=4)
    assert first["status"] == "partial" and first["backend_attempts_this_invocation"] == 4
    second = run_batch(config, larger, source_root=sources, output=tmp_path / "run", transports=transports)
    assert second["backend_attempts_this_invocation"] == 7 and second["all_slots_accounted"]
    assert sum(s["status"] == "contract_invalid" for s in second["slots"]) == 3


def test_unsupported_parameters_count_no_dispatch_and_frontier_swap_independent(setup, tmp_path):
    sources, config, corpus, tasks, paper_input, index = setup
    config["classification"]["profile_id"] = config["roles"]["locals"][2]
    config["classification"]["profile_sha256"] = digest(config["profiles"][2])
    config["profiles"][0]["capabilities"]["supported_parameters"].remove("top_k")
    transports = {p["profile_id"]: mock_transport for p in config["profiles"]}
    first = run_batch(config, corpus, source_root=sources, output=tmp_path / "run", transports=transports)
    assert first["backend_attempts_this_invocation"] == 8
    assert sum(s["status"] == "invalid_request" for s in first["slots"]) == 3
    config["profiles"][-1]["model_id"] = "SWAPPED-FRONTIER"
    second = run_batch(config, corpus, source_root=sources, output=tmp_path / "run", transports=transports)
    assert second["backend_attempts_this_invocation"] == 1


def test_invalid_task_fails_before_dispatch_and_parent_cycle_rejected(setup, tmp_path):
    sources, config, corpus, tasks, paper_input, index = setup
    broken = copy.deepcopy(tasks)
    broken["classification"]["user_template"] = broken["classification"]["user_template"].replace("{{UNIT_CATALOG_JSON}}", "")
    broken["classification"] = seal(broken["classification"], "task_sha256")
    config["task_hashes"] = {k: t["task_sha256"] for k, t in broken.items()}
    result = run_batch(config, corpus, source_root=sources, output=tmp_path / "runs", tasks=broken)
    assert result["status"] == "blocked" and result["model_calls"] == 0
    response = mock_transport({"model": "fixture", "messages": render_task(tasks["extraction"], paper_input, index)})
    payload = json.loads(response["text"])
    claim = payload["claims"][0]; claim["parent_claim_id"] = claim["claim_id"]
    assert any("parent_cycle" in e for e in extraction_errors(payload, paper_input))


def test_classifier_binding_does_not_automatically_follow_profile_swap(setup, tmp_path):
    sources, config, corpus, tasks, paper_input, index = setup
    config["profiles"][-1]["model_id"] = "NEW-REFERENCE-AND-CLASSIFIER-DEPLOYMENT"
    result = run_batch(config, corpus, source_root=sources, output=tmp_path / "run")
    assert result["status"] == "blocked" and "classifier_profile_pin_required_or_mismatched" in result["errors"]
