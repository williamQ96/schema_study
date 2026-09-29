"""Portable scheduler packet checks using a real offline worker run."""
import copy
import json
from pathlib import Path
import shutil
import tempfile

import pytest

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import digest, identity, read_json, seal
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category.scheduler import default_policy, run_scheduler
from high_fidelity_schema_study.four_category.scheduled_packet import (
    build_scheduled_packet, verify_scheduled_packet,
)
from .test_complete_scheduler import configure as configure_complete


@pytest.fixture(scope="module")
def scheduler_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("scheduled-packet")
    sources = root / "sources"
    config, corpus = make_fixture(sources)
    config["classification"]["profile_id"] = config["roles"]["locals"][0]
    config["classification"]["profile_sha256"] = profile_hash(config["profiles"][0])
    policy = default_policy()
    policy.update(poll_s=.03, heartbeat_s=.03)
    run = root / "run"
    summary = run_scheduler(config, corpus, policy, root=run, sources=sources,
                            leases=root / "leases", start_watchdog=False)
    assert summary["terminal"]
    packet = root / "packet"
    report = build_scheduled_packet(run, sources, packet)
    assert report["status"] == "pass", report
    return root, sources, packet, corpus


def _copy_packet(scheduler_run, tmp_path):
    _, sources, original, _ = scheduler_run
    target = tmp_path / "relocated"
    shutil.copytree(original, target)
    return sources, target


@pytest.fixture(scope="module")
def complete_run():
    # Two digest components below group folders exceed Windows MAX_PATH under
    # pytest's normal, deeply nested base directory.
    short_base = Path.home() / ".codex" / "tmp"
    with tempfile.TemporaryDirectory(prefix="scp-", dir=short_base if short_base.is_dir() else None) as folder:
        root = Path(folder)
        sources = root / "sources"
        config, corpus = make_fixture(sources)
        config["classification"]["profile_id"] = config["roles"]["locals"][0]
        config["classification"]["profile_sha256"] = profile_hash(config["profiles"][0])
        configure_complete(config)
        policy = default_policy()
        policy.update(poll_s=.03, heartbeat_s=.03)
        run = root / "run"
        summary = run_scheduler(config, corpus, policy, root=run, sources=sources,
                                leases=root / "leases", start_watchdog=False)
        assert summary["terminal"]
        packet = root / "packet"
        report = build_scheduled_packet(run, sources, packet)
        assert report["status"] == "pass", report
        planned_local = sum(job["kind"] == "local_extraction" for job in read_json(run / "condition.json")["jobs"])
        assert report["coverage"]["local_extractions"] == planned_local
        assert report["coverage"]["complete_generation_cells"] == planned_local
        yield root, sources, packet


def _rehash_member(packet, relative):
    manifest = read_json(packet / "manifest.json")
    new = identity(packet / relative, packet)
    manifest["files"] = [new if row["path"] == relative else row for row in manifest["files"]]
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_relocates_and_replays_without_run_database(scheduler_run, tmp_path):
    sources, packet = _copy_packet(scheduler_run, tmp_path)
    condition = read_json(packet / "condition.json")["condition"]
    result = verify_scheduled_packet(packet, sources, expected_condition=condition)
    assert result["status"] == "pass", result
    assert result["byte_integrity"] == "pass"
    assert result["derivation_validity"] == "pass"
    assert result["semantic_accuracy"] is None
    assert result["coverage"]["selected_jobs"] == 12
    assert result["external_condition_anchor"] == "supplied"
    assert verify_scheduled_packet(packet, sources)["external_condition_anchor"] == "missing"


def test_rejects_wrong_condition_and_external_source(scheduler_run, tmp_path):
    sources, packet = _copy_packet(scheduler_run, tmp_path)
    wrong = verify_scheduled_packet(packet, sources, expected_condition="different")
    assert wrong["byte_integrity"] == "pass"
    assert "external_condition_identity_mismatch" in wrong["derivation_errors"]
    other = tmp_path / "other_sources"
    shutil.copytree(sources, other)
    paper = next(other.rglob("*.txt"))
    paper.write_bytes(paper.read_bytes() + b"tampered")
    damaged = verify_scheduled_packet(packet, other)
    assert damaged["status"] == "fail"
    assert damaged["byte_integrity"] == "pass"
    assert damaged["derivation_validity"] == "fail"


def test_rejects_modified_artifact_even_when_manifest_is_rehashed(scheduler_run, tmp_path):
    sources, packet = _copy_packet(scheduler_run, tmp_path)
    relative = next(row["path"] for row in read_json(packet / "manifest.json")["files"]
                    if row["path"].startswith("indexes/"))
    saved = read_json(packet / relative)
    saved["semantic_review"] = "established"
    (packet / relative).write_text(json.dumps(saved), encoding="utf-8")
    _rehash_member(packet, relative)
    report = verify_scheduled_packet(packet, sources)
    assert report["byte_integrity"] == "pass", report
    assert report["derivation_validity"] == "fail"
    assert any("saved_artifact_derivation_mismatch" in error for error in report["derivation_errors"])


def test_paper_selection_keeps_condition_but_only_its_jobs(scheduler_run, tmp_path):
    _, sources, _, corpus = scheduler_run
    paper_id = corpus["papers"][0]["paper_id"]
    target = tmp_path / "selected"
    report = build_scheduled_packet(scheduler_run[0] / "run", sources, target, paper_id=paper_id)
    assert report["status"] == "pass", report
    selection = read_json(target / "manifest.json")["selection"]
    assert selection["paper_ids"] == [paper_id]
    assert len(selection["job_ids"]) == 12


def test_complete_group_packet_rejects_missing_group_record(complete_run):
    root, sources, original = complete_run
    packet = root / "missing"
    shutil.copytree(original, packet)
    relative = next(row["path"] for row in read_json(packet / "manifest.json")["files"]
                    if row["path"].startswith("classification_groups/") and row["path"].endswith(".json"))
    (packet / relative).unlink()
    manifest = read_json(packet / "manifest.json")
    manifest["files"] = [row for row in manifest["files"] if row["path"] != relative]
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = verify_scheduled_packet(packet, sources)
    assert report["byte_integrity"] == "pass", report
    assert report["derivation_validity"] == "fail"
    assert any("group_execution_replay" in error for error in report["derivation_errors"])


def test_rehashed_summary_cannot_claim_full_generation(complete_run):
    root, sources, original = complete_run
    packet = root / "summary_forgery"
    shutil.copytree(original, packet)
    summary = read_json(packet / "summary.json")
    summary["group_execution"]["full_local_inference_coverage"] = True
    (packet / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _rehash_member(packet, "summary.json")
    report = verify_scheduled_packet(packet, sources)
    assert report["byte_integrity"] == "pass"
    assert "summary_full_local_coverage_mismatch" in report["derivation_errors"]


def test_extra_group_record_rejected_even_when_manifest_lists_it(complete_run):
    root, sources, original = complete_run
    packet = root / "extra_record"
    shutil.copytree(original, packet)
    relative = next(row["path"] for row in read_json(packet / "manifest.json")["files"]
                    if row["path"].startswith("classification_groups/") and row["path"].endswith("attempt-1.json"))
    extra = relative.replace("attempt-1.json", "attempt-2.json")
    shutil.copyfile(packet / relative, packet / extra)
    manifest = read_json(packet / "manifest.json")
    manifest["files"].append(identity(packet / extra, packet))
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = verify_scheduled_packet(packet, sources)
    assert report["byte_integrity"] == "pass", report
    assert report["derivation_validity"] == "fail"
    assert any("group_execution_replay" in error for error in report["derivation_errors"])


def test_authentic_model_rejection_remains_verified_failure(complete_run):
    from high_fidelity_schema_study.four_category.complete_groups import (
        _body, _child_job, _context, _generation_returned, verify_complete,
    )
    from high_fidelity_schema_study.four_category.execution_artifacts import build_artifact
    from high_fidelity_schema_study.four_category.grouped_extraction import mock_group_transport
    from high_fidelity_schema_study.four_category.scheduler_tasks import paper_context
    from high_fidelity_schema_study.four_category.workflow import _attempt, tasks_for_config

    root, sources, _ = complete_run
    condition = read_json(root / "run" / "condition.json")
    paper_id = condition["corpus"]["papers"][0]["paper_id"]
    packet = root / "invalid_model"
    assert build_scheduled_packet(root / "run", sources, packet, paper_id=paper_id)["status"] == "pass"
    job = next(row for row in condition["jobs"] if row["kind"] == "local_extraction")
    profile = next(row for row in condition["config"]["profiles"] if row["profile_id"] == job["profile_id"])
    task = tasks_for_config(condition["config"])["extraction"]
    paper_input = paper_context(condition, job, sources)["input"]
    index = read_json(packet / "indexes" / (digest(paper_id) + ".json"))
    policy = condition["config"]["extraction_grouping"]
    module, groups = _context(task, paper_input, index, policy)
    group = groups[0]
    child = _child_job(module, job, group, index)
    def malformed_answer(request):
        response = mock_group_transport(request)
        response["text"] = "{broken}"
        return response
    rejected = _attempt(child, profile, task, paper_input, index, None, None, 1,
                        allow_live=False, transport=malformed_answer,
                        counter=None, extraction_group=group)
    assert rejected["status"] != "success" and _generation_returned(rejected), rejected["backend_result"]
    relative_group = f"extraction_groups/{job['job_id']}/{group['group_id']}/attempt-1.json"
    (packet / relative_group).write_text(json.dumps(rejected), encoding="utf-8")
    _rehash_member(packet, relative_group)
    relative_result = f"results/{job['job_id']}/attempt-1.json"
    result = read_json(packet / relative_result)
    outcomes = result["body"]["group_outcomes"]
    first = outcomes[0]
    first["attempt_refs"] = [module.record_ref(packet / relative_group, packet)]
    first["selected_record_sha256"] = rejected["record_sha256"]
    first["status"] = rejected["status"]
    first["generation_returned"] = True
    result["body"] = _body(groups, result["body"]["group_plan_sha256"], outcomes)
    result = seal({key: value for key, value in result.items() if key != "result_sha256"}, "result_sha256")
    (packet / relative_result).write_text(json.dumps(result), encoding="utf-8")
    _rehash_member(packet, relative_result)
    replay_errors, records = verify_complete(job, profile, task, paper_input, index, policy, 1,
                                             result["body"], packet)
    assert not replay_errors
    artifact = build_artifact(job, profile, task, paper_input, index, policy, records)
    relative_artifact = f"extractions/{job['job_id']}.json"
    (packet / relative_artifact).write_text(json.dumps(artifact), encoding="utf-8")
    _rehash_member(packet, relative_artifact)
    summary = read_json(packet / "summary.json")
    matrix = summary["matrices"][0]
    cell = next(row for row in matrix["cells"] if row["model"] == job["profile_id"] and row["replicate"] == job["replicate_id"])
    cell["status"] = result["body"]["status"]
    matrix["successful_cells"] -= 1
    summary["slot_counts"]["success"] -= 1
    summary["slot_counts"][result["body"]["status"]] = 1
    complete = summary["group_execution"]
    complete["fully_admitted_local_cells"] -= 1
    complete["extraction_group_counts"]["admitted_groups"] -= 1
    row = next(row for row in complete["completed_jobs"] if row["job_id"] == job["job_id"])
    row.update(generation_status=result["body"]["generation_status"],
               admission_status=result["body"]["admission_status"], counts=result["body"]["counts"])
    (packet / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _rehash_member(packet, "summary.json")
    report = verify_scheduled_packet(packet, sources)
    assert report["status"] == "pass", report
    assert report["semantic_accuracy"] is None
