"""Portable handoff packets with byte checks and independent derivation replay."""
from __future__ import annotations

import shutil
from pathlib import Path

from .common import contained, digest, file_digest, identity, read_json, seal_errors, write_new
from .dataset import verify_dataset
from .paper import verify_paper, verify_index, build_index, category_counts, fact_view
from .tasks import load_task
from .workflow import corpus_errors, plan_jobs, replay_run


def build_packet(config: dict, corpus: dict, batch: dict, *, run_root: Path, source_root: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError("packet output must be a new directory")
    errors = corpus_errors(corpus)
    if errors:
        raise ValueError(errors)
    output.mkdir(parents=True)
    write_new(output / "experiment.json", config)
    write_new(output / "corpus.json", corpus)
    write_new(output / "batch.json", batch)
    plan = read_json(contained(run_root, "plans/" + batch["plan_sha256"] + ".json"))
    write_new(output / "plan.json", plan)
    task_paths = {}
    for kind in ("classification", "extraction"):
        task = read_json(contained(run_root, "tasks/" + plan["task_hashes"][kind] + ".json"))
        task_paths[kind] = f"tasks/{kind}.json"
        write_new(output / task_paths[kind], task)
    copied = set()
    for slot in batch["slots"]:
        for relative in slot.get("record_paths", []):
            if relative not in copied:
                path = contained(output, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(contained(run_root, relative), path)
                copied.add(relative)
    for relative in batch["index_paths"].values():
        path = contained(output, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained(run_root, relative), path)
    datasets = {}
    for dataset in corpus["datasets"]:
        relative = "datasets/" + digest(dataset["dataset_id"]) + ".json"
        original = contained(source_root, dataset["bundle_path"])
        path = contained(output, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, path)
        datasets[dataset["dataset_id"]] = relative
    files = [identity(p, output) for p in sorted(output.rglob("*")) if p.is_file()]
    manifest = {"schema_version": "four-category-packet/v1", "files": files, "hash_mode": "file_bytes",
                "tasks": task_paths, "datasets": datasets,
                "external_source_root_parameter": "source_root", "source_paths": "repository-relative paths in corpus and dataset evidence",
                "authority": "auditable_attempts_not_semantic_gold", "semantic_evaluation": "pending_independent_sampled_reference"}
    write_new(output / "manifest.json", manifest)
    return verify_packet(output, source_root=source_root)


def verify_packet(packet: Path, *, source_root: Path, expected_experiment_sha256: str | None = None) -> dict:
    byte_errors, derivation_errors, admission, facts, dataset_admission = [], [], [], [], []
    bytes_checked = False
    try:
        manifest = read_json(Path(packet) / "manifest.json")
        if manifest.get("schema_version") != "four-category-packet/v1" or manifest.get("hash_mode") != "file_bytes":
            raise ValueError("packet_manifest_version_or_hash_mode_invalid")
        declared = set()
        for member in manifest["files"]:
            path = contained(packet, member["path"])
            if member["path"] in declared:
                byte_errors.append("duplicate_member:" + member["path"])
            declared.add(member["path"])
            if member.get("hash_mode") != "file_bytes" or not path.is_file() or file_digest(path) != member["sha256"] or path.stat().st_size != member["bytes"]:
                byte_errors.append("member_identity_mismatch:" + member["path"])
        actual = {p.relative_to(packet).as_posix() for p in Path(packet).rglob("*") if p.is_file()}
        if actual != declared | {"manifest.json"}:
            byte_errors.append("packet_member_set_mismatch")
        if byte_errors:
            return {"status": "fail", "byte_integrity": "fail", "byte_errors": byte_errors,
                    "derivation_validity": "not_checked", "semantic_correctness": "not_established"}
        bytes_checked = True
        config, corpus, batch, plan = [read_json(Path(packet) / name) for name in ("experiment.json", "corpus.json", "batch.json", "plan.json")]
        if expected_experiment_sha256 is not None and digest(config) != expected_experiment_sha256:
            derivation_errors.append("external_experiment_identity_mismatch")
        tasks = {kind: read_json(contained(packet, name)) for kind, name in manifest["tasks"].items()}
        for kind, task in tasks.items():
            derivation_errors.extend(seal_errors(task, "task_sha256"))
            if task["kind"] != kind:
                derivation_errors.append("task_kind_mismatch")
        derivation_errors.extend(seal_errors(batch, "batch_sha256"))
        if plan != plan_jobs(config, corpus, tasks=tasks) or batch["plan_sha256"] != plan["plan_sha256"]:
            derivation_errors.append("plan_derivation_mismatch")
        jobs = {j["job_id"]: j for j in plan["jobs"]}
        slot_ids = [s["job_id"] for s in batch["slots"]]
        if len(slot_ids) != len(set(slot_ids)) or set(slot_ids) != set(jobs) or not batch["all_slots_accounted"]:
            derivation_errors.append("slot_accounting_mismatch")
        if batch["status"] != ("complete" if len(slot_ids) == len(jobs) else "partial"):
            derivation_errors.append("batch_status_derivation_mismatch")
        contexts, indexes, classifier_records = {}, {}, {}
        profiles = {p["profile_id"]: p for p in config["profiles"]}
        if set(batch["index_paths"]) - {p["paper_id"] for p in corpus["papers"]}:
            derivation_errors.append("index_for_unknown_paper")
        if set(manifest["datasets"]) != {d["dataset_id"] for d in corpus["datasets"]}:
            derivation_errors.append("dataset_member_set_mismatch")
        for paper in corpus["papers"]:
            pid, source = paper["paper_id"], paper["source"]
            errors = verify_paper(source, source_root)
            derivation_errors.extend(f"{pid}:{e}" for e in errors)
            if errors:
                continue
            paths = {k: contained(source_root, item["path"]) for k, item in source["artifacts"].items()}
            contexts[pid] = {"input": read_json(paths["input"]), "layout": read_json(paths["layout"]),
                             "reading": paths["reading_text"].read_text(encoding="utf-8")}
            relative = batch["index_paths"].get(pid)
            if relative:
                indexes[pid] = read_json(contained(packet, relative))
                derivation_errors.extend(f"{pid}:{e}" for e in verify_index(indexes[pid], contexts[pid]["input"], taxonomy_value=tasks["classification"]["taxonomy"]))
        for slot in batch["slots"]:
            job = jobs.get(slot["job_id"])
            if job is None:
                continue
            if slot.get("paper_id") != job["paper_id"]:
                derivation_errors.append("slot_paper_binding_mismatch")
            context = contexts.get(job["paper_id"])
            if context is None:
                continue
            records = []
            for relative in slot.get("record_paths", []):
                record = read_json(contained(packet, relative))
                records.append(record)
                if record["job"] != job:
                    derivation_errors.append("run_job_binding_mismatch")
                if record["profile"] != profiles[job["profile_id"]]:
                    derivation_errors.append("run_profile_binding_mismatch")
                kind = "classification" if job["kind"] == "classification" else "extraction"
                index = None if kind == "classification" else indexes.get(job["paper_id"])
                derivation_errors.extend(replay_run(record, tasks[kind], context["input"], index=index,
                                                    layout=context["layout"], reading_text=context["reading"]))
            if records:
                selected = records[-1]
                if selected["run_id"] != slot.get("selected_run_id") or selected["status"] != slot["status"]:
                    derivation_errors.append("attempt_selection_mismatch")
                if [r["attempt"] for r in records] != list(range(1, len(records) + 1)):
                    derivation_errors.append("attempt_order_mismatch")
                if len(records) > config["execution"]["max_attempts"] or any(r["status"] != "transport_error" for r in records[:-1]):
                    derivation_errors.append("retry_policy_mismatch")
                if selected["status"] == "success":
                    if job["kind"] == "classification":
                        classifier_records[job["paper_id"]] = selected
                    else:
                        run_facts = fact_view(selected["parsed_response"])
                        facts.append({"run_id": selected["run_id"], **category_counts(run_facts)})
            elif slot["status"] != "blocked_dependency":
                derivation_errors.append("unaccounted_generation")
            elif job["kind"] == "classification" or job["paper_id"] in indexes:
                derivation_errors.append("unjustified_blocked_dependency:" + job["job_id"])
            admission.append({"job_id": job["job_id"], "status": slot["status"], "semantic_accuracy": None})
        if set(indexes) != set(classifier_records):
            derivation_errors.append("successful_classifier_index_set_mismatch")
        for pid, index in indexes.items():
            record = classifier_records.get(pid)
            if record is None:
                derivation_errors.append("index_has_no_admitted_classifier_run:" + pid)
                continue
            producer = {"run_id": record["run_id"], "profile_sha256": record["profile_sha256"],
                        "task_sha256": record["task"]["task_sha256"], "request_sha256": record["request_sha256"]}
            if index != build_index(record["backend_result"]["raw_text"], contexts[pid]["input"], producer, taxonomy_value=tasks["classification"]["taxonomy"]):
                derivation_errors.append("index_classifier_run_derivation_mismatch:" + pid)
        for dataset in corpus["datasets"]:
            bundle = read_json(contained(packet, manifest["datasets"][dataset["dataset_id"]]))
            if bundle.get("bundle_sha256") != dataset["bundle_sha256"]:
                derivation_errors.append("dataset_bundle_binding_mismatch:" + dataset["dataset_id"])
            if bundle.get("dataset_id") != dataset["dataset_id"]:
                derivation_errors.append("dataset_identity_mismatch")
            if bundle.get("parser", {}).get("sample_limit") != config["dataset_parser"]["sample_limit"]:
                derivation_errors.append("dataset_sampling_config_mismatch")
            if bundle["taxonomy_sha256"] != digest(tasks["extraction"]["taxonomy"]):
                derivation_errors.append("dataset_taxonomy_mismatch")
            derivation_errors.extend(verify_dataset(bundle, source_root, taxonomy=tasks["extraction"]["taxonomy"]))
            dataset_admission.append({"dataset_id": dataset["dataset_id"], "parse_status": bundle["status"],
                                      **category_counts(bundle["facts"]), "semantic_accuracy": None})
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        (derivation_errors if bytes_checked else byte_errors).append("packet_verification:" + str(exc))
    return {"status": "pass" if not byte_errors and not derivation_errors else "fail",
            "byte_integrity": "pass" if bytes_checked and not byte_errors else "fail", "byte_errors": byte_errors,
            "derivation_validity": "not_checked" if not bytes_checked else "pass" if not derivation_errors else "fail", "derivation_errors": derivation_errors,
            "admission": admission, "paper_fact_counts": facts,
            "dataset_admission": dataset_admission,
            "semantic_correctness": "not_established", "accuracy": None,
            "note": "Authentic failed/invalid attempts remain failures; provenance does not convert them to correct extractions."}
