"""Portable packet for a terminal matrix-scheduler run.

The packet contains immutable scheduler requests/results and selected group
attempts. Original paper and dataset sources remain external at source_root.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil

from .common import contained, digest, file_digest, identity, read_json, seal_errors, write_new
from .dataset import verify_dataset
from .paper import verify_paper
from .scheduler import TERMINAL
from .scheduler_tasks import verify_result
from .extraction_protocol import GROUPED_PROTOCOLS
from .workflow import plan_jobs, tasks_for_config

VERSION = "scheduled-paper-packet/v1"
LINEAGE_VERSION = "scheduled-paper-packet/v2"
RETRYABLE_OUTER = {"transport_error", "infrastructure_failed"}
ATTEMPT_NAME = re.compile(r"attempt-([1-9][0-9]*)\.json\Z")


def _jobs(condition: dict) -> list[dict]:
    config, corpus = condition["config"], condition["corpus"]
    plan = plan_jobs(config, corpus)
    if plan["plan_sha256"] != condition["logical_plan_sha256"]:
        raise ValueError("condition_logical_plan_mismatch")
    jobs = list(plan["jobs"])
    for ds in corpus["datasets"]:
        body = {"kind": "dataset_parse", "dataset": ds, "dependencies": [],
                "parser_config": config["dataset_parser"], "source_identity": digest(ds)}
        body["job_id"] = digest(body)
        jobs.append(body)
    if condition["jobs"] != jobs:
        raise ValueError("condition_job_set_mismatch")
    tasks = tasks_for_config(config)
    if plan["task_hashes"] != {kind: task["task_sha256"] for kind, task in tasks.items()}:
        raise ValueError("condition_task_version_mismatch")
    return jobs


def _selection(condition: dict, paper_id: str | None) -> dict:
    corpus = condition["corpus"]
    all_papers = {paper["paper_id"] for paper in corpus["papers"]}
    if paper_id is not None and paper_id not in all_papers:
        raise ValueError("selected_paper_not_in_condition")
    papers = all_papers if paper_id is None else {paper_id}
    datasets = ({dataset["dataset_id"] for dataset in corpus["datasets"]} if paper_id is None else
                {match["dataset_id"] for match in corpus["matches"] if match["paper_id"] == paper_id})
    jobs = [job["job_id"] for job in _jobs(condition)
            if job.get("paper_id") in papers or job["kind"] == "dataset_parse" and job["dataset"]["dataset_id"] in datasets]
    return {"paper_id": paper_id, "paper_ids": sorted(papers), "dataset_ids": sorted(datasets),
            "job_ids": jobs}


def _attempt_paths(root: Path, job_id: str, kind: str) -> list[str]:
    folder = contained(root, f"{kind}/{job_id}")
    if not folder.exists():
        return []
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("attempt_folder_invalid:" + kind + ":" + job_id)
    paths = []
    for path in folder.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("attempt_file_invalid:" + str(path))
        if not ATTEMPT_NAME.fullmatch(path.name):
            raise ValueError("attempt_file_name_invalid:" + str(path))
        paths.append(path.relative_to(root).as_posix())
    return sorted(paths)


def _group_paths(root: Path, job_id: str, job_kind: str) -> list[str]:
    paths = []
    for kind in ("classification_groups", "extraction_groups"):
        folder = contained(root, f"{kind}/{job_id}")
        if not folder.exists():
            continue
        if (kind == "classification_groups") != (job_kind == "classification"):
            raise ValueError("group_directory_job_kind_mismatch:" + job_id)
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError("group_folder_invalid:" + job_id)
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError("group_symlink_invalid:" + str(path))
            if path.is_file():
                paths.append(path.relative_to(root).as_posix())
    return sorted(paths)


def _artifact_path(job: dict) -> str | None:
    if job["kind"] == "classification":
        return "indexes/" + digest(job["paper_id"]) + ".json"
    if job["kind"] == "local_extraction":
        return "extractions/" + job["job_id"] + ".json"
    return None


def _summary_status(summary: dict, job: dict) -> str | None:
    if job["kind"] == "soft_reference":
        return "deferred"
    if job["kind"] == "dataset_parse":
        rows = [row for row in summary["dataset_jobs"] if row["dataset_id"] == job["dataset"]["dataset_id"]]
        return rows[0]["status"] if len(rows) == 1 else None
    if job["kind"] == "local_extraction":
        matrices = [row for row in summary["matrices"] if row["paper_id"] == job["paper_id"]]
        if len(matrices) != 1:
            return None
        cells = [cell for cell in matrices[0]["cells"] if cell["model"] == job["profile_id"] and
                 cell["replicate"] == job["replicate_id"]]
        return cells[0]["status"] if len(cells) == 1 else None
    return None  # Classification has no per-job status in scheduler summary.


def build_scheduled_packet(run_root, source_root, output, *, paper_id=None) -> dict:
    run_root, source_root, output = Path(run_root).resolve(), Path(source_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("packet output must be a new directory")
    if output.is_relative_to(run_root) or output.is_relative_to(source_root):
        raise ValueError("packet_output_must_be_outside_run_and_source_roots")
    condition = read_json(run_root / "condition.json")
    summary = read_json(run_root / "summary.json")
    if seal_errors(condition, "condition") or not summary.get("terminal"):
        raise ValueError("terminal_sealed_condition_required")
    selection = _selection(condition, paper_id)
    lineage_members = []
    if condition["policy"].get("group_cache_import") is not None:
        from .cache_lineage import verify_cache_lineage
        lineage_errors, lineage_members = verify_cache_lineage(
            run_root, condition, selection["job_ids"], source_root=source_root)
        if lineage_errors:
            raise ValueError("cache_lineage_invalid:" + ",".join(lineage_errors))
    by_id = {job["job_id"]: job for job in condition["jobs"]}
    members = {"condition.json", "summary.json"}
    members.update(lineage_members)
    for job_id in selection["job_ids"]:
        job = by_id[job_id]
        if job["kind"] == "soft_reference":
            continue
        members.update(_attempt_paths(run_root, job_id, "requests"))
        members.update(_attempt_paths(run_root, job_id, "results"))
        members.update(_group_paths(run_root, job_id, job["kind"]))
        artifact = _artifact_path(job)
        if artifact is not None and contained(run_root, artifact).is_file():
            members.add(artifact)
    output.mkdir(parents=True)
    for relative in sorted(members):
        destination = contained(output, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained(run_root, relative), destination)
    manifest = {"schema_version": LINEAGE_VERSION if lineage_members else VERSION, "hash_mode": "file_bytes", "selection": selection,
                "files": [identity(contained(output, relative), output) for relative in sorted(members)],
                "external_source_root_parameter": "source_root", "semantic_accuracy": None}
    write_new(output / "manifest.json", manifest)
    return verify_scheduled_packet(output, source_root)


def _verify_bytes(packet: Path, manifest: dict) -> list[str]:
    errors = []
    if manifest.get("schema_version") not in {VERSION, LINEAGE_VERSION} or manifest.get("hash_mode") != "file_bytes":
        return ["manifest_version_or_hash_mode_invalid"]
    declared = set()
    for item in manifest.get("files", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes", "hash_mode"}:
            errors.append("member_shape_invalid")
            continue
        relative = item["path"]
        if relative in declared:
            errors.append("duplicate_member:" + relative)
        declared.add(relative)
        path = contained(packet, relative)
        if (item["hash_mode"] != "file_bytes" or not path.is_file() or path.is_symlink()
                or file_digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]):
            errors.append("member_identity_mismatch:" + relative)
    actual = set()
    for path in packet.rglob("*"):
        if path.is_symlink():
            errors.append("packet_symlink_invalid:" + str(path))
        if path.is_file():
            actual.add(path.relative_to(packet).as_posix())
    if actual != declared | {"manifest.json"}:
        errors.append("packet_member_set_mismatch")
    return errors


def _verify_derivation(packet: Path, sources: Path, manifest: dict, expected_condition: str | None) -> tuple[list[str], dict]:
    errors = []
    condition, summary = read_json(packet / "condition.json"), read_json(packet / "summary.json")
    errors.extend(seal_errors(condition, "condition"))
    has_lineage = condition["policy"].get("group_cache_import") is not None
    if manifest.get("schema_version") != (LINEAGE_VERSION if has_lineage else VERSION):
        errors.append("packet_lineage_version_mismatch")
    if expected_condition is not None and condition.get("condition") != expected_condition:
        errors.append("external_condition_identity_mismatch")
    jobs = _jobs(condition)
    by_id = {job["job_id"]: job for job in jobs}
    selection = manifest["selection"]
    if selection != _selection(condition, selection.get("paper_id")):
        errors.append("selection_derivation_mismatch")
    lineage_members = []
    if has_lineage:
        from .cache_lineage import verify_cache_lineage
        lineage_errors, lineage_members = verify_cache_lineage(
            packet, condition, selection["job_ids"], source_root=sources)
        errors.extend("cache_lineage:" + issue for issue in lineage_errors)
    if (summary.get("schema_version") != "matrix-scheduler-summary/v1" or
            summary.get("condition") != condition.get("condition") or summary.get("terminal") is not True or
            summary.get("execution_mode") != ("live" if condition["live"] else "synthetic_mock") or
            summary.get("semantic_accuracy") is not None):
        errors.append("terminal_summary_binding_mismatch")
    if (len(summary["matrices"]) != len(condition["corpus"]["papers"]) or
            {row["paper_id"] for row in summary["matrices"]} != {row["paper_id"] for row in condition["corpus"]["papers"]}):
        errors.append("summary_paper_set_mismatch")
    if (len(summary["dataset_jobs"]) != len(condition["corpus"]["datasets"]) or
            {row["dataset_id"] for row in summary["dataset_jobs"]} != {row["dataset_id"] for row in condition["corpus"]["datasets"]}):
        errors.append("summary_dataset_set_mismatch")
    for matrix in summary["matrices"]:
        paper_jobs = [job for job in jobs if job["kind"] == "local_extraction" and job["paper_id"] == matrix["paper_id"]]
        expected_cells = {(job["profile_id"], job["replicate_id"]) for job in paper_jobs}
        actual_cells = [(cell["model"], cell["replicate"]) for cell in matrix["cells"]]
        if len(actual_cells) != len(expected_cells) or set(actual_cells) != expected_cells:
            errors.append("summary_matrix_cell_set_mismatch:" + matrix["paper_id"])
        if matrix.get("successful_cells") != sum(cell["status"] == "success" for cell in matrix["cells"]):
            errors.append("summary_matrix_success_count_mismatch:" + matrix["paper_id"])
        if matrix.get("terminal") is not True:
            errors.append("summary_matrix_not_terminal:" + matrix["paper_id"])
    for paper in condition["corpus"]["papers"]:
        if paper["paper_id"] in selection["paper_ids"]:
            errors.extend(paper["paper_id"] + ":" + issue for issue in verify_paper(paper["source"], sources))
    for dataset in condition["corpus"]["datasets"]:
        if dataset["dataset_id"] in selection["dataset_ids"]:
            bundle = read_json(contained(sources, dataset["bundle_path"]))
            if bundle.get("bundle_sha256") != dataset["bundle_sha256"]:
                errors.append("dataset_source_bundle_mismatch:" + dataset["dataset_id"])
            errors.extend(dataset["dataset_id"] + ":" + issue for issue in verify_dataset(bundle, sources))
    expected_files = {"condition.json", "summary.json"}
    expected_files.update(lineage_members)
    statuses = {}
    result_count = generated_count = admitted_count = 0
    grouped_completed = []
    for job_id in selection["job_ids"]:
        job = by_id[job_id]
        requests = _attempt_paths(packet, job_id, "requests")
        results = _attempt_paths(packet, job_id, "results")
        groups = _group_paths(packet, job_id, job["kind"])
        grouped = (job.get("group_execution", {}).get("version") == "all-groups/v1" or
                   job["kind"] == "classification" and condition["config"]["classification"].get("protocol") in {"classification-quotes/v2", "classification-anchors/v3", "classification-anchors/v13r2"} or
                   job["kind"] == "local_extraction" and condition["config"].get("extraction_input_protocol") in GROUPED_PROTOCOLS)
        if groups and not grouped:
            errors.append("unexpected_group_records:" + job_id)
        expected_files.update(requests + results + groups)
        artifact = _artifact_path(job)
        if artifact is not None and contained(packet, artifact).is_file():
            expected_files.add(artifact)
        summary_status = _summary_status(summary, job)
        if job["kind"] == "soft_reference":
            if requests or results or groups or summary_status != "deferred" or not condition["policy"].get("defer_soft_reference"):
                errors.append("soft_reference_deferral_mismatch:" + job_id)
            statuses[job_id] = "deferred"
            continue
        if not results:
            # A blocked dependency has no worker result; prove it using the
            # classifier final state and absence of a navigation index below.
            if job["kind"] == "local_extraction" and summary_status == "blocked_dependency" and not requests and not groups and not (artifact and contained(packet, artifact).is_file()):
                statuses[job_id] = "blocked_dependency"
                continue
            errors.append("selected_job_result_missing:" + job_id)
            continue
        if {Path(path).name for path in requests} != {Path(path).name for path in results}:
            errors.append("outer_attempt_request_result_count_mismatch:" + job_id)
        sequence = []
        for relative in results:
            name = Path(relative).name
            number = int(ATTEMPT_NAME.fullmatch(name).group(1))
            request_path = f"requests/{job_id}/attempt-{number}.json"
            if request_path not in requests or relative != f"results/{job_id}/attempt-{number}.json":
                errors.append("outer_attempt_pair_invalid:" + job_id)
                continue
            envelope, result = read_json(contained(packet, request_path)), read_json(contained(packet, relative))
            expected_index = "indexes/" + digest(job["paper_id"]) + ".json" if job["kind"] == "local_extraction" else None
            if (envelope.get("condition") != condition["condition"] or envelope.get("job_id") != job_id or
                    envelope.get("job") != job or envelope.get("attempt") != number or
                    envelope.get("index_path") != expected_index or envelope.get("request_id") != digest({k: v for k, v in envelope.items() if k != "request_id"}) or
                    envelope.get("worker_id") not in {worker["worker_id"] for worker in condition["policy"]["workers"]}):
                errors.append("outer_envelope_binding_mismatch:" + relative)
            result_errors, derived = verify_result(condition, envelope, result, sources, packet)
            errors.extend(relative + ":" + issue for issue in result_errors)
            status = result.get("body", {}).get("status")
            if derived is not None and number == max(int(ATTEMPT_NAME.fullmatch(Path(path).name).group(1)) for path in results):
                if artifact is None or not contained(packet, artifact).is_file() or read_json(contained(packet, artifact)) != derived:
                    errors.append("saved_artifact_derivation_mismatch:" + job_id)
            elif derived is None and number == max(int(ATTEMPT_NAME.fullmatch(Path(path).name).group(1)) for path in results) and artifact and contained(packet, artifact).is_file():
                errors.append("unjustified_saved_artifact:" + job_id)
            sequence.append((number, status, result, derived))
        sequence.sort()
        if [row[0] for row in sequence] != list(range(1, len(sequence) + 1)):
            errors.append("outer_attempt_sequence_invalid:" + job_id)
        if len(sequence) > condition["policy"]["max_attempts"] or any(row[1] not in RETRYABLE_OUTER for row in sequence[:-1]):
            errors.append("outer_retry_policy_mismatch:" + job_id)
        if sequence and sequence[-1][1] in RETRYABLE_OUTER and sequence[-1][0] < condition["policy"]["max_attempts"]:
            errors.append("outer_retry_stopped_early:" + job_id)
        if sequence:
            final = sequence[-1]
            statuses[job_id] = final[1]
            if final[1] not in TERMINAL or final[1] == "deferred":
                errors.append("outer_final_status_invalid:" + job_id)
            if summary_status is not None and summary_status != final[1]:
                errors.append("summary_job_status_mismatch:" + job_id)
            final_path = f"results/{job_id}/attempt-{final[0]}.json"
            if job["kind"] == "local_extraction":
                matrix = next(row for row in summary["matrices"] if row["paper_id"] == job["paper_id"])
                cell = next(row for row in matrix["cells"] if row["model"] == job["profile_id"] and row["replicate"] == job["replicate_id"])
                if cell.get("result") != final_path:
                    errors.append("summary_final_result_path_mismatch:" + job_id)
            elif job["kind"] == "dataset_parse":
                dataset_row = next(row for row in summary["dataset_jobs"] if row["dataset_id"] == job["dataset"]["dataset_id"])
                if dataset_row.get("result") != final_path:
                    errors.append("summary_final_result_path_mismatch:" + job_id)
            result_count += 1
            if job["kind"] == "local_extraction":
                body = final[2]["body"]
                if body.get("generation_status") == "complete" and body.get("counts", {}).get("returned_groups") == body.get("counts", {}).get("planned_groups"):
                    generated_count += 1
                if body.get("admission_status") == "full":
                    admitted_count += 1
            if job["kind"] in {"classification", "local_extraction"}:
                body = final[2]["body"]
                if body.get("schema_version") == "group-execution-result/v2":
                    grouped_completed.append({"job_id": job_id, "kind": job["kind"], "paper_id": job["paper_id"],
                                              "generation_status": body["generation_status"],
                                              "admission_status": body["admission_status"], "counts": body["counts"]})
    for job_id in selection["job_ids"]:
        job = by_id[job_id]
        if statuses.get(job_id) == "blocked_dependency":
            dependencies = job["dependencies"]
            if len(dependencies) != 1 or statuses.get(dependencies[0]) in {None, "success", "completed_with_rejections", "generation_incomplete"}:
                errors.append("blocked_dependency_not_proven:" + job_id)
            classifier_artifact = _artifact_path(by_id[dependencies[0]]) if dependencies else None
            if classifier_artifact and contained(packet, classifier_artifact).is_file():
                errors.append("blocked_dependency_has_navigation_index:" + job_id)
    if selection["paper_id"] is None and Counter(statuses.values()) != Counter(summary["slot_counts"]):
        errors.append("summary_slot_counts_mismatch")
    if condition["config"]["execution"].get("grouped_mode") == "all-groups/v1":
        published = summary.get("group_execution", {})
        if published.get("version") != "all-groups/v1":
            errors.append("summary_group_execution_version_mismatch")
        published_selected = [row for row in published.get("completed_jobs", []) if row.get("job_id") in selection["job_ids"]]
        if published_selected != grouped_completed:
            errors.append("summary_group_completed_jobs_mismatch")
        if selection["paper_id"] is None:
            local = [row for row in grouped_completed if row["kind"] == "local_extraction"]
            classifier = [row for row in grouped_completed if row["kind"] == "classification"]
            if (published.get("planned_local_cells") != sum(job["kind"] == "local_extraction" for job in jobs) or
                    published.get("returned_local_cells") != sum(row["generation_status"] == "complete" for row in local) or
                    published.get("fully_admitted_local_cells") != sum(row["admission_status"] == "full" for row in local) or
                    published.get("planned_classifier_jobs") != sum(job["kind"] == "classification" for job in jobs) or
                    published.get("returned_classifier_jobs") != sum(row["generation_status"] == "complete" for row in classifier)):
                errors.append("summary_group_coverage_count_mismatch")
            for name, rows in (("classification_group_counts", classifier), ("extraction_group_counts", local)):
                expected = {key: sum(row["counts"][key] for row in rows)
                            for key in ("planned_groups", "accounted_groups", "returned_groups", "admitted_groups", "missing_generation_groups")}
                if published.get(name) != expected:
                    errors.append("summary_group_counts_mismatch:" + name)
            expected_full = bool(condition["live"]) and len(local) == sum(job["kind"] == "local_extraction" for job in jobs) and all(row["generation_status"] == "complete" for row in local)
            if published.get("full_local_inference_coverage") is not expected_full:
                errors.append("summary_full_local_coverage_mismatch")
    if {item["path"] for item in manifest["files"]} != expected_files:
        errors.append("packet_member_derivation_mismatch")
    selected_local_jobs = sum(by_id[job_id]["kind"] == "local_extraction" for job_id in selection["job_ids"])
    coverage = {"selected_jobs": len(selection["job_ids"]), "verified_results": result_count,
                "local_extractions": selected_local_jobs,
                "complete_generation_cells": generated_count, "fully_admitted_cells": admitted_count,
                "real_generation_goal_met": bool(not errors and condition["live"] and selected_local_jobs > 0 and
                                                 generated_count == selected_local_jobs)}
    return errors, coverage


def verify_scheduled_packet(packet, source_root, *, expected_condition=None) -> dict:
    packet, sources = Path(packet).resolve(), Path(source_root).resolve()
    byte_errors, derivation_errors, coverage = [], [], None
    bytes_checked = False
    try:
        manifest = read_json(packet / "manifest.json")
        byte_errors = _verify_bytes(packet, manifest)
        if not byte_errors:
            bytes_checked = True
            derivation_errors, coverage = _verify_derivation(packet, sources, manifest, expected_condition)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        (derivation_errors if bytes_checked else byte_errors).append("packet_verification:" + str(exc))
    return {"status": "pass" if not byte_errors and not derivation_errors else "fail",
            "byte_integrity": "pass" if not byte_errors else "fail", "byte_errors": byte_errors,
            "derivation_validity": "not_checked" if byte_errors else "pass" if not derivation_errors else "fail",
            "derivation_errors": derivation_errors, "coverage": coverage,
            "external_condition_anchor": "supplied" if expected_condition is not None else "missing",
            "semantic_accuracy": None, "semantic_correctness": "not_established"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run-root", type=Path, required=True)
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--paper-id")
    verify = sub.add_parser("verify")
    verify.add_argument("--packet", type=Path, required=True)
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--expected-condition")
    args = parser.parse_args(argv)
    report = (build_scheduled_packet(args.run_root, args.source_root, args.output, paper_id=args.paper_id)
              if args.command == "build" else
              verify_scheduled_packet(args.packet, args.source_root, expected_condition=args.expected_condition))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
