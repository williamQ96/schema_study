"""Offline fidelity replay and immutable preparation; no network or inference calls."""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path

from . import extraction_v10, extraction_v11, preprocessing, preprocessing_pilot
from .common import contained, digest, file_digest, read_json, seal, seal_errors, write_new
from .dataset_catalog import build_catalog, verify_catalog, load_aliases
from .disabled_navigation import make_disabled_index
from .fidelity import evaluate, visibility_candidates


def verify_archive(root: Path, manifest: dict, repo: Path) -> dict:
    checked = 0
    for key, base in (("files", root), ("repository_references", repo)):
        for row in manifest[key]:
            path = contained(base, row["path"])
            if file_digest(path) != row["file_bytes_sha256"]:
                raise ValueError("historical_archive_changed:" + row["path"])
            checked += 1
    return {"checked_files": checked, "mismatches": 0}


def _load(config: dict, root: Path):
    pilot = contained(root, config["pilot_root"])
    archive = pilot / "artifact-manifest-v1.json"
    if file_digest(archive) != config["artifact_manifest_file_bytes_sha256"]:
        raise ValueError("archive_manifest_identity_changed")
    integrity = verify_archive(pilot, read_json(archive), root)
    workspace = pilot / "collected-v2/workspace"
    return pilot, workspace, integrity


def replay_batch(config: dict, root: Path, output: Path) -> dict:
    pilot, workspace, integrity = _load(config, root)
    if output.exists():
        raise ValueError("immutable_output_already_exists")
    catalogs = {b["paper_id"]: build_catalog(b, root) for b in config["dataset_bindings"]}
    aliases = {pid: [] for pid in catalogs}
    for pid, ref in config.get("alias_documents", {}).items():
        path = contained(root, ref["path"])
        if file_digest(path) != ref["file_bytes_sha256"]:
            raise ValueError("alias_document_identity_changed")
        aliases[pid] = load_aliases(read_json(path), catalogs[pid], root)
    source_manifest = read_json(workspace / "input-manifest.json")
    plan = read_json(workspace / "extraction-plan.json")
    continuation = read_json(pilot / "analysis-catalog-v1.json")
    if (seal_errors(plan, "extraction_plan_sha256") or seal_errors(continuation, "catalog_sha256")
            or continuation["extraction_plan_sha256"] != plan["extraction_plan_sha256"]):
        raise ValueError("pilot_catalog_plan_binding_mismatch")
    rows, package_cache = [], {}
    for request_row, origin_row in zip(plan["requests"], continuation["rows"], strict=True):
        ordinal = request_row["ordinal"]
        if (ordinal != origin_row["global_ordinal"] or request_row["case_id"] != origin_row["case_id"]
                or request_row["condition_id"] != origin_row["condition_id"]):
            raise ValueError("request_origin_mismatch")
        try:
            saved = read_json(contained(workspace, request_row["path"]))
            package = read_json(contained(workspace, request_row["package_path"]))
            if digest(saved) != request_row["request_canonical_sha256"] or digest(package) != request_row["package_canonical_sha256"]:
                raise ValueError("request_package_hash_mismatch")
            group = request_row["group"]
            key = package["package_sha256"]
            if key not in package_cache:
                errors = preprocessing.verify_condition(package, workspace)
                if errors:
                    raise ValueError("condition_replay_failed:" + str(errors[:2]))
                selected = package["mineru_source"] if package["config"]["mineru_enabled"] else package["baseline_source"]
                package_cache[key] = read_json(contained(workspace, selected["artifacts"]["input"]["path"]))
            paper = package_cache[key]
            if saved["messages"] != preprocessing.render_group(package, workspace, group, task=saved["source_task"]):
                raise ValueError("saved_prompt_derivation_mismatch")
            if saved["response_schema"] != preprocessing.response_schema(package, workspace, group, task=saved["source_task"]):
                raise ValueError("saved_schema_derivation_mismatch")
            origin = origin_row["origin"]
            key = origin["probe_root_key"]
            probe = pilot / "collected-v2/outputs" / {"original": "qwen", "continuation": "qwen-continuation-v1"}[key]
            result, backend = preprocessing_pilot._probe_result(probe, origin["local_ordinal"], saved)
            admission = preprocessing.admit_response(backend.get("raw_text"), package, workspace, group, task=saved["source_task"]) if backend["status"] == "success" else {"status": backend["status"], "payload": None}
            report = evaluate(admission.get("payload"), paper, catalogs[paper["paper_id"]], aliases=aliases[paper["paper_id"]],
                              admission_status=admission["status"], admission_errors=admission.get("validation_errors", []),
                              target_windows=group["window_ids"])
            report_path = f"runs/{ordinal:04d}.json"
            write_new(output / report_path, report)
            rows.append({"ordinal": ordinal, "paper_id": paper["paper_id"],
                         "condition_id": request_row["condition_id"], "case_id": request_row["case_id"],
                         "status": "evaluated", "admission": admission["status"], "report_path": report_path,
                         "report_sha256": report["report_sha256"], "decision_counts": report["decision_counts"],
                         "decisions": report["deduplicated_decisions"], "raw_field_records": report["raw_field_records"],
                         "automatic_support_rate": report["automatic_support_rate"],
                         "locator_precision": report["locator_precision"],
                         "accepted_object_ids": report["accepted_object_ids"],
                         "source_request_canonical_sha256": digest(saved), "origin": origin,
                         "generation_duration_s": result.get("generation_duration_s")})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            rows.append({"ordinal": ordinal, "condition_id": request_row["condition_id"],
                         "case_id": request_row["case_id"], "status": "failed", "error": str(exc),
                         "formal_precision": None, "formal_recall": None})
    for pid, catalog in catalogs.items():
        write_new(output / f"catalogs/{pid}.json", catalog)
        src = source_manifest["sources"][pid]["baseline"]
        pdf = contained(workspace, src["artifacts"]["pdf"]["path"])
        vis = visibility_candidates(catalog, pdf, src["artifacts"]["pdf"]["sha256"], aliases=aliases[pid])
        write_new(output / f"visibility/{pid}.json", vis)
    arms = {}
    for condition in sorted({r["condition_id"] for r in rows}):
        selected = [r for r in rows if r["condition_id"] == condition and r["status"] == "evaluated"]
        counts = Counter()
        by_paper = defaultdict(list)
        for r in selected:
            counts.update(r["decision_counts"])
            by_paper[r["paper_id"]].append(r)
        paper_rates = {pid: sum(r["decision_counts"].get("supported", 0) for r in prs) / sum(r["decisions"] for r in prs)
                       if sum(r["decisions"] for r in prs) else None for pid, prs in by_paper.items()}
        rates = [v for v in paper_rates.values() if v is not None]
        arms[condition] = {"runs_evaluated": len(selected), "decision_counts": dict(counts),
                           "paper_rates": paper_rates, "paper_macro_automatic_support_rate": sum(rates) / len(rates) if rates else None,
                           "pooled_automatic_support_rate": counts["supported"] / counts.total() if counts.total() else None,
                           "interpretation": "region_decisions_not_unique_paper_recall; missing_papers_not_imputed"}
    report = seal({"schema_version": "fidelity-offline-batch/v1", "config_canonical_sha256": digest(config),
                   "historical_integrity": integrity, "rows": rows, "arms": arms,
                   "formal_precision": None, "formal_recall": None,
                   "limitations": ["no_independent_semantic_gold", "cross_parser_assigned_regions_differ_in_historical_pilot",
                                    "rule_support_is_not_measured_accuracy", "alias_declarations_only_no_semantic_guessing",
                                    "unsupported_names_not_automatically_false_fields"]}, "report_sha256")
    write_new(output / "config.json", config)
    write_new(output / "report.json", report)
    _freeze(output)
    return report


def _freeze(output: Path) -> None:
    members = [{"path": p.relative_to(output).as_posix(), "file_bytes_sha256": file_digest(p)}
               for p in sorted(output.rglob("*")) if p.is_file()]
    write_new(output / "manifest.json", {"schema_version": "fidelity-release-files/v1", "files": members,
                                        "self_exclusion": "manifest.json"})


def verify_release(config: dict, root: Path, output: Path) -> dict:
    """Verify bytes AND rederive all reports, including a resealed report attack."""
    import tempfile
    manifest = read_json(output / "manifest.json")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file() and p != output / "manifest.json"}
    declared = [r["path"] for r in manifest["files"]]
    if len(declared) != len(set(declared)) or set(declared) != actual:
        raise ValueError("release_member_set_mismatch")
    for row in manifest["files"]:
        if file_digest(contained(output, row["path"])) != row["file_bytes_sha256"]:
            raise ValueError("release_bytes_changed:" + row["path"])
    # Temporary artifacts are created under the explicitly selected output parent.
    with tempfile.TemporaryDirectory(prefix="fidelity-verify-", dir=output.resolve().parent) as temp:
        replay = Path(temp) / "replay"
        if (output / "plan.json").is_file():
            prepare_probe(config, root, replay)
        else:
            replay_batch(config, root, replay)
        if read_json(replay / "manifest.json") != manifest:
            raise ValueError("release_derivation_mismatch")
    return {"byte_integrity": "pass", "derivation": "pass", "semantic_accuracy": None}


def prepare_probe(config: dict, root: Path, output: Path) -> dict:
    """Prepare four v11 conditions and an unchanged V10 contract reference.

    No dataset binding or catalog is read here. PDF pages are shared responsibility
    boundaries; historical V10 groups retain their original edge context.
    """
    pilot, workspace, integrity = _load(config, root)
    if output.exists():
        raise ValueError("immutable_output_already_exists")
    plan = read_json(workspace / "plan.json")
    task = extraction_v11.make_task()
    requests = []
    for condition in plan["conditions"]:
        for case in plan["cases"]:
            pid = case["paper_id"]
            package = read_json(workspace / f"conditions/{condition['condition_id']}/{pid}.json")
            errors = preprocessing.verify_condition(package, workspace)
            if errors:
                raise ValueError(errors)
            src = package["mineru_source"] if condition["mineru_enabled"] else package["baseline_source"]
            paper = read_json(contained(workspace, src["artifacts"]["input"]["path"]))
            index = make_disabled_index(paper)
            group = extraction_v11.plan_region(paper, [case["page"]])
            hints = preprocessing._hints(package, paper, group)
            profile = read_json(workspace / "profiles/qwen.json")
            # Preserve frozen model identity, actual parameters and decoding mode.
            request = {"source_profile": profile, "source_task": task,
                       "messages": extraction_v11.render_group(task, paper, index, group, auxiliary_hints=hints),
                       "response_schema": extraction_v11.response_schema(task, paper, index, group),
                       "parameters": copy.deepcopy(plan["extraction_parameters"]),
                       "diagnostic_binding": {"condition_id": condition["condition_id"], "case_id": case["case_id"],
                                              "group": group, "package_sha256": package["package_sha256"],
                                              "auxiliary_hints_canonical_sha256": digest(hints),
                                              "purpose": "prepared_not_executed_fidelity_probe"}}
            name = f"requests/{len(requests)+1:04d}.json"
            write_new(output / name, request)
            requests.append({"path": name, "request_canonical_sha256": digest(request),
                             "condition_id": condition["condition_id"], "case_id": case["case_id"],
                             "paper_id": pid, "group": group})
    # Original contract and grouping are preserved. Evaluate only target-page
    # claims after assembly; neighboring windows remain declared edge context.
    for case in plan["cases"]:
        pid = case["paper_id"]
        paper = read_json(workspace / f"baseline/{pid}/input.json")
        index, old_task = make_disabled_index(paper), extraction_v10.make_task()
        windows = extraction_v10.window_catalog(paper)
        for group in extraction_v10.plan_groups(paper):
            target = [w for w in group["window_ids"] if windows[w]["page"] == case["page"]]
            if not target:
                continue
            request = {"source_profile": read_json(workspace / "profiles/qwen.json"), "source_task": old_task,
                       "messages": extraction_v10.render_group(old_task, paper, index, group),
                       "response_schema": extraction_v10.response_schema(old_task, paper, index, group),
                       "parameters": copy.deepcopy(plan["extraction_parameters"]),
                       "diagnostic_binding": {"condition_id": "V10_reference", "case_id": case["case_id"],
                                              "group": group, "scored_window_ids": target,
                                              "purpose": "prepared_not_executed_common_page_reference"}}
            name = f"requests/{len(requests)+1:04d}.json"
            write_new(output / name, request)
            requests.append({"path": name, "request_canonical_sha256": digest(request), "condition_id": "V10_reference",
                             "case_id": case["case_id"], "paper_id": pid, "group": group})
    result = seal({"schema_version": "fidelity-probe-plan/v1", "requests": requests,
                   "historical_integrity": integrity, "status": "prepared_not_executed",
                   "semantic_gold_status": "not_established", "common_responsibility": "same_pdf_pages",
                   "repetitions": 1, "sampling": False, "scope": "engineering_probe_not_confirmatory",
                   "dataset_content_in_requests": False}, "plan_sha256")
    write_new(output / "plan.json", result)
    _freeze(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("catalog", "verify-catalog", "replay", "prepare-probe", "verify-release"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command == "catalog":
        write_new(args.output, build_catalog(config, args.root))
    elif args.command == "verify-catalog":
        errors = verify_catalog(read_json(args.output), config, args.root)
        if errors:
            raise SystemExit(str(errors))
        print("catalog byte identities and source derivation verified")
    elif args.command == "replay":
        report = replay_batch(config, args.root, args.output)
        print({"runs": len(report["rows"]), "failed": sum(r["status"] == "failed" for r in report["rows"])})
    elif args.command == "verify-release":
        print(verify_release(config, args.root, args.output))
    else:
        result = prepare_probe(config, args.root, args.output)
        print({"requests": len(result["requests"]), "status": result["status"]})


if __name__ == "__main__":
    main()
