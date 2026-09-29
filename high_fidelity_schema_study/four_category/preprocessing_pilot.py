"""Portable, bounded 2x2 development workload; never launches the production queue."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import shutil

from . import extraction_v10 as extraction, structural_classifier as auxiliary
from .backends import _normalize, profile_hash, request_for
from .common import contained, digest, file_digest, read_json, seal, seal_errors, write_new
from .mineru_adapter import build_mineru_bundle
from .paper import prepare_paper, unit_catalog, verify_paper


def _normalized(value):
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()


def select_case(paper_input: dict, case: dict) -> dict:
    """Select before generation by PDF page and declared text anchor, not an answer."""
    windows = extraction.window_catalog(paper_input)
    originals = {u["unit_id"]: u for page in paper_input["pages"]
                 for u in page["text_regions"]}
    matches = [row for row in windows if row["page"] == case["page"]
               and _normalized(case["anchor"]) in _normalized(row["text"])]
    if not matches:
        raise ValueError("prespecified_case_anchor_not_found:" + case["case_id"])
    # Recovery regions remain in full input, but an original recognized match is
    # the prespecified focus when both representations contain the anchor.
    matches.sort(key=lambda row: ("recovery" in originals.get(row["unit_id"], {}).get("text_origin", ""),
                                  row["window_index"]))
    anchor = matches[0]
    group = next(g for g in extraction.plan_groups(paper_input) if anchor["window_index"] in g["window_ids"])
    return {"case": copy.deepcopy(case), "anchor_window": anchor, "group": group,
            "matching_windows": [r["window_index"] for r in matches]}


def _copy_source(source: dict, upstream: Path, target: Path, root: Path) -> dict:
    target.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, identity in source["artifacts"].items():
        old = contained(upstream, identity["path"])
        if file_digest(old) != identity["sha256"]:
            raise ValueError("upstream_source_bytes_changed:" + name)
        dest = target / (name + old.suffix)
        shutil.copyfile(old, dest)
        artifacts[name] = {**identity, "path": dest.relative_to(root).as_posix()}
    migrated = seal({**source, "artifacts": artifacts}, "source_sha256")
    errors = verify_paper(migrated, root)
    if errors:
        raise ValueError(errors)
    write_new(target / "source.json", migrated)
    return migrated


def _saved_request(profile, task, messages, schema, parameters, binding):
    return {"source_profile": copy.deepcopy(profile), "source_task": copy.deepcopy(task),
            "messages": messages, "response_schema": schema, "parameters": copy.deepcopy(parameters),
            "diagnostic_binding": copy.deepcopy(binding)}


def prepare(plan_path: Path, baseline_root: Path, parser_root: Path, profiles: Path, output: Path) -> dict:
    plan = read_json(plan_path)
    if (plan["paper_ids"] != ["P05", "P06", "P07"] or len(plan["cases"]) != 6
            or plan["repetitions"] != 1 or {c["condition_id"] for c in plan["conditions"]} != {"M0A0", "M1A0", "M0A1", "M1A1"}):
        raise ValueError("unsupported_pilot_load; use_a_new_version_for_another_design")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "plan.json", plan)
    corpus = read_json(baseline_root / "corpus.json")
    by_id = {row["paper_id"]: row["source"] for row in corpus["papers"]}
    profile_values = {slot: read_json(profiles / (slot + "-profile.json"))
                      for slot in {plan["primary_extraction_slot"], plan["auxiliary_slot"]}}
    for slot, profile in profile_values.items():
        write_new(output / "profiles" / (slot + ".json"), profile)
    task = auxiliary.make_task()
    cases, requests, sources = [], [], {}
    for pid in plan["paper_ids"]:
        baseline = _copy_source(by_id[pid], baseline_root, output / "baseline" / pid, output)
        native = read_json(contained(output, baseline["artifacts"]["input"]["path"]))
        parsed = parser_root / pid
        receipt = read_json(parsed / "receipt.json")
        if receipt["status"] != "success" or receipt["source_pdf_sha256"] != native["source_pdf_sha256"]:
            raise ValueError("parser_receipt_failed_or_wrong_pdf")
        # Preserve the entire raw SDK package, images, configuration and receipt.
        raw_target = output / "parser" / pid
        shutil.copytree(parsed, raw_target)
        for item in receipt["files"]:
            if file_digest(contained(raw_target, item["path"])) != item["file_bytes_sha256"]:
                raise ValueError("parser_receipt_member_mismatch:" + item["path"])
        raw, parser_id = read_json(raw_target / "middle.json"), read_json(raw_target / "parser_identity.json")
        layout, reading, paper = build_mineru_bundle(native, raw, parser_id, pdf_sha256=native["source_pdf_sha256"])
        derived = output / "mineru" / pid
        derived.mkdir(parents=True)
        write_new(derived / "layout.json", layout)
        write_new(derived / "input.json", paper)
        (derived / "reading_text.txt").write_text(reading, encoding="utf-8", newline="\n")
        new_source = prepare_paper(derived / "layout.json", derived / "reading_text.txt", derived / "input.json",
                                  contained(output, baseline["artifacts"]["pdf"]["path"]), root=output,
                                  mineru_raw_path=raw_target / "middle.json",
                                  baseline_input_path=contained(output, baseline["artifacts"]["input"]["path"]),
                                  parser_identity_path=raw_target / "parser_identity.json")
        write_new(derived / "source.json", new_source)
        sources[pid] = {"baseline": baseline, "mineru": new_source,
                        "upstream_source_sha256": by_id[pid]["source_sha256"]}
        for parser_name, source_input in (("baseline", native), ("mineru", paper)):
            selected = [select_case(source_input, case) for case in plan["cases"] if case["paper_id"] == pid]
            # Large nonselected tables stay in the full source. This explicit
            # grouping policy is a bound, not a claim that all groups fit a model.
            groups = auxiliary.plan_groups(source_input, {"max_units": 512, "max_target_chars": 60000})
            chosen = sorted({g["index"] for case in selected for g in groups
                             if case["anchor_window"]["unit_id"] in g["unit_ids"]})
            for case in selected:
                cases.append({"paper_id": pid, "parser": parser_name, **case})
            for gi in chosen:
                group = groups[gi]
                binding = {"paper_id": pid, "parser": parser_name, "group": group,
                           "paper_input_canonical_sha256": digest(source_input), "plan_sha256": digest(plan),
                           "purpose": "development_auxiliary_not_gold"}
                request = _saved_request(profile_values[plan["auxiliary_slot"]], task,
                                         auxiliary.render_group(task, source_input, group),
                                         auxiliary.response_schema(task, source_input, group),
                                         plan["auxiliary_parameters"], binding)
                ordinal = len(requests) + 1
                name = f"auxiliary-inputs/{ordinal:04d}.json"
                write_new(output / name, request)
                requests.append({"ordinal": ordinal, "path": name, "paper_id": pid, "parser": parser_name,
                                 "group_index": gi, "request_canonical_sha256": digest(request)})
    if len(requests) > plan["limits"]["max_auxiliary_calls"]:
        raise ValueError("auxiliary_call_budget_exceeded")
    manifest = seal({"schema_version": "mineru-structural-pilot-inputs/v1", "plan_sha256": digest(plan),
                     "sources": sources, "cases": cases, "auxiliary_requests": requests,
                     "semantic_accuracy": None, "selected_before_model_outputs": True}, "manifest_sha256")
    write_new(output / "input-manifest.json", manifest)
    return manifest


def paper_input(root, manifest, pid, parser_name):
    source = manifest["sources"][pid][parser_name]
    return read_json(contained(root, source["artifacts"]["input"]["path"]))


def _manifest(root):
    plan, manifest = read_json(root / "plan.json"), read_json(root / "input-manifest.json")
    if seal_errors(manifest, "manifest_sha256") or manifest["plan_sha256"] != digest(plan):
        raise ValueError("pilot_plan_or_manifest_identity_mismatch")
    if set(manifest["sources"]) != set(plan["paper_ids"]):
        raise ValueError("pilot_source_set_mismatch")
    expected_cases, expected_requests = [], []
    for pid in plan["paper_ids"]:
        for parser_name in ("baseline", "mineru"):
            source = manifest["sources"][pid][parser_name]
            errors = verify_paper(source, root)
            if errors:
                raise ValueError("pilot_source_invalid:" + str(errors[:3]))
            paper = paper_input(root, manifest, pid, parser_name)
            selected = [select_case(paper, case) for case in plan["cases"] if case["paper_id"] == pid]
            expected_cases.extend({"paper_id": pid, "parser": parser_name, **case} for case in selected)
            groups = auxiliary.plan_groups(paper, {"max_units": 512, "max_target_chars": 60000})
            indexes = sorted({g["index"] for case in selected for g in groups
                              if case["anchor_window"]["unit_id"] in g["unit_ids"]})
            for gi in indexes:
                ordinal = len(expected_requests) + 1
                expected_requests.append({"ordinal": ordinal, "path": f"auxiliary-inputs/{ordinal:04d}.json",
                                          "paper_id": pid, "parser": parser_name, "group_index": gi})
    if manifest["cases"] != expected_cases:
        raise ValueError("pilot_case_selection_derivation_mismatch")
    actual = [{k: r.get(k) for k in ("ordinal", "path", "paper_id", "parser", "group_index")}
              for r in manifest["auxiliary_requests"]]
    if actual != expected_requests:
        raise ValueError("pilot_auxiliary_request_coverage_mismatch")
    return plan, manifest


def _probe_result(probe_root: Path, ordinal: int, saved: dict):
    directory = probe_root / "requests" / f"{ordinal:04d}"
    actual_saved = read_json(directory / "saved_request.json")
    identity, result = read_json(directory / "identity.json"), read_json(directory / "result.json")
    if actual_saved != saved or result["identity_sha256"] != digest(identity):
        raise ValueError("probe_request_or_identity_mismatch")
    if identity["request_sha256"] != digest({"messages": saved["messages"], "parameters": saved["parameters"],
                                            "response_schema": saved["response_schema"]}):
        raise ValueError("probe_rendered_request_mismatch")
    if identity["candidate_profile_sha256"] != profile_hash(saved["source_profile"]):
        raise ValueError("probe_profile_mismatch")
    if result["status"] == "completed":
        if result["request"] != request_for(saved["source_profile"], saved["messages"], saved["parameters"],
                                            response_schema=saved["response_schema"]):
            raise ValueError("probe_actual_request_mismatch")
        try:
            backend = _normalize({}, saved["source_profile"]["backend"], result["generation"], profile=saved["source_profile"])
        except (KeyError, ValueError, TypeError) as exc:
            backend = {"status": "unavailable", "raw_text": None,
                       "error": type(exc).__name__ + ":" + str(exc)[:500]}
        if backend["status"] == "success" and (not isinstance(backend.get("raw_text"), str)
                                                or not backend["raw_text"].strip()):
            backend = {**backend, "status": "unavailable", "error": "missing_or_empty_response_text"}
    else:
        backend = {"status": "transport_error", "raw_text": None}
    return result, backend


def import_auxiliary(root: Path, probe_root: Path) -> dict:
    from . import preprocessing
    plan, manifest = _manifest(root)
    task = auxiliary.make_task()
    profile = read_json(root / "profiles" / (plan["auxiliary_slot"] + ".json"))
    bundles, outcomes = {}, []
    for pid in plan["paper_ids"]:
        for parser_name in ("baseline", "mineru"):
            paper = paper_input(root, manifest, pid, parser_name)
            selected = [r for r in manifest["auxiliary_requests"] if r["paper_id"] == pid and r["parser"] == parser_name]
            records = []
            for row in selected:
                saved = read_json(root / row["path"])
                group = saved["diagnostic_binding"]["group"]
                if (saved["messages"] != auxiliary.render_group(task, paper, group)
                        or saved["response_schema"] != auxiliary.response_schema(task, paper, group)
                        or saved["source_task"] != task or saved["source_profile"] != profile
                        or saved["parameters"] != plan["auxiliary_parameters"]
                        or digest(saved) != row["request_canonical_sha256"]):
                    raise ValueError("auxiliary_request_derivation_mismatch")
                result, backend = _probe_result(probe_root, row["ordinal"], saved)
                record_status = backend["status"] if backend["status"] in {
                    "success", "transport_error", "truncated", "invalid_request", "unavailable"} else "unavailable"
                records.append(auxiliary.make_record(backend.get("raw_text"), paper, task, profile, group,
                                                      backend_status=record_status))
                outcomes.append({**row, "backend_status": backend["status"],
                                 "output_tokens": result.get("output_tokens"), "duration_s": result.get("generation_duration_s")})
            bundle = auxiliary.build_annotations(paper, task, profile, records,
                                                  selected_groups=[r["group_index"] for r in selected])
            bundles[(pid, parser_name)] = bundle
            write_new(root / "annotations" / f"{pid}-{parser_name}.json", bundle)
    primary = read_json(root / "profiles" / (plan["primary_extraction_slot"] + ".json"))
    extraction_task, requests = preprocessing.make_extraction_task(), []
    for condition in plan["conditions"]:
        parser_name = "mineru" if condition["mineru_enabled"] else "baseline"
        for pid in plan["paper_ids"]:
            source = manifest["sources"][pid]
            identity = read_json(root / "parser" / pid / "parser_identity.json")
            parser_profile = {k: identity[k] for k in preprocessing.PARSER_PROFILE_KEYS} if condition["mineru_enabled"] else None
            config = preprocessing.make_config(mineru_enabled=condition["mineru_enabled"],
                auxiliary_enabled=condition["auxiliary_enabled"], parser_profile=parser_profile,
                auxiliary_profile=profile if condition["auxiliary_enabled"] else None)
            package = preprocessing.prepare_condition(root, source["baseline"], config,
                mineru_source=source["mineru"] if condition["mineru_enabled"] else None,
                annotations=bundles[(pid, parser_name)] if condition["auxiliary_enabled"] else None)
            package_name = f"conditions/{condition['condition_id']}/{pid}.json"
            write_new(root / package_name, package)
            for selected in manifest["cases"]:
                if selected["paper_id"] != pid or selected["parser"] != parser_name:
                    continue
                group = selected["group"]
                binding = {"condition_id": condition["condition_id"], "case_id": selected["case"]["case_id"],
                           "package_path": package_name, "package_canonical_sha256": digest(package),
                           "plan_sha256": digest(plan), "group": group,
                           "purpose": "bounded_development_not_confirmatory_accuracy"}
                saved = _saved_request(primary, extraction_task,
                    preprocessing.render_group(package, root, group, task=extraction_task),
                    preprocessing.response_schema(package, root, group, task=extraction_task),
                    plan["extraction_parameters"], binding)
                ordinal, name = len(requests) + 1, f"extraction-inputs/{len(requests)+1:04d}.json"
                write_new(root / name, saved)
                requests.append({"ordinal": ordinal, "path": name, **binding,
                                 "request_canonical_sha256": digest(saved)})
    if len(requests) != 24 or len(requests) > plan["limits"]["max_extraction_calls"]:
        raise ValueError("extraction_call_budget_mismatch")
    output = seal({"schema_version": "mineru-structural-extraction-plan/v1", "requests": requests,
              "auxiliary_outcomes": outcomes, "plan_sha256": digest(plan), "task_sha256": extraction_task["task_sha256"]}
              , "extraction_plan_sha256")
    write_new(root / "extraction-plan.json", output)
    return output


def analyze(root: Path, probe_root: Path, output: Path) -> dict:
    from . import preprocessing
    plan = read_json(root / "extraction-plan.json")
    source_plan, source_manifest = _manifest(root)
    if seal_errors(plan, "extraction_plan_sha256") or plan["plan_sha256"] != digest(source_plan):
        raise ValueError("extraction_plan_identity_mismatch")
    expected_cases = [(condition["condition_id"], case["case_id"]) for condition in source_plan["conditions"]
                      for case in source_plan["cases"]]
    if [(r["condition_id"], r["case_id"]) for r in plan["requests"]] != expected_cases:
        raise ValueError("extraction_condition_case_coverage_mismatch")
    rows = []
    primary = read_json(root / "profiles" / (source_plan["primary_extraction_slot"] + ".json"))
    for ordinal, row in enumerate(plan["requests"], 1):
        if row["ordinal"] != ordinal or row["path"] != f"extraction-inputs/{ordinal:04d}.json":
            raise ValueError("extraction_request_order_mismatch")
        saved, package = read_json(root / row["path"]), read_json(root / row["package_path"])
        if digest(saved) != row["request_canonical_sha256"] or digest(package) != row["package_canonical_sha256"]:
            raise ValueError("frozen_request_or_package_changed")
        group = row["group"]
        condition = next(c for c in source_plan["conditions"] if c["condition_id"] == row["condition_id"])
        parser_name = "mineru" if condition["mineru_enabled"] else "baseline"
        selected = next(c for c in source_manifest["cases"] if c["case"]["case_id"] == row["case_id"] and c["parser"] == parser_name)
        if (group != selected["group"] or saved["source_profile"] != primary
                or saved["parameters"] != source_plan["extraction_parameters"]
                or any(package["config"][key] != condition[key] for key in ("mineru_enabled", "auxiliary_enabled"))):
            raise ValueError("extraction_condition_group_or_parameters_mismatch")
        if saved["messages"] != preprocessing.render_group(package, root, group, task=saved["source_task"]):
            raise ValueError("request_prompt_replay_mismatch")
        if saved["response_schema"] != preprocessing.response_schema(package, root, group, task=saved["source_task"]):
            raise ValueError("request_schema_replay_mismatch")
        result, backend = _probe_result(probe_root, row["ordinal"], saved)
        admission = preprocessing.admit_response(backend["raw_text"], package, root, group, task=saved["source_task"]) if backend["status"] == "success" else {"status": backend["status"], "errors": []}
        rows.append({"case_id": row["case_id"], "condition_id": row["condition_id"],
                     "status": result["status"], "admission": admission,
                     **{k: result.get(k) for k in ("output_tokens", "generation_duration_s", "time_to_first_token_s", "truncated", "normal_stop")},
                     "semantic_accuracy": None, "result_file_bytes_sha256": file_digest(probe_root / "requests" / f"{row['ordinal']:04d}" / "result.json")})
    summary = {"schema_version": "mineru-structural-development-analysis/v1", "rows": rows,
               "scope": "Selected cases, one extraction model and one seed; no independent gold or population accuracy estimate",
               "plan_sha256": plan["plan_sha256"], "semantic_accuracy": None}
    write_new(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    for name in ("plan", "baseline-root", "parser-root", "profiles", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p = subs.add_parser("import-auxiliary")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--probe-root", type=Path, required=True)
    p = subs.add_parser("analyze")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--probe-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.plan, args.baseline_root, args.parser_root, args.profiles, args.output)
        print({"auxiliary_requests": len(result["auxiliary_requests"]), "case_selections": len(result["cases"])})
    elif args.command == "import-auxiliary":
        result = import_auxiliary(args.root, args.probe_root)
        print({"extraction_requests": len(result["requests"]), "auxiliary_results": len(result["auxiliary_outcomes"])})
    else:
        result = analyze(args.root, args.probe_root, args.output)
        print({"rows": len(result["rows"]), "semantic_accuracy": None})


if __name__ == "__main__":
    main()
