"""Replay V11/V10 admissions and assemble diagnostics on identical PDF pages.

This is a local, downstream analysis. Dataset catalogs are never read by the
paper request builder or GPU runner. Semantic precision/recall remain missing.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from high_fidelity_schema_study.four_category import backends, extraction_v10, extraction_v11
from high_fidelity_schema_study.four_category.common import ROOT, digest, file_digest, read_json, write_new
from high_fidelity_schema_study.four_category.disabled_navigation import make_disabled_index
from high_fidelity_schema_study.four_category.fidelity import evaluate


def merge_page(payloads, targets, all_windows=None):
    """Keep malformed attempts; exclude only claims provably outside this page.

    Rebase subject IDs across original groups instead of concatenating indices.
    Evidence validity and object deduplication are delegated to the evaluator.
    """
    merged = {"mentions": [], "facts": []}
    all_windows = targets if all_windows is None else set(all_windows)
    excluded = {"mentions": 0, "facts": 0}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        mentions = payload.get("mentions") if isinstance(payload.get("mentions"), list) else []
        facts = payload.get("facts") if isinstance(payload.get("facts"), list) else []
        scored_facts = []
        for fact in facts:
            wid = fact.get("primary_window") if isinstance(fact, dict) else None
            if type(wid) is int and wid in all_windows and wid not in targets:
                excluded["facts"] += 1
            else:
                scored_facts.append(fact)
        subjects = {f["subject_mention"] for f in scored_facts if isinstance(f, dict)
                    and type(f.get("subject_mention")) is int and 0 <= f["subject_mention"] < len(mentions)}
        mapping = {}
        for old, mention in enumerate(mentions):
            refs = mention.get("source_windows") if isinstance(mention, dict) else None
            # Empty/malformed references are invalid attempts, not exclusions.
            outside = bool(isinstance(refs, list) and refs and all(type(w) is int and w in all_windows for w in refs)
                           and not set(refs) & targets)
            if outside and old not in subjects:
                excluded["mentions"] += 1
                continue
            mapping[old] = len(merged["mentions"])
            merged["mentions"].append(copy.deepcopy(mention))
        for fact in scored_facts:
            fact = copy.deepcopy(fact)
            if isinstance(fact, dict):
                old = fact.get("subject_mention")
                fact["subject_mention"] = mapping.get(old) if type(old) is int else None
            merged["facts"].append(fact)
    return merged, excluded


def analyze(collected, output, experiment=None):
    experiment = experiment or ROOT / "data/experiments/v11_gpu_2026_09_28_v1"
    frozen = ROOT / "data/experiments/fidelity_validation_2026_09_27_v1"
    workspace = ROOT / "data/experiments/mineru_structural_2026_09_27_v1/collected-v2/workspace"
    plan = read_json(frozen / "probe-inputs/plan.json")
    stage = read_json(experiment / "stage-manifest.json")
    pre = read_json(collected / "audit/cpu-preflight.json")
    assert pre["stage_manifest_file_bytes_sha256"] == file_digest(experiment / "stage-manifest.json")
    origins = {}
    for name in ("qualification", "comparison"):
        order = collected / "audit" / (name + "-order.json")
        if not order.exists():
            continue
        for index, ordinal in enumerate(read_json(order)["ordinals"], 1):
            if ordinal in origins:
                raise ValueError("duplicate_comparison_observation")
            origins[ordinal] = collected / "outputs" / name / "requests" / f"{index:04d}"
    pages, rows = defaultdict(list), []
    for ordinal, row in enumerate(plan["requests"], 1):
        saved_path = frozen / "probe-inputs" / row["path"]
        saved = read_json(saved_path)
        assert digest(saved) == row["request_canonical_sha256"]
        assert file_digest(saved_path) == stage["files"]["inputs/" + row["path"]]
        condition, pid = row["condition_id"], row["paper_id"]
        src = "baseline" if condition == "V10_reference" or condition.startswith("M0") else "mineru"
        paper = read_json(workspace / src / pid / "input.json")
        assert file_digest(workspace / src / pid / "input.json") == stage["files"][f"workspace/{src}/{pid}/input.json"]
        index = make_disabled_index(paper)
        record = {"ordinal": ordinal, "case_id": row["case_id"], "condition_id": condition,
                  "admission": "missing", "payload": None, "errors": []}
        folder = origins.get(ordinal)
        try:
            if folder is None or not (folder / "result.json").exists():
                raise ValueError("result_missing")
            result, identity = read_json(folder / "result.json"), read_json(folder / "identity.json")
            probe = folder.parent.parent
            checkpoint = read_json(probe / "checkpoint_verification.json")
            assert checkpoint["status"] == "pass"
            actual_code = read_json(probe / "executing_code_identity.json")
            assert digest(actual_code) == identity["executing_code_identity_sha256"] == pre["executing_source_canonical_sha256"]
            assert all(stage["files"]["source/high_fidelity_schema_study/"+name] == h for name, h in actual_code.items())
            assert read_json(folder / "saved_request.json") == saved
            assert result["identity_sha256"] == digest(identity)
            assert identity["source_request_file_sha256"] == file_digest(saved_path)
            assert identity["request_sha256"] == digest({k: saved[k] for k in ("messages", "parameters", "response_schema")})
            assert identity["candidate_profile_sha256"] == backends.profile_hash(saved["source_profile"])
            record.update(result_file_bytes_sha256=file_digest(folder / "result.json"),
                          generation_duration_s=result.get("generation_duration_s"), output_tokens=result.get("output_tokens"),
                          normal_stop=result.get("normal_stop"), peak_reserved_bytes=result.get("peak_reserved_bytes"))
            if result["status"] != "completed":
                record.update(admission="runtime_failed", errors=[result.get("error_type", result["status"])])
            else:
                assert result["request"] == backends.request_for(saved["source_profile"], saved["messages"], saved["parameters"], response_schema=saved["response_schema"])
                backend = backends._normalize({}, saved["source_profile"]["backend"], result["generation"], profile=saved["source_profile"])
                if condition == "V10_reference":
                    payload, _ = extraction_v10.parse_response(backend["raw_text"])
                    errors = extraction_v10.validate_response(payload, paper, index, row["group"], saved["source_task"])
                    record.update(payload=payload, errors=errors,
                                  admission="contract_invalid" if errors else "incomplete" if any(c["state"] == "overflow" for c in payload["coverage"]) else "success")
                else:
                    admitted = extraction_v11.admit_response(backend["raw_text"], paper, index, row["group"])
                    payload = admitted.get("materialized_v10_payload")
                    if payload is None and isinstance(admitted.get("payload"), dict):
                        # Materialize diagnostic quotes if safe, even for invalid
                        # admissions. No repair or promotion of the original label.
                        try:
                            payload = extraction_v11.to_v10(admitted["payload"], paper, row["group"])
                        except (ValueError, KeyError, TypeError):
                            pass
                    record.update(payload=payload, admission=admitted["status"], errors=admitted.get("errors", []))
                if not result.get("normal_stop") or result.get("truncated"):
                    record["admission"] = "incomplete"
        except (ValueError, KeyError, TypeError, AssertionError, OSError) as exc:
            record.update(admission="missing" if folder is None else "verification_failed", errors=[type(exc).__name__ + ":" + str(exc)[:300]])
        rows.append(record)
        pages[(condition, row["case_id"], pid, src)].append((row, record))
    summaries = []
    for (condition, case, pid, src), records in pages.items():
        paper = read_json(workspace / src / pid / "input.json")
        windows = extraction_v10.window_catalog(paper)
        if condition == "V10_reference":
            targets = {wid for row, _ in records for wid in read_json(frozen / "probe-inputs" / row["path"])["diagnostic_binding"]["scored_window_ids"]}
        else:
            targets = set(records[0][0]["group"]["window_ids"])
        assert len({windows[w]["page"] for w in targets}) == 1
        payload, excluded = merge_page([r["payload"] for _, r in records], targets, range(len(windows)))
        statuses = Counter(r["admission"] for _, r in records)
        status = "success" if set(statuses) == {"success"} else "incomplete_or_invalid"
        catalog = read_json(frozen / "replay-final/catalogs" / (pid + ".json"))
        report = evaluate(payload, paper, catalog, admission_status=status,
                          admission_errors=[e for _, r in records for e in r["errors"]], target_windows=sorted(targets))
        name = f"pages/{condition}-{case}.json"
        write_new(output / name, report)
        summaries.append({"condition": condition, "case_id": case, "paper_id": pid,
                          "pdf_page": windows[next(iter(targets))]["page"], "planned_requests": len(records),
                          "admissions": dict(statuses), "out_of_page_excluded": excluded,
                          "supported_object_ids_diagnostic": report["supported_object_ids_diagnostic"],
                          "accepted_object_ids": report["accepted_object_ids"], "decision_counts": report["decision_counts"],
                          "locator_precision": report["locator_precision"], "uncovered_candidate_cells": len(report["uncovered_feature_cells"]),
                          "generation_seconds": (sum(r["generation_duration_s"] for _, r in records)
                                                 if all(type(r.get("generation_duration_s")) in (int, float) for _, r in records) else None),
                          "observed_partial_generation_seconds": sum(r.get("generation_duration_s") or 0 for _, r in records),
                          "report_path": name, "report_sha256": report["report_sha256"]})
    result = {"schema_version": "v11-same-page-diagnostics/v1", "pages": summaries,
              "requests": [{k: v for k, v in r.items() if k != "payload"} for r in rows],
              "timing_scope": "sum of observed generation durations; preprocessing and cold loading excluded; missing runs are not zero-cost observations",
              "formal_precision": None, "formal_recall": None, "G_visible": None,
              "repetitions": 1, "scope": "development_probe_not_confirmatory"}
    write_new(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.collected, args.output)
    print({"pages": len(report["pages"]), "requests": len(report["requests"]), "semantic_accuracy": None})
