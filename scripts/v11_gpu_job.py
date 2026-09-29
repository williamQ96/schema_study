"""Frozen V11 qualification followed by the predeclared 69-request comparison.

Run inside the existing structured-output Apptainer image. Only this job's
audit/outputs/cache are writable; paper prompts never read dataset catalogs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from telegram_codex_bridge import atomic, read, sha


def verify_stage(job):
    for name, expected in read(job / "stage-manifest.json")["files"].items():
        if sha(job / name) != expected:
            raise ValueError("staged_bytes_changed:" + name)


def source(job, row):
    condition, pid = row["condition_id"], row["paper_id"]
    if condition == "V10_reference":
        return read(job / "workspace" / f"baseline/{pid}/input.json"), None
    package = read(job / "workspace" / f"conditions/{condition}/{pid}.json")
    selected = package["mineru_source"] if package["config"]["mineru_enabled"] else package["baseline_source"]
    return read(job / "workspace" / selected["artifacts"]["input"]["path"]), package


def select_qualification(rows):
    """Select before generation: maximum context, grammar bytes, and coverage."""
    eligible = [r for r in rows if r["condition_id"] != "V10_reference"]
    selected = []
    for key in ("input_tokens", "schema_bytes", "coverage_windows"):
        ordinal = max(eligible, key=lambda r: (r[key], -r["ordinal"]))["ordinal"]
        if ordinal not in selected:
            selected.append(ordinal)
    for row in sorted(eligible, key=lambda r: (-r["input_tokens"], r["ordinal"])):
        if len(selected) >= 3:
            break
        if row["ordinal"] not in selected:
            selected.append(row["ordinal"])
    return selected


def preflight(job):
    import torch
    from types import SimpleNamespace
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig
    from high_fidelity_schema_study.four_category import backends, structured_output, extraction_v10, extraction_v11, preprocessing
    from high_fidelity_schema_study.four_category.common import digest
    from high_fidelity_schema_study.four_category.disabled_navigation import make_disabled_index
    from high_fidelity_schema_study.four_category.inference_probe import validate_input, executing_code_identity
    assert torch.cuda.device_count() == 0
    torch.set_num_threads(2)
    verify_stage(job)
    plan = read(job / "inputs/plan.json")
    profile = read(job / "profile.json")
    tokenizer = AutoTokenizer.from_pretrained(profile["model_id"], revision=profile["revision"], local_files_only=True, trust_remote_code=False)
    conf = AutoConfig.from_pretrained(profile["model_id"], local_files_only=True, trust_remote_code=False)
    gen = GenerationConfig.from_pretrained(profile["model_id"], local_files_only=True)
    stub = SimpleNamespace(generation_config=gen, get_output_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(conf.get_text_config().vocab_size, 1))))
    rows = []
    for ordinal, row in enumerate(plan["requests"], 1):
        saved = read(job / "inputs" / row["path"])
        assert digest(saved) == row["request_canonical_sha256"]
        validate_input(saved, profile)
        paper, package = source(job, row)
        index, group, task = make_disabled_index(paper), row["group"], saved["source_task"]
        if package is not None:
            assert not preprocessing.verify_condition(package, job / "workspace")
            hints = preprocessing._hints(package, paper, group)
            messages = extraction_v11.render_group(task, paper, index, group, auxiliary_hints=hints)
            schema = extraction_v11.response_schema(task, paper, index, group)
        else:
            messages = extraction_v10.render_group(task, paper, index, group)
            schema = extraction_v10.response_schema(task, paper, index, group)
        assert saved["messages"] == messages and saved["response_schema"] == schema, "derived_input_mismatch"
        tokens = tokenizer.apply_chat_template(saved["messages"], tokenize=True, add_generation_prompt=True,
                                             return_dict=False, **profile["runtime"].get("chat_template_kwargs", {}))
        request = backends.request_for(profile, saved["messages"], saved["parameters"], response_schema=schema)
        structured_output.logits_processor(request, tokenizer, stub)
        assert len(tokens) + saved["parameters"]["max_output_tokens"] <= profile["context_window"]
        rows.append({"ordinal": ordinal, "condition_id": row["condition_id"], "case_id": row["case_id"],
                     "input_tokens": len(tokens), "schema_bytes": len(json.dumps(schema)),
                     "coverage_windows": len(group["window_ids"]), "grammar_compiled": True})
    result = {"status": "pass", "requests": rows, "generation_calls": 0, "device_count": 0,
              "qualification_ordinals": select_qualification(rows),
              "selection": "pre-generation maxima of input tokens, schema bytes and coverage; fill to three by input tokens",
              "executing_source_canonical_sha256": digest(executing_code_identity()),
              "profile_deployment_identity_is_historical": True,
              "stage_manifest_file_bytes_sha256": sha(job / "stage-manifest.json")}
    atomic(job / "audit/cpu-preflight.json", result)
    print(json.dumps({"status": "pass", "requests": len(rows), "qualification_ordinals": result["qualification_ordinals"],
                      "max_input_tokens": max(r["input_tokens"] for r in rows)}), flush=True)


def admission(job, row, saved, result):
    from high_fidelity_schema_study.four_category import backends, extraction_v10, extraction_v11
    from high_fidelity_schema_study.four_category.disabled_navigation import make_disabled_index
    paper, _ = source(job, row)
    if result.get("status") != "completed" or not result.get("normal_stop") or result.get("truncated"):
        return {"status": "runtime_or_truncation_failure", "errors": [result.get("error_type", "abnormal_stop")], "payload": None}
    backend = backends._normalize({}, saved["source_profile"]["backend"], result["generation"], profile=saved["source_profile"])
    if backend["status"] != "success":
        return {"status": backend["status"], "errors": [], "payload": None}
    if row["condition_id"] != "V10_reference":
        return extraction_v11.admit_response(backend["raw_text"], paper, make_disabled_index(paper), row["group"])
    try:
        payload, _ = extraction_v10.parse_response(backend["raw_text"])
        errors = extraction_v10.validate_response(payload, paper, make_disabled_index(paper), row["group"], saved["source_task"])
        incomplete = any(c["state"] == "overflow" for c in payload.get("coverage", []))
        return {"status": "contract_invalid" if errors else "incomplete" if incomplete else "success", "errors": errors, "payload": payload}
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "contract_invalid", "errors": [type(exc).__name__], "payload": None}


def qualification_gate(records, total_bytes, reserve_bytes=8 * 1024**3):
    errors = []
    for ordinal, result, admitted in records:
        if result.get("status") != "completed" or admitted["status"] != "success":
            errors.append(f"request_{ordinal}:" + admitted["status"])
        peaks = result.get("peak_reserved_bytes")
        if not isinstance(peaks, list) or len(peaks) != 1 or type(peaks[0]) not in (int, float):
            errors.append(f"request_{ordinal}:memory_metric_missing")
        elif peaks[0] > total_bytes - reserve_bytes:
            errors.append(f"request_{ordinal}:memory_reserve_below_8GiB")
    return {"status": "pass" if not errors else "failed", "errors": errors,
            "required_free_reserve_bytes": reserve_bytes, "gpu_total_bytes": total_bytes,
            "semantic_accuracy": None}


def execute(job):
    import torch
    from high_fidelity_schema_study.four_category.inference_probe import run_probe
    verify_stage(job)
    pre = read(job / "audit/cpu-preflight.json")
    assert pre["status"] == "pass" and pre["stage_manifest_file_bytes_sha256"] == sha(job / "stage-manifest.json")
    plan, runtime = read(job / "inputs/plan.json"), read(job / "runtime.json")
    selection = pre["qualification_ordinals"]
    assert len(selection) == 3 and len(set(selection)) == 3
    atomic(job / "audit/job-state.json", {"phase": "qualification", "time": time.time(), "ordinals": selection})
    def run(ordinals, name):
        paths = [job / "inputs" / plan["requests"][n - 1]["path"] for n in ordinals]
        atomic(job / "audit" / (name + "-order.json"), {"ordinals": ordinals})
        summary = run_probe(paths, job / "profile.json", job / "outputs" / name,
                            Path(runtime["leases"]), [runtime["gpu_id"]], allow_live=True)
        records = []
        for index, ordinal in enumerate(ordinals, 1):
            folder = job / "outputs" / name / "requests" / f"{index:04d}"
            result = read(folder / "result.json")
            admitted = admission(job, plan["requests"][ordinal-1], read(paths[index-1]), result)
            atomic(folder / "admission.json", admitted)
            records.append((ordinal, result, admitted))
        return summary, records
    try:
        summary, records = run(selection, "qualification")
        total = torch.cuda.get_device_properties(0).total_memory
        qualified = qualification_gate(records, total)
        qualified["ordinals"] = selection
        qualified["runtime_status"] = summary["status"]
        if summary["status"] != "completed":
            qualified.update(status="failed")
        atomic(job / "audit/qualification.json", qualified)
        if qualified["status"] != "pass":
            atomic(job / "audit/job-state.json", {"phase": "qualification_failed", "time": time.time()})
            return 2
        # The first three responses remain comparison observations: no rerun or
        # best-of selection after viewing their answers.
        remaining = [n for n in range(1, len(plan["requests"])+1) if n not in selection]
        atomic(job / "audit/job-state.json", {"phase": "comparison", "time": time.time(), "remaining": len(remaining)})
        summary, more = run(remaining, "comparison")
        atomic(job / "audit/execution-summary.json", {"status": summary["status"], "requests": len(records)+len(more),
                                                       "successful_admissions": sum(a["status"] == "success" for _, _, a in records+more),
                                                       "semantic_accuracy": None})
        atomic(job / "audit/job-state.json", {"phase": "comparison_completed" if summary["status"] == "completed" else "comparison_failed", "time": time.time()})
        return 0 if summary["status"] == "completed" else 3
    except Exception as exc:
        atomic(job / "audit/execution-error.json", {"error_type": type(exc).__name__, "error": str(exc)[:1500], "time": time.time()})
        prior = read(job / "audit/job-state.json")["phase"]
        atomic(job / "audit/job-state.json", {"phase": "qualification_failed" if prior == "qualification" else "comparison_failed", "time": time.time()})
        return 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "execute"])
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.job)
    else:
        raise SystemExit(execute(args.job))
