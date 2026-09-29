"""Offline, immutable experiment planning and qualification for Mercury scheduler V2.

This module never loads a model. The optional runner accepts explicitly marked
mock adapters only; it cannot run or qualify a service backend.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time

from . import VERSION
from .io import read, write_once


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _blocked(reason):
    return {"status": "blocked", "reason": reason}


_IDENTITY_FIELDS = ("profile_id", "engine", "image", "image_sha256", "revision",
                    "checkpoint_identity", "model_id", "tokenizer_id")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _profile_errors(profile, *, backend=None):
    errors = ["profile_missing:" + key for key in _IDENTITY_FIELDS if not isinstance(profile.get(key), str) or not profile[key].strip()]
    for key in _IDENTITY_FIELDS:
        value = profile.get(key)
        if isinstance(value, str) and re.search(r"REPLACE|TODO|TBD|UNSELECTED|UNKNOWN", value, re.I):
            errors.append("profile_placeholder:" + key)
    effective_backend = backend or profile.get("backend")
    if effective_backend not in {"service", "mock", "transformers"}:
        errors.append("unsupported_backend")
    if effective_backend != "mock":
        if not _SHA256.fullmatch(str(profile.get("image_sha256", ""))):
            errors.append("image_sha256_must_be_immutable_digest")
        if not _SHA256.fullmatch(str(profile.get("checkpoint_identity", ""))):
            errors.append("checkpoint_identity_must_be_sha256")
        if not _COMMIT.fullmatch(str(profile.get("revision", ""))):
            errors.append("revision_must_be_full_commit")
        if not re.search(r"@[0-9a-f]{40,64}$", str(profile.get("tokenizer_id", ""))):
            errors.append("tokenizer_id_must_include_full_revision")
        if not re.search(r"@sha256:[0-9a-f]{64}$", str(profile.get("image", ""))):
            errors.append("image_must_use_sha256_reference")
        elif not str(profile.get("image", "")).endswith("@sha256:" + str(profile.get("image_sha256", ""))):
            errors.append("image_reference_digest_mismatch")
    return errors


def _condition_id(profile, factors):
    return digest({"profile_sha256": digest(profile), "factors": factors})[:24]


def _valid_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema_version") != "mercury-scheduler-v2-qualification/v2":
        return ["unsupported_plan_schema"]
    candidate = dict(plan)
    claimed = candidate.pop("plan_id", None)
    errors = []
    if claimed != digest(candidate):
        errors.append("plan_identity_mismatch")
    request_ids = [r.get("request_id") for r in plan.get("frozen_requests", []) if isinstance(r, dict)]
    if len(request_ids) != len(plan.get("frozen_requests", [])) or len(request_ids) != len(set(request_ids)):
        errors.append("plan_request_ids_invalid_or_duplicate")
    seen = set()
    for row in plan.get("conditions", []):
        key = _condition_id(row.get("profile", {}), row.get("factors", {}))
        if (row.get("condition_id") != key or row.get("profile_sha256") != digest(row.get("profile", {}))
                or row.get("status") not in {"planned", "blocked"} or key in seen):
            errors.append("plan_condition_identity_invalid_or_duplicate")
        seen.add(key)
    baseline = plan.get("baseline", {})
    if baseline.get("profile_sha256") != digest(baseline.get("profile", {})):
        errors.append("plan_baseline_identity_invalid")
    return errors


def build_plan(config):
    """Build a versioned factorial matrix; every unsupported cell remains visible."""
    models = config.get("models", [])
    requests = config.get("frozen_requests", [])
    factors = config.get("factors", {})
    concurrency = factors.get("concurrency", [1, 2, 4])
    prefixes = factors.get("prefix_cache", [False, True])
    gpus = factors.get("gpu_count", [1, 2])
    if not models or not requests:
        raise ValueError("models and frozen_requests must be nonempty")
    required_request_fields = {"request_id", "input", "task", "messages", "seed", "generation_parameters",
                               "longest_context", "dense_table", "prior_failure_tags"}
    for req in requests:
        missing = required_request_fields - set(req)
        if missing:
            raise ValueError(f"frozen request {req.get('request_id')} missing {sorted(missing)}")
        if not isinstance(req["generation_parameters"], dict) or not isinstance(req["messages"], list):
            raise ValueError(f"frozen request {req.get('request_id')} has invalid messages or generation_parameters")
        if req["generation_parameters"].get("seed") != req["seed"]:
            raise ValueError(f"frozen request {req.get('request_id')} seed must match generation_parameters.seed")
        if not set(config.get("required_parameters", [])).issubset(req["generation_parameters"]):
            raise ValueError(f"frozen request {req.get('request_id')} is missing required generation parameters")
    request_ids = [r["request_id"] for r in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("frozen request IDs must be unique")
    if any(not isinstance(req["request_id"], str) or not req["request_id"].strip() for req in requests):
        raise ValueError("frozen request IDs must be nonempty strings")
    if not any(r["longest_context"] for r in requests) or not any(r["dense_table"] for r in requests) or not any(r["prior_failure_tags"] for r in requests):
        raise ValueError("frozen_requests must cover longest context, dense table, and prior failure tags")

    rows = []
    profile_ids = [m.get("profile_id") for m in models]
    if any(not isinstance(pid, str) or not pid for pid in profile_ids) or len(profile_ids) != len(set(profile_ids)):
        raise ValueError("model profile_id values must be unique and nonempty")
    for model in models:
        profile_errors = _profile_errors(model)
        missing = profile_errors + ["profile_missing:" + k for k in ("supports_seed", "supports_grammar", "supported_parameters") if k not in model]
        if not isinstance(model.get("supports_seed"), bool):
            missing.append("supports_seed_must_be_boolean")
        if not isinstance(model.get("supports_grammar"), bool):
            missing.append("supports_grammar_must_be_boolean")
        declared_parameters = model.get("supported_parameters", [])
        if not isinstance(declared_parameters, list) or any(not isinstance(k, str) for k in declared_parameters):
            missing.append("supported_parameters_must_be_string_list")
            declared_parameters = []
        request_parameters = {k for req in requests for k in req["generation_parameters"]}
        if not request_parameters.issubset(set(declared_parameters)):
            missing.append("unsupported_frozen_request_parameters:" + ",".join(sorted(request_parameters - set(declared_parameters))))
        for c in concurrency:
            for cache in prefixes:
                for gpu in gpus:
                    factors_selected = {"concurrency": c, "prefix_cache": cache, "gpu_count": gpu}
                    status = "planned"
                    reason = None
                    if missing:
                        status, reason = "blocked", "profile_missing:" + ",".join(missing)
                    elif model.get("backend") == "transformers" and c > 1:
                        status, reason = "blocked", "legacy_transformers_parallel_execution_prohibited"
                    elif c not in (1, 2, 4) or gpu not in (1, 2) or not isinstance(cache, bool):
                        status, reason = "blocked", "unsupported_factor_level"
                    required = set(config.get("required_parameters", []))
                    if not required.issubset(set(model.get("supported_parameters", []))):
                        status, reason = "blocked", "unsupported_required_parameters:" + ",".join(sorted(required - set(model.get("supported_parameters", []))))
                    if config.get("grammar_required") and not model.get("supports_grammar", False):
                        status, reason = "blocked", "grammar_unsupported"
                    if config.get("seed_required", True) and not model.get("supports_seed", False):
                        status, reason = "blocked", "seed_unsupported"
                    cid = _condition_id(model, factors_selected)
                    rows.append({"condition_id": cid, "profile_sha256": digest(model),
                                 "model_id": model.get("model_id"), "profile": copy.deepcopy(model),
                                 "factors": factors_selected, "status": status, "reason": reason})

    baseline_profile = config.get("baseline_profile", {})
    baseline = {"kind": "baseline", "backend": "existing_single_request_transformers", "concurrency": 1,
                "prefix_cache": False, "gpu_count": 1, "profile": copy.deepcopy(baseline_profile),
                "profile_sha256": digest(baseline_profile), "profile_errors": _profile_errors(baseline_profile, backend="transformers")}
    comparisons = [{"kind": "single_factor", "factor": factor, "from": baseline,
                    "to": {**baseline, factor: value}}
                   for factor, values in (("concurrency", concurrency), ("prefix_cache", prefixes), ("gpu_count", gpus))
                   for value in values]
    combos = [{"kind": "combination", "factors": row["factors"], "condition_id": row["condition_id"],
               "profile_id": row["profile"].get("profile_id")}
              for row in rows]
    plan = {"schema_version": "mercury-scheduler-v2-qualification/v2", "scheduler_version": VERSION,
            "created_at": config.get("created_at", "UNSET"), "config_sha256": digest(config),
            "baseline": baseline, "frozen_requests": copy.deepcopy(requests), "conditions": rows,
            "comparisons": comparisons, "combinations": combos,
            "guardrails": {"minimum_free_vram_gib": 12, "duration_includes_model_loading_and_validation": True,
                           "metric": "admitted_units_per_hour", "accuracy_claim": False}}
    plan["plan_id"] = digest(plan)
    return plan


def _metric_errors(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return [name + "_missing_or_nonfinite"]
    if value < 0 or (not allow_zero and value == 0):
        return [name + "_out_of_range"]
    return []


def _record_errors(record, req, condition_id, profile):
    errors = []
    if not isinstance(record, dict):
        return ["response_not_object"]
    required = ("request_id", "status", "input", "task", "messages", "seed", "generation_parameters",
                "condition_id", "profile_id", "profile_sha256", "free_vram_gib", "duration_seconds")
    errors.extend("record_missing:" + key for key in required if key not in record)
    if record.get("request_id") != req["request_id"]:
        errors.append("request_id_mismatch_or_missing")
    if record.get("status") != "success":
        errors.append("request_failed_or_invalid")
    for flag in ("oom", "truncated", "contract_regression", "misrouted", "provenance_error"):
        if type(record.get(flag)) is not bool:
            errors.append("guardrail_not_measured:" + flag)
        elif record[flag]:
            errors.append(flag)
    errors.extend(_metric_errors(record.get("free_vram_gib"), "free_vram_gib"))
    if isinstance(record.get("free_vram_gib"), (int, float)) and not isinstance(record.get("free_vram_gib"), bool) and math.isfinite(record["free_vram_gib"]) and record["free_vram_gib"] < 12:
        errors.append("minimum_12_gib_reserve_violated")
    errors.extend(_metric_errors(record.get("duration_seconds"), "duration_seconds"))
    for key in ("input", "task", "messages", "seed", "generation_parameters"):
        if record.get(key) != req[key]:
            errors.append("request_parity_mismatch:" + key)
    if record.get("condition_id") != condition_id:
        errors.append("condition_mismatch")
    if record.get("profile_id") != profile.get("profile_id") or record.get("profile_sha256") != digest(profile):
        errors.append("profile_identity_mismatch")
    return errors


def qualify(plan, records):
    """Judge supplied per-request records. Absence is unqualified, never success."""
    plan_errors = _valid_plan(plan)
    if plan_errors:
        return {"schema_version": "mercury-scheduler-v2-qualification-report/v2", "plan_id": plan.get("plan_id") if isinstance(plan, dict) else None,
                "plan_errors": plan_errors, "conditions": [], "all_conditions_qualified": False}
    record_input_errors = []
    if isinstance(records, dict) and records.get("schema_version") == "mercury-scheduler-v2-mock-records/v1":
        if records.get("plan_id") != plan.get("plan_id"):
            record_input_errors.append("record_plan_identity_mismatch")
        condition_records = records.get("conditions", {})
        records = {**(condition_records if isinstance(condition_records, dict) else {}),
                   "baseline": records.get("baseline") or {}}
    baseline_block = records.get("baseline", {}) if isinstance(records, dict) else {}
    baseline = baseline_block.get("requests", {}) if isinstance(baseline_block, dict) else {}
    base_profile = plan["baseline"].get("profile", {})
    baseline_errors = list(plan["baseline"].get("profile_errors", [])) + record_input_errors
    if not isinstance(baseline_block, dict):
        baseline_errors.append("baseline_record_block_invalid")
        baseline_block = {}
    if not baseline_errors:
        for req in plan["frozen_requests"]:
            if req["request_id"] not in baseline:
                baseline_errors.append("baseline_missing_request:" + req["request_id"])
            else:
                baseline_errors.extend(req["request_id"] + ":" + e for e in _record_errors(
                    baseline[req["request_id"]], req, "baseline", base_profile))
    baseline_errors.extend(_metric_errors(baseline_block.get("duration_seconds_including_model_loading_and_validation"), "baseline_duration_seconds_including_model_loading_and_validation"))
    baseline_errors.extend(_metric_errors(baseline_block.get("admitted_units_per_hour"), "baseline_admitted_units_per_hour", allow_zero=True))
    outcomes = []
    for row in plan["conditions"]:
        key = row["condition_id"]
        if row["status"] == "blocked":
            outcomes.append({"condition_id": key, "status": "blocked", "reason": row["reason"]})
            continue
        condition_block = records.get(key, {}) if isinstance(records, dict) else {}
        if not isinstance(condition_block, dict):
            condition_block = {}
        supplied = condition_block.get("requests", {})
        if not isinstance(supplied, dict):
            supplied = {}
        missing = [r["request_id"] for r in plan["frozen_requests"] if r["request_id"] not in supplied]
        errors = {r["request_id"]: _record_errors(supplied[r["request_id"]], r, key, row["profile"])
                  for r in plan["frozen_requests"] if r["request_id"] in supplied}
        errors = {k: v for k, v in errors.items() if v}
        extra = sorted(set(supplied) - {r["request_id"] for r in plan["frozen_requests"]})
        errors_block = _metric_errors(condition_block.get("duration_seconds_including_model_loading_and_validation"), "duration_seconds_including_model_loading_and_validation")
        errors_block.extend(_metric_errors(condition_block.get("admitted_units_per_hour"), "admitted_units_per_hour", allow_zero=True))
        status = "qualified" if not missing and not errors and not errors_block and not extra and not baseline_errors else "unqualified"
        if row["profile"].get("backend") == "mock":
            status = "mock_only"
        outcomes.append({"condition_id": key, "status": status, "missing_requests": missing,
                         "baseline_errors": baseline_errors, "extra_request_ids": extra,
                         "errors": errors, "condition_errors": errors_block,
                         "qualification_note": "mock_profiles_cannot_qualify_real_service_performance" if status == "mock_only" else None,
                         "admitted_units_per_hour": condition_block.get("admitted_units_per_hour"),
                         "baseline_admitted_units_per_hour": baseline_block.get("admitted_units_per_hour")})
    return {"schema_version": "mercury-scheduler-v2-qualification-report/v2", "plan_id": plan["plan_id"],
            "conditions": outcomes, "all_conditions_qualified": bool(outcomes) and all(x["status"] == "qualified" for x in outcomes)}


def run_mock(plan, adapter, timeout_s=30):
    """Run mock profiles only; one isolated bounded session per condition cell.

    Adapter call signature is ``adapter(request, profile, factors, condition_id)``.
    Stateful adapters may provide ``open_session(profile, factors, condition_id)``
    and ``invoke(request, profile, factors, condition_id, session)``. Calls that
    exceed timeout are recorded as timed out; executor shutdown waits for running
    mock callbacks before the next cell so callbacks cannot overlap conditions.
    """
    errors = _valid_plan(plan)
    if errors:
        raise ValueError("invalid_plan:" + ",".join(errors))
    if any(row["profile"].get("backend") != "mock" for row in plan["conditions"]):
        raise ValueError("run_mock_accepts_mock_profiles_only")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s_must_be_positive_finite")
    results = {}
    eligible = [r for r in plan["conditions"] if r["status"] == "planned"]
    for row in eligible:
        c = row["factors"]["concurrency"]
        cid = row["condition_id"]
        started = time.monotonic()
        records = {}
        session = adapter.open_session(copy.deepcopy(row["profile"]), copy.deepcopy(row["factors"]), cid) if hasattr(adapter, "open_session") else None
        def invoke(req):
            if hasattr(adapter, "invoke"):
                return adapter.invoke(copy.deepcopy(req), copy.deepcopy(row["profile"]),
                                      copy.deepcopy(row["factors"]), cid, session)
            return adapter(copy.deepcopy(req), copy.deepcopy(row["profile"]),
                           copy.deepcopy(row["factors"]), cid)
        pool = ThreadPoolExecutor(max_workers=c)
        futures = {pool.submit(invoke, req): req for req in plan["frozen_requests"]}
        timed_out = set()
        try:
            for future in as_completed(futures, timeout=timeout_s):
                req = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {"request_id": req["request_id"], "status": "error", "error": type(exc).__name__}
                records[req["request_id"]] = record
        except TimeoutError:
            timed_out = {req["request_id"] for future, req in futures.items() if not future.done()}
            for future in futures:
                if not future.done():
                    future.cancel()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        for request_id in timed_out:
            records[request_id] = {"request_id": request_id, "status": "timeout", "condition_id": cid}
        elapsed = time.monotonic() - started
        results[cid] = {"requests": records, "duration_seconds_including_model_loading_and_validation": elapsed,
                        "admitted_units_per_hour": None, "timeout_request_ids": sorted(timed_out)}
    return {"schema_version": "mercury-scheduler-v2-mock-records/v1", "plan_id": plan["plan_id"],
            "execution_class": "mock_only", "baseline": None, "baseline_required": True,
            "conditions": results}


class _DemoMockAdapter:
    """Deterministic, explicitly unmeasured adapter for CLI round-trip demos."""
    def __call__(self, request, profile, factors, condition_id):
        return {"request_id": request["request_id"], "status": "mock_only_unmeasured",
                "condition_id": condition_id, "profile_id": profile["profile_id"],
                "profile_sha256": digest(profile), "factors": factors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan"); p.add_argument("config"); p.add_argument("output")
    p = sub.add_parser("report"); p.add_argument("plan"); p.add_argument("records"); p.add_argument("output")
    p = sub.add_parser("run-mock"); p.add_argument("plan"); p.add_argument("records_output"); p.add_argument("report_output")
    args = parser.parse_args(argv)
    if args.command == "plan":
        value = build_plan(read(args.config))
        write_once(args.output, value)
        print(value["plan_id"])
    elif args.command == "report":
        value = qualify(read(args.plan), read(args.records))
        write_once(args.output, value)
        print(value["all_conditions_qualified"])
    else:
        plan = read(args.plan)
        records = run_mock(plan, _DemoMockAdapter())
        report = qualify(plan, records)
        write_once(args.records_output, records)
        write_once(args.report_output, report)
        print(json.dumps({"records": args.records_output, "report": args.report_output,
                          "execution_class": records["execution_class"],
                          "all_conditions_qualified": report["all_conditions_qualified"]}, sort_keys=True))


if __name__ == "__main__":
    main()
