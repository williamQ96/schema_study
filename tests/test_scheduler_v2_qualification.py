import copy
import time

import pytest

from scheduler_v2.qualification import build_plan, qualify, run_mock, digest


def profile(backend='mock', model='fixture-model'):
    return {"profile_id": model + '-profile', "backend": backend, "engine": "fixture-engine",
            "image": "fixture-image", "image_sha256": "fixture-image-digest", "revision": "fixture-revision",
            "checkpoint_identity": "fixture-checkpoint", "model_id": model, "tokenizer_id": "fixture-tokenizer",
            "supports_seed": True, "supports_grammar": True,
            "supported_parameters": ["max_output_tokens", "temperature", "seed"]}


def baseline_profile():
    return {"profile_id": "baseline-profile", "backend": "transformers", "engine": "transformers-v4.1",
            "image": "oci://registry.example/mercury@sha256:" + "a" * 64, "image_sha256": "a" * 64,
            "revision": "b" * 40, "checkpoint_identity": "c" * 64,
            "model_id": "frozen-baseline-model", "tokenizer_id": "baseline-tokenizer@" + "d" * 40}


def config(models=None):
    req = {"request_id": "r1", "input": {"id": 1}, "task": "extract", "messages": [{"role": "user", "content": "x"}],
           "seed": 7, "generation_parameters": {"max_output_tokens": 10, "temperature": 0, "seed": 7},
           "longest_context": True, "dense_table": True, "prior_failure_tags": ["long_context"]}
    return {"models": models or [profile()], "baseline_profile": baseline_profile(), "frozen_requests": [req],
            "required_parameters": ["max_output_tokens", "temperature"], "grammar_required": True, "seed_required": True,
            "factors": {"concurrency": [1, 2, 4], "prefix_cache": [False, True], "gpu_count": [1, 2]}}


def _request_record(req, condition_id, prof_id, profile_sha):
    return {**{key: copy.deepcopy(req[key]) for key in ("request_id", "input", "task", "messages", "seed", "generation_parameters")},
            "status": "success", "condition_id": condition_id, "profile_id": prof_id,
            "profile_sha256": profile_sha, "free_vram_gib": 12.0, "duration_seconds": 1.0,
            "oom": False, "truncated": False, "contract_regression": False,
            "misrouted": False, "provenance_error": False}


def complete_records(plan):
    base = plan["baseline"]["profile"]
    records = {"baseline": {"requests": {}, "duration_seconds_including_model_loading_and_validation": 2.0,
                             "admitted_units_per_hour": 10.0}}
    req = plan["frozen_requests"][0]
    records["baseline"]["requests"][req["request_id"]] = _request_record(req, "baseline", base["profile_id"], digest(base))
    for row in plan["conditions"]:
        cid = row["condition_id"]
        records[cid] = {"requests": {req["request_id"]: _request_record(req, cid, row["profile"]["profile_id"], digest(row["profile"]))},
                        "duration_seconds_including_model_loading_and_validation": 2.0,
                        "admitted_units_per_hour": 12.0}
    return records


def test_full_matrix_ids_bind_model_profile_and_block_unsupported_capabilities():
    plan = build_plan(config())
    assert len(plan["conditions"]) == 12
    assert len({row["condition_id"] for row in plan["conditions"]}) == 12
    two = build_plan(config([profile(model='fixture-a'), profile(model='fixture-b')]))
    assert len(two["conditions"]) == 24
    assert len({row["condition_id"] for row in two["conditions"]}) == 24
    assert all(row["status"] == "planned" for row in two["conditions"])
    cfg = config(); cfg["models"][0]["supports_seed"] = False
    assert all(row["status"] == "blocked" and row["reason"] == "seed_unsupported" for row in build_plan(cfg)["conditions"])
    cfg = config(); cfg["models"][0]["backend"] = "transformers"
    assert all(row["status"] == "blocked" for row in build_plan(cfg)["conditions"] if row["factors"]["concurrency"] > 1)


def test_real_service_placeholders_are_blocked_and_duplicate_requests_rejected():
    service = profile(backend='service', model='served-model')
    service.update(engine='REPLACE_ENGINE', image='repo:latest', image_sha256='0', revision='main', checkpoint_identity='unknown',
                   tokenizer_id='tokenizer:latest')
    plan = build_plan(config([service]))
    assert all(row["status"] == "blocked" for row in plan["conditions"])
    cfg = config(); cfg["frozen_requests"].append(copy.deepcopy(cfg["frozen_requests"][0]))
    with pytest.raises(ValueError, match="unique"):
        build_plan(cfg)
    cfg = config(); cfg["frozen_requests"][0]["generation_parameters"].pop("temperature")
    with pytest.raises(ValueError, match="required generation parameters"):
        build_plan(cfg)


def test_complete_measurements_qualify_but_cross_model_records_do_not():
    plan = build_plan(config([profile(model='fixture-a'), profile(model='fixture-b')]))
    records = complete_records(plan)
    assert not qualify(plan, records)["all_conditions_qualified"]
    assert all(row["status"] == "mock_only" for row in qualify(plan, records)["conditions"])
    first = plan["conditions"][0]
    second = next(row for row in plan["conditions"] if row["profile"]["profile_id"] != first["profile"]["profile_id"] and row["factors"] == first["factors"])
    records[second["condition_id"]] = copy.deepcopy(records[first["condition_id"]])
    outcome = next(row for row in qualify(plan, records)["conditions"] if row["condition_id"] == second["condition_id"])
    assert outcome["status"] == "mock_only"
    assert "profile_identity_mismatch" in outcome["errors"]["r1"]


def test_malformed_baseline_or_minimal_candidate_records_fail_closed():
    plan = build_plan(config())
    records = complete_records(plan)
    records["baseline"]["requests"]["r1"] = {"request_id": "r1", "status": "success"}
    report = qualify(plan, records)
    assert not report["all_conditions_qualified"]
    assert any("record_missing:free_vram_gib" in error for error in report["conditions"][0]["baseline_errors"])
    records = complete_records(plan)
    cid = plan["conditions"][0]["condition_id"]
    records[cid]["requests"]["r1"] = {"request_id": "r1", "status": "success"}
    report = qualify(plan, records)
    assert report["conditions"][0]["status"] == "mock_only"
    assert "record_missing:duration_seconds" in report["conditions"][0]["errors"]["r1"]
    records = complete_records(plan); records["baseline"]["admitted_units_per_hour"] = float('nan')
    assert "baseline_admitted_units_per_hour_missing_or_nonfinite" in qualify(plan, records)["conditions"][0]["baseline_errors"]


def test_plan_hash_is_checked_before_report_and_missing_records_are_unqualified():
    plan = build_plan(config())
    assert not qualify(plan, {})["all_conditions_qualified"]
    tampered = copy.deepcopy(plan); tampered["conditions"][0]["factors"]["gpu_count"] = 99
    assert "plan_identity_mismatch" in qualify(tampered, {})["plan_errors"]


def test_real_profile_requires_measured_guardrails_not_absent_flags():
    service = {**profile(backend='service'), **baseline_profile(), 'backend': 'service'}
    plan = build_plan(config([service]))
    records = complete_records(plan)
    assert qualify(plan, records)['all_conditions_qualified']
    cid = plan['conditions'][0]['condition_id']
    records[cid]['requests']['r1'].pop('oom')
    report = qualify(plan, records)
    assert not report['all_conditions_qualified']
    assert 'guardrail_not_measured:oom' in report['conditions'][0]['errors']['r1']


def test_service_template_plans_as_blocked_not_falsely_ready():
    import json
    from pathlib import Path
    config_path = Path(__file__).parents[1] / "high_fidelity_schema_study" / "config" / "mercury_scheduler_v2_performance.example.json"
    plan = build_plan(json.loads(config_path.read_text(encoding="utf-8")))
    assert all(row["status"] == "blocked" for row in plan["conditions"])
    assert plan["baseline"]["profile_errors"]


def test_mock_runner_session_cache_and_timeout_are_only_harness_demonstrations():
    plan = build_plan(config())
    class Adapter:
        def __init__(self): self.opens = 0; self.calls = 0; self.factor_keys = set(); self.seen_factors = set()
        def open_session(self, profile, factors, condition_id): self.opens += 1; self.factor_keys.add(condition_id); return object()
        def invoke(self, request, profile, factors, condition_id, session):
            self.calls += 1
            self.seen_factors.add((factors['concurrency'], factors['prefix_cache'], factors['gpu_count']))
            return _request_record(request, condition_id, profile["profile_id"], digest(profile))
        def __call__(self, request, profile, factors, condition_id): raise AssertionError("session dispatch expected")
    adapter = Adapter()
    result = run_mock(plan, adapter, timeout_s=2)
    assert len(result["conditions"]) == 12 and adapter.opens == 12 and adapter.calls == 12
    assert len(adapter.factor_keys) == 12
    assert adapter.seen_factors == {(c, cache, gpu) for c in (1, 2, 4) for cache in (False, True) for gpu in (1, 2)}
    invalid = run_mock(plan, lambda request, prof, factors, cid: {"request_id": "wrong-id", "status": "success"}, timeout_s=2)
    invalid_report = qualify(plan, invalid)
    assert "request_id_mismatch_or_missing" in invalid_report["conditions"][0]["errors"]["r1"]
    slow = run_mock(plan, lambda request, prof, factors, cid: (time.sleep(.05) or {"request_id": request["request_id"]}), timeout_s=.001)
    assert all(cell["timeout_request_ids"] == ["r1"] for cell in slow["conditions"].values())
    service_plan = build_plan(config([profile(backend='service')]))
    with pytest.raises(ValueError, match="mock_profiles_only"):
        run_mock(service_plan, lambda *args: {})


def test_run_mock_cli_round_trip_writes_explicitly_unqualified_artifacts(tmp_path):
    from scheduler_v2.qualification import main
    from scheduler_v2.io import write_once, read
    plan = build_plan(config())
    plan_path = tmp_path / "plan.json"; records_path = tmp_path / "records.json"; report_path = tmp_path / "report.json"
    write_once(plan_path, plan)
    main(["run-mock", str(plan_path), str(records_path), str(report_path)])
    records, report = read(records_path), read(report_path)
    assert records["execution_class"] == "mock_only" and records["baseline_required"] is True
    assert report["all_conditions_qualified"] is False
    assert all(cell["status"] == "mock_only" for cell in report["conditions"])
    assert qualify(plan, records) == report
