"""Fixed V13R2 output-interface qualification using the production attempt path.

Run inside the prepared container with ``--job /job --profile-id PROFILE``.
This deliberately substitutes a disabled navigation index for extraction. Its
results measure only this isolated output interface, never production fidelity.
Each selected group is generated once; failures are retained without retries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from high_fidelity_schema_study.four_category import backends, classification_v13r2
from high_fidelity_schema_study.four_category.classification_v2 import plan_groups as classification_groups
from high_fidelity_schema_study.four_category.common import contained, digest, file_digest, read_json, write_new
from high_fidelity_schema_study.four_category.disabled_navigation import make_disabled_index, verify_disabled_index
from high_fidelity_schema_study.four_category.extraction_protocol import module_for_task
from high_fidelity_schema_study.four_category.grouped_classification import group_job as classification_job
from high_fidelity_schema_study.four_category.grouped_extraction import group_job as extraction_job
from high_fidelity_schema_study.four_category.paper import unit_catalog, verify_paper
from high_fidelity_schema_study.four_category.resident_worker import TransformersSession
from high_fidelity_schema_study.four_category.inference_probe import ProgressRecorder
from high_fidelity_schema_study.four_category.workflow import _attempt, plan_jobs, replay_run, tasks_for_config


EXTRACTION_TARGETS = (("P01", 9), ("P01", 10), ("P03", 4), ("P06", 8), ("P08", 9))
CLASSIFICATION_TARGETS = (("P01", 11), ("P03", 4), ("P06", 8), ("P08", 9))
SEED = 1729
INDEX_CONDITION = "disabled_navigation_for_isolated_output_interface_qualification_only"


def _active_readiness(config, plan):
    """Use resident-scheduler readiness, where soft reference is deferred."""
    active = set(config['roles']['locals']) | {config['classification']['profile_id']}
    profiles = {p['profile_id']: p for p in config['profiles']}
    if config['execution'].get('deferred_roles') != ['soft_reference']:
        raise ValueError('probe_requires_explicit_deferred_soft_reference')
    if config['status'] != 'frozen' or config['inference_enabled'] is not True:
        raise ValueError('probe_experiment_not_frozen_and_enabled')
    if config['roles']['soft_reference'] in active:
        raise ValueError('deferred_reference_used_by_active_role')
    for pid in active:
        problems = backends.validate_profile(profiles[pid], for_execution=True)
        if problems:
            raise ValueError('active_profile_invalid:' + pid + ':' + repr(problems))
    if config['classification']['profile_sha256'] != backends.profile_hash(profiles[config['classification']['profile_id']]):
        raise ValueError('classifier_profile_pin_mismatch')
    for row in plan['jobs']:
        if row['kind'] == 'soft_reference':
            continue
        if row['profile_id'] not in active or backends.preflight_parameters(profiles[row['profile_id']], row['parameters']):
            raise ValueError('active_job_parameters_invalid:' + row['job_id'])


def _select_cases(config: dict, corpus: dict, plan: dict, tasks: dict, profile_id: str,
                  source_root: Path) -> tuple[list[dict], dict]:
    profile = next((p for p in config["profiles"] if p["profile_id"] == profile_id), None)
    if profile is None or profile["backend"] != "transformers" or profile_id not in config["roles"]["locals"]:
        raise ValueError("probe_requires_assigned_local_transformers_profile")
    if profile.get("runtime", {}).get("structured_output", {}).get("channel") not in {"muse-atem/v2", "json/v2"}:
        raise ValueError("probe_requires_versioned_repaired_grammar_profile")
    if config["extraction_input_protocol"] != "extraction-page-regions/v13r2":
        raise ValueError("probe_requires_v13r2_extraction")
    if config["classification"]["protocol"] != "classification-anchors/v13r2":
        raise ValueError("probe_requires_v13r2_classifier")
    _active_readiness(config, plan)
    replicate = [r for r in config["replicates"] if r["seed"] == SEED]
    if len(replicate) != 1 or replicate[0]["replicate_id"] != 1:
        raise ValueError("probe_requires_rep1_seed1729")
    papers = {p["paper_id"]: p for p in corpus["papers"]}
    required = {p for p, _ in EXTRACTION_TARGETS + CLASSIFICATION_TARGETS}
    contexts = {}
    for paper_id in sorted(required):
        source = papers[paper_id]["source"]
        if not {"mineru_raw", "baseline_input", "parser_identity"} <= set(source["artifacts"]):
            raise ValueError("probe_requires_frozen_mineru_source:" + paper_id)
        issues = verify_paper(source, source_root)
        if issues:
            raise ValueError("paper_source_invalid:" + paper_id + ":" + repr(issues[:3]))
        paper = read_json(contained(source_root, source["artifacts"]["input"]["path"]))
        index = make_disabled_index(paper, tasks["extraction"]["taxonomy"])
        if verify_disabled_index(index, paper, tasks["extraction"]["taxonomy"]):
            raise ValueError("disabled_index_derivation_invalid:" + paper_id)
        contexts[paper_id] = {"paper": paper, "index": index,
                              "source_sha256": source["source_sha256"]}
    jobs = plan["jobs"]
    cases = []
    extraction = module_for_task(tasks["extraction"])
    for paper_id, page in EXTRACTION_TARGETS:
        outer = next((j for j in jobs if j["kind"] == "local_extraction" and j["paper_id"] == paper_id
                      and j["profile_id"] == profile_id and j["replicate_id"] == 1), None)
        if outer is None or outer["parameters"].get("seed") != SEED:
            raise ValueError("extraction_job_missing:" + paper_id)
        context = contexts[paper_id]
        groups = extraction.plan_groups(context["paper"], config["extraction_grouping"])
        selected = [g for g in groups if g["pages"] == [page]]
        if len(selected) != 1:
            raise ValueError(f"extraction_page_group_missing_or_ambiguous:{paper_id}:{page}")
        group = selected[0]
        cases.append({"kind": "extraction", "paper_id": paper_id, "target_pages": [page],
                      "group": group, "job": extraction_job(outer, group, context["index"]),
                      # P08 p9 is a survey/hardware table used for context stress;
                      # requiring dataset fields there would reward false positives.
                      "target_positive": (paper_id, page) in {("P01", 9), ("P03", 4), ("P06", 8)}})
    if profile_id == config["classification"]["profile_id"]:
        for paper_id, page in CLASSIFICATION_TARGETS:
            outer = next((j for j in jobs if j["kind"] == "classification" and j["paper_id"] == paper_id
                          and j["profile_id"] == profile_id), None)
            if outer is None:
                raise ValueError("classification_job_missing:" + paper_id)
            paper = contexts[paper_id]["paper"]
            groups = classification_groups(paper, config["classification"]["grouping"])
            units = unit_catalog(paper)
            selected = next((g for g in groups if any(units[uid]["page"] == page for uid in g["unit_ids"])), None)
            if selected is None:
                raise ValueError(f"classification_page_group_missing:{paper_id}:{page}")
            existing = next((c for c in cases if c["kind"] == "classification" and
                             c["paper_id"] == paper_id and c["group"]["group_id"] == selected["group_id"]), None)
            if existing:
                existing["target_pages"].append(page)
            else:
                cases.append({"kind": "classification", "paper_id": paper_id,
                              "target_pages": [page], "group": selected,
                              "job": classification_job(outer, selected), "target_positive": None})
    return cases, contexts


def _summary(case: dict, record: dict, replay_errors: list[str], record_path: Path) -> dict:
    backend = record["backend_result"]
    raw = backend.get("raw_response")
    raw = raw if isinstance(raw, dict) else {}
    payload = record.get("parsed_response") or {}
    objects = payload.get("objects", []) if case["kind"] == "extraction" else []
    entries = payload.get("entries", []) if case["kind"] == "classification" else []
    classification_counts = {state: sum(e.get("state") == state for e in entries if isinstance(e, dict))
                             for state in ("classified", "none", "uncertain")}
    field_count = sum(obj.get("kind") in {"field", "variable"} for obj in objects if isinstance(obj, dict))
    applied = backend.get("structured_output_applied")
    metrics = (raw.get("runtime_identity") or {}).get("request_metrics") or {}
    output_tokens = (raw.get("usage") or backend.get("usage") or {}).get("output_tokens")
    channel_tokens = metrics.get("muse_channels") or {}
    failures = []
    if record["status"] != "success":
        failures.append("attempt_" + record["status"])
    if replay_errors:
        failures.append("replay_failed")
    if not isinstance(applied, dict) or applied.get("synthetic") is not False:
        failures.append("applied_grammar_absent")
    if case["kind"] == "extraction" and case["target_positive"] and field_count == 0:
        failures.append("target_positive_without_field_candidates")
    return {"kind": case["kind"], "paper_id": case["paper_id"],
            "target_pages": case["target_pages"], "group_id": case["group"]["group_id"],
            "job_id": case["job"]["job_id"], "record_path": record_path.as_posix(),
            "record_sha256": record["record_sha256"], "status": record["status"],
            "backend_status": backend.get("status"), "finish_reason": backend.get("finish_reason"),
            "validation_errors": record.get("validation_errors", []), "replay_errors": replay_errors,
            "structured_output_applied": applied,
            "raw_channel_text": raw.get("decoded_with_special_tokens"),
            "response_channel_normalization": backend.get("response_channel_normalization"),
            "channel_tokens": metrics.get("muse_channels"),
            "reasoning_tokens": channel_tokens.get("reasoning_tokens"),
            "final_tokens": channel_tokens.get("final_tokens", output_tokens),
            "output_tokens": output_tokens,
            "peak_allocated_bytes": metrics.get("peak_allocated_bytes"),
            "peak_reserved_bytes": metrics.get("peak_reserved_bytes"),
            "accepted": record["status"] == "success" and not replay_errors and
                        isinstance(applied, dict) and applied.get("synthetic") is False,
            "object_candidates": len(objects), "field_candidates": field_count,
            "classification_counts": classification_counts,
            "target_positive": case["target_positive"], "quality_assessment": "not_established",
            "qualification_failures": failures}


def run_probe(job: Path, profile_id: str, *, session_factory=TransformersSession) -> dict:
    job = job.resolve()
    config_path, corpus_path = job / "config.json", job / "corpus.json"
    config, corpus = read_json(config_path), read_json(corpus_path)
    manifest = read_json(job / 'science-manifest.json')
    for name, expected in manifest['files'].items():
        if file_digest(job / 'source/high_fidelity_schema_study' / name) != expected:
            raise ValueError('probe_source_changed:' + name)
    source_root = job / "source_bundle"
    tasks = tasks_for_config(config)
    plan = plan_jobs(config, corpus, tasks=tasks)
    cases, contexts = _select_cases(config, corpus, plan, tasks, profile_id, source_root)
    profile = next(p for p in config["profiles"] if p["profile_id"] == profile_id)
    output = job / "audit" / "v13-repair-probe" / profile_id
    if output.exists():
        raise ValueError("probe_output_already_exists_no_returned_output_retry")
    output.mkdir(parents=True)
    declared = [{"kind": c["kind"], "paper_id": c["paper_id"], "target_pages": c["target_pages"],
                 "group_id": c["group"]["group_id"], "job_id": c["job"]["job_id"]} for c in cases]
    write_new(output / "declared_scope.json", {"cases": declared, "seed": SEED,
                                               "index_condition": INDEX_CONDITION})
    events = []
    ordinal = [None]
    monitor = ProgressRecorder(output)
    monitor.start()
    def event(name, **kwargs):
        events.append({'event': name, **kwargs})
        monitor.event(name, ordinal[0], **kwargs)
    session = session_factory(lambda phase, **kwargs: monitor.phase(phase, ordinal[0], **kwargs), event)
    old_transport = backends._transformers_transport
    summaries = []
    try:
        backends._transformers_transport = session
        for number, case in enumerate(cases, 1):
            ordinal[0] = number
            monitor.phase('preparing', number, force=True)
            context = contexts[case["paper_id"]]
            task = tasks[case["kind"]]
            index = context["index"] if case["kind"] == "extraction" else None
            record_path = output / f"case-{number:02d}.record.json"
            try:
                record = _attempt(case["job"], profile, task, context["paper"], index,
                                  None, None, 1, allow_live=True, transport=None,
                                  counter=session.count,
                                  classification_group=case["group"] if case["kind"] == "classification" else None,
                                  extraction_group=case["group"] if case["kind"] == "extraction" else None)
                write_new(record_path, record)
                replay_errors = replay_run(read_json(record_path), task, context["paper"], index=index,
                                           classification_group=case["group"] if case["kind"] == "classification" else None,
                                           extraction_group=case["group"] if case["kind"] == "extraction" else None)
                row = _summary(case, record, replay_errors, record_path.relative_to(job))
                row["record_file_sha256"] = file_digest(record_path)
            except Exception as exc:
                row = {"kind": case["kind"], "paper_id": case["paper_id"],
                       "target_pages": case["target_pages"], "group_id": case["group"]["group_id"],
                       "status": "infrastructure_failed", "qualification_failures":
                       [type(exc).__name__ + ":" + str(exc)], "record_path": None}
            write_new(output / f"case-{number:02d}.summary.json", row)
            summaries.append(row)
            print(json.dumps({"case": number, "kind": row["kind"], "paper_id": row["paper_id"],
                              "status": row["status"], "failures": row["qualification_failures"]}), flush=True)
    finally:
        backends._transformers_transport = old_transport
        session.cache.close()
        monitor.close()
    write_new(output / "telemetry.json", {"events": events})
    failures = [{"case": i + 1, "errors": row["qualification_failures"]} for i, row in enumerate(summaries)
                if row["qualification_failures"]]
    report = {"schema_version": "v13-repair-output-probe/v1",
              "status": "pass" if not failures and len(summaries) == len(cases) else "fail",
              "scope": "isolated_output_interface_qualification_not_production_or_fidelity_evaluation",
              "index_condition": INDEX_CONDITION, "navigation_index_source": "deterministic_disabled_index",
              "config_file_sha256": file_digest(config_path), "corpus_file_sha256": file_digest(corpus_path),
              "config_sha256": digest(config), "corpus_sha256": digest(corpus),
              "science_manifest_file_sha256": file_digest(job / 'science-manifest.json'),
              "source_tree_sha256": manifest['source_tree_sha256'],
              "job_plan_sha256": plan["plan_sha256"], "task_sha256": {k: v["task_sha256"] for k, v in tasks.items()},
              "profile_id": profile_id, "profile_sha256": backends.profile_hash(profile),
              "paper_sources": {p: c["source_sha256"] for p, c in contexts.items()},
              "disabled_indexes": {p: c["index"]["index_sha256"] for p, c in contexts.items()},
              "seed": SEED, "attempts_per_case": 1, "declared_cases": declared,
              "case_summaries": summaries, "failures": failures,
              "semantic_correctness": "not_established"}
    write_new(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    args = parser.parse_args()
    report = run_probe(args.job, args.profile_id)
    print(json.dumps({"status": report["status"], "profile_id": args.profile_id,
                      "cases": len(report["case_summaries"]), "failures": len(report["failures"])}), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
