"""Navigation index that preserves rejected classifier groups as unavailable.

Every group has one selected, sealed attempt. A failed attempt contributes no
classification prediction; its source units remain visible to extraction.
"""
from __future__ import annotations

import copy
from collections import Counter

import jsonschema

from .backends import profile_hash
from .classification_v2 import DEFAULT_POLICY, plan_groups
from .classification_v3 import make_task, parse_response, resolve_entries
from .common import ROOT, digest, read_json, seal, seal_errors, taxonomy
from .paper import source_identity, unit_catalog
from .evidence_schema import validate_paper_input

INDEX_VERSION = "paper-category-index/v4"


def _parent_job(child: dict, group: dict) -> dict:
    if child.get("classification_group") != group or not isinstance(child.get("parent_job_id"), str):
        raise ValueError("classification_child_group_or_parent_mismatch")
    parent = copy.deepcopy(child)
    expected_id = parent.pop("parent_job_id")
    parent.pop("classification_group")
    parent.pop("job_id")
    if digest(parent) != expected_id:
        raise ValueError("classification_parent_job_identity_mismatch")
    parent["job_id"] = expected_id
    expected_child = copy.deepcopy(parent)
    expected_child.pop("job_id")
    expected_child["parent_job_id"] = expected_id
    expected_child["classification_group"] = copy.deepcopy(group)
    expected_child["job_id"] = digest(expected_child)
    if child != expected_child:
        raise ValueError("classification_child_job_identity_mismatch")
    return parent


def _record_binding(record: dict, paper_input: dict, task: dict, group: dict) -> dict:
    from .workflow import replay_run
    errors = replay_run(record, task, paper_input, classification_group=group)
    if errors:
        raise ValueError("classification_record_replay_failed:" + ",".join(errors[:4]))
    result = record["backend_result"]
    if not result.get("dispatch_started") and (result.get("raw_response") is not None or result.get("raw_text") is not None):
        raise ValueError("classification_undispatched_generation_claim")
    job = record["job"]
    parent = _parent_job(job, group)
    if (job.get("kind") != "classification" or job.get("paper_id") != paper_input["paper_id"]
            or not isinstance(job.get("source_sha256"), str) or not job["source_sha256"]
            or job.get("task_sha256") != task["task_sha256"]
            or job.get("profile_sha256") != record["profile_sha256"]
            or job.get("profile_id") != record["profile"]["profile_id"]
            or record["profile_sha256"] != profile_hash(record["profile"])
            or record.get("index_sha256") is not None):
        raise ValueError("classification_record_job_profile_source_mismatch")
    return parent


def _generation_observed(result: dict) -> bool:
    return (result.get('status') in {'success', 'truncated', 'refused'}
            and result.get('dispatch_started') is True
            and result.get('raw_response') is not None)


def build_index_v4(paper_input: dict, task: dict, groups: list[dict], records: list[dict]) -> dict:
    """Build an index from exactly one replayable selected attempt per group."""
    validate_paper_input(paper_input)
    repair = task.get("admission_rules_version") == "classification-anchors/v13r2"
    if repair:
        from .classification_v13r2 import make_task as repair_task, resolve_entries as repair_entries
        expected_task = repair_task(task.get("taxonomy"))
    else:
        expected_task = make_task(task.get("taxonomy"))
    if task != expected_task:
        raise ValueError("classification_v3_task_identity_mismatch")
    if not isinstance(groups, list) or not isinstance(records, list) or len(records) != len(groups):
        raise ValueError("classification_group_record_count_mismatch")
    policy = groups[0]["policy"] if groups else DEFAULT_POLICY
    if groups != plan_groups(paper_input, policy):
        raise ValueError("classification_group_plan_mismatch")
    if not groups:
        raise ValueError("classification_requires_source_units")
    catalog = unit_catalog(paper_input)
    entries, outcomes = [], []
    common_parent = common_profile = common_source = None
    for group, record in zip(groups, records):
        parent = _record_binding(record, paper_input, task, group)
        if common_parent is None:
            common_parent = parent
            common_profile = record["profile"]
            common_source = parent.get("source_sha256")
        elif parent != common_parent or record["profile"] != common_profile or parent.get("source_sha256") != common_source:
            raise ValueError("classification_shared_parent_profile_source_mismatch")
        result = record["backend_result"]
        status = record["status"]
        derivation = {"group_id": group["group_id"], "run_id": record["run_id"],
                      "record_sha256": record["record_sha256"], "outcome": status}
        observed = _generation_observed(result)
        outcomes.append({"group_id": group["group_id"], "status": status,
                         "backend_status": result["status"], "generation_observed": observed,
                         "unit_count": len(group["unit_ids"]), "run_id": record["run_id"],
                         "record_sha256": record["record_sha256"]})
        if status == "success":
            payload, _ = parse_response(result["raw_text"])
            resolved = repair_entries(payload, paper_input, group) if repair else resolve_entries(payload, paper_input, group)
            entries.extend({"unit_id": entry["unit_id"], "page": entry["page"],
                            "availability": "available", "prediction": entry,
                            "derivation": copy.deepcopy(derivation)} for entry in resolved)
        else:
            entries.extend({"unit_id": uid, "page": catalog[uid]["page"],
                            "availability": "unavailable", "prediction": None,
                            "machine_reason": status, "derivation": copy.deepcopy(derivation)}
                           for uid in group["unit_ids"])
    if [entry["unit_id"] for entry in entries] != list(catalog):
        raise ValueError("classification_complete_coverage_mismatch")
    availability = Counter(entry["availability"] for entry in entries)
    group_availability = Counter("available" if row["status"] == "success" else "unavailable" for row in outcomes)
    backend_groups = Counter(row["backend_status"] for row in outcomes)
    backend_units = Counter()
    for row in outcomes:
        backend_units[row["backend_status"]] += row["unit_count"]
    return seal({"schema_version": INDEX_VERSION,
                 "taxonomy_sha256": digest(task["taxonomy"]),
                 "source_identity": source_identity(paper_input),
                 "paper_input_canonical_sha256": digest(paper_input),
                 "task": copy.deepcopy(task), "group_policy": copy.deepcopy(policy),
                 "groups": copy.deepcopy(groups), "records": copy.deepcopy(records),
                 "parent_job": copy.deepcopy(common_parent), "profile": copy.deepcopy(common_profile),
                 "source_sha256": common_source,
                 "entries": entries, "group_outcomes": outcomes,
                 "availability_counts": {"groups_available": group_availability["available"],
                                         "groups_unavailable": group_availability["unavailable"],
                                         "units_available": availability["available"],
                                         "units_unavailable": availability["unavailable"]},
                 "generation_coverage": {"groups_by_backend_status": dict(sorted(backend_groups.items())),
                                         "units_by_backend_status": dict(sorted(backend_units.items())),
                                         "groups_with_output": sum(row["generation_observed"] for row in outcomes),
                                         "units_with_output": sum(row["unit_count"] for row in outcomes if row["generation_observed"])},
                 "authority": "automated_fallible_navigation_not_gold",
                 "semantic_review": "not_established"}, "index_sha256")


def verify_index_v4(index: dict, paper_input: dict, taxonomy_value: dict | None = None) -> list[str]:
    errors = seal_errors(index, "index_sha256")
    try:
        expected_taxonomy = taxonomy() if taxonomy_value is None else taxonomy_value
        if index["task"]["taxonomy"] != expected_taxonomy or index["taxonomy_sha256"] != digest(expected_taxonomy):
            errors.append("index_taxonomy_mismatch")
        replayed = build_index_v4(paper_input, index["task"], index["groups"], index["records"])
        if replayed != index:
            errors.append("index_raw_derivation_mismatch")
    except (KeyError, TypeError, ValueError, IndexError, jsonschema.ValidationError) as exc:
        errors.append("index_replay:" + str(exc))
    return errors
