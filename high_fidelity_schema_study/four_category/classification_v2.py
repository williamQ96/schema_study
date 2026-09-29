"""Bounded, quote-grounded four-category classification without model offsets.

This module deliberately leaves the v1 task and index readers untouched.
"""
from __future__ import annotations

import copy
import re

import jsonschema

from .common import ROOT, canonical_bytes, digest, read_json, seal, seal_errors, strict_json, taxonomy
from .paper import source_identity, unit_catalog
from .evidence_schema import validate_paper_input

TASK_VERSION = "four-category-task/v2"
GROUP_VERSION = "four-category-classification-group/v2"
RESPONSE_VERSION = "paper-category-response/v2"
INDEX_VERSION = "paper-category-index/v2"
DEFAULT_POLICY = {"max_units": 32, "max_target_chars": 16000}
MARKERS = {"TAXONOMY_JSON", "TARGET_SCHEMA_JSON", "SOURCE_IDENTITY_JSON", "UNIT_CATALOG_JSON", "TARGET_GROUP_JSON"}


def _valid_input(paper_input: dict) -> None:
    validate_paper_input(paper_input)
    unit_catalog(paper_input)


def _task_errors(task: dict) -> list[str]:
    errors = seal_errors(task, "task_sha256")
    if task.get("schema_version") != TASK_VERSION or task.get("kind") != "classification" or task.get("admission_rules_version") != "classification-quotes/v2":
        errors.append("task_version_mismatch")
    if task.get("output_schema") != read_json(ROOT / "templates/paper_category_response_v2.schema.json"):
        errors.append("task_schema_mismatch")
    template = task.get("user_template")
    if not isinstance(template, str) or not isinstance(task.get("system_prompt"), str):
        errors.append("task_prompt_type_mismatch")
    elif set(re.findall(r"\{\{([A-Z_]+)\}\}", template)) != MARKERS or len(re.findall(r"\{\{([A-Z_]+)\}\}", template)) != len(MARKERS):
        errors.append("task_marker_mismatch")
    return errors


def make_task(taxonomy_value: dict | None = None) -> dict:
    value = taxonomy() if taxonomy_value is None else taxonomy_value
    if not isinstance(value, dict):
        raise ValueError("taxonomy must be an object")
    return seal({"kind": "classification", "schema_version": TASK_VERSION,
                 "admission_rules_version": "classification-quotes/v2",
                 "system_prompt": (ROOT / "templates/four_category_classification_system_v2.txt").read_text(encoding="utf-8"),
                 "user_template": (ROOT / "templates/four_category_classification_user_v2.txt").read_text(encoding="utf-8"),
                 "output_schema": read_json(ROOT / "templates/paper_category_response_v2.schema.json"),
                 "taxonomy": copy.deepcopy(value)}, "task_sha256")


def _policy(policy: dict | None) -> dict:
    p = copy.deepcopy(DEFAULT_POLICY if policy is None else policy)
    if not isinstance(p, dict) or set(p) != set(DEFAULT_POLICY) or any(type(p[k]) is not int or p[k] <= 0 for k in DEFAULT_POLICY):
        raise ValueError("classification group policy requires positive integer max_units and max_target_chars")
    return p


def plan_groups(paper_input: dict, policy: dict | None = None) -> list[dict]:
    _valid_input(paper_input)
    p = _policy(policy)
    catalog = unit_catalog(paper_input)
    if paper_input.get("schema_version") == "paper-evidence-input/v5" and not catalog:
        raise ValueError("v5_input_has_no_visible_source_units")
    partitions: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for uid, unit in catalog.items():
        length = len(unit["text"])
        if length > p["max_target_chars"]:
            raise ValueError("unit_exceeds_max_target_chars:" + uid)
        if current and (len(current) == p["max_units"] or chars + length > p["max_target_chars"]):
            partitions.append(current)
            current, chars = [], 0
        current.append(uid)
        chars += length
    if current:
        partitions.append(current)
    input_hash = digest(paper_input)
    total = len(partitions)
    return [seal({"schema_version": GROUP_VERSION, "group_id": digest({"paper_input_canonical_sha256": input_hash, "policy": p, "index": i, "unit_ids": ids}),
                  "unit_ids": ids, "index": i, "total": total, "policy": p,
                  "paper_input_canonical_sha256": input_hash}, "group_sha256") for i, ids in enumerate(partitions)]


def _group_errors(group: dict, paper_input: dict) -> list[str]:
    if not isinstance(group, dict):
        return ["classification_group_object_required"]
    errors = seal_errors(group, "group_sha256")
    try:
        if group["schema_version"] != GROUP_VERSION or group["paper_input_canonical_sha256"] != digest(paper_input):
            errors.append("classification_group_binding_mismatch")
        planned = plan_groups(paper_input, group["policy"])
        if group not in planned or planned[group["index"]] != group:
            errors.append("classification_group_plan_mismatch")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        errors.append("classification_group_invalid:" + str(exc))
    return errors


def render_group(task: dict, paper_input: dict, group: dict) -> list[dict]:
    errors = _task_errors(task) + _group_errors(group, paper_input)
    if errors:
        raise ValueError(errors)
    catalog = list(unit_catalog(paper_input).values())
    values = {"TAXONOMY_JSON": task["taxonomy"], "TARGET_SCHEMA_JSON": task["output_schema"],
              "SOURCE_IDENTITY_JSON": source_identity(paper_input), "UNIT_CATALOG_JSON": catalog,
              "TARGET_GROUP_JSON": {"group_id": group["group_id"], "unit_ids": group["unit_ids"]}}
    rendered = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: canonical_bytes(values[m[1]]).decode("utf-8"), task["user_template"])
    return [{"role": "system", "content": task["system_prompt"]}, {"role": "user", "content": rendered}]


def parse_response(raw: str) -> tuple[dict, dict]:
    if not isinstance(raw, str):
        raise ValueError("response must be text")
    stripped = raw.strip()
    if stripped.startswith("```"):
        match = re.fullmatch(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```", stripped)
        if match is None:
            raise ValueError("invalid json code fence envelope")
        body, kind = match.group(1), "json_code_fence"
    else:
        body, kind = raw, "bare_json"
    payload = strict_json(body)
    return payload, {"kind": kind}


def _source_units(paper_input: dict) -> dict[str, dict]:
    """Retain original metadata for each catalog unit without adding it to prompts."""
    found: dict[str, dict] = {}
    def visit(value):
        if isinstance(value, dict):
            if "unit_id" in value and "text" in value:
                found[value["unit_id"]] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(paper_input["pages"])
    return found


def validate_response(payload: dict, paper_input: dict, group: dict) -> list[str]:
    errors = _group_errors(group, paper_input)
    if errors:
        return errors
    schema = read_json(ROOT / "templates/paper_category_response_v2.schema.json")
    errors.extend("classification_schema:" + e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(payload))
    if errors:
        return errors
    if payload["source_identity"] != source_identity(paper_input):
        errors.append("classification_source_identity_mismatch")
    expected = group["unit_ids"]
    actual = [e["unit_id"] for e in payload["entries"]]
    if len(actual) != len(set(actual)):
        errors.append("duplicate_classification_unit")
    if actual != expected:
        errors.append("classification_target_order_or_coverage_mismatch")
    catalog = unit_catalog(paper_input)
    sources = _source_units(paper_input)
    for entry in payload["entries"]:
        uid = entry["unit_id"]
        unit = catalog.get(uid)
        if unit is None:
            continue
        state, cats, quotes = entry["state"], entry["categories"], entry["evidence_quotes"]
        if state == "classified" and (not cats or not quotes):
            errors.append("classified_entry_requires_categories_and_quotes:" + uid)
        if state != "classified" and (cats or quotes):
            errors.append("nonclassified_entry_has_categories_or_quotes:" + uid)
        if not unit["text"] and state != "none":
            errors.append("empty_unit_must_be_none:" + uid)
        for quote in quotes:
            if unit["text"].count(quote) != 1:
                errors.append("quote_absent_or_ambiguous:" + uid)
            source = sources.get(uid, {})
            span = source.get("document_char_span")
            if span is not None and (not isinstance(span, dict) or type(span.get("start")) is not int or type(span.get("end")) is not int or span["end"] - span["start"] != len(unit["text"])):
                errors.append("document_span_length_mismatch:" + uid)
    return errors


def resolve_entries(payload: dict, paper_input: dict, group: dict) -> list[dict]:
    errors = validate_response(payload, paper_input, group)
    if errors:
        raise ValueError(errors)
    catalog, sources = unit_catalog(paper_input), _source_units(paper_input)
    entries = []
    for entry in payload["entries"]:
        uid = entry["unit_id"]
        unit = catalog[uid]
        local, document = [], []
        original = sources.get(uid, {}).get("document_char_span")
        for quote in entry["evidence_quotes"]:
            start = unit["text"].index(quote)
            end = start + len(quote)
            local.append({"start": start, "end": end, "quote": quote})
            if original is not None:
                document.append({"start": original["start"] + start, "end": original["start"] + end,
                                 "quote": quote, "unit_document_char_span": copy.deepcopy(original)})
        entries.append({"unit_id": uid, "page": unit["page"], "state": entry["state"],
                        "categories": copy.deepcopy(entry["categories"]), "evidence_spans": local,
                        "document_evidence_spans": document, "rationale": entry["rationale"]})
    return entries


def _record_errors(record: dict, paper_input: dict, task: dict, group: dict) -> list[str]:
    from .backends import profile_hash
    from .workflow import replay_run
    errors = seal_errors(record, "record_sha256")
    try:
        errors.extend(replay_run(record, task, paper_input, classification_group=group))
        messages = render_group(task, paper_input, group)
        if record["schema_version"] != "four-category-run/v1" or record["classification_group"] != group:
            errors.append("record_group_mismatch")
        if record["task"] != task or record["messages"] != messages or record["request_sha256"] != digest(messages):
            errors.append("record_request_mismatch")
        if record["profile_sha256"] != profile_hash(record["profile"]):
            errors.append("record_profile_mismatch")
        if record["paper_input_canonical_sha256"] != digest(paper_input) or record["index_sha256"] is not None:
            errors.append("record_input_mismatch")
        job = record["job"]
        if job["task_sha256"] != task["task_sha256"] or job["profile_sha256"] != record["profile_sha256"] or job["classification_group"] != group:
            errors.append("record_job_mismatch")
        if job.get("kind") != "classification" or job.get("paper_id") != paper_input["paper_id"] or job.get("job_id") != digest({k: v for k, v in job.items() if k != "job_id"}):
            errors.append("record_job_identity_mismatch")
        if record["run_id"] != digest({"job_id": job["job_id"], "index_sha256": None, "attempt": record["attempt"]}):
            errors.append("record_run_id_mismatch")
        if record["status"] != "success" or record["backend_result"]["status"] != "success" or record["validation_errors"]:
            errors.append("record_not_successful")
        payload, normalization = parse_response(record["backend_result"]["raw_text"])
        if record["parsed_response"] != payload or record["response_normalization"] != normalization:
            errors.append("record_raw_replay_mismatch")
        errors.extend(validate_response(payload, paper_input, group))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append("record_replay:" + str(exc))
    return errors


def build_index_v2(paper_input: dict, task: dict, groups: list[dict], records: list[dict]) -> dict:
    _valid_input(paper_input)
    errors = _task_errors(task)
    if errors:
        raise ValueError(errors)
    if groups != plan_groups(paper_input, groups[0]["policy"] if groups else None):
        raise ValueError("classification_group_plan_mismatch")
    if len(records) != len(groups):
        raise ValueError("classification_group_record_count_mismatch")
    entries = []
    for group, record in zip(groups, records):
        errors = _record_errors(record, paper_input, task, group)
        if errors:
            raise ValueError(errors)
        payload, _ = parse_response(record["backend_result"]["raw_text"])
        entries.extend(resolve_entries(payload, paper_input, group))
    if [e["unit_id"] for e in entries] != list(unit_catalog(paper_input)):
        raise ValueError("classification_complete_coverage_mismatch")
    return seal({"schema_version": INDEX_VERSION, "taxonomy_sha256": digest(task["taxonomy"]),
                 "source_identity": source_identity(paper_input), "paper_input_canonical_sha256": digest(paper_input),
                 "task": copy.deepcopy(task), "group_policy": copy.deepcopy(groups[0]["policy"] if groups else DEFAULT_POLICY),
                 "groups": copy.deepcopy(groups), "records": copy.deepcopy(records), "entries": entries,
                 "authority": "automated_fallible_navigation_not_gold", "semantic_review": "not_established"}, "index_sha256")


def verify_index_v2(index: dict, paper_input: dict, taxonomy_value: dict | None = None) -> list[str]:
    errors = seal_errors(index, "index_sha256")
    try:
        task = index["task"]
        expected_taxonomy = taxonomy() if taxonomy_value is None else taxonomy_value
        if task["taxonomy"] != expected_taxonomy or index["taxonomy_sha256"] != digest(expected_taxonomy):
            errors.append("index_taxonomy_mismatch")
        replayed = build_index_v2(paper_input, task, index["groups"], index["records"])
        if replayed != index:
            errors.append("index_raw_derivation_mismatch")
    except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as exc:
        errors.append("index_replay:" + str(exc))
    return errors
