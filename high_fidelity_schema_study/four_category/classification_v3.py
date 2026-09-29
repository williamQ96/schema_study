"""Anchored quote classification: the model cites unit-local option IDs, never text."""
from __future__ import annotations

import copy
import re

import jsonschema

from .classification_v2 import _group_errors, _source_units, parse_response, plan_groups
from .common import ROOT, canonical_bytes, digest, read_json, seal, seal_errors, taxonomy
from .paper import source_identity, unit_catalog
from .evidence_schema import validate_paper_input

TASK_VERSION = "four-category-task/v4"
RESPONSE_VERSION = "paper-category-response/v3"
INDEX_VERSION = "paper-category-index/v3"
MARKERS = {"TAXONOMY_JSON", "TARGET_SCHEMA_JSON", "SOURCE_IDENTITY_JSON", "UNIT_CATALOG_JSON", "TARGET_GROUP_JSON", "QUOTE_OPTIONS_JSON"}


def make_task(taxonomy_value: dict | None = None) -> dict:
    value = taxonomy() if taxonomy_value is None else taxonomy_value
    if not isinstance(value, dict):
        raise ValueError("taxonomy must be an object")
    return seal({"kind": "classification", "schema_version": TASK_VERSION,
                 "admission_rules_version": "classification-anchors/v3",
                 "system_prompt": (ROOT / "templates/four_category_classification_system_v3.txt").read_text(encoding="utf-8"),
                 "user_template": (ROOT / "templates/four_category_classification_user_v3.txt").read_text(encoding="utf-8"),
                 "output_schema": read_json(ROOT / "templates/paper_category_response_v3.schema.json"),
                 "taxonomy": copy.deepcopy(value)}, "task_sha256")


def task_errors(task: dict) -> list[str]:
    errors = seal_errors(task, "task_sha256")
    if task.get("kind") != "classification" or task.get("schema_version") != TASK_VERSION or task.get("admission_rules_version") != "classification-anchors/v3":
        errors.append("anchor_task_version_mismatch")
    if task.get("output_schema") != read_json(ROOT / "templates/paper_category_response_v3.schema.json"):
        errors.append("anchor_output_schema_mismatch")
    template = task.get("user_template")
    if not isinstance(task.get("system_prompt"), str) or not isinstance(template, str):
        errors.append("anchor_prompt_type_mismatch")
    elif len(re.findall(r"\{\{([A-Z_]+)\}\}", template)) != len(MARKERS) or set(re.findall(r"\{\{([A-Z_]+)\}\}", template)) != MARKERS:
        errors.append("anchor_markers_mismatch")
    return errors


def _segments(text: str) -> list[tuple[int, int, str]]:
    """Partition by Unicode code points; prefer boundaries in the latter half."""
    parts = []
    start = 0
    while start < len(text):
        hard_end = min(start + 240, len(text))
        end = hard_end
        if hard_end < len(text):
            lower = start + 120
            newline = text.rfind("\n", lower, hard_end)
            if newline >= lower:
                end = newline + 1
            else:
                whitespace = max((i for i in range(lower, hard_end) if text[i].isspace()), default=-1)
                if whitespace >= lower:
                    end = whitespace + 1
        parts.append((start, end, text[start:end]))
        start = end
    return parts


def quote_options(paper_input: dict, group: dict) -> list[dict]:
    errors = _group_errors(group, paper_input)
    if errors:
        raise ValueError(errors)
    catalog = unit_catalog(paper_input)
    return [{"unit_id": uid, "options": [{"quote_id": f"q{i:04d}", "text": chunk}
                                         for i, (_, _, chunk) in enumerate(_segments(catalog[uid]["text"]))]}
            for uid in group["unit_ids"]]


def render_group(task: dict, paper_input: dict, group: dict) -> list[dict]:
    errors = task_errors(task) + _group_errors(group, paper_input)
    if errors:
        raise ValueError(errors)
    values = {"TAXONOMY_JSON": task["taxonomy"], "TARGET_SCHEMA_JSON": task["output_schema"],
              "SOURCE_IDENTITY_JSON": source_identity(paper_input),
              "UNIT_CATALOG_JSON": list(unit_catalog(paper_input).values()),
              "TARGET_GROUP_JSON": {"group_id": group["group_id"], "unit_ids": group["unit_ids"]},
              "QUOTE_OPTIONS_JSON": quote_options(paper_input, group)}
    user = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: canonical_bytes(values[m[1]]).decode("utf-8"), task["user_template"])
    return [{"role": "system", "content": task["system_prompt"]}, {"role": "user", "content": user}]


def validate_response(payload: dict, paper_input: dict, group: dict) -> list[str]:
    errors = _group_errors(group, paper_input)
    if errors:
        return errors
    schema = read_json(ROOT / "templates/paper_category_response_v3.schema.json")
    errors.extend("classification_schema:" + e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(payload))
    if errors:
        return errors
    if payload["source_identity"] != source_identity(paper_input):
        errors.append("classification_source_identity_mismatch")
    actual = [entry["unit_id"] for entry in payload["entries"]]
    if len(actual) != len(set(actual)):
        errors.append("duplicate_classification_unit")
    if actual != group["unit_ids"]:
        errors.append("classification_target_order_or_coverage_mismatch")
    catalog = unit_catalog(paper_input)
    options = {row["unit_id"]: {o["quote_id"] for o in row["options"]} for row in quote_options(paper_input, group)}
    originals = _source_units(paper_input)
    for entry in payload["entries"]:
        uid = entry["unit_id"]
        unit = catalog.get(uid)
        if unit is None:
            continue
        state, cats, ids = entry["state"], entry["categories"], entry["evidence_quote_ids"]
        if state == "classified" and (not cats or not ids):
            errors.append("classified_entry_requires_categories_and_quote_ids:" + uid)
        if state != "classified" and (cats or ids):
            errors.append("nonclassified_entry_has_categories_or_quote_ids:" + uid)
        if not unit["text"] and state != "none":
            errors.append("empty_unit_must_be_none:" + uid)
        for qid in ids:
            if qid not in options.get(uid, set()):
                errors.append("quote_id_missing_or_wrong_unit:" + uid + ":" + qid)
        span = originals.get(uid, {}).get("document_char_span")
        if ids and span is not None and (not isinstance(span, dict) or type(span.get("start")) is not int or type(span.get("end")) is not int or span["end"] - span["start"] != len(unit["text"])):
            errors.append("document_span_length_mismatch:" + uid)
    return errors


def resolve_entries(payload: dict, paper_input: dict, group: dict) -> list[dict]:
    errors = validate_response(payload, paper_input, group)
    if errors:
        raise ValueError(errors)
    catalog, originals = unit_catalog(paper_input), _source_units(paper_input)
    lookup = {uid: {f"q{i:04d}": (start, end, chunk) for i, (start, end, chunk) in enumerate(_segments(catalog[uid]["text"]))}
              for uid in group["unit_ids"]}
    entries = []
    for entry in payload["entries"]:
        uid = entry["unit_id"]
        unit = catalog[uid]
        original_span = originals.get(uid, {}).get("document_char_span")
        local, document = [], []
        for qid in entry["evidence_quote_ids"]:
            start, end, quote = lookup[uid][qid]
            local.append({"start": start, "end": end, "quote": quote})
            if original_span is not None:
                document.append({"start": original_span["start"] + start, "end": original_span["start"] + end,
                                 "quote": quote, "unit_document_char_span": copy.deepcopy(original_span)})
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


def build_index_v3(paper_input: dict, task: dict, groups: list[dict], records: list[dict]) -> dict:
    validate_paper_input(paper_input)
    errors = task_errors(task)
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
    if [entry["unit_id"] for entry in entries] != list(unit_catalog(paper_input)):
        raise ValueError("classification_complete_coverage_mismatch")
    return seal({"schema_version": INDEX_VERSION, "taxonomy_sha256": digest(task["taxonomy"]),
                 "source_identity": source_identity(paper_input), "paper_input_canonical_sha256": digest(paper_input),
                 "task": copy.deepcopy(task), "group_policy": copy.deepcopy(groups[0]["policy"] if groups else {"max_units": 32, "max_target_chars": 16000}),
                 "groups": copy.deepcopy(groups), "records": copy.deepcopy(records), "entries": entries,
                 "authority": "automated_fallible_navigation_not_gold", "semantic_review": "not_established"}, "index_sha256")


def verify_index_v3(index: dict, paper_input: dict, taxonomy_value: dict | None = None) -> list[str]:
    errors = seal_errors(index, "index_sha256")
    try:
        task = index["task"]
        expected_taxonomy = taxonomy() if taxonomy_value is None else taxonomy_value
        if task["taxonomy"] != expected_taxonomy or index["taxonomy_sha256"] != digest(expected_taxonomy):
            errors.append("index_taxonomy_mismatch")
        replayed = build_index_v3(paper_input, task, index["groups"], index["records"])
        if replayed != index:
            errors.append("index_raw_derivation_mismatch")
    except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as exc:
        errors.append("index_replay:" + str(exc))
    return errors
