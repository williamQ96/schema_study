"""Nested object facts and multi-name source-region coverage. V11 stays frozen."""
from __future__ import annotations

import copy
import jsonschema
from . import extraction_v10 as v10, extraction_v11 as v11
from .common import canonical_bytes, digest, seal, seal_errors
from .feature_regions_v2 import regions

VERSION = "four-category-task/v12"
RESPONSE_VERSION = "paper-extraction-group-response/v7"
SYSTEM = v11.SYSTEM + """
Facts are nested inside their owning object. Do not output subject IDs or source
window numbers as object references. Link targets remain explicit named objects
with their own source evidence. A feature inventory need not look like a CSV:
grouped cells under Main features / Subfeatures, or feature-family names, may
enumerate several dataset fields in one cell. Extract each individually supported
feature, including subfeatures. Family names alone are groups, not fields. A
descriptive feature list can be dataset schema; lack of a row/column field axis
does not mean that it has no fields. Use grouped_cells or mixed when appropriate.
Review every candidate region and list ALL emitted_labels, using the identical
paper wording used in objects. If segmentation or coverage is unclear, report
partial or uncertain rather than complete. Inspect same-page recovery text when
cell text has run together; never guess an intended name or silently correct
letters, channels or indices. Preserve duplicate paper labels in their context.
All candidate rules and model coverage statements are fallible, not gold.
"""


def make_task():
    return seal({"schema_version": VERSION, "kind": "extraction", "system_prompt": SYSTEM,
                 "base_schema": v10.make_task()["output_schema"], "taxonomy": v10.make_task()["taxonomy"],
                 "grouping": "common_pdf_pages/v1", "anchors": "joint_window_quote/v1",
                 "roles": "paper-feature-regions/v2", "object_binding": "nested_facts/v1"}, "task_sha256")


plan_region = v11.plan_region
anchors = v11.anchors
_record = v11._record


def context(paper, group):
    tables = [t for t in regions(paper) if t["page"] in group["pages"]]
    return {"pages": group["pages"], "window_ids": group["window_ids"], "anchors": anchors(paper, group),
            "table_candidates": tables,
            "feature_cell_candidates": [dict(c, table_id=t["table_id"]) for t in tables for c in t["feature_cells"]]}


def response_schema(task, paper, index, group):
    if (task != make_task() or seal_errors(group, "group_sha256")
            or group != plan_region(paper, group["pages"], **group["limits"])):
        raise ValueError("v12_task_or_region_derivation_mismatch")
    schema = v11.response_schema(v11.make_task(), paper, index, group)
    schema["$id"] = RESPONSE_VERSION
    schema["properties"]["schema_version"] = {"const": RESPONSE_VERSION}
    obj = schema["properties"].pop("mentions")
    fact = schema["properties"].pop("facts")
    fact["items"]["properties"].pop("subject_mention")
    fact["items"]["required"].remove("subject_mention")
    obj["items"]["properties"]["facts"] = fact
    obj["items"]["required"].append("facts")
    schema["properties"]["objects"] = obj
    schema["required"] = [x for x in schema["required"] if x not in {"mentions", "facts"}] + ["objects"]
    ctx = context(paper, group)
    reviews = [_record({"table_id": {"const": t["table_id"]},
        "purpose": {"enum": ["dataset_schema", "summary_statistics", "raw_data", "model_results", "other", "uncertain"]},
        "field_axis": {"enum": ["rows", "columns", "grouped_cells", "mixed", "neither", "uncertain"]},
        "rationale": {"type": "string", "minLength": 1},
        "source_windows": {"$ref": "#/$defs/sourceWindows"}}) for t in ctx["table_candidates"]]
    coverage = [_record({"unit_id": {"const": c["unit_id"]},
        "decision": {"enum": ["complete", "partial", "not_field", "uncertain"]},
        "emitted_labels": {"type": "array", "uniqueItems": True,
                           "items": {"type": "string", "minLength": 1}, "maxItems": group["limits"]["max_mentions"]},
        "rationale": {"type": "string", "minLength": 1}}) for c in ctx["feature_cell_candidates"]]
    for key, entries in (("table_reviews", reviews), ("feature_coverage", coverage)):
        schema["properties"][key] = {"type": "array", "minItems": len(entries), "maxItems": len(entries), "items": {}}
        if entries:
            schema["properties"][key]["prefixItems"] = entries
    return schema


def render_group(task, paper, index, group, *, auxiliary_hints=None):
    body = {"task_sha256": task["task_sha256"], "target_schema": response_schema(task, paper, index, group),
            "full_source": v10._full_source(paper, index, v10.make_task()), "target": context(paper, group),
            "auxiliary_hints": auxiliary_hints or {"status": "disabled"}}
    return [{"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": canonical_bytes(body).decode("utf-8")}]


def to_v10(payload, paper, group):
    refs = {a["anchor_id"]: a for a in anchors(paper, group)}
    result = {"schema_version": v10.RESPONSE_VERSION, "coverage": copy.deepcopy(payload["coverage"]), "mentions": [], "facts": []}
    for i, obj in enumerate(payload["objects"]):
        mention = copy.deepcopy(obj)
        facts = mention.pop("facts")
        result["mentions"].append(mention)
        for fact in facts:
            anchor = refs[fact.pop("primary_anchor")]
            result["facts"].append(dict(fact, subject_mention=i, primary_window=anchor["window_id"], primary_quote=anchor["quote"]))
    return result


def validate_response(payload, paper, index, group, task=None):
    schema = response_schema(make_task() if task is None else task, paper, index, group)
    errors = ["v12_schema:" + e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(payload)]
    if errors:
        return errors
    windows, ctx = v10.window_catalog(paper), context(paper, group)
    targets = set(group["window_ids"])
    if [c["window_id"] for c in payload["coverage"]] != group["window_ids"]:
        errors.append("coverage_order_mismatch")
    facts = [f for o in payload["objects"] for f in o["facts"]]
    if len(facts) > group["limits"]["max_facts"]:
        errors.append("total_fact_limit_exceeded")
    for obj in payload["objects"]:
        if not targets.intersection(obj["source_windows"]):
            errors.append("object_without_assigned_page_evidence")
        if len({digest(f) for f in obj["facts"]}) != len(obj["facts"]):
            errors.append("duplicate_object_fact")
    for review, table in zip(payload["table_reviews"], ctx["table_candidates"]):
        if not any(windows[w]["unit_id"] in table["unit_ids"] for w in review["source_windows"]):
            errors.append("table_review_without_table_evidence:" + table["table_id"])
    for row, cell in zip(payload["feature_coverage"], ctx["feature_cell_candidates"]):
        if row["decision"] == "complete" and not row["emitted_labels"]:
            errors.append("complete_region_without_objects:" + cell["unit_id"])
        if row["decision"] == "not_field" and row["emitted_labels"]:
            errors.append("nonfield_region_has_objects:" + cell["unit_id"])
        for label in row["emitted_labels"]:
            matches = [o for o in payload["objects"] if o["kind"] in {"field", "variable"}
                       and o["normalized_label"] == label and any(windows[w]["unit_id"] in
                           {cell["unit_id"], cell["row_unit_id"]} for w in o["source_windows"])]
            if not matches:
                errors.append("region_label_without_bound_object:" + cell["unit_id"] + ":" + label)
    return errors


def completeness(payload, paper, group):
    reasons = []
    if any(c["state"] != "reviewed" for c in payload["coverage"]):
        reasons.append("source_coverage_uncertain_or_overflow")
    if any(c["decision"] in {"partial", "uncertain"} for c in payload["feature_coverage"]):
        reasons.append("feature_region_coverage_unresolved")
    windows = v10.window_catalog(paper)
    for table, review in zip(context(paper, group)["table_candidates"], payload["table_reviews"]):
        if table["feature_inventory_cue"] or review["purpose"] == "dataset_schema":
            objects = [o for o in payload["objects"] if o["kind"] in {"field", "variable"}
                       and any(windows[w]["unit_id"] in table["unit_ids"] for w in o["source_windows"])]
            if not objects:
                reasons.append("feature_inventory_without_fields:" + table["table_id"])
    if len(payload["objects"]) >= group["limits"]["max_mentions"] or sum(len(o["facts"]) for o in payload["objects"]) >= group["limits"]["max_facts"]:
        reasons.append("output_limit_reached")
    return reasons


def admit_response(raw, paper, index, group):
    try:
        payload, _ = v10.parse_response(raw)
        errors = validate_response(payload, paper, index, group)
        if errors:
            return {"status": "contract_invalid", "errors": errors, "payload": payload}
        reasons = completeness(payload, paper, group)
        return {"status": "incomplete" if reasons else "success", "errors": [], "completeness_reasons": reasons,
                "payload": payload, "materialized_v10_payload": to_v10(payload, paper, group),
                "semantic_correctness": "not_established"}
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "contract_invalid", "errors": [str(exc)], "payload": None}
