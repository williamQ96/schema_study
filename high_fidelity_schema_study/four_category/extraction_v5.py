"""Bounded, replayable paper-only extraction of atomic observations.

Source windows own facts, not semantic objects.  Repeated mentions remain
distinct until an independently reviewed identity decision is available.
"""
from __future__ import annotations

import copy
import re

import jsonschema

from .classification_v3 import _segments
from .common import ROOT, canonical_bytes, digest, read_json, seal, seal_errors, strict_json, taxonomy
from .extraction_view import _source_units, make_view
from .paper import source_identity, unit_catalog, verify_index

TASK_VERSION = "four-category-task/v5"
GROUP_VERSION = "four-category-extraction-group/v1"
RESPONSE_VERSION = "paper-extraction-group-response/v1"
BUNDLE_VERSION = "paper-derived-observations/v5"
DEFAULT_POLICY = {"max_windows": 24, "max_target_chars": 4000, "max_mentions": 16, "max_facts": 24}
MARKERS = {"SOURCE_IDENTITY_JSON", "TAXONOMY_JSON", "TARGET_SCHEMA_JSON", "FULL_SOURCE_JSON", "TARGET_GROUP_JSON"}
SCHEMA_PATH = ROOT / "templates/paper_extraction_observations_v5.schema.json"


def make_task(taxonomy_value: dict | None = None) -> dict:
    tax = taxonomy() if taxonomy_value is None else taxonomy_value
    if not isinstance(tax, dict):
        raise ValueError("taxonomy must be an object")
    return seal({"kind": "extraction", "schema_version": TASK_VERSION,
                 "admission_rules_version": "extraction-anchors/v3",
                 "system_prompt": (ROOT / "templates/paper_to_schema_system_v6.txt").read_text(encoding="utf-8"),
                 "user_template": (ROOT / "templates/paper_to_schema_user_v6.txt").read_text(encoding="utf-8"),
                 "output_schema": read_json(SCHEMA_PATH), "taxonomy": copy.deepcopy(tax)}, "task_sha256")


def task_errors(task: dict) -> list[str]:
    errors = seal_errors(task, "task_sha256")
    try:
        if task != make_task(task["taxonomy"]):
            errors.append("extraction_v5_task_mismatch")
        markers = re.findall(r"\{\{([A-Z_]+)\}\}", task["user_template"])
        if len(markers) != len(MARKERS) or set(markers) != MARKERS:
            errors.append("extraction_v5_template_markers_invalid")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("extraction_v5_task_invalid:" + str(exc))
    return errors


def quote_catalog(paper_input: dict) -> dict[str, list[dict]]:
    return {uid: [{"quote_id": f"q{i:04d}", "text": chunk}
                  for i, (_, _, chunk) in enumerate(_segments(unit["text"]))]
            for uid, unit in unit_catalog(paper_input).items()}


def _policy(policy: dict | None) -> dict:
    value = copy.deepcopy(DEFAULT_POLICY if policy is None else policy)
    if (not isinstance(value, dict) or set(value) != set(DEFAULT_POLICY)
            or any(type(value[key]) is not int or value[key] <= 0 for key in DEFAULT_POLICY)):
        raise ValueError("invalid_extraction_group_policy")
    return value


def _windows(paper_input: dict) -> list[tuple[str, int]]:
    catalog = unit_catalog(paper_input)
    quotes = quote_catalog(paper_input)
    return [(uid + "#" + quote["quote_id"], len(quote["text"]))
            for uid in catalog for quote in quotes[uid]]


def plan_groups(paper_input: dict, policy: dict | None = None) -> list[dict]:
    p = _policy(policy)
    windows = _windows(paper_input)
    groups: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for wid, length in windows:
        if length > p["max_target_chars"]:
            raise ValueError("window_exceeds_target_chars:" + wid)
        if current and (len(current) >= p["max_windows"] or chars + length > p["max_target_chars"]):
            groups.append(current)
            current, chars = [], 0
        current.append(wid)
        chars += length
    if current:
        groups.append(current)
    identity = digest(paper_input)
    return [seal({"schema_version": GROUP_VERSION,
                  "group_id": digest({"paper_input_canonical_sha256": identity, "policy": p,
                                      "index": i, "window_ids": ids}),
                  "paper_input_canonical_sha256": identity, "policy": copy.deepcopy(p),
                  "index": i, "total": len(groups), "window_ids": ids}, "group_sha256")
            for i, ids in enumerate(groups)]


def group_errors(group: dict, paper_input: dict) -> list[str]:
    errors = seal_errors(group, "group_sha256")
    try:
        planned = plan_groups(paper_input, group["policy"])
        if group["schema_version"] != GROUP_VERSION or group["paper_input_canonical_sha256"] != digest(paper_input):
            errors.append("extraction_group_identity_mismatch")
        if type(group["index"]) is not int or not 0 <= group["index"] < len(planned) or planned[group["index"]] != group:
            errors.append("extraction_group_plan_mismatch")
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        errors.append("extraction_group_invalid:" + str(exc))
    return errors


def response_schema(task: dict, paper_input: dict, index: dict, group: dict) -> dict:
    errors = task_errors(task) + group_errors(group, paper_input) + verify_index(index, paper_input, taxonomy_value=task["taxonomy"])
    if errors:
        raise ValueError(errors)
    schema = copy.deepcopy(task["output_schema"])
    ids = group["window_ids"]
    coverage = schema["properties"]["coverage"]
    coverage["minItems"] = coverage["maxItems"] = len(ids)
    coverage["prefixItems"] = [dict(coverage["items"], properties={**coverage["items"]["properties"], "window_id": {"const": wid}}) for wid in ids]
    schema["properties"]["mentions"]["maxItems"] = group["policy"]["max_mentions"]
    schema["properties"]["facts"]["maxItems"] = group["policy"]["max_facts"]
    return schema


def _full_source(paper_input: dict, index: dict, task: dict) -> dict:
    view = make_view(paper_input, index, taxonomy_value=task["taxonomy"])
    quotes = quote_catalog(paper_input)
    units = copy.deepcopy(view["units"])
    for row in units:
        row[3] = [[q["quote_id"], q["text"]] for q in quotes[row[0]]]
    return {"schema_version": "paper-extraction-quoted-source/v1", "source_identity": view["source_identity"],
            "paper_input_canonical_sha256": view["paper_input_canonical_sha256"],
            "index_sha256": view["index_sha256"], "taxonomy_sha256": view["taxonomy_sha256"],
            "authority": view["authority"], "unit_columns": ["unit_id", "page", "source_kind", "quote_options", "reading_order", "bbox"],
            "units": units, "page_columns": view["page_columns"], "table_columns": view["table_columns"],
            "column_columns": view["column_columns"], "row_columns": view["row_columns"],
            "pages": view["pages"], "index_columns": view["index_columns"], "index_entries": view["index_entries"]}


def render_group(task: dict, paper_input: dict, index: dict, group: dict) -> list[dict]:
    schema = response_schema(task, paper_input, index, group)
    values = {"SOURCE_IDENTITY_JSON": source_identity(paper_input), "TAXONOMY_JSON": task["taxonomy"],
              "TARGET_SCHEMA_JSON": schema, "FULL_SOURCE_JSON": _full_source(paper_input, index, task),
              "TARGET_GROUP_JSON": {"group_id": group["group_id"], "window_ids": group["window_ids"],
                                    "max_mentions": group["policy"]["max_mentions"], "max_facts": group["policy"]["max_facts"]}}
    user = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: canonical_bytes(values[m[1]]).decode("utf-8"), task["user_template"])
    return [{"role": "system", "content": task["system_prompt"]}, {"role": "user", "content": user}]


def parse_response(raw: str) -> tuple[dict, dict]:
    if not isinstance(raw, str):
        raise ValueError("response must be text")
    # No prose, fence, channel marker, or posthoc repair belongs to this lane.
    return strict_json(raw), {"kind": "bare_json"}


def _anchor_valid(anchor: dict, quotes: dict[str, list[dict]]) -> bool:
    return isinstance(anchor, dict) and anchor.get("quote_id") in {q["quote_id"] for q in quotes.get(anchor.get("unit_id"), [])}


def _anchor_window(anchor: dict) -> str:
    return anchor["unit_id"] + "#" + anchor["quote_id"]


def validate_response(payload: dict, paper_input: dict, index: dict, group: dict, task: dict | None = None) -> list[str]:
    task = make_task() if task is None else task
    try:
        schema = response_schema(task, paper_input, index, group)
    except (KeyError, TypeError, ValueError) as exc:
        return ["extraction_group_context_invalid:" + str(exc)]
    errors = ["extraction_schema:" + error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(payload)]
    if errors:
        return errors
    quotes = quote_catalog(paper_input)
    targets = set(group["window_ids"])
    actual = [row["window_id"] for row in payload["coverage"]]
    if actual != group["window_ids"]:
        errors.append("extraction_target_order_or_coverage_mismatch")
    for i, mention in enumerate(payload["mentions"]):
        if not _anchor_valid(mention["name_anchor"], quotes):
            errors.append(f"mention_name_anchor_invalid:{i}")
        elif mention["name"] not in next(q["text"] for q in quotes[mention["name_anchor"]["unit_id"]] if q["quote_id"] == mention["name_anchor"]["quote_id"]):
            errors.append(f"mention_name_not_in_anchor:{i}")
    seen = set()
    for i, fact in enumerate(payload["facts"]):
        if fact["subject_mention"] >= len(payload["mentions"]):
            errors.append(f"fact_subject_missing:{i}")
        primary = fact["primary_anchor"]
        if not _anchor_valid(primary, quotes) or _anchor_window(primary) not in targets:
            errors.append(f"fact_primary_anchor_foreign:{i}")
        elif payload["coverage"][group["window_ids"].index(_anchor_window(primary))]["state"] == "overflow":
            errors.append(f"fact_on_overflow_window:{i}")
        for support in fact["support_anchors"]:
            if not _anchor_valid(support, quotes):
                errors.append(f"fact_support_anchor_invalid:{i}")
        target = fact["target"]
        if (fact["predicate"] in {"relationship", "same_as"}) != (target is not None):
            errors.append(f"fact_target_shape_invalid:{i}")
        if target is not None and (not _anchor_valid(target["anchor"], quotes)
                                   or target["name"] not in next((q["text"] for q in quotes.get(target["anchor"]["unit_id"], []) if q["quote_id"] == target["anchor"]["quote_id"]), "")):
            errors.append(f"fact_target_anchor_invalid:{i}")
        if (fact["status"] == "unknown") != (fact["value"] is None):
            errors.append(f"fact_status_value_conflict:{i}")
        if fact["status"] == "inferred" and not fact["basis"]:
            errors.append(f"fact_inference_basis_required:{i}")
        key = canonical_bytes(fact)
        if key in seen:
            errors.append(f"duplicate_fact:{i}")
        seen.add(key)
    return errors


def completion_status(payload: dict) -> str:
    """Uncertain review is explicit coverage; overflow blocks aggregation."""
    states = {row["state"] for row in payload["coverage"]}
    return "incomplete" if "overflow" in states else "success"


def _materialize_anchor(anchor: dict, paper_input: dict) -> dict:
    unit = unit_catalog(paper_input)[anchor["unit_id"]]
    options = {f"q{i:04d}": (start, end, text)
               for i, (start, end, text) in enumerate(_segments(unit["text"]))}
    start, end, text = options[anchor["quote_id"]]
    original = _source_units(paper_input)[anchor["unit_id"]]
    doc = original.get("document_char_span")
    if doc is not None and (type(doc.get("start")) is not int or type(doc.get("end")) is not int
                            or doc["end"] - doc["start"] != len(unit["text"])):
        raise ValueError("unit_document_span_length_mismatch:" + anchor["unit_id"])
    return {"unit_id": anchor["unit_id"], "quote_id": anchor["quote_id"], "page": unit["page"],
            "source_kind": unit["source_kind"], "quote_or_cell_text": text,
            "unit_char_span": {"start": start, "end": end},
            "document_char_span": None if doc is None else {"start": doc["start"] + start, "end": doc["start"] + end},
            "source_identity": source_identity(paper_input)}


def build_bundle(paper_input: dict, index: dict, task: dict, groups: list[dict], records: list[dict]) -> dict:
    from .workflow import replay_run
    if groups != plan_groups(paper_input, groups[0]["policy"] if groups else None):
        raise ValueError("extraction_group_plan_mismatch")
    if len(groups) != len(records):
        raise ValueError("extraction_group_record_count_mismatch")
    mentions, facts, coverage, refs = [], [], [], []
    binding = None
    for group, record in zip(groups, records):
        replay_errors = replay_run(record, task, paper_input, index=index, extraction_group=group)
        if replay_errors:
            raise ValueError("extraction_group_record_replay_failed:" + ",".join(replay_errors[:4]))
        if record["status"] != "success" or record["task"] != task or record.get("extraction_group") != group:
            raise ValueError("extraction_group_record_not_successful")
        current_binding = (record["profile_sha256"], record["job"]["parent_job_id"], record["index_sha256"])
        if record["index_sha256"] != index["index_sha256"] or (binding is not None and binding != current_binding):
            raise ValueError("extraction_group_cross_record_binding_mismatch")
        binding = current_binding
        payload, normalization = parse_response(record["backend_result"]["raw_text"])
        if record["parsed_response"] != payload or record.get("response_normalization") != normalization:
            raise ValueError("extraction_group_raw_replay_mismatch")
        errors = validate_response(payload, paper_input, index, group, task)
        if errors:
            raise ValueError(errors)
        if completion_status(payload) != "success":
            raise ValueError("extraction_group_incomplete")
        prefix = group["group_id"]
        identity_prefix = record["job"]["parent_job_id"] + ":" + prefix
        coverage.extend({"group_id": prefix, **copy.deepcopy(row)} for row in payload["coverage"])
        for i, mention in enumerate(payload["mentions"]):
            mentions.append({"mention_id": identity_prefix + ":m" + str(i), "group_id": prefix,
                             **copy.deepcopy(mention),
                             "name_evidence": _materialize_anchor(mention["name_anchor"], paper_input)})
        for i, fact in enumerate(payload["facts"]):
            row = copy.deepcopy(fact)
            row["subject_mention_id"] = identity_prefix + ":m" + str(row.pop("subject_mention"))
            facts.append({"fact_id": identity_prefix + ":f" + str(i), "group_id": prefix, **row,
                          "primary_evidence": _materialize_anchor(fact["primary_anchor"], paper_input),
                          "support_evidence": [_materialize_anchor(a, paper_input) for a in fact["support_anchors"]],
                          "target_evidence": None if fact["target"] is None else _materialize_anchor(fact["target"]["anchor"], paper_input)})
        refs.append({"group_id": prefix, "run_id": record["run_id"], "record_sha256": record["record_sha256"]})
    return seal({"schema_version": BUNDLE_VERSION, "source_identity": source_identity(paper_input),
                 "paper_input_canonical_sha256": digest(paper_input), "index_sha256": index["index_sha256"],
                 "task_sha256": task["task_sha256"], "groups": copy.deepcopy(groups),
                 "profile_sha256": binding[0] if binding else None,
                 "parent_job_id": binding[1] if binding else None,
                 "record_refs": refs, "coverage": coverage, "mentions": mentions, "facts": facts,
                 "observation_count": len(facts), "mention_record_count": len(mentions),
                 "resolved_object_count": None, "semantic_correctness": "not_established",
                 "identity_resolution": "unresolved_model_hypotheses_only"}, "bundle_sha256")


def verify_bundle(bundle: dict, paper_input: dict, index: dict, task: dict, records: list[dict] | None = None) -> list[str]:
    errors = seal_errors(bundle, "bundle_sha256")
    if records is None:
        return errors + ["extraction_bundle_records_required_for_replay"]
    try:
        if bundle != build_bundle(paper_input, index, task, bundle["groups"], records):
            errors.append("extraction_bundle_derivation_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("extraction_bundle_replay:" + str(exc))
    return errors
