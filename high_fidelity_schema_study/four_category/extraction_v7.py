"""Bounded paper extraction with one global source-window pointer namespace."""
from __future__ import annotations

import copy
import re

import jsonschema

from .classification_v3 import _segments
from .common import ROOT, canonical_bytes, digest, read_json, seal, seal_errors, strict_json, taxonomy
from .extraction_view import _source_units, make_view
from .paper import source_identity, unit_catalog, verify_index

TASK_VERSION = "four-category-task/v7"
GROUP_VERSION = "four-category-extraction-group/v3"
RESPONSE_VERSION = "paper-extraction-group-response/v3"
BUNDLE_VERSION = "paper-derived-observations/v7"
DEFAULT_POLICY = {"max_windows": 8, "max_target_chars": 1200, "max_mentions": 32, "max_facts": 48}
MARKERS = {"SOURCE_IDENTITY_JSON", "TAXONOMY_JSON", "TARGET_SCHEMA_JSON", "FULL_SOURCE_JSON", "TARGET_GROUP_JSON"}
SCHEMA_PATH = ROOT / "templates/paper_extraction_observations_v7.schema.json"


def make_task(taxonomy_value: dict | None = None) -> dict:
    tax = taxonomy() if taxonomy_value is None else taxonomy_value
    if not isinstance(tax, dict):
        raise ValueError("taxonomy must be an object")
    return seal({"kind": "extraction", "schema_version": TASK_VERSION,
                 "admission_rules_version": "extraction-pointers/v5",
                 "system_prompt": (ROOT / "templates/paper_to_schema_system_v8.txt").read_text(encoding="utf-8"),
                 "user_template": (ROOT / "templates/paper_to_schema_user_v8.txt").read_text(encoding="utf-8"),
                 "output_schema": read_json(SCHEMA_PATH), "taxonomy": copy.deepcopy(tax)}, "task_sha256")


def task_errors(task: dict) -> list[str]:
    errors = seal_errors(task, "task_sha256")
    try:
        if task != make_task(task["taxonomy"]):
            errors.append("extraction_v7_task_mismatch")
        markers = re.findall(r"\{\{([A-Z_]+)\}\}", task["user_template"])
        if len(markers) != len(MARKERS) or set(markers) != MARKERS:
            errors.append("extraction_v7_template_markers_invalid")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("extraction_v7_task_invalid:" + str(exc))
    return errors


def quote_catalog(paper_input: dict) -> dict[str, list[dict]]:
    """Every source character appears once, with globally numbered windows."""
    result: dict[str, list[dict]] = {}
    index = 0
    for uid, unit in unit_catalog(paper_input).items():
        rows = []
        for i, (start, end, chunk) in enumerate(_segments(unit["text"])):
            rows.append({"window_index": index, "quote_id": f"q{i:04d}", "text": chunk,
                         "start": start, "end": end})
            index += 1
        result[uid] = rows
    return result


def window_catalog(paper_input: dict) -> list[dict]:
    catalog, quotes = unit_catalog(paper_input), quote_catalog(paper_input)
    return [{"window_index": q["window_index"], "unit_id": uid,
             "quote_id": q["quote_id"], "text": q["text"], "start": q["start"], "end": q["end"],
             "page": unit["page"], "source_kind": unit["source_kind"]}
            for uid, unit in catalog.items() for q in quotes[uid]]


def _policy(policy: dict | None) -> dict:
    value = copy.deepcopy(DEFAULT_POLICY if policy is None else policy)
    if (not isinstance(value, dict) or set(value) != set(DEFAULT_POLICY)
            or any(type(value[key]) is not int or value[key] <= 0 for key in DEFAULT_POLICY)):
        raise ValueError("invalid_extraction_group_policy")
    return value


def _windows(paper_input: dict) -> list[tuple[int, int]]:
    return [(row["window_index"], len(row["text"])) for row in window_catalog(paper_input)]


def plan_groups(paper_input: dict, policy: dict | None = None) -> list[dict]:
    p = _policy(policy)
    windows = _windows(paper_input)
    groups: list[list[int]] = []
    current: list[int] = []
    chars = 0
    for wid, length in windows:
        if length > p["max_target_chars"]:
            raise ValueError("window_exceeds_target_chars:" + str(wid))
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
    facts = schema["properties"]["facts"]
    facts["maxItems"] = group["policy"]["max_facts"]
    facts["items"]["properties"]["primary_window"] = {"type": "integer", "enum": ids}
    facts["items"]["properties"]["support_windows"]["items"]["maximum"] = len(window_catalog(paper_input)) - 1
    schema["$defs"]["sourceWindows"]["items"]["maximum"] = len(window_catalog(paper_input)) - 1
    return schema


def _full_source(paper_input: dict, index: dict, task: dict) -> dict:
    view = make_view(paper_input, index, taxonomy_value=task["taxonomy"])
    quotes = quote_catalog(paper_input)
    units = copy.deepcopy(view["units"])
    for row in units:
        row[3] = [[q["window_index"], q["text"]] for q in quotes[row[0]]]
    return {"schema_version": "paper-extraction-numbered-source/v3", "source_identity": view["source_identity"],
            "paper_input_canonical_sha256": view["paper_input_canonical_sha256"],
            "index_sha256": view["index_sha256"], "taxonomy_sha256": view["taxonomy_sha256"],
            "authority": view["authority"], "unit_columns": ["unit_id", "page", "source_kind", "numbered_windows", "reading_order", "bbox"],
            "units": units, "page_columns": view["page_columns"], "table_columns": view["table_columns"],
            "column_columns": view["column_columns"], "row_columns": view["row_columns"],
            "pages": view["pages"], "index_columns": view["index_columns"], "index_entries": view["index_entries"],
            "window_count": len(window_catalog(paper_input))}


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


def _window_evidence(window_index: int, paper_input: dict) -> dict:
    windows = window_catalog(paper_input)
    if type(window_index) is not int or not 0 <= window_index < len(windows):
        raise ValueError("window_index_invalid")
    row = windows[window_index]
    original = _source_units(paper_input)[row["unit_id"]]
    doc = original.get("document_char_span")
    if doc is not None and (type(doc.get("start")) is not int or type(doc.get("end")) is not int
                            or doc["end"] - doc["start"] != len(unit_catalog(paper_input)[row["unit_id"]]["text"])):
        raise ValueError("unit_document_span_length_mismatch:" + row["unit_id"])
    return {"window_index": window_index, "unit_id": row["unit_id"],
            "quote_id": row["quote_id"], "page": row["page"], "source_kind": row["source_kind"],
            "quote_or_cell_text": row["text"], "unit_char_span": {"start": row["start"], "end": row["end"]},
            "document_char_span": None if doc is None else {"start": doc["start"] + row["start"],
                                                               "end": doc["start"] + row["end"]},
            "source_identity": source_identity(paper_input)}


def validate_response(payload: dict, paper_input: dict, index: dict, group: dict, task: dict | None = None) -> list[str]:
    task = make_task() if task is None else task
    try:
        schema = response_schema(task, paper_input, index, group)
    except (KeyError, TypeError, ValueError) as exc:
        return ["extraction_group_context_invalid:" + str(exc)]
    errors = ["extraction_schema:" + error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(payload)]
    if errors:
        return errors
    windows = window_catalog(paper_input)
    targets = set(group["window_ids"])
    actual = [row["window_id"] for row in payload["coverage"]]
    if actual != group["window_ids"]:
        errors.append("extraction_target_order_or_coverage_mismatch")
    for i, mention in enumerate(payload["mentions"]):
        if any(type(w) is not int or not 0 <= w < len(windows) for w in mention["source_windows"]):
            errors.append(f"mention_source_window_invalid:{i}")
    seen = set()
    for i, fact in enumerate(payload["facts"]):
        if fact["subject_mention"] >= len(payload["mentions"]):
            errors.append(f"fact_subject_missing:{i}")
        primary = fact["primary_window"]
        if primary not in targets:
            errors.append(f"fact_primary_window_foreign:{i}")
        elif payload["coverage"][group["window_ids"].index(primary)]["state"] == "overflow":
            errors.append(f"fact_on_overflow_window:{i}")
        if any(type(s) is not int or not 0 <= s < len(windows) for s in fact["support_windows"]):
            errors.append(f"fact_support_window_invalid:{i}")
        claim = fact["claim"]
        if claim["kind"] == "link":
            if any(type(w) is not int or not 0 <= w < len(windows) for w in claim["target"]["source_windows"]):
                errors.append(f"fact_target_source_window_invalid:{i}")
        key = canonical_bytes(fact)
        if key in seen:
            errors.append(f"duplicate_fact:{i}")
        seen.add(key)
    return errors


def completion_status(payload: dict, group: dict) -> str:
    """Explicit overflow or exact cap saturation blocks paper assembly."""
    states = {row["state"] for row in payload["coverage"]}
    saturated = (len(payload["mentions"]) >= group["policy"]["max_mentions"]
                 or len(payload["facts"]) >= group["policy"]["max_facts"])
    return "incomplete" if "overflow" in states or saturated else "success"


def materialize_group(paper_input: dict, index: dict, task: dict, group: dict, record: dict) -> dict:
    """Admit and materialize exactly one complete group; no partial salvage."""
    from .workflow import replay_run
    replay_errors = replay_run(record, task, paper_input, index=index, extraction_group=group)
    if replay_errors:
        raise ValueError("extraction_group_record_replay_failed:" + ",".join(replay_errors[:4]))
    if (record["status"] != "success" or record["task"] != task
            or record.get("extraction_group") != group or record["index_sha256"] != index["index_sha256"]):
        raise ValueError("extraction_group_record_not_successful_or_bound")
    payload, normalization = parse_response(record["backend_result"]["raw_text"])
    if record["parsed_response"] != payload or record.get("response_normalization") != normalization:
        raise ValueError("extraction_group_raw_replay_mismatch")
    errors = validate_response(payload, paper_input, index, group, task)
    if errors:
        raise ValueError(errors)
    if completion_status(payload, group) != "success":
        raise ValueError("extraction_group_incomplete")
    prefix = group["group_id"]
    identity_prefix = record["job"]["parent_job_id"] + ":" + prefix
    coverage = [{"group_id": prefix, **copy.deepcopy(row)} for row in payload["coverage"]]
    mentions = [{"mention_id": identity_prefix + ":m" + str(i), "group_id": prefix,
                 **copy.deepcopy(mention),
                 "source_evidence": [_window_evidence(w, paper_input) for w in mention["source_windows"]],
                 "normalized_label_authority": "model_asserted_not_verified"}
                for i, mention in enumerate(payload["mentions"])]
    facts = []
    for i, fact in enumerate(payload["facts"]):
        row = copy.deepcopy(fact)
        row["subject_mention_id"] = identity_prefix + ":m" + str(row.pop("subject_mention"))
        facts.append({"fact_id": identity_prefix + ":f" + str(i), "group_id": prefix, **row,
                      "primary_evidence": _window_evidence(fact["primary_window"], paper_input),
                      "support_evidence": [_window_evidence(w, paper_input) for w in fact["support_windows"]],
                      "target_evidence": None if fact["claim"]["kind"] != "link" else
                      [_window_evidence(w, paper_input) for w in fact["claim"]["target"]["source_windows"]]})
    return {"group_id": prefix, "record_sha256": record["record_sha256"],
            "mentions": mentions, "facts": facts, "coverage": coverage}


def build_bundle(paper_input: dict, index: dict, task: dict, groups: list[dict], records: list[dict]) -> dict:
    if groups != plan_groups(paper_input, groups[0]["policy"] if groups else None):
        raise ValueError("extraction_group_plan_mismatch")
    if len(groups) != len(records):
        raise ValueError("extraction_group_record_count_mismatch")
    mentions, facts, coverage, refs = [], [], [], []
    binding = None
    for group, record in zip(groups, records):
        part = materialize_group(paper_input, index, task, group, record)
        current_binding = (record["profile_sha256"], record["job"]["parent_job_id"], record["index_sha256"])
        if binding is not None and binding != current_binding:
            raise ValueError("extraction_group_cross_record_binding_mismatch")
        binding = current_binding
        coverage.extend(part["coverage"])
        mentions.extend(part["mentions"])
        facts.extend(part["facts"])
        refs.append({"group_id": part["group_id"], "run_id": record["run_id"],
                     "record_sha256": part["record_sha256"]})
    return seal({"schema_version": BUNDLE_VERSION, "source_identity": source_identity(paper_input),
                 "paper_input_canonical_sha256": digest(paper_input), "index_sha256": index["index_sha256"],
                 "task_sha256": task["task_sha256"], "groups": copy.deepcopy(groups),
                 "profile_sha256": binding[0] if binding else None,
                 "parent_job_id": binding[1] if binding else None,
                 "record_refs": refs, "coverage": coverage, "mentions": mentions, "facts": facts,
                 "observation_count": len(facts), "mention_record_count": len(mentions),
                 "resolved_object_count": None, "semantic_correctness": "not_established",
                 "identity_resolution": "unresolved_model_hypotheses_only",
                 "label_evidence_scope": "cited_window_context_not_name_equivalence_or_entailment"}, "bundle_sha256")


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
