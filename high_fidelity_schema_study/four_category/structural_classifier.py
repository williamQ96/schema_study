"""Independent, fallible paper structure classification with exact source anchors.

This lane never reads dataset-side files and never invokes a model. It produces
navigation candidates, not established schema facts or semantic accuracy.
"""
from __future__ import annotations

import copy
import re

import jsonschema

from .backends import profile_hash
from .common import ROOT, canonical_bytes, digest, read_json, seal, seal_errors, strict_json
from .evidence_schema import validate_paper_input
from .paper import source_identity, unit_catalog

TASK_VERSION = "paper-structural-task/v1"
GROUP_VERSION = "paper-structural-group/v1"
RESPONSE_VERSION = "paper-structural-response/v1"
BUNDLE_VERSION = "paper-structural-annotations/v1"
RECORD_VERSION = "paper-structural-group-record/v1"
DEFAULT_POLICY = {"max_units": 128, "max_target_chars": 16000}
SCHEMA_PATH = ROOT / "templates/paper_structure_roles_v1.schema.json"
MARKERS = {"SOURCE_IDENTITY_JSON", "GROUP_SOURCE_JSON", "TARGET_SCHEMA_JSON"}


def make_task() -> dict:
    return seal({"kind": "structural_classification", "schema_version": TASK_VERSION,
                 "admission_rules_version": "structure-exact-quotes/v1",
                 "system_prompt": (ROOT / "templates/paper_structure_system_v1.txt").read_text(encoding="utf-8"),
                 "user_template": (ROOT / "templates/paper_structure_user_v1.txt").read_text(encoding="utf-8"),
                 "output_schema": read_json(SCHEMA_PATH)}, "task_sha256")


def task_errors(task: dict) -> list[str]:
    errors = seal_errors(task, "task_sha256")
    try:
        if task != make_task():
            errors.append("structural_task_mismatch")
        markers = re.findall(r"\{\{([A-Z_]+)\}\}", task["user_template"])
        if len(markers) != len(MARKERS) or set(markers) != MARKERS:
            errors.append("structural_task_markers_invalid")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("structural_task_invalid:" + str(exc))
    return errors


def _policy(value: dict | None) -> dict:
    policy = copy.deepcopy(DEFAULT_POLICY if value is None else value)
    if (not isinstance(policy, dict) or set(policy) != set(DEFAULT_POLICY)
            or any(type(policy[key]) is not int or policy[key] <= 0 for key in DEFAULT_POLICY)):
        raise ValueError("invalid_structural_group_policy")
    return policy


def _tables(paper_input: dict) -> dict[str, dict]:
    tables = {}
    for page in paper_input["pages"]:
        for table in page["tables"]:
            tid = table["table_id"]
            if tid in tables:
                raise ValueError("duplicate_table_id:" + tid)
            ids = [table["caption"]["unit_id"]]
            ids.extend(unit["unit_id"] for unit in table.get("header_units", []))
            for row in table["rows"]:
                ids.append(row["unit_id"])
                ids.extend(unit["unit_id"] for unit in row["cells"])
            tables[tid] = {"table_id": tid, "page": page["page"], "unit_ids": ids,
                           "caption_unit_id": ids[0],
                           "header_unit_ids": [unit["unit_id"] for unit in table.get("header_units", [])],
                           "rows": [{"row_unit_id": row["unit_id"],
                                     "cell_unit_ids": [unit["unit_id"] for unit in row["cells"]]}
                                    for row in table["rows"]],
                           "columns": copy.deepcopy(table["columns"]), "bbox": table.get("bbox")}
    return tables


def _nearby(page: dict, table: dict) -> list[str]:
    bbox = table.get("bbox")
    regions = page["text_regions"]
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return [u["unit_id"] for u in regions[-2:]]
    def distance(unit):
        box = unit.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            return float("inf")
        return max(0, box[1] - bbox[3], bbox[1] - box[3])
    ranked = sorted(enumerate(regions), key=lambda pair: (distance(pair[1]), pair[0]))[:2]
    return [unit["unit_id"] for _, unit in ranked]


def plan_groups(paper_input: dict, policy: dict | None = None) -> list[dict]:
    validate_paper_input(paper_input)
    p = _policy(policy)
    catalog, tables = unit_catalog(paper_input), _tables(paper_input)
    blocks: list[tuple[list[str], list[str], list[str]]] = []
    for page in paper_input["pages"]:
        current: list[str] = []
        chars = 0
        for unit in page["text_regions"]:
            uid = unit["unit_id"]
            length = len(catalog[uid]["text"])
            if length > p["max_target_chars"]:
                raise ValueError("unit_exceeds_max_target_chars:" + uid)
            if current and (len(current) >= p["max_units"] or chars + length > p["max_target_chars"]):
                blocks.append((current, [], []))
                current, chars = [], 0
            current.append(uid)
            chars += length
        if current:
            blocks.append((current, [], []))
        for table in page["tables"]:
            item = tables[table["table_id"]]
            ids = item["unit_ids"]
            if len(ids) > p["max_units"] or sum(len(catalog[uid]["text"]) for uid in ids) > p["max_target_chars"]:
                raise ValueError("table_exceeds_structural_group_policy:" + table["table_id"])
            blocks.append((ids, [table["table_id"]], _nearby(page, table)))
    expected = list(catalog)
    actual = [uid for ids, _, _ in blocks for uid in ids]
    if actual != expected:
        raise ValueError("structural_group_source_coverage_mismatch")
    identity = digest(paper_input)
    return [seal({"schema_version": GROUP_VERSION,
                  "group_id": digest({"version": GROUP_VERSION, "paper": identity, "policy": p,
                                      "index": i, "unit_ids": ids, "table_ids": tids}),
                  "paper_input_canonical_sha256": identity, "policy": copy.deepcopy(p),
                  "index": i, "total": len(blocks), "unit_ids": ids, "table_ids": tids,
                  "context_unit_ids": context}, "group_sha256")
            for i, (ids, tids, context) in enumerate(blocks)]


def group_errors(group: dict, paper_input: dict) -> list[str]:
    errors = seal_errors(group, "group_sha256")
    try:
        planned = plan_groups(paper_input, group["policy"])
        if group["schema_version"] != GROUP_VERSION or group["paper_input_canonical_sha256"] != digest(paper_input):
            errors.append("structural_group_identity_mismatch")
        if type(group["index"]) is not int or not 0 <= group["index"] < len(planned) or planned[group["index"]] != group:
            errors.append("structural_group_plan_mismatch")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append("structural_group_invalid:" + str(exc))
    return errors


def quote_options(text: str) -> list[str]:
    candidates = [text, text.strip()]
    for part in re.split(r"[\n\t]", text):
        candidates.extend((part, part.strip()))
    return list(dict.fromkeys(value for value in candidates if value.strip()))


def response_schema(task: dict, paper_input: dict, group: dict) -> dict:
    errors = task_errors(task) + group_errors(group, paper_input)
    if errors:
        raise ValueError(errors)
    schema = copy.deepcopy(task["output_schema"])
    units = unit_catalog(paper_input)
    ids = group["unit_ids"]
    schema["properties"]["group_id"] = {"const": group["group_id"]}
    reviews = schema["properties"]["unit_reviews"]
    reviews["minItems"] = reviews["maxItems"] = len(ids)
    reviews["prefixItems"] = [{**copy.deepcopy(reviews["items"]), "properties":
                               {**reviews["items"]["properties"], "unit_id": {"const": uid}}} for uid in ids]
    roles = schema["properties"]["table_roles"]
    roles["minItems"] = roles["maxItems"] = len(group["table_ids"])
    if group["table_ids"]:
        roles["items"]["properties"]["table_id"] = {"type": "string", "enum": group["table_ids"]}
    options = list(dict.fromkeys(option for uid in ids for option in quote_options(units[uid]["text"])))
    evidence = schema["$defs"]["evidence"]["properties"]
    evidence["unit_id"] = {"type": "string", "enum": ids}
    if options:
        evidence["quote"] = {"type": "string", "enum": options}
    else:
        schema["properties"]["mentions"]["maxItems"] = 0
        schema["properties"]["links"]["maxItems"] = 0
    return schema


def _group_source(paper_input: dict, group: dict) -> dict:
    units, tables = unit_catalog(paper_input), _tables(paper_input)
    def entry(uid):
        unit = units[uid]
        return {**unit, "quote_options": quote_options(unit["text"])}
    return {"group_id": group["group_id"], "unit_ids": group["unit_ids"],
            "units": [entry(uid) for uid in group["unit_ids"]],
            "tables": [{key: copy.deepcopy(value) for key, value in tables[tid].items() if key != "bbox"}
                       for tid in group["table_ids"]],
            "context_units": [entry(uid) for uid in group["context_unit_ids"]]}


def render_group(task: dict, paper_input: dict, group: dict) -> list[dict]:
    schema = response_schema(task, paper_input, group)
    values = {"SOURCE_IDENTITY_JSON": source_identity(paper_input),
              "GROUP_SOURCE_JSON": _group_source(paper_input, group), "TARGET_SCHEMA_JSON": schema}
    user = re.sub(r"\{\{([A-Z_]+)\}\}", lambda match: canonical_bytes(values[match[1]]).decode("utf-8"),
                  task["user_template"])
    return [{"role": "system", "content": task["system_prompt"]}, {"role": "user", "content": user}]


def _response_errors(payload: dict, paper_input: dict, group: dict, task: dict) -> list[str]:
    schema = response_schema(task, paper_input, group)
    errors = ["structural_schema:" + error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(payload)]
    if errors:
        return errors
    ids, units, tables = group["unit_ids"], unit_catalog(paper_input), _tables(paper_input)
    if [row["unit_id"] for row in payload["unit_reviews"]] != ids:
        errors.append("structural_unit_coverage_mismatch")
    if [row["table_id"] for row in payload["table_roles"]] != group["table_ids"]:
        errors.append("structural_table_coverage_mismatch")
    for role in payload["table_roles"]:
        table = tables.get(role["table_id"])
        if table is None:
            continue
        expected = [column["column_index"] for column in table["columns"]]
        actual = [column["column_index"] for column in role["columns"]]
        if actual != expected:
            errors.append("structural_column_coverage_mismatch:" + role["table_id"])
    mention_ids = [row["mention_id"] for row in payload["mentions"]]
    if mention_ids != list(range(len(mention_ids))):
        errors.append("structural_mention_ids_not_contiguous")
    seen_mentions = set()
    for i, mention in enumerate(payload["mentions"]):
        key = canonical_bytes({key: value for key, value in mention.items() if key != "mention_id"})
        if key in seen_mentions:
            errors.append(f"structural_duplicate_mention:{i}")
        seen_mentions.add(key)
    seen_links = set()
    for i, link in enumerate(payload["links"]):
        if (link["source_mention_id"] not in mention_ids or link["target_mention_id"] not in mention_ids
                or link["source_mention_id"] == link["target_mention_id"]):
            errors.append(f"structural_link_endpoint_invalid:{i}")
        key = canonical_bytes(link)
        if key in seen_links:
            errors.append(f"structural_duplicate_link:{i}")
        seen_links.add(key)
    evidence = [("table", role["evidence"]) for role in payload["table_roles"]]
    evidence += [("column", column["evidence"]) for role in payload["table_roles"] for column in role["columns"]]
    evidence += [("mention", row["evidence"]) for row in payload["mentions"]]
    evidence += [("link", row["evidence"]) for row in payload["links"]]
    review_states = {row["unit_id"]: row["state"] for row in payload["unit_reviews"]}
    for i, (kind, item) in enumerate(evidence):
        uid = item["unit_id"]
        if uid not in ids or not item["quote"].strip() or item["quote"] not in quote_options(units[uid]["text"]):
            errors.append(f"structural_evidence_not_exact_target:{kind}:{i}")
        if kind in {"mention", "link"} and review_states.get(uid) == "none":
            errors.append(f"structural_evidence_on_none_review:{kind}:{i}")
    return errors


def admit(raw_text: str, paper_input: dict, group: dict, task: dict | None = None) -> dict:
    task = make_task() if task is None else task
    try:
        if not isinstance(raw_text, str):
            raise ValueError("raw_text_required")
        payload = strict_json(raw_text)
        errors = _response_errors(payload, paper_input, group, task)
        return {"status": "admitted" if not errors else "unavailable", "payload": payload,
                "errors": errors, "normalization": {"kind": "bare_json"}}
    except (ValueError, TypeError, KeyError, IndexError, jsonschema.SchemaError) as exc:
        return {"status": "unavailable", "payload": None, "errors": ["structural_parse_or_context:" + str(exc)],
                "normalization": None}


def make_record(raw_text: str | None, paper_input: dict, task: dict, profile: dict, group: dict,
                *, backend_status: str = "success") -> dict:
    messages = render_group(task, paper_input, group)
    if backend_status == "success" and not isinstance(raw_text, str):
        raise ValueError("successful_structural_record_requires_raw_text")
    if backend_status not in {"success", "transport_error", "truncated", "invalid_request", "unavailable"}:
        raise ValueError("invalid_structural_backend_status")
    return seal({"schema_version": RECORD_VERSION, "group": copy.deepcopy(group),
                 "task_sha256": task["task_sha256"], "profile_sha256": profile_hash(profile),
                 "messages": messages, "request_sha256": digest(messages),
                 "backend_status": backend_status, "raw_text": raw_text}, "record_sha256")


def _record_errors(record: dict, paper_input: dict, task: dict, profile: dict, group: dict) -> list[str]:
    errors = seal_errors(record, "record_sha256")
    try:
        if record != make_record(record["raw_text"], paper_input, task, profile, group,
                                 backend_status=record["backend_status"]):
            errors.append("structural_record_derivation_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("structural_record_invalid:" + str(exc))
    return errors


def _selection(selected_groups, total: int) -> list[int]:
    if selected_groups is None:
        return list(range(total))
    if (not isinstance(selected_groups, list) or any(type(i) is not int or not 0 <= i < total for i in selected_groups)
            or selected_groups != sorted(set(selected_groups))):
        raise ValueError("invalid_structural_group_selection")
    return selected_groups


def build_annotations(paper_input: dict, task: dict, profile: dict, group_records: list[dict],
                      *, selected_groups: list[int] | None = None) -> dict:
    if task_errors(task):
        raise ValueError("structural_task_invalid")
    if not isinstance(profile, dict) or not isinstance(group_records, list):
        raise ValueError("structural_profile_or_records_invalid")
    policy = group_records[0]["group"]["policy"] if group_records else None
    groups = plan_groups(paper_input, policy)
    selected = _selection(selected_groups, len(groups))
    records = {}
    for record in group_records:
        index = record["group"]["index"]
        if index not in selected or index in records:
            raise ValueError("structural_record_unselected_or_duplicate")
        records[index] = record
    outcomes, annotations = [], []
    for group in groups:
        i = group["index"]
        if i not in selected:
            status, reason, payload, record_sha = "unreviewed", "not_selected", None, None
        elif i not in records:
            status, reason, payload, record_sha = "unavailable", "missing_record", None, None
        else:
            record = records[i]
            record_sha = record.get("record_sha256")
            issues = _record_errors(record, paper_input, task, profile, group)
            if issues:
                status, reason, payload = "unavailable", "record_invalid:" + ",".join(issues[:3]), None
            elif record["backend_status"] != "success":
                status, reason, payload = "unavailable", "backend_" + record["backend_status"], None
            else:
                admitted = admit(record["raw_text"], paper_input, group, task)
                status = "admitted" if admitted["status"] == "admitted" else "unavailable"
                reason = None if status == "admitted" else "response_invalid:" + ",".join(admitted["errors"][:3])
                payload = admitted["payload"] if status == "admitted" else None
        outcomes.append({"group_id": group["group_id"], "index": i, "unit_ids": group["unit_ids"],
                         "status": status, "reason": reason, "record_sha256": record_sha})
        if payload is not None:
            annotations.append({"group_id": group["group_id"], "unit_reviews": copy.deepcopy(payload["unit_reviews"]),
                                "table_roles": copy.deepcopy(payload["table_roles"]),
                                "mentions": copy.deepcopy(payload["mentions"]), "links": copy.deepcopy(payload["links"])})
    unit_outcomes = [{"unit_id": uid, "group_id": outcome["group_id"], "status": outcome["status"]}
                     for outcome in outcomes for uid in outcome["unit_ids"]]
    return seal({"schema_version": BUNDLE_VERSION, "source_identity": source_identity(paper_input),
                 "paper_input_canonical_sha256": digest(paper_input), "task": copy.deepcopy(task),
                 "profile": copy.deepcopy(profile), "profile_sha256": profile_hash(profile),
                 "groups": groups, "selected_group_indexes": selected,
                 "records": copy.deepcopy(group_records), "group_outcomes": outcomes,
                 "unit_outcomes": unit_outcomes, "annotations": annotations,
                 "authority": "fallible_auxiliary_navigation_not_schema_truth",
                 "semantic_correctness": "not_established", "dataset_input_used": False}, "bundle_sha256")


def verify_annotations(bundle: dict, paper_input: dict, task: dict | None = None,
                       profile: dict | None = None, group_records: list[dict] | None = None) -> list[str]:
    errors = seal_errors(bundle, "bundle_sha256")
    try:
        task = bundle["task"] if task is None else task
        profile = bundle["profile"] if profile is None else profile
        records = bundle["records"] if group_records is None else group_records
        expected = build_annotations(paper_input, task, profile, records,
                                     selected_groups=bundle["selected_group_indexes"])
        if expected != bundle:
            errors.append("structural_bundle_derivation_mismatch")
    except (KeyError, TypeError, ValueError, IndexError, jsonschema.SchemaError) as exc:
        errors.append("structural_bundle_replay:" + str(exc))
    return errors
