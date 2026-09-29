"""V13 repair classifier: group-local context and a bound generation contract.

The v3 classifier remains frozen. This task has its own identity, while its
response and resolved navigation entries retain the established v3 shape.
"""
from __future__ import annotations

import copy
import re

import jsonschema

from .classification_v2 import _group_errors, parse_response, plan_groups
from .classification_v3 import quote_options, resolve_entries as _resolve_entries
from .common import ROOT, canonical_bytes, read_json, seal, taxonomy
from .paper import source_identity, unit_catalog

TASK_VERSION = "four-category-task/v13r2"
PROTOCOL = "classification-anchors/v13r2"
RESPONSE_VERSION = "paper-category-response/v3"
MARKERS = {"TAXONOMY_JSON", "TARGET_SCHEMA_JSON", "SOURCE_IDENTITY_JSON",
           "UNIT_CATALOG_JSON", "TARGET_GROUP_JSON", "QUOTE_OPTIONS_JSON"}


def make_task(taxonomy_value: dict | None = None) -> dict:
    value = taxonomy() if taxonomy_value is None else taxonomy_value
    if not isinstance(value, dict):
        raise ValueError("taxonomy must be an object")
    system = (ROOT / "templates/four_category_classification_system_v3.txt").read_text(encoding="utf-8")
    system = system.replace("Use the full canonical paper catalog for context and only target units for entries.",
                            "The supplied source context consists exactly of target-group units; classify only these units.")
    user = (ROOT / "templates/four_category_classification_user_v3.txt").read_text(encoding="utf-8")
    user = user.replace("FULL CANONICAL PAPER UNIT CATALOG", "TARGET-GROUP SOURCE UNITS")
    return seal({"kind": "classification", "schema_version": TASK_VERSION,
                 "admission_rules_version": PROTOCOL, "system_prompt": system,
                 "user_template": user,
                 "output_schema": read_json(ROOT / "templates/paper_category_response_v3.schema.json"),
                 "taxonomy": copy.deepcopy(value)}, "task_sha256")


def task_errors(task: dict) -> list[str]:
    if not isinstance(task, dict):
        return ["classification_v13r2_task_object_required"]
    try:
        return [] if task == make_task(task["taxonomy"]) else ["classification_v13r2_task_identity_mismatch"]
    except (KeyError, TypeError, ValueError):
        return ["classification_v13r2_task_invalid"]


def response_schema(task: dict, paper_input: dict, group: dict) -> dict:
    errors = task_errors(task) + _group_errors(group, paper_input)
    if errors:
        raise ValueError(errors)
    schema = copy.deepcopy(task["output_schema"])
    identity = source_identity(paper_input)
    for key, value in identity.items():
        schema["properties"]["source_identity"]["properties"][key] = {"const": value}
    catalog = unit_catalog(paper_input)
    options = quote_options(paper_input, group)
    base = schema["properties"]["entries"]["items"]
    entries = []
    for uid, row in zip(group["unit_ids"], options):
        quote_ids = [option["quote_id"] for option in row["options"]]
        states = ["none"] if not catalog[uid]["text"] else ["classified", "none", "uncertain"]
        variants = []
        for state in states:
            item = copy.deepcopy(base)
            item["properties"]["unit_id"] = {"const": uid}
            item["properties"]["state"] = {"const": state}
            if state == "classified":
                item["properties"]["categories"]["minItems"] = 1
                item["properties"]["evidence_quote_ids"]["minItems"] = 1
                item["properties"]["evidence_quote_ids"]["items"] = {"enum": quote_ids}
            else:
                item["properties"]["categories"]["maxItems"] = 0
                item["properties"]["evidence_quote_ids"]["maxItems"] = 0
            variants.append(item)
        entries.append({"anyOf": variants})
    schema["properties"]["entries"] = {"type": "array", "minItems": len(entries),
                                       "maxItems": len(entries), "items": {}, "prefixItems": entries}
    return schema


def render_group(task: dict, paper_input: dict, group: dict) -> list[dict]:
    schema = response_schema(task, paper_input, group)
    catalog = unit_catalog(paper_input)
    values = {"TAXONOMY_JSON": task["taxonomy"], "TARGET_SCHEMA_JSON": schema,
              "SOURCE_IDENTITY_JSON": source_identity(paper_input),
              "UNIT_CATALOG_JSON": [catalog[uid] for uid in group["unit_ids"]],
              "TARGET_GROUP_JSON": {"group_id": group["group_id"], "unit_ids": group["unit_ids"]},
              "QUOTE_OPTIONS_JSON": quote_options(paper_input, group)}
    user = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: canonical_bytes(values[m[1]]).decode("utf-8"), task["user_template"])
    return [{"role": "system", "content": task["system_prompt"]}, {"role": "user", "content": user}]


def validate_response(payload: dict, paper_input: dict, group: dict) -> list[str]:
    try:
        schema = response_schema(make_task(), paper_input, group)
    except ValueError as exc:
        return [str(exc)]
    return ["classification_schema:" + error.message for error in
            jsonschema.Draft202012Validator(schema).iter_errors(payload)]


def resolve_entries(payload: dict, paper_input: dict, group: dict) -> list[dict]:
    errors = validate_response(payload, paper_input, group)
    if errors:
        raise ValueError(errors)
    return _resolve_entries(payload, paper_input, group)
