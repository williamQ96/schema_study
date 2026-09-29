"""Evidence-text-complete compact prompt view for v4 schema extraction.

The source paper and category index stay immutable. This projection omits layout
audit metadata and classification quote proofs from the model request; replay
always rebuilds it from those bound artifacts.
"""
from __future__ import annotations

import copy
import re

import jsonschema

from .common import ROOT, canonical_bytes, digest, read_json, seal, seal_errors, taxonomy
from .paper import source_identity, unit_catalog, verify_index
from .evidence_schema import validate_paper_input

TASK_VERSION = "four-category-task/v3"
VIEW_VERSION = "paper-categorized-request/v2"
VIEW_VERSION_WITH_AVAILABILITY = "paper-categorized-request/v3"
MARKERS = {"SOURCE_IDENTITY_JSON", "TAXONOMY_JSON", "TARGET_SCHEMA_JSON", "COMPACT_CATEGORIZED_INPUT_JSON"}


def make_task(taxonomy_value: dict | None = None) -> dict:
    value = taxonomy() if taxonomy_value is None else taxonomy_value
    if not isinstance(value, dict):
        raise ValueError("taxonomy must be an object")
    return seal({"kind": "extraction", "schema_version": TASK_VERSION,
                 "admission_rules_version": "extraction-compact/v2",
                 "system_prompt": (ROOT / "templates/paper_to_schema_system_v4.txt").read_text(encoding="utf-8"),
                 "user_template": (ROOT / "templates/paper_to_schema_user_v5.txt").read_text(encoding="utf-8"),
                 "output_schema": read_json(ROOT / "templates/paper_derived_schema_v4.schema.json"),
                 "taxonomy": copy.deepcopy(value)}, "task_sha256")


def task_errors(task: dict) -> list[str]:
    errors = seal_errors(task, "task_sha256")
    if task.get("kind") != "extraction" or task.get("schema_version") != TASK_VERSION or task.get("admission_rules_version") != "extraction-compact/v2":
        errors.append("compact_extraction_task_version_mismatch")
    if task.get("output_schema") != read_json(ROOT / "templates/paper_derived_schema_v4.schema.json"):
        errors.append("compact_extraction_output_schema_mismatch")
    template = task.get("user_template")
    if not isinstance(task.get("system_prompt"), str) or not isinstance(template, str):
        errors.append("compact_extraction_prompt_type_mismatch")
    elif len(re.findall(r"\{\{([A-Z_]+)\}\}", template)) != len(MARKERS) or set(re.findall(r"\{\{([A-Z_]+)\}\}", template)) != MARKERS:
        errors.append("compact_extraction_markers_mismatch")
    return errors


def _source_units(paper_input: dict) -> dict[str, dict]:
    found: dict[str, dict] = {}
    def visit(value):
        if isinstance(value, dict):
            if "unit_id" in value and "text" in value:
                uid = value["unit_id"]
                if uid in found:
                    raise ValueError("duplicate source unit:" + uid)
                found[uid] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(paper_input["pages"])
    return found


def _page_rows(paper_input: dict) -> list[list]:
    rows = []
    for page in paper_input["pages"]:
        tables = []
        for table in page["tables"]:
            columns = [[col["column_index"], col["header"]] for col in table["columns"]]
            table_rows = [[row["unit_id"], [cell["unit_id"] for cell in row["cells"]]] for row in table["rows"]]
            tables.append([table["table_id"], table.get("table_number"), copy.deepcopy(table.get("bbox")),
                           table["caption"]["unit_id"], columns,
                           [unit["unit_id"] for unit in table["header_units"]], table_rows])
        rows.append([page["page"], page["layout_mode"], [unit["unit_id"] for unit in page["text_regions"]], tables])
    return rows


def make_view(paper_input: dict, index: dict, *, taxonomy_value: dict | None = None) -> dict:
    validate_paper_input(paper_input)
    tax = taxonomy() if taxonomy_value is None else taxonomy_value
    navigation_v4 = index.get("schema_version") in {"paper-category-index/v4", "paper-category-index/disabled-v1"}
    if navigation_v4:
        errors = verify_index(index, paper_input, taxonomy_value=tax)
    else:
        errors = verify_index(index, paper_input, taxonomy_value=tax)
    if errors:
        raise ValueError(errors)
    catalog = unit_catalog(paper_input)
    originals = _source_units(paper_input)
    if set(catalog) != set(originals):
        raise ValueError("source_catalog_unit_coverage_mismatch")
    ids = list(catalog)
    entries = index["entries"]
    if [entry["unit_id"] for entry in entries] != ids:
        raise ValueError("index_unit_order_mismatch")
    is_v4 = paper_input.get("schema_version") in {"paper-evidence-input/v4", "paper-evidence-input/v5",
                                                     "paper-evidence-input/v6"}
    units = [[uid, unit["page"], unit["source_kind"], unit["text"],
              originals[uid].get("reading_order"), copy.deepcopy(originals[uid].get("bbox")),
              *([originals[uid].get("structure_status")] if is_v4 else [])]
             for uid, unit in catalog.items()]
    if navigation_v4:
        projection = [[entry["availability"],
                       entry["prediction"]["state"] if entry["availability"] == "available" else None,
                       copy.deepcopy(entry["prediction"]["categories"]) if entry["availability"] == "available" else None]
                      for entry in entries]
        index_columns = ["availability", "state", "categories"]
    else:
        projection = [[entry["state"], copy.deepcopy(entry["categories"])] for entry in entries]
        index_columns = ["state", "categories"]
    return seal({"schema_version": VIEW_VERSION_WITH_AVAILABILITY if navigation_v4 else VIEW_VERSION, "source_identity": source_identity(paper_input),
                 "paper_input_canonical_sha256": digest(paper_input), "index_sha256": index["index_sha256"],
                 "taxonomy_sha256": digest(tax), "authority": index["authority"] if navigation_v4 else "automated_fallible_navigation_not_gold",
                 "scope": "evidence_text_complete_with_page_table_topology",
                 "omitted_metadata": ["document_char_span", "document_token_span", "region_char_span",
                                      "source_word_indexes", "classification_quotes", "classification_rationale",
                                      "classification_derived_spans", "layout_detection_audit"],
                 "unit_columns": ["unit_id", "page", "source_kind", "text", "reading_order", "bbox",
                                 *( ["structure_status"] if is_v4 else [])],
                 "units": units,
                 "page_columns": ["page", "layout_mode", "text_region_unit_ids", "tables"],
                 "table_columns": ["table_id", "table_number", "bbox", "caption_unit_id", "columns", "header_unit_ids", "rows"],
                 "column_columns": ["column_index", "header"], "row_columns": ["row_unit_id", "cell_unit_ids"],
                 "pages": _page_rows(paper_input),
                 "index_columns": index_columns, "index_entries": projection}, "view_sha256")


def verify_view(view: dict, paper_input: dict, index: dict, *, taxonomy_value: dict | None = None) -> list[str]:
    errors = seal_errors(view, "view_sha256")
    try:
        if view != make_view(paper_input, index, taxonomy_value=taxonomy_value):
            errors.append("compact_view_derivation_mismatch")
    except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as exc:
        errors.append("compact_view_replay:" + str(exc))
    return errors


def render_task_v2(task: dict, paper_input: dict, index: dict) -> list[dict]:
    errors = task_errors(task)
    if errors:
        raise ValueError(errors)
    view = make_view(paper_input, index, taxonomy_value=task["taxonomy"])
    values = {"SOURCE_IDENTITY_JSON": source_identity(paper_input), "TAXONOMY_JSON": task["taxonomy"],
              "TARGET_SCHEMA_JSON": task["output_schema"], "COMPACT_CATEGORIZED_INPUT_JSON": view}
    user = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: canonical_bytes(values[m[1]]).decode("utf-8"), task["user_template"])
    return [{"role": "system", "content": task["system_prompt"]}, {"role": "user", "content": user}]
