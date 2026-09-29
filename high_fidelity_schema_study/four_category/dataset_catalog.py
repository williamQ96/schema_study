"""Scoped dataset objects, derived from replayed evidence, never from model output."""
from __future__ import annotations

import copy
import re
from pathlib import Path

from .common import contained, digest, file_digest, read_json, seal, seal_errors
from .dataset import verify_dataset

VERSION = "scoped-dataset-catalog/v1"


def build_catalog(binding: dict, root: Path) -> dict:
    """Explicit resource selection; unselected workbook sheets remain recorded.

    v1 certifies header inventories for CSV/TSV/XLSX only. Other formats fail
    explicitly rather than treating sampled paths as a complete schema.
    """
    if binding.get("schema_version") != "dataset-scope-binding/v1":
        raise ValueError("scope_binding_version")
    tables, excluded, sources = [], [], []
    scopes = set()
    for resource in binding["resources"]:
        scope = resource["scope_id"]
        if not isinstance(scope, str) or not scope or scope in scopes:
            raise ValueError("duplicate_or_invalid_scope")
        scopes.add(scope)
        path = contained(root, resource["evidence_path"])
        if file_digest(path) != resource["evidence_file_bytes_sha256"]:
            raise ValueError("dataset_evidence_bytes_changed:" + scope)
        bundle = read_json(path)
        if bundle.get("status") != "pass":
            raise ValueError("dataset_evidence_not_pass:" + scope)
        errors = verify_dataset(bundle, contained(root, resource["source_root"]))
        if errors:
            raise ValueError("dataset_derivation_invalid:" + str(errors[:3]))
        fmt = bundle["parser"]["format"]
        if fmt not in {"csv", "tsv", "xlsx"}:
            raise ValueError("catalog_inventory_unsupported_format:" + fmt)
        sheet = resource.get("sheet")
        if (fmt == "xlsx" and not isinstance(sheet, str)) or (fmt != "xlsx" and sheet is not None):
            raise ValueError("explicit_sheet_selection_required_for_xlsx_only")
        ev = {e["evidence_id"]: e for e in bundle["evidence"]}
        fields = []
        for fact in bundle["facts"]:
            if fact["predicate"] != "name":
                continue
            refs = [ev[eid] for eid in fact["evidence_ids"]]
            if fmt == "xlsx" and any(e["locator"].get("sheet") != sheet for e in refs):
                excluded.append({"scope_id": scope, "reason": "outside_selected_data_sheet",
                                 "fact": copy.deepcopy(fact), "evidence": copy.deepcopy(refs)})
                continue
            name = fact["value"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("unnamed_column_requires_explicit_resolution:" + scope)
            if fact["coverage"].get("mode") != "full":
                raise ValueError("incomplete_header_inventory")
            if fmt == "xlsx" and any(not re.fullmatch(r"[A-Z]+1", e["locator"].get("cell", "")) for e in refs):
                raise ValueError("header_not_first_row")
            related = [copy.deepcopy(f) for f in bundle["facts"] if f["subject_id"] == fact["subject_id"]]
            fields.append({"object_id": scope + "::" + fact["subject_id"], "scope_id": scope,
                           "name": name, "source_subject_id": fact["subject_id"],
                           "role": resource.get("field_roles", {}).get(name, "unspecified"),
                           "name_fact": copy.deepcopy(fact), "name_evidence": copy.deepcopy(refs),
                           "property_facts": related})
        if not fields:
            raise ValueError("selected_scope_has_no_fields:" + scope)
        if not set(resource.get("field_roles", {})) <= {f["name"] for f in fields}:
            raise ValueError("field_role_references_unknown_name")
        if any(f["role"] not in {"predictor", "target", "identifier", "unspecified"} for f in fields):
            raise ValueError("invalid_field_role")
        tables.append({"scope_id": scope, "inventory": "complete_selected_header",
                       "sheet": sheet, "fields": fields, "dataset_sources": bundle["sources"],
                       "parser": bundle["parser"], "bundle_canonical_sha256": bundle["bundle_sha256"]})
        sources.append({"evidence_path": resource["evidence_path"],
                        "file_bytes_sha256": file_digest(path), "dataset_sources": bundle["sources"]})
    if not tables:
        raise ValueError("empty_dataset_scope")
    return seal({"schema_version": VERSION, "paper_id": binding["paper_id"],
                 "binding_canonical_sha256": digest(binding), "sources": sources,
                 "tables": tables, "excluded_name_facts": excluded,
                 "object_count": sum(len(t["fields"]) for t in tables),
                 "authority": "scoped_observed_dataset_objects_not_paper_visibility",
                 "aliases": []}, "catalog_sha256")


def verify_catalog(catalog: dict, binding: dict, root: Path) -> list[str]:
    errors = seal_errors(catalog, "catalog_sha256")
    try:
        if catalog != build_catalog(binding, root):
            errors.append("catalog_derivation_mismatch")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errors.append("catalog_replay:" + str(exc))
    return errors


def fields(catalog: dict) -> list[dict]:
    if seal_errors(catalog, "catalog_sha256") or catalog.get("schema_version") != VERSION:
        raise ValueError("invalid_catalog_identity")
    return [f for t in catalog["tables"] for f in t["fields"]]


def load_aliases(document: dict, catalog: dict, root: Path) -> list[dict]:
    """Read explicit alias records from bound JSON metadata, not LLM assertions.

    The external metadata origin remains an explicit trust boundary. A source
    hash proves identity, not that a user-created codebook is authoritative.
    """
    objects = {f["object_id"]: f for f in fields(catalog)}
    if document.get("catalog_sha256") != catalog["catalog_sha256"]:
        raise ValueError("alias_catalog_mismatch")
    result = []
    for row in document["entries"]:
        obj = objects[row["object_id"]]
        path = contained(root, row["source_path"])
        if file_digest(path) != row["source_file_bytes_sha256"]:
            raise ValueError("alias_source_changed")
        value = read_json(path)
        pointer = row["json_pointer"]
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise ValueError("alias_pointer_required")
        for part in pointer[1:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        if (not isinstance(row.get("alias"), str) or not row["alias"].strip()
                or value != {"name": obj["name"], "alias": row["alias"], "scope_id": obj["scope_id"]}):
            raise ValueError("alias_declaration_mismatch")
        result.append({**copy.deepcopy(row), "basis": "bound_metadata_alias_declaration"})
    return result
