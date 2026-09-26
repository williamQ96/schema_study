"""Full-paper input, automatic index admission and v4 extraction validation.

Semantic correctness is deliberately not inferred from a passing locator gate.
All new calls consume v3 evidence; historical readers remain in their old modules.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import jsonschema

from ..paper_layout_evidence import validate_layout_bundle, resolve_evidence_quote
from ..paper_layout_evidence_v3 import word_coverage_audit
from .common import ROOT, CATEGORIES, contained, digest, file_digest, identity, read_json, seal, seal_errors, strict_json, taxonomy


def source_identity(paper_input: dict) -> dict:
    return {"document_id": paper_input["paper_id"], "paper_sha256": paper_input["source_pdf_sha256"],
            "preprocessing_sha256": paper_input["layout_preprocessing_sha256"]}


def unit_catalog(paper_input: dict) -> dict[str, dict]:
    units = {}
    for page in paper_input["pages"]:
        items = [(u, u.get("region_kind", "paragraph")) for u in page["text_regions"]]
        for table in page["tables"]:
            items.append((table["caption"], "table_caption"))
            items.extend((u, "table_header") for u in table.get("header_units", []))
            for row in table["rows"]:
                items.append((row, "table_row"))
                items.extend((u, "table_cell") for u in row["cells"])
        for unit, kind in items:
            uid = unit["unit_id"]
            if uid in units:
                raise ValueError(f"duplicate unit: {uid}")
            units[uid] = {"unit_id": uid, "page": page["page"], "source_kind": kind, "text": unit["text"]}
    return units


def prepare_paper(layout_path: Path, reading_path: Path, input_path: Path, pdf_path: Path, *, root: Path) -> dict:
    paths = {"layout": layout_path, "reading_text": reading_path, "input": input_path, "pdf": pdf_path}
    artifacts = {name: identity(Path(path), root) for name, path in paths.items()}
    paper_input = read_json(input_path)
    value = seal({"schema_version": "four-category-paper-source/v1", "source_identity": source_identity(paper_input),
                  "artifacts": artifacts}, "source_sha256")
    errors = verify_paper(value, root)
    if errors:
        raise ValueError(errors)
    return value


def verify_paper(source: dict, root: Path) -> list[str]:
    errors = seal_errors(source, "source_sha256")
    try:
        paths = {}
        for name in ("layout", "reading_text", "input", "pdf"):
            item = source["artifacts"][name]
            path = contained(root, item["path"])
            paths[name] = path
            if item.get("hash_mode") != "file_bytes" or file_digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                errors.append(f"{name}:file_identity_mismatch")
        layout, paper_input = read_json(paths["layout"]), read_json(paths["input"])
        errors.extend("input_schema:" + e.message for e in jsonschema.Draft202012Validator(read_json(ROOT / "templates/paper_evidence_input_v3.schema.json")).iter_errors(paper_input))
        reading = paths["reading_text"].read_text(encoding="utf-8")
        if layout.get("schema_version") != "paper-evidence-layout/v3" or paper_input.get("schema_version") != "paper-evidence-input/v3":
            errors.append("new_lane_requires_v3_evidence")
        if file_digest(paths["pdf"]) != layout["source_pdf_sha256"]:
            errors.append("pdf_layout_identity_mismatch")
        if source_identity(paper_input) != source["source_identity"]:
            errors.append("source_identity_mismatch")
        errors.extend(validate_layout_bundle(layout, reading, paper_input))
        if word_coverage_audit(layout)["status"] != "pass":
            errors.append("source_word_coverage_failed")
        unit_catalog(paper_input)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"paper_source:{exc}")
    return errors


def classification_errors(response: dict, paper_input: dict) -> list[str]:
    schema = read_json(ROOT / "templates/paper_category_response_v1.schema.json")
    errors = ["classification_schema:" + e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(response)]
    if errors:
        return errors
    if response["source_identity"] != source_identity(paper_input):
        errors.append("classification_source_identity_mismatch")
    catalog = unit_catalog(paper_input)
    ids = [e["unit_id"] for e in response["entries"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_classification_unit")
    if set(ids) != set(catalog):
        errors.append("classification_unit_set_mismatch")
    for entry in response["entries"]:
        unit = catalog.get(entry["unit_id"])
        if not unit:
            continue
        if entry["page"] != unit["page"]:
            errors.append("classification_page_mismatch:" + entry["unit_id"])
        if entry["state"] == "classified":
            if not entry["categories"] or not entry["evidence_spans"]:
                errors.append("classified_entry_requires_categories_and_evidence")
        elif entry["categories"]:
            errors.append("noncategory_state_has_categories")
        for span in entry["evidence_spans"]:
            if not (0 <= span["start"] < span["end"] <= len(unit["text"])) or unit["text"][span["start"]:span["end"]] != span["quote"]:
                errors.append("classification_span_quote_mismatch:" + entry["unit_id"])
    return errors


def build_index(raw_text: str, paper_input: dict, producer: dict, *, taxonomy_value: dict | None = None) -> dict:
    response = strict_json(raw_text)
    errors = classification_errors(response, paper_input)
    if errors:
        raise ValueError(errors)
    for key in ("run_id", "profile_sha256", "task_sha256", "request_sha256"):
        if not producer.get(key):
            raise ValueError(f"index producer requires {key}")
    order = {uid: i for i, uid in enumerate(unit_catalog(paper_input))}
    return seal({"schema_version": "paper-category-index/v1", "taxonomy_sha256": digest(taxonomy_value or taxonomy()),
                 "source_identity": source_identity(paper_input), "paper_input_canonical_sha256": digest(paper_input),
                 "entries": sorted(response["entries"], key=lambda x: order[x["unit_id"]]),
                 "raw_response": raw_text, "producer": copy.deepcopy(producer),
                 "authority": "automated_fallible_navigation_not_gold", "semantic_review": "not_established"}, "index_sha256")


def verify_index(index: dict, paper_input: dict, *, taxonomy_value: dict | None = None) -> list[str]:
    errors = seal_errors(index, "index_sha256")
    try:
        replayed = build_index(index["raw_response"], paper_input, index["producer"], taxonomy_value=taxonomy_value)
        if replayed != index:
            errors.append("classification_raw_derivation_mismatch")
    except (ValueError, TypeError, KeyError, jsonschema.SchemaError) as exc:
        errors.append(f"classification_replay:{exc}")
    return errors


def categorized_input(paper_input: dict, index: dict, *, taxonomy_value: dict | None = None) -> dict:
    tax = taxonomy_value or taxonomy()
    errors = verify_index(index, paper_input, taxonomy_value=tax)
    if errors:
        raise ValueError(errors)
    return seal({"schema_version": "paper-categorized-input/v1", "taxonomy_sha256": digest(tax),
                 "paper_input": copy.deepcopy(paper_input),
                 "category_index": {"index_sha256": index["index_sha256"], "entries": copy.deepcopy(index["entries"]),
                                    "authority": "automated_fallible_navigation_not_gold"}}, "categorized_input_sha256")


def verify_categorized(value: dict, paper_input: dict, index: dict, *, taxonomy_value: dict | None = None) -> list[str]:
    try:
        return [] if value == categorized_input(paper_input, index, taxonomy_value=taxonomy_value) else ["categorized_input_derivation_mismatch"]
    except (ValueError, TypeError, KeyError) as exc:
        return [f"categorized_input:{exc}"]


def pointer(value: Any, path: str) -> Any:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("invalid JSON pointer")
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not part.isdecimal() or str(int(part)) != part:
                raise ValueError("invalid array index")
            value = value[int(part)]
        else:
            value = value[part]
    return value


def assertion_pointers(claim: dict) -> list[str]:
    targets = ["/reported_name"]
    for name in ("parent_claim_id", "canonical_path"):
        if claim.get(name) is not None:
            targets.append("/" + name)
    for name in ("datatype", "shape_or_dimensions"):
        if claim[name]["status"] != "unknown":
            targets.append("/" + name)
    for name, value in claim["semantic_annotations"].items():
        if value["status"] != "unknown":
            targets.append("/semantic_annotations/" + name)
    for name in ("constraints", "relationships", "encoding", "syntax"):
        targets.extend(f"/{name}/{i}" for i, value in enumerate(claim[name]) if value["status"] != "unknown")
    return targets


def extraction_errors(payload: dict, paper_input: dict, *, layout: dict | None = None, reading_text: str | None = None) -> list[str]:
    schema = read_json(ROOT / "templates/paper_derived_schema_v4.schema.json")
    errors = ["extraction_schema:" + e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(payload)]
    if errors:
        return errors
    if payload["source_document"] != source_identity(paper_input):
        errors.append("extraction_source_identity_mismatch")
    ids = [c["claim_id"] for c in payload["claims"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_claim_id")
    catalog = unit_catalog(paper_input)
    parents = {c["claim_id"]: c["parent_claim_id"] for c in payload["claims"]}
    for cid in parents:
        seen, parent = {cid}, parents[cid]
        while parent in parents:
            if parent in seen:
                errors.append(f"{cid}:parent_cycle")
                break
            seen.add(parent)
            parent = parents[parent]
    for claim in payload["claims"]:
        cid = claim["claim_id"]
        if claim["parent_claim_id"] is not None and claim["parent_claim_id"] not in ids:
            errors.append(f"{cid}:invalid_parent")
        if any(r["target_claim_id"] not in ids for r in claim["relationships"]):
            errors.append(f"{cid}:invalid_relationship_target")
        evs = {e["evidence_id"]: e for e in claim["evidence"]}
        if len(evs) != len(claim["evidence"]):
            errors.append(f"{cid}:duplicate_evidence_id")
        for ev in evs.values():
            unit = catalog.get(ev["unit_id"])
            if not unit or ev["document_id"] != paper_input["paper_id"] or ev["page"] != unit["page"]:
                errors.append(f"{cid}:invalid_evidence_identity")
                continue
            if ev["source_kind"] != unit["source_kind"]:
                errors.append(f"{cid}:evidence_kind_mismatch")
            if layout is not None and reading_text is not None:
                if resolve_evidence_quote(layout, reading_text, ev["unit_id"], ev["quote_or_cell_text"])["status"] != "resolved":
                    errors.append(f"{cid}:unresolved_quote")
            elif ev["quote_or_cell_text"] not in unit["text"]:
                errors.append(f"{cid}:quote_not_found")
            if ev["supports"] != "whole_claim":
                try:
                    pointer(claim, ev["supports"])
                except (ValueError, KeyError, IndexError, TypeError):
                    errors.append(f"{cid}:unresolved_support_pointer")
        required = set(assertion_pointers(claim))
        targets = []
        for ann in claim["category_annotations"]:
            target = ann["target_pointer"]
            targets.append(target)
            if target not in required:
                errors.append(f"{cid}:category_target_is_not_asserted_fact:{target}")
            for eid in ann["evidence_ids"]:
                if eid not in evs or evs[eid]["supports"] != target:
                    errors.append(f"{cid}:category_evidence_target_mismatch")
        if len(targets) != len(set(targets)):
            errors.append(f"{cid}:duplicate_category_target")
        if set(targets) != required:
            errors.append(f"{cid}:assertion_category_coverage_mismatch")
        qualified = [claim["datatype"], claim["shape_or_dimensions"], *claim["semantic_annotations"].values(), *claim["encoding"], *claim["syntax"]]
        for value in qualified:
            if (value["status"] == "unknown") != (value["value"] is None):
                errors.append(f"{cid}:qualified_status_value_conflict")
        if claim["claim_state"] == "inferred" and not claim["inference_basis"]:
            errors.append(f"{cid}:inference_basis_required")
    return errors


def fact_view(payload: dict) -> list[dict]:
    """Unique assertions; overlapping categories never multiply the global count."""
    facts = []
    for claim in payload["claims"]:
        for ann in claim["category_annotations"]:
            target = ann["target_pointer"]
            raw = pointer(claim, target)
            state = raw.get("status", claim["claim_state"]) if isinstance(raw, dict) else claim["claim_state"]
            facts.append({"fact_id": claim["claim_id"] + "#" + target, "subject_id": claim["claim_id"],
                          "predicate": target, "value": raw.get("value", raw) if isinstance(raw, dict) else raw,
                          "categories": ann["categories"], "basis": "declared" if state in {"supported", "reported"} else "unknown" if state == "conflicted" else state,
                          "source_state": state,
                          "basis_authority": "model_asserted_not_semantically_adjudicated",
                          "evidence_ids": ann["evidence_ids"], "origin": "paper"})
    return facts


def category_counts(facts: list[dict]) -> dict:
    unique = {f["fact_id"]: f for f in facts}
    if len(unique) != len(facts):
        raise ValueError("duplicate fact identity")
    return {"unique_facts": len(unique), "categories": {c: sum(c in f["categories"] for f in facts) for c in CATEGORIES},
            "category_counts_overlap": True}
