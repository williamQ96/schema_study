"""Spatial layout with a separately auditable, conservative visibility projection.

Every frozen source word remains in layout and audit reading text. Only words
independently proven nonpainting are omitted from the compact model input.
Unknown visibility is retained; this is not OCR or semantic adjudication.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .paper_layout_evidence import (assign_document_spans, canonical_json_bytes,
    compact_model_input, iter_layout_units, sha256_bytes, sha256_file, text_region)
from .paper_layout_evidence_v4 import (build_spatial_page, compact_model_input_v4,
    validate_layout_bundle_v4)
from .paper_pdf_visibility import (POLICY_VERSION, build_visibility_bundle,
    proven_exclusion_indices, validate_visibility_ledger)

LAYOUT_VERSION = "paper-evidence-layout/v5"
INPUT_VERSION = "paper-evidence-input/v5"
PROFILE = "frozen-pdf-words-spatial-visibility-v5"
AUDIT_REGION = "proven_nonvisible_source_audit"


def _seal_layout(layout: dict) -> None:
    layout.pop("preprocessing_sha256", None)
    layout["preprocessing_sha256"] = sha256_bytes(canonical_json_bytes(layout))


def compact_model_input_v5(layout: dict) -> dict:
    if layout.get("schema_version") != LAYOUT_VERSION:
        raise ValueError("v5_layout_required")
    result = compact_model_input(layout)
    result["schema_version"] = INPUT_VERSION
    for source_page, page in zip(layout["pages"], result["pages"], strict=True):
        source_regions = {r["unit_id"]: r for r in source_page["regions"]}
        page["text_regions"] = [r for r in page["text_regions"] if r["region_kind"] != AUDIT_REGION]
        for region in page["text_regions"]:
            if "structure_status" in source_regions[region["unit_id"]]:
                region["structure_status"] = source_regions[region["unit_id"]]["structure_status"]
    ledger = layout["visibility_ledger"]
    result["visibility_projection"] = {
        "policy_version": POLICY_VERSION, "ledger_sha256": ledger["ledger_sha256"],
        "excluded_source_word_count": ledger["counts"].get("proven_nonvisible", 0),
        "retained_visibility_uncertain_word_count": ledger["counts"].get("uncertain", 0),
        "rule": "Only fully proven nonpainting words excluded; uncertain words retained. Audit offsets refer to complete source reading text.",
    }
    result.pop("model_input_sha256", None)
    result["model_input_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def build_bundle(v4_layout: dict, *, pdf_path: Path) -> tuple[dict, str, dict, list[dict], dict]:
    import pdfplumber
    if v4_layout.get("schema_version") != "paper-evidence-layout/v4":
        raise ValueError("v4_spatial_ancestor_required")
    body = dict(v4_layout); ancestor_hash = body.pop("preprocessing_sha256", None)
    if ancestor_hash != sha256_bytes(canonical_json_bytes(body)):
        raise ValueError("v4_ancestor_hash_mismatch")
    if sha256_file(Path(pdf_path)) != v4_layout["source_pdf_sha256"]:
        raise ValueError("pdf_identity_mismatch")
    ledger, native_observations = build_visibility_bundle(v4_layout, pdf_path)
    pages, decisions = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for source_page, pdf_page in zip(v4_layout["pages"], pdf.pages, strict=True):
            excluded = proven_exclusion_indices(ledger, source_page["page"])
            selected = copy.deepcopy(source_page)
            selected["source_words"] = [w for w in selected["source_words"] if w["word_index"] not in excluded]
            selected["source_word_count"] = len(selected["source_words"])
            page, page_decisions = build_spatial_page(selected, pdf_page)
            page["source_words"] = copy.deepcopy(source_page["source_words"])
            page["source_word_count"] = len(page["source_words"])
            page["proven_nonvisible_source_word_indexes"] = sorted(excluded)
            if excluded:
                audit = text_region(f"v5-p{page['page']:04d}-nonvisible-audit", page["page"], AUDIT_REGION, None,
                                    [w for w in page["source_words"] if w["word_index"] in excluded])
                audit["reading_order"] = len(page["regions"]) + 1
                audit["model_input_disposition"] = "excluded_with_pdf_glyph_proof"
                page["regions"].append(audit)
            pages.append(page); decisions.extend(page_decisions)

    def rename(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"unit_id", "region_id", "table_id", "row_id"} and isinstance(item, str) and item.startswith("v4-"):
                    value[key] = "v5-" + item[3:]
                else:
                    rename(item)
        elif isinstance(value, list):
            for item in value:
                rename(item)
    rename(pages)
    reading = assign_document_spans(pages)
    layout = copy.deepcopy(v4_layout)
    layout.update(schema_version=LAYOUT_VERSION, preprocessing_profile=PROFILE,
                  derived_from_v4_preprocessing_sha256=ancestor_hash, pages=pages,
                  visibility_ledger=ledger, page_count=len(pages),
                  region_count=sum(len(p["regions"]) for p in pages),
                  table_count=sum(len(p["tables"]) for p in pages),
                  layout_token_count=sum(len(r["tokens"]) for p in pages for r in p["regions"]),
                  reading_text_sha256=sha256_bytes(reading.encode("utf-8")))
    layout["span_conventions"]["audit_reading_text"] = "Complete saved source-word inventory, including proof-labelled nonvisible audit regions omitted from model input."
    _seal_layout(layout)
    model_input = compact_model_input_v5(layout)
    mapping = []
    for old_page, page in zip(v4_layout["pages"], pages, strict=True):
        old_units = [(u["unit_id"], set(u.get("source_word_indexes", [])))
                     for u in iter_layout_units({"pages": [old_page]})]
        for unit in iter_layout_units({"pages": [page]}):
            words = set(unit.get("source_word_indexes", []))
            mapping.append({"v5_unit_id": unit["unit_id"], "page": page["page"],
                            "v4_overlaps": [{"unit_id": uid, "source_word_overlap": len(words & indexes)}
                                            for uid, indexes in old_units if words & indexes]})
    audit = {"schema_version": "paper-layout-spatial-visibility-audit/v5", "paper_id": layout["paper_id"],
             "visibility_counts": ledger["counts"], "visibility_ledger_sha256": ledger["ledger_sha256"],
             "candidate_decisions": decisions, "release_blockers": [d for d in decisions if d.get("release_blocker")],
             "visibility_native_observations": native_observations,
             "source_words_retained_in_audit": layout["layout_token_count"],
             "semantic_correctness": "not_established"}
    errors = validate_layout_bundle_v5(layout, reading, model_input)
    if errors:
        raise ValueError(errors)
    return layout, reading, model_input, mapping, audit


def validate_layout_bundle_v5(layout: dict, reading_text: str, model_input: dict, *, pdf_path: Path | None = None) -> list[str]:
    """Check all derivations; supplying the PDF also independently replays proof."""
    if layout.get("schema_version") != LAYOUT_VERSION or model_input.get("schema_version") != INPUT_VERSION:
        return ["v5_schema_pair_required"]
    errors = []
    try:
        if layout.get("preprocessing_profile") != PROFILE:
            errors.append("v5_profile_mismatch")
        body = dict(layout); observed = body.pop("preprocessing_sha256", None)
        if observed != sha256_bytes(canonical_json_bytes(body)):
            errors.append("v5_preprocessing_hash_mismatch")
        if model_input != compact_model_input_v5(layout):
            errors.append("v5_compact_derivation_mismatch")
        ledger = layout["visibility_ledger"]
        body = dict(ledger); seal = body.pop("ledger_sha256", None)
        if seal != sha256_bytes(canonical_json_bytes(body)) or ledger["policy_version"] != POLICY_VERSION:
            errors.append("visibility_ledger_identity_mismatch")
        if ledger["source_pdf_sha256"] != layout["source_pdf_sha256"] or ledger["source_layout_preprocessing_sha256"] != layout["derived_from_v4_preprocessing_sha256"]:
            errors.append("visibility_ancestor_identity_mismatch")
        totals = {"uncertain": 0, "proven_nonvisible": 0}
        for page, proof_page in zip(layout["pages"], ledger["pages"], strict=True):
            if page["page"] != proof_page["page"] or page["source_words"] != [w["source_word"] for w in proof_page["words"]]:
                errors.append("visibility_source_inventory_mismatch")
            excluded = proven_exclusion_indices(ledger, page["page"])
            if sorted(excluded) != page["proven_nonvisible_source_word_indexes"]:
                errors.append("visibility_disposition_mismatch")
            for record in proof_page["words"]:
                totals[record["status"]] += 1
                if record["status"] == "uncertain" and record != {"source_word": record["source_word"], "status": "uncertain", "reason": "visibility_not_proven", "glyphs": []}:
                    errors.append("visibility_uncertain_record_not_canonical")
                if record["status"] == "proven_nonvisible" and (not record["glyphs"] or any(g["status"] != "proven_nonvisible" or g.get("font_embedded") is not True for g in record["glyphs"])):
                    errors.append("visibility_exclusion_without_glyph_proof")
            audited = [i for r in page["regions"] if r["region_kind"] == AUDIT_REGION for i in r["source_word_indexes"]]
            if len(audited) != len(excluded) or set(audited) != excluded:
                errors.append("visibility_audit_partition_mismatch")
        if {k:v for k,v in totals.items() if v} != ledger["counts"]:
            errors.append("visibility_counts_mismatch")
        # V4's structural validator still verifies *all* raw words and spans.
        proxy = copy.deepcopy(layout)
        proxy["schema_version"] = "paper-evidence-layout/v4"
        proxy["preprocessing_profile"] = "frozen-pdf-words-spatial-table-v4"
        _seal_layout(proxy)
        errors.extend(validate_layout_bundle_v4(proxy, reading_text, compact_model_input_v4(proxy)))
        if pdf_path is not None:
            ancestor_view = {"source_pdf_sha256": layout["source_pdf_sha256"],
                             "preprocessing_sha256": layout["derived_from_v4_preprocessing_sha256"], "pages": layout["pages"]}
            validate_visibility_ledger(ledger, ancestor_view, pdf_path)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        errors.append("v5_validation:" + str(exc))
    return errors
