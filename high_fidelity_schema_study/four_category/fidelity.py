"""Conservative automatic evidence checks; never upgrades proxy scores to gold."""
from __future__ import annotations

import copy
import re
import unicodedata
from collections import Counter
from pathlib import Path

from . import extraction_v10 as extraction
from .common import digest, file_digest, seal
from .dataset_catalog import fields

VERSION = "automatic-paper-dataset-verification/v1"


def lexical(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return re.sub(r"[\s_\-]+", " ", value).strip().casefold()


def contains(text: str, name: str) -> bool:
    needle = lexical(name)
    return bool(needle) and bool(re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", lexical(text)))


def table_roles(paper: dict) -> list[dict]:
    """Paper-only, inspectable rules; handles TD-only first-row header candidates.

    These are navigation/diagnostic rules, not authoritative semantic labels.
    No dataset objects or field-name list enter this function.
    """
    axes = {"feature", "features", "variable", "variables", "attribute", "attributes"}
    statistics = {"minimum", "maximum", "mean", "standard deviation", "std", "median"}
    result = []
    for page in paper["pages"]:
        for table in page["tables"]:
            rows = table["rows"]
            headers = [c.get("header", "") for c in table["columns"]]
            column_ids = [c["column_index"] for c in table["columns"]]
            header_origin, first = "declared_header_units", 0
            if not any(lexical(h) in axes for h in headers) and rows:
                trial = [c["text"] for c in rows[0]["cells"]]
                if any(lexical(h) in axes for h in trial):
                    headers, header_origin, first = trial, "first_row_candidate", 1
                    column_ids = [c["column_index"] for c in rows[0]["cells"]]
            axis = [i for i, h in enumerate(headers) if lexical(h) in axes]
            caption = table["caption"]["text"]
            stat_cols = [i for i, h in enumerate(headers) if lexical(h) in statistics]
            purpose = "summary_statistics" if len(axis) == 1 and len(stat_cols) >= 2 else (
                "feature_list_candidate" if len(axis) == 1 else "unresolved")
            mentions = []
            if len(axis) == 1:
                for row in rows[first:]:
                    cells = [c for c in row["cells"] if c.get("column_index") == column_ids[axis[0]]]
                    if len(cells) == 1 and cells[0]["text"].strip():
                        mentions.append({"label": cells[0]["text"], "unit_id": cells[0]["unit_id"],
                                         "row_unit_id": row["unit_id"]})
            unit_ids = [table["caption"]["unit_id"]] + [u["unit_id"] for u in table.get("header_units", [])]
            unit_ids += [u for row in rows for u in [row["unit_id"], *[c["unit_id"] for c in row["cells"]]]]
            header_ids = [u["unit_id"] for u in table.get("header_units", [])]
            if first:
                header_ids += [rows[0]["unit_id"], *[c["unit_id"] for c in rows[0]["cells"]]]
            result.append({"table_id": table["table_id"], "page": page["page"], "caption": caption,
                           "bbox": table.get("bbox"), "purpose": purpose, "header_origin": header_origin,
                           "feature_axis": column_ids[axis[0]] if len(axis) == 1 else None,
                           "headers": headers, "descriptor_labels": [headers[i] for i in stat_cols] if purpose == "summary_statistics" else [],
                           "feature_cells": mentions, "unit_ids": unit_ids, "header_unit_ids": header_ids,
                           "authority": "automatic_role_rule_not_gold"})
    return result


def correspondence(label: str, catalog: dict, *, scope_id: str | None = None,
                   aliases: list[dict] = (), canonical_id: str | None = None) -> dict:
    objects = fields(catalog)
    if scope_id is not None and scope_id not in {t["scope_id"] for t in catalog["tables"]}:
        return {"status": "unresolved", "reason": "unknown_explicit_scope", "candidates": []}
    eligible = [f for f in objects if scope_id is None or f["scope_id"] == scope_id]
    if canonical_id is not None:
        candidates = [f for f in eligible if f["object_id"] == canonical_id and lexical(f["name"]) == lexical(label)]
        basis = "scoped_identity_and_name"
    else:
        candidates = [f for f in eligible if f["name"] == label]
        basis = "exact_unique_name"
        if not candidates:
            candidates = [f for f in eligible if lexical(f["name"]) == lexical(label)]
            basis = "conservative_lexical_name"
        if not candidates:
            ids = {a["object_id"] for a in aliases if lexical(a["alias"]) == lexical(label)}
            candidates = [f for f in eligible if f["object_id"] in ids]
            basis = "bound_metadata_alias_declaration"
    return {"status": "matched" if len(candidates) == 1 else "unresolved",
            "reason": basis if len(candidates) == 1 else ("ambiguous_scope_or_name" if candidates else "no_supported_mapping"),
            "candidates": [{"object_id": f["object_id"], "name": f["name"], "scope_id": f["scope_id"],
                            "name_evidence": f["name_evidence"]} for f in candidates]}


def _evidence(mention: dict, paper: dict, windows: list[dict]) -> tuple[list[dict], list[str]]:
    refs, errors = [], []
    ids = mention.get("source_windows")
    if not isinstance(ids, list) or not ids:
        return refs, ["missing_source_windows"]
    for wid in ids:
        if type(wid) is not int or not 0 <= wid < len(windows):
            errors.append("invalid_source_window")
            continue
        refs.append({**windows[wid], "source_pdf_sha256": paper["source_pdf_sha256"]})
    return refs, errors


def paper_support(label: str, refs: list[dict], roles: list[dict], windows: list[dict] = ()) -> dict:
    named = [r for r in refs if contains(r["text"], label)]
    if not named:
        return {"status": "unresolved", "reason": "cited_windows_do_not_establish_name_equivalence"}
    uids = {r["unit_id"] for r in named}
    positives, negatives = [], []
    for table in roles:
        if any(lexical(label) == lexical(cell["label"]) and
               uids.intersection({cell["unit_id"], cell["row_unit_id"]}) for cell in table["feature_cells"]):
            positives.append(table["table_id"])
        if (uids.intersection(table["header_unit_ids"]) and
                lexical(label) in {lexical(s) for s in table["descriptor_labels"]}):
            negatives.append(table["table_id"])
    if positives and negatives:
        return {"status": "unresolved", "reason": "conflicting_source_roles"}
    if positives:
        return {"status": "supported", "reason": "feature_axis_cell_rule", "table_ids": positives,
                "basis": "automatic_semantic_rule_not_independent_gold"}
    if negatives:
        return {"status": "rejected", "reason": "statistic_descriptor_as_field", "table_ids": negatives,
                "basis": "automatic_semantic_rule_not_independent_gold"}
    # Bounded prose rule: a named measurement definition plus a nearby explicit
    # feature-description introduction. A word occurrence or a bare colon alone
    # cannot pass. This is calibrated-rule support, not an infallible entailment
    # oracle; retain the added context and its exact source window.
    definition = re.compile(r"(?<!\w)" + re.escape(lexical(label)) +
                            r"\s*:\s*(?:it\s+)?(?:gives?|measures?|represents?|denotes?)\b")
    for ref in named:
        if not definition.search(lexical(ref["text"])):
            continue
        nearby = [w for w in windows if w["page"] == ref["page"] and
                  0 <= ref["window_index"] - w["window_index"] <= 6]
        introductions = [w for w in nearby if re.search(
            r"\bfeatures?\b.{0,100}\b(?:descriptions?|definitions?|given below|extracted)\b", lexical(w["text"]))]
        if introductions:
            return {"status": "supported", "reason": "named_measurement_definition_with_feature_context_rule",
                    "context_evidence": introductions, "basis": "automatic_semantic_rule_not_independent_gold"}
    return {"status": "unresolved", "reason": "name_present_but_object_role_unverified"}


def evaluate(payload: dict, paper: dict, catalog: dict, *, aliases: list[dict] = (),
             admission_status: str = "not_replayed", admission_errors: list[str] = (),
             target_windows: list[int] | None = None) -> dict:
    """Raw diagnostics survive invalid runs; formal semantic metrics stay null."""
    if catalog["paper_id"] != paper["paper_id"]:
        raise ValueError("paper_catalog_binding_mismatch")
    objects = fields(catalog)
    windows, roles = extraction.window_catalog(paper), table_roles(paper)
    targets = set(range(len(windows))) if target_windows is None else set(target_windows)
    if any(type(w) is not int or not 0 <= w < len(windows) for w in targets):
        raise ValueError("invalid_target_windows")
    rows, property_rows, errors = [], [], []
    if not isinstance(payload, dict):
        payload = {}
        errors.append("payload_not_object")
    mentions, facts = payload.get("mentions"), payload.get("facts")
    if not isinstance(mentions, list):
        mentions = []; errors.append("mentions_not_array")
    if not isinstance(facts, list):
        facts = []; errors.append("facts_not_array")
    seen = {}
    for i, mention in enumerate(mentions):
        if not isinstance(mention, dict):
            rows.append({"mention_indices": [i], "status": "rejected", "reason": "malformed_mention"})
            continue
        if mention.get("kind") not in {"field", "variable"}:
            continue
        label = mention.get("normalized_label")
        if not isinstance(label, str) or not label.strip():
            rows.append({"mention_indices": [i], "status": "rejected", "reason": "invalid_label"})
            continue
        # Different contexts stay separate unless a canonical object is resolved.
        refs, bad = _evidence(mention, paper, windows)
        match = correspondence(label, catalog, scope_id=mention.get("dataset_scope_id"), aliases=aliases)
        support = paper_support(label, refs, roles, windows)
        if support["status"] == "rejected" and match["status"] == "matched":
            support = {**support, "status": "unresolved", "reason": "dataset_match_conflicts_with_descriptor_rule"}
        if not any(r["window_index"] in targets for r in refs):
            bad.append("no_evidence_in_assigned_region")
        status = ("rejected" if bad or support["status"] == "rejected" else
                  "supported" if support["status"] == "supported" and match["status"] == "matched" else "unresolved")
        key = digest([match["candidates"][0]["object_id"], status, support["reason"]]) if match["status"] == "matched" else digest([
            label, mention.get("source_windows"), mention.get("context"), status])
        if key in seen:
            rows[seen[key]]["mention_indices"].append(i)
            rows[seen[key]]["paper_evidence"].extend(r for r in refs if r not in rows[seen[key]]["paper_evidence"])
            continue
        seen[key] = len(rows)
        rows.append({"mention_indices": [i], "label": label, "status": status,
                     "reason": bad[0] if bad else support["reason"] if support["status"] != "supported" else match["reason"],
                     "mapping": match, "paper_support": support, "paper_evidence": refs, "evidence_errors": bad})
    invalid_attempts, attempts = 0, 0
    for i, fact in enumerate(facts):
        attempts += 1
        if not isinstance(fact, dict):
            invalid_attempts += 1
            property_rows.append({"fact_index": i, "locator_valid": False, "semantic_status": "unresolved", "reason": "malformed_fact"})
            continue
        wid, quote, subject = fact.get("primary_window"), fact.get("primary_quote"), fact.get("subject_mention")
        valid = (type(wid) is int and wid in targets and isinstance(quote, str) and bool(quote.strip())
                 and quote in windows[wid]["text"] and type(subject) is int and 0 <= subject < len(mentions)
                 and isinstance(mentions[subject], dict))
        supports = fact.get("support_windows")
        valid = bool(valid and isinstance(supports, list) and all(type(w) is int and 0 <= w < len(windows) for w in supports))
        invalid_attempts += not valid
        property_rows.append({"fact_index": i, "locator_valid": valid, "claim": fact.get("claim"),
                              "semantic_status": "unresolved", "reason": "locator_does_not_prove_property_entailment" if valid else "invalid_evidence_attempt"})
    counts = Counter(r["status"] for r in rows)
    accepted = sorted({r["mapping"]["candidates"][0]["object_id"] for r in rows if r["status"] == "supported"})
    eligible_accepted = accepted if admission_status == "success" and not errors and not invalid_attempts else []
    covered_labels = {lexical(r.get("label", "")) for r in rows}
    target_uids = {windows[w]["unit_id"] for w in targets}
    uncovered = [dict(cell, table_id=t["table_id"]) for t in roles for cell in t["feature_cells"]
                 if target_uids.intersection({cell["unit_id"], cell["row_unit_id"]}) and lexical(cell["label"]) not in covered_labels]
    return seal({"schema_version": VERSION, "paper_id": paper["paper_id"],
                 "catalog_sha256": catalog["catalog_sha256"], "paper_input_canonical_sha256": digest(paper),
                 "source_pdf_sha256": paper["source_pdf_sha256"], "raw_payload_canonical_sha256": digest(payload),
                 "aliases_canonical_sha256": digest(list(aliases)), "admission_status": admission_status,
                 "admission_errors": list(admission_errors), "diagnostic_errors": errors,
                 "field_decisions": rows, "property_checks": property_rows, "table_role_diagnostics": roles,
                 "uncovered_feature_cells": uncovered, "raw_field_records": sum(len(r["mention_indices"]) for r in rows),
                 "deduplicated_decisions": len(rows), "decision_counts": dict(counts),
                 "supported_object_ids_diagnostic": accepted, "accepted_object_ids": eligible_accepted,
                 "dataset_object_count": len(objects), "evidence_attempts": attempts,
                 "invalid_evidence_attempts": invalid_attempts,
                 "locator_precision": (attempts - invalid_attempts) / attempts if attempts else None,
                 "automatic_support_rate": counts["supported"] / len(rows) if rows else None,
                 "unresolved_rate": counts["unresolved"] / len(rows) if rows else None,
                 "formal_precision": None, "formal_recall": None, "G_visible": None,
                 "authority": "automatic_diagnostics_require_independent_semantic_calibration"}, "report_sha256")


def verify_report(report: dict, payload: dict, paper: dict, catalog: dict, **options) -> list[str]:
    from .common import seal_errors
    errors = seal_errors(report, "report_sha256")
    try:
        if report != evaluate(payload, paper, catalog, **options):
            errors.append("verification_report_derivation_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("verification_report_replay:" + str(exc))
    return errors


def visibility_candidates(catalog: dict, pdf: Path, expected_pdf_sha256: str, *, aliases: list[dict] = ()) -> dict:
    """Full PDF pass independent of model outputs and parser arms. Miss != absent."""
    import pdfplumber
    if file_digest(pdf) != expected_pdf_sha256:
        raise ValueError("visibility_pdf_identity_mismatch")
    with pdfplumber.open(pdf) as doc:
        # A page box is deliberately not advertised as a precise field box.
        blocks = [{"page": i + 1, "bbox": [0, 0, float(page.width), float(page.height)],
                   "region_precision": "page", "quote": page.extract_text() or ""}
                  for i, page in enumerate(doc.pages)]
        page_count = len(doc.pages)
    rows = []
    for obj in fields(catalog):
        names = [obj["name"]] + [a["alias"] for a in aliases if a["object_id"] == obj["object_id"]]
        hits = [b for b in blocks if any(contains(b["quote"], n) for n in names)]
        rows.append({"object_id": obj["object_id"], "name": obj["name"], "visibility": "uncertain",
                     "lexical_hits": hits, "reason": "name_hit_requires_role_and_entailment" if hits else "no_lexical_hit_not_proof_of_absence"})
    return seal({"schema_version": "independent-visibility-candidates/v1", "paper_id": catalog["paper_id"],
                 "catalog_sha256": catalog["catalog_sha256"], "source_pdf_sha256": expected_pdf_sha256,
                 "aliases_canonical_sha256": digest(list(aliases)), "engine": "pdfplumber", "engine_version": pdfplumber.__version__,
                 "page_count": page_count, "objects": rows, "G_visible": None,
                 "basis": "full_pdf_text_search_no_model_outputs_no_ocr_no_absence_inference"}, "visibility_sha256")
