"""Separate V12 diagnostics; historical V1 reports and claims remain unchanged."""
from __future__ import annotations

from . import fidelity, extraction_v10
from .common import seal
from .feature_regions_v2 import regions
from .name_mapping_v2 import match, verify


def evaluate(payload, paper, catalog, *, mapping_document=None, mapping_root=None,
             admission_status="not_replayed", admission_errors=(), target_windows=None):
    entries = verify(mapping_document, catalog, mapping_root) if mapping_document else []
    old = fidelity.evaluate(payload, paper, catalog, admission_status=admission_status,
                            admission_errors=admission_errors, target_windows=target_windows)
    roles, windows = regions(paper), extraction_v10.window_catalog(paper)
    decisions = []
    for row in old["field_decisions"]:
        if "label" not in row:
            decisions.append(row)
            continue
        mapping = match(row["label"], catalog, entries)
        support = row["paper_support"]
        if support["status"] == "unresolved":
            table_ids = [t["table_id"] for t in roles if t["grouped_inventory_candidate"] and
                         any(ref["unit_id"] in t["unit_ids"] and fidelity.contains(ref["text"], row["label"])
                             for ref in row["paper_evidence"])]
            if table_ids:
                support = {"status": "supported", "reason": "named_grouped_inventory_region_rule",
                           "table_ids": table_ids, "basis": "automatic_semantic_rule_not_gold"}
        status = ("rejected" if row["evidence_errors"] or support["status"] == "rejected" else
                  "supported" if support["status"] == "supported" and mapping["status"] == "matched" else "unresolved")
        reason = (row["evidence_errors"][0] if row["evidence_errors"] else
                  support["reason"] if support["status"] != "supported" else mapping["reason"])
        decisions.append({**row, "mapping": mapping, "paper_support": support, "status": status, "reason": reason})
    eligible = (admission_status == "success" and not old["admission_errors"]
                and not old["diagnostic_errors"] and old["invalid_evidence_attempts"] == 0)
    accepted = sorted({r["mapping"]["candidates"][0]["object_id"] for r in decisions if r["status"] == "supported"}) if eligible else []
    target = set(range(len(windows))) if target_windows is None else set(target_windows)
    table_regions = [t for t in roles if any(w["window_index"] in target and w["unit_id"] in t["unit_ids"] for w in windows)]
    return seal({"schema_version": "automatic-paper-dataset-verification/v2", "historical_v1": old,
                 "admission_status": admission_status, "field_decisions": decisions,
                 "mapping_sha256": mapping_document["mapping_sha256"] if mapping_document else None,
                 "automatic_supported_object_ids": accepted, "candidate_regions": table_regions,
                 "candidate_inventory_exhaustive": False, "formal_precision": None, "formal_recall": None,
                 "interpretation": "rule-supported diagnostics; ambiguous scopes and unsupported spelling corrections remain unresolved"}, "report_sha256")
