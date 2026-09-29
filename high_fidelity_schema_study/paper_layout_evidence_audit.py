from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from .paper_layout_evidence import (
    find_evidence_units,
    sha256_file,
    validate_layout_bundle,
)


ERROR_INDEX = re.compile(r"claim\[(\d+)\]\.evidence\[(\d+)\]:exact_quote_not_in_block")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_quality_audit(experiment_root: Path) -> dict[str, Any]:
    corpus_path = experiment_root / "frozen_corpus_manifest_v1.json"
    manifest_path = experiment_root / "paper_layout_preprocessing_manifest_v2.json"
    layout_schema_path = experiment_root / "paper_evidence_layout_v2.schema.json"
    input_schema_path = experiment_root / "paper_evidence_input_v2.schema.json"
    corpus = _load(corpus_path)
    manifest = _load(manifest_path)
    layout_validator = Draft202012Validator(_load(layout_schema_path))
    input_validator = Draft202012Validator(_load(input_schema_path))

    bundle_records = []
    total_regions = total_tables = total_tokens = 0
    all_bundle_errors: list[str] = []
    p03_layout: dict[str, Any] | None = None
    p03_text = ""
    for paper in corpus["papers"]:
        lane = experiment_root / Path(paper["preprocessing_artifact"]).parent
        layout_path = lane / "evidence_layout_v2.json"
        text_path = lane / "reading_text_v2.txt"
        input_path = lane / "evidence_input_v2.json"
        layout = _load(layout_path)
        reading_text = text_path.read_text(encoding="utf-8")
        model_input = _load(input_path)
        internal = validate_layout_bundle(layout, reading_text, model_input)
        schema_errors = [error.message for error in layout_validator.iter_errors(layout)]
        input_schema_errors = [error.message for error in input_validator.iter_errors(model_input)]
        identity_ok = layout["source_pdf_sha256"] == paper["paper_sha256"]
        errors = internal + schema_errors + input_schema_errors + ([] if identity_ok else ["source_pdf_identity_mismatch"])
        all_bundle_errors.extend(f"{paper['paper_id']}:{error}" for error in errors)
        record = {
            "paper_id": paper["paper_id"],
            "candidate_id": lane.name,
            "status": "pass" if not errors else "fail",
            "source_pdf_identity": "pass" if identity_ok else "fail",
            "internal_validation_error_count": len(internal),
            "json_schema_error_count": len(schema_errors) + len(input_schema_errors),
            "page_count": layout["page_count"],
            "region_count": layout["region_count"],
            "table_candidate_count": layout["table_count"],
            "layout_token_count": layout["layout_token_count"],
            "artifacts": {
                "layout": {"path": str(layout_path), "sha256": sha256_file(layout_path)},
                "reading_text": {"path": str(text_path), "sha256": sha256_file(text_path)},
                "model_input": {"path": str(input_path), "sha256": sha256_file(input_path)},
            },
        }
        bundle_records.append(record)
        total_regions += layout["region_count"]
        total_tables += layout["table_count"]
        total_tokens += layout["layout_token_count"]
        if paper["paper_id"] == "P03":
            p03_layout, p03_text = layout, reading_text

    if p03_layout is None:
        raise ValueError("P03 layout bundle is required for the frozen qualification regression")
    recovery_records = []
    qualification_root = experiment_root / "final_16k_qualification" / "v3"
    for run_dir in sorted(qualification_root.glob("P03--*")):
        run_record = _load(run_dir / "run_record.json")
        parsed = _load(run_dir / "parsed_output.json")
        for error in run_record["validation"]["errors"]:
            match = ERROR_INDEX.fullmatch(error)
            if not match:
                continue
            claim_index, evidence_index = map(int, match.groups())
            evidence = parsed["claims"][claim_index]["evidence"][evidence_index]
            hits = find_evidence_units(
                p03_layout,
                p03_text,
                str(evidence["quote_or_cell_text"]),
                page=int(evidence["page"]),
            )
            recovery_records.append(
                {
                    "run_id": run_record["run_id"],
                    "legacy_error": error,
                    "page": evidence["page"],
                    "legacy_block_id": evidence["block_id"],
                    "quote_or_cell_text": evidence["quote_or_cell_text"],
                    "status": "recovered" if hits else "not_recovered",
                    "matching_rule": "NFKC_then_whitespace_collapse_case_sensitive_exact_v1",
                    "resolved_unit_ids": [hit["unit_id"] for hit in hits],
                    "resolved_matches": [match for hit in hits for match in hit["matches"]],
                }
            )
    recovered = sum(record["status"] == "recovered" for record in recovery_records)
    status = "qualification_pass" if not all_bundle_errors and recovered == len(recovery_records) == 12 else "qualification_fail"
    return {
        "schema_version": "paper-layout-evidence-quality-audit/v2",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "representation_and_deterministic_evidence_resolution_only_no_model_inference",
        "source_corpus_manifest": {"path": str(corpus_path), "sha256": sha256_file(corpus_path)},
        "preprocessing_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "quality_gates": {
            "bundle_count": len(bundle_records),
            "bundle_pass_count": sum(record["status"] == "pass" for record in bundle_records),
            "source_word_single_assignment_required": True,
            "character_span_exact_reconstruction_required": True,
            "layout_token_contiguity_required": True,
            "json_schema_validation_required": True,
            "legacy_exact_quote_mismatch_count": len(recovery_records),
            "legacy_exact_quote_recovered_count": recovered,
        },
        "corpus_totals": {
            "page_count": sum(record["page_count"] for record in bundle_records),
            "region_count": total_regions,
            "table_candidate_count": total_tables,
            "layout_token_count": total_tokens,
        },
        "bundle_records": bundle_records,
        "legacy_p03_recovery_records": recovery_records,
        "limitations": [
            "Table detection is deterministic geometry-based candidate detection, not semantic table adjudication.",
            "Scanned/image-only pages are not OCRed in v2.",
            "Token spans are model-neutral PDF layout-word spans, not model-tokenizer IDs.",
            "The evidence resolver allows only NFKC and whitespace collapse; it does not use fuzzy or semantic matching.",
        ],
        "errors": all_bundle_errors,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit layout-aware evidence v2")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    audit = build_quality_audit(args.experiment_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": audit["status"], "quality_gates": audit["quality_gates"]}, ensure_ascii=False))
    return 0 if audit["status"] == "qualification_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
