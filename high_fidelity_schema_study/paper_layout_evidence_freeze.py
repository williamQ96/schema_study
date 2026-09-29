from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .paper_layout_evidence import sha256_file


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_freeze_manifest(experiment_root: Path) -> dict[str, Any]:
    audit_path = experiment_root / "paper_layout_evidence_quality_audit_v2.json"
    preprocessing_manifest_path = experiment_root / "paper_layout_preprocessing_manifest_v2.json"
    protocol_path = experiment_root / "paper_layout_evidence_integration_protocol_v2.json"
    layout_schema_path = experiment_root / "paper_evidence_layout_v2.schema.json"
    input_schema_path = experiment_root / "paper_evidence_input_v2.schema.json"
    locator_schema_path = experiment_root / "paper_evidence_locator_v2.schema.json"
    audit = _load(audit_path)
    preprocessing = _load(preprocessing_manifest_path)
    if audit.get("status") != "qualification_pass":
        raise ValueError("layout evidence quality audit has not passed")
    if audit["quality_gates"]["bundle_pass_count"] != 10:
        raise ValueError("all ten layout evidence bundles must pass before freeze")
    by_paper = {record["paper_id"]: record for record in audit["bundle_records"]}
    records = []
    for record in preprocessing["records"]:
        audited = by_paper[record["paper_id"]]
        if audited["status"] != "pass":
            raise ValueError(f"bundle is not qualified: {record['paper_id']}")
        records.append(
            {
                "paper_id": record["paper_id"],
                "candidate_id": record["candidate_id"],
                "source_pdf_sha256": record["source_pdf_sha256"],
                "preprocessing_sha256": record["preprocessing_sha256"],
                "model_input_sha256": record["model_input_sha256"],
                "artifacts": audited["artifacts"],
            }
        )
    control_files = {}
    for path in [audit_path, preprocessing_manifest_path, protocol_path, layout_schema_path, input_schema_path, locator_schema_path]:
        control_files[path.name] = {"path": str(path), "sha256": sha256_file(path)}
    return {
        "schema_version": "paper-layout-evidence-freeze/v2",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "layout-aware evidence representation only; no inference outputs are included",
        "preprocessing_profile": preprocessing["preprocessing_profile"],
        "activation": "available only to a newly versioned and explicitly bound inference qualification",
        "record_count": len(records),
        "control_files": control_files,
        "records": records,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze qualified layout-aware evidence v2 artifacts")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_freeze_manifest(args.experiment_root)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": manifest["status"], "record_count": manifest["record_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
