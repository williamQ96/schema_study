"""Explicitly synthetic end-to-end fixtures. Never label these as model results."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..paper_layout_evidence import preprocess_layout_pdf
from ..paper_layout_evidence_v3 import upgrade_bundle
from .common import ROOT, digest, read_json, write_new
from .dataset import parse_dataset
from .paper import prepare_paper, source_identity, unit_catalog


def _pdf(path: Path) -> None:
    lines = ["SYNTHETIC SOFTWARE TEST PAPER - NOT RESEARCH DATA",
             "The dataset contains record_id and temperature_kg fields.",
             "The file is CSV and dates use YYYY-MM-DD syntax.",
             "A status code 0 denotes normal and code 1 denotes alarm."]
    stream = "BT /F1 11 Tf 45 730 Td " + " 0 -20 Td ".join("(" + line + ") Tj" for line in lines) + " ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode()]
    data = b"%PDF-1.4\n"; offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data)); data += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    data += f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(data)


def make_fixture(root: Path) -> tuple[dict, dict]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pdf = root / "synthetic-paper.pdf"
    _pdf(pdf)
    old, _, _ = preprocess_layout_pdf("SYNTHETIC_P01", pdf)
    layout, reading, paper_input, _ = upgrade_bundle(old)
    write_new(root / "layout.json", layout)
    write_new(root / "paper_input.json", paper_input)
    (root / "reading.txt").write_text(reading, encoding="utf-8")
    source = prepare_paper(root / "layout.json", root / "reading.txt", root / "paper_input.json", pdf, root=root)
    (root / "dataset.csv").write_text("record_id,temperature_kg,private_annotation\n001,12.5,DATASET_SECRET_CANARY\n002,,TEST_ONLY\n", encoding="utf-8")
    dataset = parse_dataset(root / "dataset.csv", root=root, sample_limit=1)
    write_new(root / "dataset_bundle.json", dataset)
    corpus = {"schema_version": "four-category-corpus/v1", "purpose": "synthetic_software_test_only",
              "papers": [{"paper_id": "SYNTHETIC_P01", "source": source, "family_id": "synthetic-paper-family"}],
              "datasets": [{"dataset_id": dataset["dataset_id"], "bundle_path": "dataset_bundle.json", "bundle_sha256": dataset["bundle_sha256"], "source_content_sha256": dataset["sources"][0]["sha256"], "family_id": "synthetic-dataset-family"}],
              "matches": [{"match_id": "SYNTHETIC_MATCH_1", "paper_id": "SYNTHETIC_P01", "dataset_id": dataset["dataset_id"],
                           "linkage": {"status": "candidate", "evidence": ["Deliberately paired synthetic software fixtures; not a research match."]}}]}
    config = read_json(ROOT / "config/four_category_experiment_v1.json")
    config["experiment_id"] = "synthetic-offline-smoke-v1"
    config["purpose"] = "synthetic_software_test_only"
    config["status"] = "frozen"
    for profile in config["profiles"]:
        profile.update(backend="mock", deployment="mock", model_id="SYNTHETIC-" + profile["profile_id"], revision="fixture-v1", context_window=1000000, status="frozen")
        profile["capabilities"]["supported_parameters"] = ["temperature", "top_p", "top_k", "do_sample", "repetition_penalty", "seed", "max_output_tokens"]
    config["classification"]["profile_id"] = config["roles"]["soft_reference"]
    config["classification"]["profile_sha256"] = digest(config["profiles"][-1])
    config["dataset_parser"]["sample_limit"] = 1
    return config, corpus


def _part(text: str, heading: str):
    return json.JSONDecoder().raw_decode(text.split(heading + "\n", 1)[1].lstrip())[0]


def mock_transport(request: dict) -> dict:
    """Deterministic fixture response; no external service or learned model."""
    messages = request["messages"]
    user = messages[-1]["content"]
    identity = _part(user, "SOURCE IDENTITY")
    if user.startswith("TASK: Build"):
        paper_input = _part(user, "FULL PAPER INPUT")
        entries = []
        for unit in unit_catalog(paper_input).values():
            # These fixed fixture tags validate mechanics, not annotation quality.
            entries.append({"unit_id": unit["unit_id"], "page": unit["page"], "state": "classified",
                            "categories": ["structure", "encoding", "value", "syntax"],
                            "evidence_spans": [{"start": 0, "end": len(unit["text"]), "quote": unit["text"]}],
                            "rationale": "Synthetic transport fixture only; not a semantic judgment."})
        result = {"schema_version": "paper-category-response/v1", "source_identity": identity, "entries": entries}
    else:
        paper_input = _part(user, "FULL PAPER AND FROZEN AUTOMATED CATEGORY INDEX")["paper_input"]
        unit = next(u for u in unit_catalog(paper_input).values() if "record_id" in u["text"])
        unknown = {"value": None, "raw_reported_value": None, "status": "unknown"}
        result = {"schema_version": "paper-derived-schema/v4", "source_document": identity, "notes": ["Synthetic transport fixture; not a model generation."],
                  "claims": [{"claim_id": "record_id", "object_kind": "field", "reported_name": "record_id", "canonical_path": None,
                              "parent_claim_id": None, "datatype": copy.deepcopy(unknown), "shape_or_dimensions": copy.deepcopy(unknown),
                              "constraints": [], "relationships": [], "encoding": [], "syntax": [],
                              "semantic_annotations": {k: copy.deepcopy(unknown) for k in ("description", "unit", "value_domain", "spatial_meaning", "temporal_meaning", "analytical_role")},
                              "evidence": [{"evidence_id": "e1", "document_id": identity["document_id"], "page": unit["page"], "unit_id": unit["unit_id"],
                                            "source_kind": unit["source_kind"], "quote_or_cell_text": "record_id", "locator": "layout_unit_v3", "supports": "/reported_name"}],
                              "category_annotations": [{"target_pointer": "/reported_name", "categories": ["structure"], "evidence_ids": ["e1"]}],
                              "claim_state": "supported", "confidence": 0.5, "inference_basis": None, "uncertainty_reason": None}]}
    return {"text": json.dumps(result, ensure_ascii=False), "model": request["model"], "finish_reason": "stop", "usage": None}
