"""Source admission and disabled navigation for the parser factorial pilot."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from high_fidelity_schema_study.four_category.common import digest, identity, seal
from high_fidelity_schema_study.four_category.disabled_navigation import (
    make_disabled_index, verify_disabled_index,
)
from high_fidelity_schema_study.four_category.extraction_view import make_view
from high_fidelity_schema_study.four_category.paper import (
    prepare_paper, source_identity, verify_index, verify_paper,
)
from high_fidelity_schema_study.four_category.evidence_schema import evidence_version, validate_layout_bundle
from high_fidelity_schema_study.four_category.mineru_adapter import build_mineru_bundle

from .test_classification_v2 import paper


def test_disabled_navigation_covers_every_unit_without_classifier_claims():
    source = paper()
    index = make_disabled_index(source)
    assert verify_index(index, source) == []
    assert all(entry["availability"] == "unavailable" and entry["prediction"] is None
               and entry["machine_reason"] == "disabled_by_experimental_condition"
               for entry in index["entries"])
    view = make_view(source, index)
    assert view["index_columns"] == ["availability", "state", "categories"]
    assert view["index_entries"] == [["unavailable", None, None] for _ in index["entries"]]


def test_resealed_disabled_navigation_tamper_rejected():
    source = paper()
    index = make_disabled_index(source)
    forged = copy.deepcopy(index)
    forged["entries"][0]["machine_reason"] = "pretend_classifier_failed"
    forged = seal(forged, "index_sha256")
    assert "disabled_navigation_derivation_mismatch" in verify_disabled_index(forged, source)
    changed_source = copy.deepcopy(source)
    changed_source["pages"][0]["text_regions"][0]["text"] += " changed"
    assert verify_index(index, changed_source)


def _v6_fixture(tmp_path):
    pdf = b"%PDF-1.4\nsynthetic parser source\n"
    pdf_sha = hashlib.sha256(pdf).hexdigest()
    baseline = paper()
    baseline["source_pdf_sha256"] = pdf_sha
    middle = {"schema": "docvortex.middle", "schema_version": "2.0", "is_full_document": True,
              "metadata": {"producer": {"name": "mineru", "version": "2.6.0"}, "file_suffix": "pdf"},
              "extensions": {"mineru": {"tier": "pipeline", "parse_mode": "auto"},
                             "docvortex_layout": {"version": 1, "pages": [
                                 {"page_idx": 0, "width_pt": 600, "height_pt": 800}]}},
              "pages": [{"page_idx": 0, "blocks": [
                  {"type": "text", "bbox": [0.02, 0.02, 0.50, 0.05], "content": "é😀 alpha"},
                  {"type": "text", "bbox": [0.02, 0.06, 0.50, 0.10], "content": "syntax: A; value: B"}]}]}
    middle_bytes = json.dumps(middle, ensure_ascii=False).encode("utf-8")
    parser_identity = seal({"schema_version": "paper-parser-identity/v1", "parser": "mineru",
                            "version": "2.6.0", "source_pdf_sha256": pdf_sha,
                            "tier": "pipeline", "ocr_mode": "auto",
                            "small_backend": "onnx", "model_manifest_sha256": "a" * 64,
                            "config_sha256": "b" * 64,
                            "raw_canonical_sha256": digest(middle),
                            "raw_file_bytes_sha256": hashlib.sha256(middle_bytes).hexdigest()},
                           "parser_identity_sha256")
    layout, reading, paper_input = build_mineru_bundle(baseline, middle, parser_identity,
                                                       pdf_sha256=pdf_sha)
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {name: tmp_path / filename for name, filename in {
        "layout": "layout.json", "reading_text": "reading.txt", "input": "input.json",
        "pdf": "paper.pdf", "mineru_raw": "middle.json", "baseline_input": "baseline.json",
        "parser_identity": "parser_identity.json"}.items()}
    for name, value in (("layout", layout), ("input", paper_input), ("mineru_raw", middle),
                        ("baseline_input", baseline), ("parser_identity", parser_identity)):
        paths[name].write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    paths["reading_text"].write_text(reading, encoding="utf-8")
    paths["pdf"].write_bytes(pdf)
    return paths, layout, reading, paper_input


def _source(paths, paper_input, root):
    return seal({"schema_version": "four-category-paper-source/v1",
                 "source_identity": source_identity(paper_input),
                 "artifacts": {name: identity(path, root) for name, path in paths.items()}},
                "source_sha256")


def test_v6_source_rebuild_and_disabled_view(tmp_path):
    paths, layout, reading, paper_input = _v6_fixture(tmp_path)
    assert evidence_version(layout, paper_input) == "v6"
    source = prepare_paper(paths["layout"], paths["reading_text"], paths["input"], paths["pdf"],
                           root=tmp_path, mineru_raw_path=paths["mineru_raw"],
                           baseline_input_path=paths["baseline_input"],
                           parser_identity_path=paths["parser_identity"])
    assert verify_paper(source, tmp_path) == []
    assert validate_layout_bundle(layout, reading, paper_input, pdf_path=paths["pdf"],
                                  baseline_input=json.loads(paths["baseline_input"].read_text(encoding="utf-8")),
                                  middle_json=json.loads(paths["mineru_raw"].read_text(encoding="utf-8")),
                                  parser_identity=json.loads(paths["parser_identity"].read_text(encoding="utf-8"))) == []
    view = make_view(paper_input, make_disabled_index(paper_input))
    assert "structure_status" in view["unit_columns"]
    assert len(view["units"]) == len(make_disabled_index(paper_input)["entries"])


def test_v6_requires_external_origins_and_rejects_tampering(tmp_path):
    paths, layout, reading, paper_input = _v6_fixture(tmp_path)
    with pytest.raises(ValueError, match="v6_source_requires_bound_external_origins"):
        prepare_paper(paths["layout"], paths["reading_text"], paths["input"], paths["pdf"], root=tmp_path)
    source = _source(paths, paper_input, tmp_path)
    paths["mineru_raw"].write_text(paths["mineru_raw"].read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert "mineru_raw:file_identity_mismatch" in verify_paper(source, tmp_path)

    paths, layout, reading, paper_input = _v6_fixture(tmp_path)
    forged = copy.deepcopy(paper_input)
    forged["pages"][0]["text_regions"][0]["text"] = "invented source text"
    # A fresh source seal and file identity cannot legitimize a new compact projection.
    paths["input"].write_text(json.dumps(forged, ensure_ascii=False), encoding="utf-8")
    errors = verify_paper(_source(paths, forged, tmp_path), tmp_path)
    assert any("mineru_paper_input_derivation_mismatch" in e for e in errors), errors


def test_v6_rejects_external_origin_and_pdf_identity_mismatch(tmp_path):
    paths, layout, reading, paper_input = _v6_fixture(tmp_path)
    baseline = json.loads(paths["baseline_input"].read_text(encoding="utf-8"))
    baseline["paper_id"] = "other_paper"
    paths["baseline_input"].write_text(json.dumps(baseline), encoding="utf-8")
    errors = verify_paper(_source(paths, paper_input, tmp_path), tmp_path)
    assert "v6_baseline_origin_identity_mismatch" in errors

    paths, layout, reading, paper_input = _v6_fixture(tmp_path)
    parser = json.loads(paths["parser_identity"].read_text(encoding="utf-8"))
    parser["version"] = "different-version"
    paths["parser_identity"].write_text(json.dumps(parser), encoding="utf-8")
    errors = verify_paper(_source(paths, paper_input, tmp_path), tmp_path)
    assert any("parser_identity_mismatch" in e for e in errors), errors

    paths, layout, reading, paper_input = _v6_fixture(tmp_path)
    parser = json.loads(paths["parser_identity"].read_text(encoding="utf-8"))
    parser["raw_file_bytes_sha256"] = "f" * 64
    parser = seal(parser, "parser_identity_sha256")
    paths["parser_identity"].write_text(json.dumps(parser), encoding="utf-8")
    errors = verify_paper(_source(paths, paper_input, tmp_path), tmp_path)
    assert "v6_parser_raw_file_identity_mismatch" in errors

    paths, layout, reading, paper_input = _v6_fixture(tmp_path)
    paths["pdf"].write_bytes(b"%PDF-1.4\nother pdf\n")
    errors = verify_paper(_source(paths, paper_input, tmp_path), tmp_path)
    assert "pdf_layout_identity_mismatch" in errors
