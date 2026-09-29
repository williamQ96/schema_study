import copy

import jsonschema
import pytest

from high_fidelity_schema_study.four_category.common import ROOT, digest, read_json
from high_fidelity_schema_study.four_category.mineru_adapter import build_mineru_bundle, validate_mineru_bundle
from high_fidelity_schema_study.four_category.paper import unit_catalog
from high_fidelity_schema_study.paper_layout_evidence import resolve_evidence_quote


PDF = "a" * 64
def parser_identity(middle):
    value = {"schema_version": "paper-parser-identity/v1", "parser": "mineru", "version": "4.0.8",
             "tier": "standard", "ocr_mode": "txt", "source_pdf_sha256": PDF,
             "raw_canonical_sha256": digest(middle), "raw_file_bytes_sha256": "d" * 64,
             "model_manifest_sha256": "e" * 64, "config_sha256": "f" * 64}
    value["parser_identity_sha256"] = digest(value)
    return value


def native_unit(uid, text, kind="body_column"):
    return {"unit_id": uid, "page": 1, "region_kind": kind, "reading_order": 0,
            "bbox": [10, 10, 300, 50], "text": text,
            "document_char_span": {"start": 0, "end": len(text)},
            "document_token_span": {"start": 0, "end": len(text.split())}}


def fixtures():
    baseline = {"schema_version": "paper-evidence-input/v5", "paper_id": "P01",
                "source_pdf_sha256": PDF, "layout_preprocessing_sha256": "b" * 64,
                "span_conventions": {}, "pages": [{"page": 1, "layout_mode": "baseline",
                "text_regions": [native_unit("native-header", "Methods"), native_unit("native-missing", "Missing PDF words")],
                "tables": []}], "model_input_sha256": "c" * 64}
    middle = {"schema": "docvortex.middle", "schema_version": "2.0", "is_full_document": True,
              "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": "4.0.8"}},
              "extensions": {"mineru": {"tier": "standard", "parse_mode": "txt"},
                             "docvortex_layout": {"version": 1, "pages": [{"page_idx": 0, "width_pt": 612, "height_pt": 792}]}},
              "pages": [{"page_idx": 0, "blocks": [
                  {"type": "title", "bbox": [0.01, 0.01, 0.50, 0.07], "content": [{"type": "text", "content": "Methods"}]},
                  {"type": "table", "bbox": [0.01, 0.12, 0.70, 0.28], "blocks": [
                      {"type": "table_caption", "content": [{"type": "text", "content": "Table 1"}]},
                      {"type": "table_body", "lines": [{"spans": [{"type": "table", "html": "<table><tr><th>Field</th><th>Meaning</th></tr><tr><td rowspan='2'>temp</td><td>air</td></tr><tr><td>water</td></tr></table>"}]}]}
                  ]}
              ]}]}
    return baseline, middle


def build():
    baseline, middle = fixtures()
    identity = parser_identity(middle)
    return baseline, middle, identity, build_mineru_bundle(baseline, middle, identity, pdf_sha256=PDF)


def test_header_data_merged_topology_and_native_recovery():
    baseline, middle, identity, (layout, reading, inp) = build()
    table = inp["pages"][0]["tables"][0]
    assert table["structure_status"] == "parsed_html"
    assert [c["header"] for c in table["columns"]] == ["Field", "Meaning"]
    assert table["rows"][0]["row_role"] == "header"
    assert table["rows"][1]["row_role"] == "data"
    assert table["rows"][1]["cells"][0]["row_span"] == 2
    assert table["rows"][2]["cells"][0]["column_index"] == 1
    assert table["bbox"][2] == pytest.approx(428.4)
    assert layout["pages"][0]["blocks"][1]["raw"] == middle["pages"][0]["blocks"][1]
    assert any(u["text"] == "Missing PDF words" and u["text_origin"] == "native_baseline_recovery"
               for u in inp["pages"][0]["text_regions"])
    assert inp["pages"][0]["text_regions"][0]["native_alignment"]["status"] == "exact_native_text"
    assert not any(u.get("native_unit_id") == "native-header" for u in inp["pages"][0]["text_regions"])
    assert layout["pages"][0]["coverage_ledger"]["exactly_covered_native_tokens"] >= 1
    assert "Missing PDF words" in reading
    assert len(unit_catalog(inp)) > 5
    body = inp["pages"][0]["text_regions"][0]
    cell = table["rows"][1]["cells"][0]
    for unit, quote in ((body, "Methods"), (cell, "temp")):
        resolved = resolve_evidence_quote(layout, reading, unit["unit_id"], quote)
        assert resolved["status"] == "resolved"
        match = resolved["matches"][0]
        span = match["document_char_span"]
        assert reading[span["start"]:span["end"]] == quote
        assert match["document_token_span"]["end"] > match["document_token_span"]["start"]
    assert table["rows"][1]["text_origin"] == "mineru_recognized"
    assert table["rows"][1]["region_kind"] == "table_row"
    jsonschema.validate(inp, read_json(ROOT / "templates/paper_evidence_input_v6.schema.json"))
    jsonschema.validate(layout, read_json(ROOT / "templates/paper_evidence_layout_v6.schema.json"))
    assert validate_mineru_bundle(baseline, middle, identity, layout, reading, inp, pdf_sha256=PDF) == []


def test_resealed_input_and_layout_drift_are_rejected():
    baseline, middle, identity, (layout, reading, inp) = build()
    corrupt = copy.deepcopy(inp)
    corrupt["pages"][0]["text_regions"][0]["text"] = "invented"
    corrupt["model_input_sha256"] = digest({k: v for k, v in corrupt.items() if k != "model_input_sha256"})
    assert "mineru_paper_input_derivation_mismatch" in validate_mineru_bundle(
        baseline, middle, identity, layout, reading, corrupt, pdf_sha256=PDF)
    changed = copy.deepcopy(layout)
    changed["pages"][0]["blocks"][0]["raw"]["type"] = "invented"
    changed["preprocessing_sha256"] = digest({k: v for k, v in changed.items() if k != "preprocessing_sha256"})
    assert "mineru_layout_derivation_mismatch" in validate_mineru_bundle(
        baseline, middle, identity, changed, reading, inp, pdf_sha256=PDF)


def test_page_geometry_pdf_and_protocol_binding():
    baseline, middle = fixtures()
    bad = copy.deepcopy(middle)
    bad["pages"][0]["blocks"][0]["bbox"] = [0, 0, 1.1, 0.5]
    with pytest.raises(ValueError, match="bbox_out_of_page"):
        build_mineru_bundle(baseline, bad, parser_identity(bad), pdf_sha256=PDF)
    bad = copy.deepcopy(middle)
    bad["pages"][0]["page_idx"] = 1
    with pytest.raises(ValueError, match="page_binding"):
        build_mineru_bundle(baseline, bad, parser_identity(bad), pdf_sha256=PDF)
    with pytest.raises(ValueError, match="baseline_pdf_mismatch"):
        build_mineru_bundle(baseline, middle, parser_identity(middle), pdf_sha256="d" * 64)
    bad = copy.deepcopy(middle)
    bad["schema"] = "docvortex.model"
    with pytest.raises(ValueError, match="docvortex_middle_v2"):
        build_mineru_bundle(baseline, bad, parser_identity(bad), pdf_sha256=PDF)


def test_unparsed_and_unsupported_blocks_are_explicit():
    baseline, middle = fixtures()
    middle["pages"][0]["blocks"][1]["blocks"][1]["lines"][0]["spans"][0]["html"] = "<table><tr><td>broken"
    middle["pages"][0]["blocks"].append({"type": "new_unsupported", "bbox": [0, 0, 0.2, 0.2], "content": "Not silently dropped"})
    identity = parser_identity(middle)
    layout, reading, inp = build_mineru_bundle(baseline, middle, identity, pdf_sha256=PDF)
    assert inp["pages"][0]["tables"][0]["structure_status"] == "unparsed_or_missing_html"
    assert any(u["region_kind"] == "unsupported_block" and u["structure_status"] == "unresolved"
               for u in inp["pages"][0]["text_regions"])
    assert "Not silently dropped" in reading
    assert validate_mineru_bundle(baseline, middle, identity, layout, reading, inp, pdf_sha256=PDF) == []


def test_all_td_first_row_does_not_invent_header():
    baseline, middle = fixtures()
    middle["pages"][0]["blocks"][1]["blocks"][1]["lines"][0]["spans"][0]["html"] = (
        "<table><tr><td>Field</td><td>Meaning</td></tr><tr><td>temp</td><td>air</td></tr></table>")
    _, _, inp = build_mineru_bundle(baseline, middle, parser_identity(middle), pdf_sha256=PDF)
    table = inp["pages"][0]["tables"][0]
    assert table["header_units"] == []
    assert all(column["header"] == "" for column in table["columns"])
    assert table["rows"][0]["row_role"] == "data"


def test_partial_native_text_is_recovered_without_dropping_unmatched_words():
    baseline, middle = fixtures()
    baseline["pages"][0]["text_regions"][0]["text"] = "Methods and Results"
    _, _, inp = build_mineru_bundle(baseline, middle, parser_identity(middle), pdf_sha256=PDF)
    recovered = [u["text"] for u in inp["pages"][0]["text_regions"]
                 if u.get("native_unit_id") == "native-header"]
    assert recovered == ["and Results"]
