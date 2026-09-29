from __future__ import annotations

import copy
import json

import pytest

from high_fidelity_schema_study.four_category.common import digest, file_digest, seal, write_new
from high_fidelity_schema_study.four_category.dataset import parse_dataset
from high_fidelity_schema_study.four_category.dataset_catalog import build_catalog, verify_catalog, load_aliases
from high_fidelity_schema_study.four_category import fidelity, extraction_v10, extraction_v11
from .test_extraction_view import paper_with_table, fixture_index


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("Area,Perimeter,hum,code,Class\n12,24,0.1,001,A\n", encoding="utf-8")
    bundle = parse_dataset(source, root=tmp_path)
    write_new(tmp_path / "evidence.json", bundle)
    binding = {"schema_version": "dataset-scope-binding/v1", "paper_id": "P01", "resources": [
        {"scope_id": "main", "evidence_path": "evidence.json", "source_root": ".",
         "evidence_file_bytes_sha256": file_digest(tmp_path / "evidence.json"),
         "field_roles": {"Class": "target"}}]}
    return binding, build_catalog(binding, tmp_path), tmp_path


def table_paper():
    p = paper_with_table()
    p["paper_id"] = "P01"
    t = p["pages"][0]["tables"][0]
    t["caption"]["text"] = "Table 7. Statistics of dataset features"
    t["columns"] = [{"column_index": i, "header": h} for i, h in enumerate(["Feature", "Minimum", "Mean"])]
    t["header_units"] = [{"unit_id": "h"+str(i), "page": 1, "text": h} for i, h in enumerate(["Feature", "Minimum", "Mean"])]
    t["rows"] = [{"row_id": "r"+str(i), "unit_id": "r"+str(i), "text": name+" | 1 | 2",
                  "cells": [{"unit_id": "c"+str(i), "text": name, "column_index": 0},
                            {"unit_id": "n"+str(i), "text": "1", "column_index": 1},
                            {"unit_id": "m"+str(i), "text": "2", "column_index": 2}]}
                 for i, name in enumerate(["Area", "Perimeter"])]
    return p


def payload(p, label, uid):
    wid = next(w["window_index"] for w in extraction_v10.window_catalog(p) if w["unit_id"] == uid)
    return {"mentions": [{"kind": "field", "normalized_label": label, "source_windows": [wid], "context": None}],
            "facts": [{"subject_mention": 0, "primary_window": wid, "primary_quote": label, "support_windows": [],
                       "claim": {"kind": "attribute", "predicate": "reported_name"}}]}


def test_catalog_replays_source_and_rejects_resealed_facts(dataset):
    binding, catalog, root = dataset
    assert catalog["object_count"] == 5
    assert verify_catalog(catalog, binding, root) == []
    changed = copy.deepcopy(catalog)
    changed["tables"][0]["fields"][0]["name"] = "imagined"
    assert "catalog_derivation_mismatch" in verify_catalog(seal(changed, "catalog_sha256"), binding, root)
    d = json.loads((root / "evidence.json").read_text())
    next(f for f in d["facts"] if f["predicate"] == "name")["value"] = "imagined"
    (root / "evidence.json").write_text(json.dumps(seal(d, "bundle_sha256")))
    binding["resources"][0]["evidence_file_bytes_sha256"] = file_digest(root / "evidence.json")
    with pytest.raises(ValueError, match="dataset_derivation_invalid"):
        build_catalog(binding, root)


def test_workbook_metadata_is_retained_but_not_a_field(tmp_path):
    from openpyxl import Workbook
    wb = Workbook(); wb.active.title = "Records"
    wb.active.append(["code", "Class"]); wb.active.append(["001", "A"])
    wb.create_sheet("Citation_Request").append(["Please cite this publication"])
    wb.save(tmp_path / "data.xlsx")
    d = parse_dataset(tmp_path / "data.xlsx", root=tmp_path)
    write_new(tmp_path / "evidence.json", d)
    b = {"schema_version": "dataset-scope-binding/v1", "paper_id": "P01", "resources": [
        {"scope_id": "records", "sheet": "Records", "source_root": ".", "evidence_path": "evidence.json",
         "evidence_file_bytes_sha256": file_digest(tmp_path / "evidence.json")}]}
    c = build_catalog(b, tmp_path)
    assert c["object_count"] == 2 and len(c["excluded_name_facts"]) == 1
    assert any(f["value"] == "001" for f in c["tables"][0]["fields"][0]["property_facts"])
    b["resources"][0].pop("sheet")
    with pytest.raises(ValueError, match="explicit_sheet"):
        build_catalog(b, tmp_path)


def test_alias_requires_replayable_declaration_and_scopes_remain_ambiguous(dataset):
    _, cat, root = dataset
    obj = cat["tables"][0]["fields"][2]
    write_new(root / "codebook.json", {"aliases": [{"name": "hum", "alias": "humidity", "scope_id": "main"}]})
    spec = {"catalog_sha256": cat["catalog_sha256"], "entries": [{"object_id": obj["object_id"], "alias": "humidity",
        "source_path": "codebook.json", "source_file_bytes_sha256": file_digest(root / "codebook.json"), "json_pointer": "/aliases/0"}]}
    assert fidelity.correspondence("humidity", cat)["status"] == "unresolved"
    aliases = load_aliases(spec, cat, root)
    assert fidelity.correspondence("humidity", cat, aliases=aliases)["status"] == "matched"
    spec["entries"][0]["alias"] = "temperature"
    with pytest.raises(ValueError, match="alias_declaration_mismatch"):
        load_aliases(spec, cat, root)
    duplicate = copy.deepcopy(cat["tables"][0]); duplicate["scope_id"] = "other"
    for f in duplicate["fields"]:
        f["scope_id"] = "other"; f["object_id"] = "other::"+f["source_subject_id"]
    cat["tables"].append(duplicate); cat = seal(cat, "catalog_sha256")
    assert fidelity.correspondence("Area", cat)["reason"] == "ambiguous_scope_or_name"
    assert fidelity.correspondence("Area", cat, scope_id="main")["status"] == "matched"
    assert fidelity.correspondence("Area", cat, scope_id="missing")["status"] == "unresolved"


def test_existence_alone_not_support_and_statistic_header_is_rejected(dataset):
    _, cat, _ = dataset
    p = table_paper()
    good = fidelity.evaluate(payload(p, "Area", "c0"), p, cat, admission_status="success")
    assert good["accepted_object_ids"] and good["formal_precision"] is None
    assert good["uncovered_feature_cells"][0]["label"] == "Perimeter"
    bad = fidelity.evaluate(payload(p, "Perimeter", "c0"), p, cat, admission_status="success")
    assert not bad["accepted_object_ids"]
    assert bad["decision_counts"] == {"unresolved": 1}
    stats = fidelity.evaluate(payload(p, "Minimum", "h1"), p, cat, admission_status="success")
    assert stats["field_decisions"][0]["reason"] == "statistic_descriptor_as_field"
    assert stats["locator_precision"] == 1.0  # Correct quote, wrong semantic role.


def test_dataset_can_itself_store_statistics_so_role_conflict_abstains(dataset):
    _, cat, _ = dataset
    p = table_paper()
    cat["tables"][0]["fields"][0]["name"] = "Minimum"
    cat = seal(cat, "catalog_sha256")
    # Pure matcher fixture: production catalogs must additionally pass source replay.
    report = fidelity.evaluate(payload(p, "Minimum", "h1"), p, cat, admission_status="success")
    assert report["decision_counts"] == {"unresolved": 1}
    assert report["field_decisions"][0]["reason"] == "dataset_match_conflicts_with_descriptor_rule"


def test_malformed_attempts_do_not_disappear_and_invalid_run_cannot_publish(dataset):
    _, cat, _ = dataset
    p = table_paper(); value = payload(p, "Area", "c0")
    value["facts"].append("not an evidence object")
    r = fidelity.evaluate(value, p, cat, admission_status="success")
    assert r["locator_precision"] == 0.5 and not r["accepted_object_ids"]
    r = fidelity.evaluate(payload(p, "Area", "c0"), p, cat, admission_status="contract_invalid")
    assert r["supported_object_ids_diagnostic"] and not r["accepted_object_ids"]
    assert fidelity.evaluate(None, p, cat)["formal_recall"] is None
    assert fidelity.evaluate({"mentions": [], "facts": []}, p, cat)["locator_precision"] is None


@pytest.mark.parametrize("source_windows", [None, {}, [None], [{"window": 1}], [True], [-1], [999999]])
def test_malformed_mention_evidence_is_isolated(dataset, source_windows):
    _, cat, _ = dataset
    p = table_paper(); value = payload(p, "Invented", "c0")
    value["mentions"][0]["source_windows"] = source_windows
    report = fidelity.evaluate(value, p, cat, admission_status="success")
    assert not report["accepted_object_ids"]
    assert report["decision_counts"] == {"rejected": 1}


def test_resealed_evaluation_and_cross_paper_are_rejected(dataset):
    _, cat, _ = dataset
    p = table_paper(); value = payload(p, "Area", "c0")
    report = fidelity.evaluate(value, p, cat, admission_status="success")
    report["accepted_object_ids"] = ["imaginary"]
    assert fidelity.verify_report(seal(report, "report_sha256"), value, p, cat,
                                  admission_status="success") == ["verification_report_derivation_mismatch"]
    p["paper_id"] = "OTHER"
    with pytest.raises(ValueError, match="binding_mismatch"):
        fidelity.evaluate(value, p, cat)


def test_role_rule_is_contextual_and_preserves_sparse_column_ids():
    p = table_paper(); t = p["pages"][0]["tables"][0]
    for col in t["columns"]: col["column_index"] += 3
    for row in t["rows"]:
        for cell in row["cells"]: cell["column_index"] += 3
    role = fidelity.table_roles(p)[0]
    assert role["feature_axis"] == 3
    assert [f["label"] for f in role["feature_cells"]] == ["Area", "Perimeter"]
    t["columns"] = [{"column_index": 3, "header": "Minimum"}]
    assert fidelity.table_roles(p)[0]["descriptor_labels"] == []


def test_definition_rule_requires_explicit_feature_context(dataset):
    _, cat, _ = dataset
    p = table_paper(); p["pages"][0]["tables"] = []
    unit = p["pages"][0]["text_regions"][0]
    unit["text"] = "Area: Gives the number of pixels."
    value = payload(p, "Area", unit["unit_id"])
    assert fidelity.evaluate(value, p, cat)["decision_counts"] == {"unresolved": 1}
    unit["text"] = "Morphological features and descriptions are given below. Area: Gives the number of pixels."
    r = fidelity.evaluate(payload(p, "Area", unit["unit_id"]), p, cat, admission_status="success")
    assert r["accepted_object_ids"]
    assert r["field_decisions"][0]["paper_support"]["context_evidence"]


def test_full_pdf_visibility_has_all_dataset_objects_and_never_infers_absence(dataset):
    _, cat, root = dataset
    stream = b"BT /F1 12 Tf 40 100 Td (Area is mentioned here.) Tj ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"+stream+b"\nendstream"]
    data = b"%PDF-1.4\n"; offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data)); data += str(i).encode()+b" 0 obj\n"+obj+b"\nendobj\n"
    start = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    data += b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n"+str(start).encode()+b"\n%%EOF"
    pdf = root / "paper.pdf"; pdf.write_bytes(data)
    v = fidelity.visibility_candidates(cat, pdf, file_digest(pdf))
    assert len(v["objects"]) == cat["object_count"]
    assert all(o["visibility"] == "uncertain" for o in v["objects"])
    assert v["G_visible"] is None
    assert next(o for o in v["objects"] if o["name"] == "Perimeter")["lexical_hits"] == []
    with pytest.raises(ValueError, match="pdf_identity"):
        fidelity.visibility_candidates(cat, pdf, "0"*64)


def empty_v11(p, group):
    ctx = extraction_v11.context(p, group)
    windows = extraction_v10.window_catalog(p)
    return {"schema_version": extraction_v11.RESPONSE_VERSION,
            "coverage": [{"window_id": w, "state": "reviewed"} for w in group["window_ids"]],
            "mentions": [], "facts": [],
            "table_reviews": [{"table_id": t["table_id"], "purpose": "uncertain", "field_axis": "uncertain",
                               "rationale": "uncertain", "source_windows": [next(w["window_index"] for w in windows if w["unit_id"] in t["unit_ids"])]}
                              for t in ctx["table_candidates"]],
            "feature_coverage": [{"unit_id": c["unit_id"], "decision": "uncertain", "mention_index": None, "rationale": "uncertain"}
                                 for c in ctx["feature_cell_candidates"]]}


def test_v11_whole_table_and_joint_anchor_contract(dataset):
    p = table_paper(); index = fixture_index(p)
    task = extraction_v11.make_task(); group = extraction_v11.plan_region(p, [1])
    assert group["window_ids"] == [w["window_index"] for w in extraction_v10.window_catalog(p)]
    response = empty_v11(p, group)
    assert extraction_v11.validate_response(response, p, index, group) == []
    response["feature_coverage"].pop()
    assert extraction_v11.validate_response(response, p, index, group)
    response = empty_v11(p, group)
    old = payload(p, "Area", "c0")
    response["mentions"] = old["mentions"]
    fact = old["facts"][0]; fact["categories"] = ["structure"]
    fact["claim"]["assertion"] = {"status": "reported", "value": "Area", "basis": None}
    anchor = next(a for a in extraction_v11.anchors(p, group) if a["quote"] == "Area")
    fact.pop("primary_window"); fact.pop("primary_quote"); fact["primary_anchor"] = anchor["anchor_id"]
    response["facts"] = [fact]
    assert extraction_v11.validate_response(response, p, index, group) == []
    converted = extraction_v11.to_v10(response, p, group)
    assert converted["facts"][0]["primary_quote"] == "Area"
    response["facts"][0]["primary_anchor"] = "invented"
    assert extraction_v11.admit_response(json.dumps(response), p, index, group)["status"] == "contract_invalid"
    prompt = extraction_v11.render_group(task, p, index, group)
    assert "main::column:" not in json.dumps(prompt)  # Dataset never enters request builder.
    with pytest.raises(ValueError, match="exceeds_bound"):
        extraction_v11.plan_region(p, [1], max_windows=1)
    altered = copy.deepcopy(group); altered["window_ids"].pop(); altered = seal(altered, "group_sha256")
    with pytest.raises(ValueError, match="derivation_mismatch"):
        extraction_v11.response_schema(task, p, index, altered)
