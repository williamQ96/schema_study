"""Format evidence fixtures for the independent dataset lane.

These tests exercise the public parser and source-byte replay, never model code.
"""

from __future__ import annotations

from datetime import datetime
import json

import pytest

from high_fidelity_schema_study.four_category.dataset import parse_dataset, verify_dataset


def evidence_for(bundle: dict, fact: dict) -> dict:
    return next(e for e in bundle["evidence"] if e["evidence_id"] == fact["evidence_ids"][0])


def test_arff_nominal_codes_and_numeric_missing_lexemes(tmp_path):
    pytest.importorskip("scipy")
    source = tmp_path / "observations.arff"
    source.write_text(
        "@RELATION observations\n"
        "@ATTRIBUTE status {'001','002'}\n"
        "@ATTRIBUTE reading NUMERIC\n"
        "@DATA\n"
        "001,1.50\n"
        "002,?\n",
        encoding="utf-8",
    )
    bundle = parse_dataset(source, root=tmp_path, sample_limit=2)
    assert bundle["status"] == "pass", bundle["issues"]
    enums = [f for f in bundle["facts"] if f["subject_id"] == "attribute:status" and f["predicate"] == "declared_enum"]
    assert enums and enums[0]["value"] == ["001", "002"]
    assert enums[0]["categories"] == ["value"] and enums[0]["basis"] == "declared"
    reading = [f for f in bundle["facts"] if f["subject_id"] == "attribute:reading" and f["predicate"] == "observed_decoded_value"]
    assert {evidence_for(bundle, f)["raw_value"]["raw_row_lexeme"] for f in reading} == {"001,1.50", "002,?"}
    assert {str(f["value"]) for f in reading} == {"1.5", "NaN"}
    json.dumps(bundle, allow_nan=False)
    assert verify_dataset(bundle, tmp_path) == []


def test_xlsx_decoded_and_original_cell_tokens(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["code", "padded", "date", "formula"])
    sheet["A2"] = "001"
    sheet["B2"] = 1
    sheet["B2"].number_format = "000"
    sheet["C2"] = datetime(2024, 1, 2)
    sheet["D2"] = "=B2+1"
    source = tmp_path / "cells.xlsx"
    workbook.save(source)
    bundle = parse_dataset(source, root=tmp_path, sample_limit=1)
    assert bundle["status"] == "pass", bundle["issues"]
    cells = {evidence_for(bundle, f)["locator"]["cell"]: evidence_for(bundle, f)["raw_value"]
             for f in bundle["facts"] if f["predicate"] == "observed_decoded_value"}
    assert cells["A2"]["decoded_value"] == "001"
    assert cells["B2"]["decoded_value"] == 1
    assert cells["B2"]["xml_value_token"] == "1" and cells["B2"]["number_format"] == "000"
    assert cells["C2"]["xml_value_token"] is not None
    assert cells["D2"]["xml_formula_token"] == "B2+1"
    assert verify_dataset(bundle, tmp_path) == []


def test_xml_namespace_repeated_siblings_and_whitespace(tmp_path):
    source = tmp_path / "rows.xml"
    source.write_text('<root xmlns:x="urn:test"><x:item code="001">  alpha  </x:item><x:item>beta</x:item></root>', encoding="utf-8")
    bundle = parse_dataset(source, root=tmp_path, sample_limit=3)
    assert bundle["status"] == "pass", bundle["issues"]
    item_facts = [f for f in bundle["facts"] if f["predicate"] == "element" and f["value"] == "item"]
    assert len(item_facts) == 2
    assert {evidence_for(bundle, f)["locator"]["occurrence"] for f in item_facts} == {1, 2}
    assert any(f["predicate"] == "namespace" and f["value"] == "urn:test" for f in bundle["facts"])
    assert any(f["predicate"] == "observed_lexical_value" and f["value"] == "  alpha  " for f in bundle["facts"])
    assert verify_dataset(bundle, tmp_path) == []


def test_xml_malformed_tail_after_sample_limit_is_failure(tmp_path):
    source = tmp_path / "broken.xml"
    source.write_text("<root><first>ok</first><later>bad</root>", encoding="utf-8")
    bundle = parse_dataset(source, root=tmp_path, sample_limit=1)
    assert bundle["status"] == "failed"
    assert not any(f["predicate"] == "serialization_format" for f in bundle["facts"])
    assert verify_dataset(bundle, tmp_path) == []


def test_jsonl_malformed_tail_after_sample_limit_is_failure(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text('{"code":"001"}\n{"code":"002"}\n{"code":\n', encoding="utf-8")
    bundle = parse_dataset(source, root=tmp_path, sample_limit=1)
    assert bundle["status"] == "failed"
    assert not any(f["predicate"] == "serialization_format" for f in bundle["facts"])
    assert verify_dataset(bundle, tmp_path) == []


def test_hdf5_declared_metadata_is_separate_from_sampled_values(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "metadata.h5"
    with h5py.File(source, "w") as handle:
        data = handle.create_dataset("temperature", data=[1, 2])
        data.attrs["units"] = "K"
    bundle = parse_dataset(source, root=tmp_path, sample_limit=1)
    assert bundle["status"] == "pass", bundle["issues"]
    units = [f for f in bundle["facts"] if f["predicate"] == "declared_unit"]
    assert any(f["value"] == "K" and f["basis"] == "declared" for f in units)
    assert all(f["predicate"] != "sample_value" for f in bundle["facts"])
    assert verify_dataset(bundle, tmp_path) == []


def test_netcdf_declared_unit_metadata(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    source = tmp_path / "metadata.nc"
    with netcdf4.Dataset(source, "w", format="NETCDF3_CLASSIC") as handle:
        handle.createDimension("record", 2)
        data = handle.createVariable("temperature", "f4", ("record",))
        data.units = "K"
        data[:] = [1, 2]
    bundle = parse_dataset(source, root=tmp_path, sample_limit=1)
    assert bundle["status"] == "pass", bundle["issues"]
    assert any(f["predicate"] == "declared_unit" and f["value"] == "K" for f in bundle["facts"])
    assert verify_dataset(bundle, tmp_path) == []
