from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from high_fidelity_schema_study.four_category.dataset import parse_dataset, verify_dataset


TAXONOMY = {"version": "test", "categories": ["structure", "encoding", "value", "syntax"]}


def test_csv_keeps_lexical_values_and_does_not_infer_controlled_vocabulary(tmp_path):
    root = tmp_path
    path = root / "values.csv"
    path.write_text("id,status,amount\n001,open,1.20\n002,closed,2.30\n", encoding="utf-8")

    bundle = parse_dataset(path, root=root, taxonomy=TAXONOMY)
    values = [f["value"] for f in bundle["facts"] if f["predicate"] == "observed_lexical_value"]

    assert values == ["001", "open", "1.20", "002", "closed", "2.30"]
    assert not any(f["predicate"] in {"enum", "allowed_values", "minimum", "maximum", "nullable"} for f in bundle["facts"])
    assert verify_dataset(bundle, root, taxonomy=TAXONOMY) == []


def test_replay_verifier_rejects_fact_tamper_even_with_recomputed_bundle_hash(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("x\n007\n", encoding="utf-8")
    bundle = parse_dataset(path, root=tmp_path, taxonomy=TAXONOMY)
    altered = deepcopy(bundle)
    altered["facts"][-1]["value"] = "forged"
    # A forged self-hash alone must not allow altered claims through.
    from high_fidelity_schema_study.four_category.dataset import _canonical, _digest
    altered["bundle_sha256"] = _digest(_canonical({k: v for k, v in altered.items() if k != "bundle_sha256"}))

    assert "derived bundle differs from replayed source" in verify_dataset(altered, tmp_path, taxonomy=TAXONOMY)


def test_sampling_is_bounded_and_explicit(tmp_path):
    path = tmp_path / "many.csv"
    path.write_text("x\n" + "\n".join(str(i) for i in range(25)) + "\n", encoding="utf-8")
    bundle = parse_dataset(path, root=tmp_path, sample_limit=3, taxonomy=TAXONOMY)
    observations = [f for f in bundle["facts"] if f["predicate"] == "observed_lexical_value"]

    assert [f["value"] for f in observations] == ["0", "1", "2"]
    assert all(f["coverage"]["mode"] == "sample" for f in observations)


def test_tsv_serialization_order_delimiter_and_quoting_are_recorded(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text('id\tvalue\n001\t" 2 "\n', encoding="utf-8")
    bundle = parse_dataset(path, root=tmp_path, taxonomy=TAXONOMY)
    facts = bundle["facts"]
    assert any(f["predicate"] == "serialization_format" and f["value"] == "tsv" for f in facts)
    assert any(f["predicate"] == "position" and f["subject_id"] == "column:2" and f["value"] == 2 for f in facts)
    assert any(f["predicate"] == "delimiter" and f["value"] == "tab" and f["categories"] == ["syntax"] for f in facts)
    assert any(f["predicate"] == "quote_character" and f["value"] == '"' for f in facts)


def test_malformed_and_unknown_formats_are_not_given_invented_schema(tmp_path):
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    failed = parse_dataset(malformed, root=tmp_path, taxonomy=TAXONOMY)
    assert failed["status"] == "failed"
    assert failed["facts"] == []

    opaque = tmp_path / "opaque.bin"
    opaque.write_bytes(b"opaque")
    unsupported = parse_dataset(opaque, root=tmp_path, taxonomy=TAXONOMY)
    assert unsupported["status"] == "unsupported"
    assert unsupported["facts"] == []


def test_json_evidence_has_real_pointer_locations(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"code":"007","values":[3,4]}', encoding="utf-8")
    bundle = parse_dataset(path, root=tmp_path, taxonomy=TAXONOMY)

    code = next(f for f in bundle["facts"] if f["predicate"] == "observed_lexical_value" and f["subject_id"] == "$/code")
    evidence = next(e for e in bundle["evidence"] if e["evidence_id"] == code["evidence_ids"][0])
    assert evidence["locator"] == {"kind": "json_pointer", "pointer": "$/code", "line": 1}
    assert code["value"] == "007"


def test_truncated_json_arrays_are_sample_covered(tmp_path):
    path = tmp_path / "array.json"
    path.write_text('{"measurements":[1,2,3,4]}', encoding="utf-8")
    bundle = parse_dataset(path, root=tmp_path, sample_limit=2, taxonomy=TAXONOMY)
    subset = [f for f in bundle["facts"] if f["subject_id"].startswith("$/measurements/")]
    assert subset and all(f["coverage"]["mode"] == "sample" for f in subset)


def test_csv_whitespace_and_mixed_observations_remain_lexical(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("value\n 001 \nword\n", encoding="utf-8")
    bundle = parse_dataset(path, root=tmp_path, taxonomy=TAXONOMY)
    vals = [f["value"] for f in bundle["facts"] if f["predicate"] == "observed_lexical_value"]
    assert vals == [" 001 ", "word"]
    assert not any(f["predicate"] in {"physical_type", "allowed_values", "declared_enum"} for f in bundle["facts"])
    assert all(f["categories"] == ["syntax"] for f in bundle["facts"] if f["predicate"] == "observed_lexical_value")
    assert all(f["fact_id"].startswith(bundle["dataset_id"] + ":f") for f in bundle["facts"])


def test_bounded_csv_io_never_uses_whole_file_read_apis(tmp_path, monkeypatch):
    path = tmp_path / "large.csv"
    path.write_text("x\n" + "\n".join(str(i) for i in range(2000)) + "\n", encoding="utf-8")
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def reject_large_read(self, *args, **kwargs):
        if self == path:
            raise AssertionError("dataset must be hashed and sniffed with bounded streaming")
        return original_read_bytes(self, *args, **kwargs)

    def reject_dataset_text(self, *args, **kwargs):
        if self == path:
            raise AssertionError("dataset text must not be loaded whole")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", reject_large_read)
    monkeypatch.setattr(Path, "read_text", reject_dataset_text)
    bundle = parse_dataset(path, root=tmp_path, sample_limit=2, taxonomy=TAXONOMY)
    observations = [f for f in bundle["facts"] if f["predicate"] == "observed_lexical_value"]
    assert bundle["status"] == "pass"
    assert len(observations) == 2
    assert all(f["coverage"]["mode"] == "sample" for f in observations)


def test_explicit_json_schema_sidecar_declarations_are_separate_and_bound(tmp_path):
    data = tmp_path / "data.json"
    schema = tmp_path / "schema.json"
    data.write_text('{"status":"unexpected"}', encoding="utf-8")
    schema.write_text('{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["status","missing"],"properties":{"status":{"type":"string","enum":["open","closed"],"pattern":"^[a-z]+$"},"missing":{"type":"integer"}}}', encoding="utf-8")
    bundle = parse_dataset(data, root=tmp_path, taxonomy=TAXONOMY, sidecars=[schema])
    declared = [f for f in bundle["facts"] if f["basis"] == "declared"]
    observed = [f for f in bundle["facts"] if f["predicate"] == "observed_lexical_value"]
    assert any(f["predicate"] == "declared_enum" and f["value"] == ["open", "closed"] for f in declared)
    assert any(f["predicate"] == "required" and f["subject_id"] == "missing" and f["value"] is True for f in declared)
    assert any(f["predicate"] == "required" and f["subject_id"] == "status" and f["value"] is True for f in declared)
    declared_type = next(f for f in declared if f["predicate"] == "declared_type")
    declared_format = next(f for f in declared if f["predicate"] == "pattern")
    assert declared_type["categories"] == ["structure"]
    assert declared_format["categories"] == ["syntax"]
    assert any(f["value"] == "unexpected" and f["basis"] == "observed" for f in observed)
    assert all(bundle["sources"][0]["path"] != e["source_path"] for e in bundle["evidence"] if e["locator"].get("kind") == "json_schema_declaration")
    assert verify_dataset(bundle, tmp_path, taxonomy=TAXONOMY) == []
    schema.write_text(schema.read_text(encoding="utf-8").replace("closed", "shut"), encoding="utf-8")
    assert "derived bundle differs from replayed source" in verify_dataset(bundle, tmp_path, taxonomy=TAXONOMY)


def test_zarr_v2_metadata_inventory_hashes_only_metadata_and_parses_tree(tmp_path):
    store = tmp_path / "sample.zarr"
    array = store / "measurements"
    array.mkdir(parents=True)
    (store / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    (store / ".zattrs").write_text('{"title":"observations"}', encoding="utf-8")
    (array / ".zarray").write_text('{"zarr_format":2,"shape":[10,2],"chunks":[5,2],"dtype":"<f4","compressor":null,"filters":null,"order":"C"}', encoding="utf-8")
    (array / ".zattrs").write_text('{"units":"m"}', encoding="utf-8")
    (array / "0.0").write_bytes(b"chunk bytes excluded")
    bundle = parse_dataset(store, root=tmp_path, taxonomy=TAXONOMY)
    source = bundle["sources"][0]
    assert bundle["status"] == "pass"
    assert source["hash_mode"] == "canonical_json_metadata_inventory"
    assert source["members"] and all(m["path"].endswith((".zgroup", ".zattrs", ".zarray")) for m in source["members"])
    assert all(f["coverage"] == {"mode": "metadata_only", "chunks_read": False} for f in bundle["facts"])
    assert any(f["predicate"] == "shape" and f["value"] == [10, 2] for f in bundle["facts"])
    assert verify_dataset(bundle, tmp_path, taxonomy=TAXONOMY) == []
    (array / "0.0").write_bytes(b"new chunk bytes; metadata coverage stays metadata-only")
    assert verify_dataset(bundle, tmp_path, taxonomy=TAXONOMY) == []
    (array / ".zarray").write_text('{bad', encoding="utf-8")
    failed = parse_dataset(store, root=tmp_path, taxonomy=TAXONOMY)
    assert failed["status"] == "failed" and failed["facts"] == []


def test_zarr_metadata_mutation_fails_replay(tmp_path):
    store = tmp_path / "sample.zarr"
    store.mkdir()
    (store / ".zarray").write_text('{"zarr_format":2,"shape":[1],"chunks":[1],"dtype":"i4"}', encoding="utf-8")
    bundle = parse_dataset(store, root=tmp_path, taxonomy=TAXONOMY)
    (store / ".zarray").write_text('{"zarr_format":2,"shape":[2],"chunks":[1],"dtype":"i4"}', encoding="utf-8")
    assert "derived bundle differs from replayed source" in verify_dataset(bundle, tmp_path, taxonomy=TAXONOMY)


def test_verifier_rejects_source_path_escape_before_replay(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    bundle = parse_dataset(path, root=tmp_path, taxonomy=TAXONOMY)
    altered = deepcopy(bundle)
    altered["sources"][0]["path"] = "../outside.csv"
    from high_fidelity_schema_study.four_category.dataset import _canonical, _digest
    altered["bundle_sha256"] = _digest(_canonical({k: v for k, v in altered.items() if k != "bundle_sha256"}))
    assert verify_dataset(altered, tmp_path, taxonomy=TAXONOMY) == ["source path escapes source_root: ../outside.csv"]


def test_native_metadata_registry_adapters_when_optional_libraries_exist(tmp_path):
    import pytest
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "sample.h5"
    with h5py.File(h5_path, "w") as h5:
        dataset = h5.create_dataset("temperature", data=[1.0, 2.0])
        dataset.attrs["units"] = "K"
        dataset.attrs["flag_values"] = [0, 1]
        dataset.attrs["flag_meanings"] = "invalid valid"
    h5_bundle = parse_dataset(h5_path, root=tmp_path, taxonomy=TAXONOMY)
    assert h5_bundle["status"] == "pass"
    physical = next(f for f in h5_bundle["facts"] if f["predicate"] == "physical_type")
    assert physical["categories"] == ["structure"] and physical["coverage"]["mode"] == "metadata_only"
    assert any(f["predicate"] == "declared_unit" and f["value"] == "K" and f["categories"] == ["value"] for f in h5_bundle["facts"])
    assert any(f["predicate"] == "declared_code_meanings" and f["categories"] == ["value"] for f in h5_bundle["facts"])
    assert verify_dataset(h5_bundle, tmp_path, taxonomy=TAXONOMY) == []

    scipy = pytest.importorskip("scipy.io")
    nc_path = tmp_path / "sample.nc"
    with scipy.netcdf_file(nc_path, "w") as nc:
        nc.createDimension("n", 2)
        var = nc.createVariable("temperature", "f", ("n",))
        var[:] = [1.0, 2.0]
        var.units = "K"
        var.flag_values = [0, 1]
        var.flag_meanings = "invalid valid"
    nc_bundle = parse_dataset(nc_path, root=tmp_path, taxonomy=TAXONOMY)
    assert nc_bundle["status"] == "pass"
    assert any(f["subject_id"].endswith("temperature") for f in nc_bundle["facts"])
    assert any(f["predicate"] == "declared_unit" and f["value"] == "K" for f in nc_bundle["facts"])
    assert any(f["predicate"] == "declared_code_meanings" for f in nc_bundle["facts"])
    assert verify_dataset(nc_bundle, tmp_path, taxonomy=TAXONOMY) == []

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"temperature": [1.0, 2.0]}), parquet_path)
    pq_bundle = parse_dataset(parquet_path, root=tmp_path, taxonomy=TAXONOMY)
    assert pq_bundle["status"] == "pass"
    assert any(f["predicate"] == "physical_type" for f in pq_bundle["facts"])
    assert verify_dataset(pq_bundle, tmp_path, taxonomy=TAXONOMY) == []
