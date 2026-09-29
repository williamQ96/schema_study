"""Offline integrity tests for the isolated MinerU runner; no model calls."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

from high_fidelity_schema_study.four_category.common import digest, file_digest, read_json, seal
from high_fidelity_schema_study.four_category import mineru_runner as runner


def _setup(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nmock source bytes\n")
    models = tmp_path / "models"
    models.mkdir()
    (models / "weights.bin").write_bytes(b"frozen model bytes")
    manifest = runner.inventory_models(models)
    # The runner deliberately clears MINERU_* variables. Replace environ with
    # an isolated mapping so both existing and test-only values are restored.
    monkeypatch.setattr(runner.os, "environ", dict(os.environ))
    monkeypatch.setenv("MINERU_API_URL", "https://must-not-be-used.example")
    versions = {"mineru": "4.0.8", "docvortex": "0.5.2", "onnxruntime": "1.0.0"}
    monkeypatch.setattr(runner.importlib.metadata, "version", versions.__getitem__)
    return pdf, models, manifest


def _middle(*, full=True, version="4.0.8", tier="basic"):
    return {"schema": "docvortex.middle", "schema_version": "2.0",
            "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": version}},
            "extensions": {"mineru": {"tier": tier, "parse_mode": "txt"},
                           "docvortex_layout": {"version": 1, "pages": [{"page_idx": 0, "width_pt": 612, "height_pt": 792}]}},
            "pages": [{"page_idx": 0, "blocks": []}], "is_full_document": full}


def _fake_result(monkeypatch, middle):
    writer_module = ModuleType("mineru.parser.writer")
    class FakeWriter:
        def __init__(self, path):
            self.path = Path(path)
    writer_module.FileBasedDataWriter = FakeWriter
    mineru_module = ModuleType("mineru")
    parser_module = ModuleType("mineru.parser")
    parser_module.writer = writer_module
    mineru_module.parser = parser_module
    monkeypatch.setitem(sys.modules, "mineru", mineru_module)
    monkeypatch.setitem(sys.modules, "mineru.parser", parser_module)
    monkeypatch.setitem(sys.modules, "mineru.parser.writer", writer_module)
    class FakeResult:
        def to_dict(self):
            return copy.deepcopy(middle)
        def markdown(self):
            return "# Offline parser result\n"
        def save(self, writer):
            writer.path.mkdir(parents=True)
            (writer.path / "asset.bin").write_bytes(b"fake asset")
    return FakeResult()


def test_successful_fake_sdk_records_complete_byte_receipt(tmp_path, monkeypatch):
    pdf, models, manifest = _setup(tmp_path, monkeypatch)
    result = _fake_result(monkeypatch, _middle())
    observed = {}
    def parse(path, **kwargs):
        observed.update(path=path, kwargs=kwargs,
                        env={key: runner.os.environ.get(key) for key in
                             ("MINERU_API_URL", "MINERU_MODEL_SOURCE", "MINERU_MODEL_SMALL_BACKEND",
                              "MINERU_TABLE_DEVICE", "CUDA_VISIBLE_DEVICES")})
        return result
    output = tmp_path / "attempt"
    receipt = runner.parse_pdf(pdf, output, model_root=models, model_manifest=manifest, parser=parse)
    assert receipt["status"] == "success"
    assert receipt["pages"] == 1
    assert observed["kwargs"] == {"tier": "basic", "ocr_mode": "txt", "page_range": ""}
    assert observed["env"] == {"MINERU_API_URL": None, "MINERU_MODEL_SOURCE": "local",
                               "MINERU_MODEL_SMALL_BACKEND": "onnx", "MINERU_TABLE_DEVICE": "cpu",
                               "CUDA_VISIBLE_DEVICES": ""}
    identity = read_json(output / "parser_identity.json")
    raw = read_json(output / "middle.json")
    assert identity["raw_canonical_sha256"] == digest(raw)
    assert identity["raw_file_bytes_sha256"] == file_digest(output / "middle.json")
    assert identity["source_pdf_sha256"] == file_digest(pdf)
    assert identity["model_manifest_sha256"] == manifest["manifest_sha256"]
    assert identity["parser_identity_sha256"] == digest({k: v for k, v in identity.items()
                                                          if k != "parser_identity_sha256"})
    assert (output / "raw_result/asset.bin").read_bytes() == b"fake asset"
    for member in receipt["files"]:
        path = output / member["path"]
        assert path.stat().st_size == member["bytes"]
        assert file_digest(path) == member["file_bytes_sha256"]
    assert read_json(output / "receipt.json") == receipt


def test_model_bytes_and_self_resealed_manifest_must_match_inventory(tmp_path, monkeypatch):
    pdf, models, manifest = _setup(tmp_path, monkeypatch)
    (models / "weights.bin").write_bytes(b"mutated model bytes")
    with pytest.raises(ValueError, match="model_inventory_mismatch"):
        runner.parse_pdf(pdf, tmp_path / "no-attempt", model_root=models, model_manifest=manifest,
                         parser=lambda *_a, **_k: pytest.fail("parser must not run"))
    assert not (tmp_path / "no-attempt").exists()
    self_resealed = copy.deepcopy(manifest)
    self_resealed["files"][0]["file_bytes_sha256"] = "0" * 64
    self_resealed = seal(self_resealed, "manifest_sha256")
    with pytest.raises(ValueError, match="model_inventory_mismatch"):
        runner.parse_pdf(pdf, tmp_path / "still-no-attempt", model_root=models,
                         model_manifest=self_resealed, parser=lambda *_a, **_k: pytest.fail("parser must not run"))


def test_nonbasic_and_dependency_version_mismatch_fail_before_parse(tmp_path, monkeypatch):
    pdf, models, manifest = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="runner_requires_basic"):
        runner.parse_pdf(pdf, tmp_path / "standard", model_root=models, model_manifest=manifest,
                         tier="standard", parser=lambda *_a, **_k: pytest.fail("parser must not run"))
    assert not (tmp_path / "standard").exists()
    monkeypatch.setattr(runner.importlib.metadata, "version",
                        lambda name: "4.0.9" if name == "mineru" else "0.5.2" if name == "docvortex" else "1.0.0")
    with pytest.raises(ValueError, match="parser_dependency_version_mismatch"):
        runner.parse_pdf(pdf, tmp_path / "wrong-version", model_root=models, model_manifest=manifest,
                         parser=lambda *_a, **_k: pytest.fail("parser must not run"))
    assert not (tmp_path / "wrong-version").exists()


def test_partial_document_is_failed_and_preserved_without_identity(tmp_path, monkeypatch):
    pdf, models, manifest = _setup(tmp_path, monkeypatch)
    result = _fake_result(monkeypatch, _middle(full=False))
    output = tmp_path / "partial"
    receipt = runner.parse_pdf(pdf, output, model_root=models, model_manifest=manifest,
                               parser=lambda *_a, **_k: result)
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "ValueError"
    assert "partial_document_not_admitted" in receipt["error"]
    assert not (output / "middle.json").exists()
    assert not (output / "parser_identity.json").exists()
    assert read_json(output / "receipt.json") == receipt
    assert all((output / member["path"]).is_file() for member in receipt["files"])


def test_source_changed_during_parse_is_failed_before_raw_admission(tmp_path, monkeypatch):
    pdf, models, manifest = _setup(tmp_path, monkeypatch)
    result = _fake_result(monkeypatch, _middle())
    def parse(path, **_kwargs):
        Path(path).write_bytes(b"changed during parse")
        return result
    output = tmp_path / "changed"
    receipt = runner.parse_pdf(pdf, output, model_root=models, model_manifest=manifest, parser=parse)
    assert receipt["status"] == "failed"
    assert "source_or_models_changed_during_parse" in receipt["error"]
    assert not (output / "middle.json").exists()
    assert not (output / "parser_identity.json").exists()
    assert read_json(output / "receipt.json") == receipt
