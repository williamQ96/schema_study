"""Optional, isolated MinerU execution with explicit local models and byte receipts.

The inference container does not import or install MinerU. Run this module in
the separate parser environment; every output directory is a new attempt.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time

from .common import digest, file_digest, now, read_json, seal, seal_errors, write_new

MINERU_VERSION = "4.0.8"
DOCVORTEX_VERSION = "0.5.2"


def inventory_models(root: Path) -> dict:
    root = Path(root).resolve(strict=True)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and not any(part.startswith(".") for part in relative.parts):
            if not path.resolve().is_relative_to(root):
                raise ValueError("model_symlink_outside_root")
            files.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                          "file_bytes_sha256": file_digest(path)})
    if not files:
        raise ValueError("empty_model_inventory")
    return seal({"schema_version": "mineru-model-inventory/v1", "files": files,
                 "hash_semantics": "file_bytes_sha256", "hidden_cache_metadata_excluded": True},
                "manifest_sha256")


def parse_pdf(pdf: Path, output: Path, *, model_root: Path, model_manifest: dict,
              tier: str = "basic", ocr_mode: str = "txt", parser=None) -> dict:
    """Execute Basic/ONNX CPU locally; no VLM or remote-service fallback.

    Standard/Advanced require a separately qualified runtime and are deliberately
    rejected by this runner. The structured adapter is independent of this tier.
    """
    if tier != "basic" or ocr_mode not in {"txt", "ocr", "auto"}:
        raise ValueError("runner_requires_basic_tier_and_explicit_ocr_mode")
    pdf, output, model_root = Path(pdf).resolve(strict=True), Path(output), Path(model_root).resolve(strict=True)
    if seal_errors(model_manifest, "manifest_sha256") or inventory_models(model_root) != model_manifest:
        raise ValueError("model_inventory_mismatch")
    versions = {key: importlib.metadata.version(key) for key in ("mineru", "docvortex", "onnxruntime")}
    if versions["mineru"] != MINERU_VERSION or versions["docvortex"] != DOCVORTEX_VERSION:
        raise ValueError("parser_dependency_version_mismatch")
    output.mkdir(parents=True, exist_ok=False)
    source_hash = file_digest(pdf)
    # Private configuration prevents a personal/default remote API or LLM-aided
    # postprocessor from changing this experimental condition.
    private_home = output.resolve() / "parser_home"
    private_home.mkdir()
    configuration = {"model": {"source": "local", "base_dir": str(model_root),
                                "small_backend": "onnx"}}
    config_path = private_home / "config.yaml"
    # JSON is a YAML subset; avoid an extra configuration serializer.
    write_new(config_path, configuration)
    for key in list(os.environ):
        if key.startswith("MINERU_"):
            os.environ.pop(key)
    os.environ.update(MINERU_HOME=str(private_home), MINERU_CONFIG=str(config_path),
                      MINERU_MODEL_SOURCE="local", MINERU_MODEL_SMALL_BACKEND="onnx",
                      MINERU_TABLE_DEVICE="cpu", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    request = {"schema_version": "mineru-parse-request/v1", "source_pdf_sha256": source_hash,
               "tier": tier, "ocr_mode": ocr_mode, "page_range": "",
               "small_backend": "onnx", "table_device": "cpu",
               "model_manifest_sha256": model_manifest["manifest_sha256"],
               "config_sha256": digest(configuration), "dependencies": versions,
               "network_policy": "local_models_no_remote_inference", "started_at": now()}
    write_new(output / "request.json", request)
    write_new(output / "model-manifest.json", model_manifest)
    started = time.monotonic()
    try:
        if parser is None:
            from mineru.parser import parse
            parser = parse
        result = parser(str(pdf), tier=tier, ocr_mode=ocr_mode, page_range="")
        middle = result.to_dict()
        if middle.get("schema") != "docvortex.middle" or middle.get("schema_version") != "2.0":
            raise ValueError("unexpected_mineru_intermediate_contract")
        if middle.get("metadata", {}).get("producer", {}).get("version") != MINERU_VERSION:
            raise ValueError("returned_parser_version_mismatch")
        if middle.get("extensions", {}).get("mineru", {}).get("tier") != tier:
            raise ValueError("returned_parser_tier_mismatch")
        if middle.get("is_full_document") is not True:
            raise ValueError("partial_document_not_admitted")
        if file_digest(pdf) != source_hash or inventory_models(model_root) != model_manifest:
            raise ValueError("source_or_models_changed_during_parse")
        write_new(output / "middle.json", middle)
        (output / "reading.md").write_text(result.markdown(), encoding="utf-8", newline="\n")
        # Save image assets and raw model output independently of the authoritative
        # middle.json; each is retained as produced by the pinned SDK.
        from mineru.parser.writer import FileBasedDataWriter
        result.save(FileBasedDataWriter(str(output / "raw_result")))
        identity = seal({"schema_version": "paper-parser-identity/v1", "parser": "mineru",
                         "version": MINERU_VERSION, "docvortex_version": DOCVORTEX_VERSION,
                         "source_pdf_sha256": source_hash, "tier": tier, "ocr_mode": ocr_mode,
                         "small_backend": "onnx", "table_device": "cpu",
                         "raw_canonical_sha256": digest(middle),
                         "raw_file_bytes_sha256": file_digest(output / "middle.json"),
                         "model_manifest_sha256": model_manifest["manifest_sha256"],
                         "config_sha256": request["config_sha256"], "request_sha256": digest(request),
                         "dependencies": versions}, "parser_identity_sha256")
        write_new(output / "parser_identity.json", identity)
        receipt = {"status": "success", "duration_s": time.monotonic() - started,
                   "pages": len(middle["pages"]), "parser_identity_sha256": identity["parser_identity_sha256"]}
    except Exception as exc:
        receipt = {"status": "failed", "duration_s": time.monotonic() - started,
                   "error_type": type(exc).__name__, "error": str(exc)[:2000]}
    receipt.update(schema_version="mineru-parse-receipt/v1", ended_at=now(),
                   source_pdf_sha256=source_hash, runner_file_bytes_sha256=file_digest(Path(__file__)))
    receipt["files"] = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size,
                         "file_bytes_sha256": file_digest(p)} for p in sorted(output.rglob("*")) if p.is_file()]
    write_new(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    inventory = subs.add_parser("inventory")
    inventory.add_argument("--model-root", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    run = subs.add_parser("parse")
    run.add_argument("--pdf", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-root", type=Path, required=True)
    run.add_argument("--model-manifest", type=Path, required=True)
    run.add_argument("--tier", choices=["basic"], default="basic")
    run.add_argument("--ocr-mode", choices=["txt", "ocr", "auto"], default="txt")
    args = parser.parse_args(argv)
    if args.command == "inventory":
        value = inventory_models(args.model_root)
        write_new(args.output, value)
        print({"files": len(value["files"]), "manifest_sha256": value["manifest_sha256"]})
        return 0
    result = parse_pdf(args.pdf, args.output, model_root=args.model_root,
                       model_manifest=read_json(args.model_manifest), tier=args.tier, ocr_mode=args.ocr_mode)
    print({key: value for key, value in result.items() if key != "files"})
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
