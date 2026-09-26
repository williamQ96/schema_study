"""Manifest-driven deterministic dataset jobs; no model backend is imported.

Source inventory and parser/taxonomy identities determine the cache. Reused
bundles are replay-verified, so caching avoids new artifact creation rather
than promising to avoid source reads. Every file job has an isolated outcome.
"""
from __future__ import annotations

from pathlib import Path
import platform
from importlib import metadata

from . import dataset as adapter
from .common import contained, digest, file_digest, identity, read_json, seal, seal_errors, taxonomy, write_new


def parse_batch(manifest: dict, *, source_root: Path, output: Path, sample_limit: int = 200,
                taxonomy_value: dict | None = None, max_jobs: int | None = None) -> dict:
    if manifest.get("schema_version") != "four-category-dataset-jobs/v1":
        raise ValueError("dataset jobs manifest version invalid")
    jobs = manifest.get("sources")
    if not isinstance(jobs, list) or any(not isinstance(j, dict) or not isinstance(j.get("source_id"), str) for j in jobs):
        raise ValueError("dataset jobs require named source objects")
    if len({j["source_id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate dataset source_id")
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit <= 0:
        raise ValueError("sample_limit must be a positive integer")
    root, output = Path(source_root), Path(output)
    tax = taxonomy_value or taxonomy()
    parser_sha = file_digest(Path(adapter.__file__))
    runtime = {"python": platform.python_version(), "packages": {}}
    for name in ("h5py", "scipy", "pyarrow", "openpyxl", "netCDF4"):
        try:
            runtime["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            runtime["packages"][name] = None
    results, reused = [], 0
    for job in jobs:
        if max_jobs is not None and len(results) >= max_jobs:
            break
        try:
            path = contained(root, job["path"])
            sidecars = [contained(root, p) for p in job.get("sidecars", [])]
            if path.is_dir():
                members = [identity(p, root) for p in sorted(path.rglob("*")) if p.is_file() and p.name in {".zgroup", ".zarray", ".zattrs", ".zmetadata"}]
                if not members:
                    raise ValueError("unsupported directory; no Zarr-v2 metadata")
            else:
                members = [identity(path, root)]
            members += [identity(p, root) for p in sidecars]
            key = digest({"source_members": members, "format_hint": job.get("format_hint"),
                          "sample_limit": sample_limit, "taxonomy": tax, "parser_file_bytes_sha256": parser_sha, "runtime": runtime})
            receipt_path = output / "receipts" / (key + ".json")
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                if seal_errors(receipt, "receipt_sha256") or receipt["cache_key"] != key:
                    raise ValueError("dataset_cache_receipt_invalid")
                bundle = read_json(contained(output, receipt["bundle_path"]))
                if bundle["bundle_sha256"] != receipt["bundle_sha256"] or adapter.verify_dataset(bundle, root, taxonomy=tax):
                    raise ValueError("dataset_cache_replay_invalid")
                reused += 1
            else:
                bundle = adapter.parse_dataset(path, root=root, format_hint=job.get("format_hint"),
                                               sample_limit=sample_limit, taxonomy=tax, sidecars=sidecars)
                relative = "bundles/" + bundle["bundle_sha256"] + ".json"
                bundle_path = contained(output, relative)
                if not bundle_path.exists():
                    write_new(bundle_path, bundle)
                elif read_json(bundle_path) != bundle:
                    raise ValueError("dataset_bundle_cache_collision")
                receipt = seal({"cache_key": key, "bundle_path": relative, "bundle_sha256": bundle["bundle_sha256"],
                                "source_members": members, "parser_file_bytes_sha256": parser_sha,
                                "runtime": runtime,
                                "taxonomy_sha256": digest(tax)}, "receipt_sha256")
                write_new(receipt_path, receipt)
            results.append({"source_id": job["source_id"], "dataset_id": bundle["dataset_id"],
                            "status": bundle["status"], "issues": bundle["issues"], "cache_key": key,
                            "bundle_path": receipt["bundle_path"], "bundle_sha256": bundle["bundle_sha256"],
                            "family_id": job.get("family_id")})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            results.append({"source_id": job["source_id"], "status": "failed", "issues": [str(exc)], "bundle_path": None})
    result = seal({"schema_version": "four-category-dataset-batch/v1", "manifest_sha256": digest(manifest),
                   "taxonomy_sha256": digest(tax), "sample_limit": sample_limit,
                   "status": "complete" if len(results) == len(jobs) else "partial",
                   "results": results, "reused_this_invocation": reused, "model_calls": 0,
                   "cache_policy": "immutable output reuse with source derivation replay"}, "batch_sha256")
    path = output / "batches" / (result["batch_sha256"] + ".json")
    if not path.exists():
        write_new(path, result)
    return result
