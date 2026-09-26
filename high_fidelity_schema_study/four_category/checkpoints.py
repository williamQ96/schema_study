"""Complete byte-level inventory for a local Transformers checkpoint.

File symlinks are common in Hugging Face snapshots. Their targets are read as
bytes; directory symlinks are rejected so traversal cannot silently omit data.
The manifest covers every file, including metadata and custom model code.
"""
from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

from .common import digest, file_digest


SCHEMA_VERSION = "mercury-checkpoint/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _inventory(model_root: Path) -> list[dict]:
    root = Path(model_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint root must be an existing non-symlink directory")
    files: list[dict] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                if not path.is_file():
                    raise ValueError(f"symlink is not a regular file: {relative}")
            elif entry.is_dir(follow_symlinks=False):
                visit(path)
                continue
            elif not entry.is_file(follow_symlinks=False):
                raise ValueError(f"unsupported checkpoint entry: {relative}")
            before = path.stat()
            sha256 = file_digest(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"checkpoint file changed while hashing: {relative}")
            files.append({"path": relative, "bytes": after.st_size, "sha256": sha256})

    visit(root)
    files.sort(key=lambda row: row["path"])
    if "config.json" not in {row["path"] for row in files}:
        raise ValueError("checkpoint root requires config.json")
    return files


def build_manifest(model_root: Path, revision: str) -> dict:
    """Hash every checkpoint file without loading model weights into memory."""
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a nonempty string")
    manifest = {"schema_version": SCHEMA_VERSION, "revision": revision,
                "files": _inventory(Path(model_root)), "hash_mode": "file_bytes"}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def _manifest_errors(manifest: dict, expected_revision: str | None) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest_must_be_object"]
    errors = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest_schema_version_invalid")
    if manifest.get("hash_mode") != "file_bytes":
        errors.append("manifest_hash_mode_invalid")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        errors.append("manifest_revision_invalid")
    elif expected_revision is not None and revision != expected_revision:
        errors.append("manifest_revision_mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        return errors + ["manifest_files_must_be_list"]
    paths = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"manifest_file_invalid:{index}")
            continue
        path = row.get("path")
        if (not isinstance(path, str) or not path or "\\" in path or
                PurePosixPath(path).is_absolute() or
                any(part in {"", ".", ".."} for part in path.split("/")) or
                re.match(r"^[A-Za-z]:", path)):
            errors.append(f"manifest_path_unsafe:{index}")
        else:
            paths.append(path)
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append(f"manifest_size_invalid:{index}")
        sha256 = row.get("sha256")
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            errors.append(f"manifest_sha256_invalid:{index}")
        if set(row) != {"path", "bytes", "sha256"}:
            errors.append(f"manifest_file_fields_invalid:{index}")
    if len(paths) != len(set(paths)):
        errors.append("manifest_duplicate_paths")
    if paths != sorted(paths):
        errors.append("manifest_files_not_sorted")
    if "config.json" not in paths:
        errors.append("manifest_config_missing")
    if set(manifest) != {"schema_version", "revision", "files", "hash_mode", "manifest_sha256"}:
        errors.append("manifest_fields_invalid")
    claimed = manifest.get("manifest_sha256")
    if not isinstance(claimed, str) or not _SHA256.fullmatch(claimed):
        errors.append("manifest_self_hash_invalid")
    else:
        try:
            unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            if digest(unsigned) != claimed:
                errors.append("manifest_self_hash_mismatch")
        except (TypeError, ValueError):
            errors.append("manifest_noncanonical_json")
    return errors


def verify_manifest(manifest: dict, model_root: Path, expected_revision: str | None = None) -> list[str]:
    """Rehash the complete checkpoint and return errors rather than dispatching."""
    errors = _manifest_errors(manifest, expected_revision)
    if errors:
        return errors
    try:
        current = _inventory(Path(model_root))
    except (OSError, ValueError) as exc:
        return [f"checkpoint_inventory_failed:{type(exc).__name__}:{exc}"]
    expected = {row["path"]: row for row in manifest["files"]}
    actual = {row["path"]: row for row in current}
    errors.extend(f"checkpoint_file_missing:{path}" for path in sorted(expected.keys() - actual.keys()))
    errors.extend(f"checkpoint_file_added:{path}" for path in sorted(actual.keys() - expected.keys()))
    for path in sorted(expected.keys() & actual.keys()):
        if actual[path]["bytes"] != expected[path]["bytes"]:
            errors.append(f"checkpoint_file_size_mismatch:{path}")
        if actual[path]["sha256"] != expected[path]["sha256"]:
            errors.append(f"checkpoint_file_sha256_mismatch:{path}")
    return errors


__all__ = ["build_manifest", "verify_manifest"]
