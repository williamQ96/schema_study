"""Portable, append-only repair artifacts. No model calls or legacy writes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(), "sha256": file_hash(path), "hash_mode": "file_bytes", "bytes": path.stat().st_size}


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes artifact root: {relative}")
    return path


def verify_inventory(root: Path, records: list[dict]) -> list[dict]:
    failures = []
    for record in records:
        try:
            path = contained(root, record["path"])
            mode = record.get("hash_mode", "file_bytes")
            if mode not in {"file_bytes", "canonical_json_utf8_sorted_compact"}:
                raise ValueError("unknown hash mode")
            actual = file_hash(path) if mode == "file_bytes" else canonical_hash(read_json(path))
            if actual != record["sha256"]:
                failures.append({"path": record["path"], "problem": "hash_mismatch"})
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"path": record.get("path"), "problem": str(exc)})
    return failures
