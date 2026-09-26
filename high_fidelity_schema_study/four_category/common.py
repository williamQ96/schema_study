"""Portable identities and immutable artifacts used by the new experiment lane."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("structure", "encoding", "value", "syntax")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return strict_json(Path(path).read_text(encoding="utf-8"))


def taxonomy() -> dict:
    return read_json(ROOT / "templates/four_category_taxonomy_v1.json")


def seal(value: dict, field: str) -> dict:
    value = copy.deepcopy(value)
    value.pop(field, None)
    value[field] = digest(value)
    return value


def seal_errors(value: dict, field: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{field}:object_required"]
    try:
        return [] if seal(value, field).get(field) == value.get(field) else [f"{field}:mismatch"]
    except (TypeError, ValueError):
        return [f"{field}:noncanonical_json"]


def write_new(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def contained(root: Path, relative: str) -> Path:
    candidate = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError(f"path outside root: {relative}")
    return candidate


def identity(path: Path, root: Path) -> dict:
    path = Path(path).resolve()
    return {"path": path.relative_to(Path(root).resolve()).as_posix(), "sha256": file_digest(path),
            "hash_mode": "file_bytes", "bytes": path.stat().st_size}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_number(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_number)
    if not isinstance(value, dict):
        raise ValueError("response must be one JSON object")
    return value
