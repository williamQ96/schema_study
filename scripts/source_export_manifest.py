"""Build or verify the public source byte inventory (excluding itself)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "source-export-manifest.json"
VERSION = "schema-study-public-export/v2"


def inventory() -> list[dict]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    names = sorted(p.decode("utf-8") for p in result.stdout.split(b"\0") if p)
    rows = []
    for name in names:
        if name == MANIFEST.name:
            continue
        path = ROOT / name
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("missing_or_escaping_source:" + name)
        data = path.read_bytes()
        rows.append({"path": name, "bytes": len(data),
                     "sha256_file_bytes": hashlib.sha256(data).hexdigest()})
    return rows


def document() -> dict:
    return {"schema_version": VERSION,
            "snapshot": "V13 revision 2 source; GPU output qualification pending",
            "scope": "all tracked public files except this self-referential manifest",
            "source_text_newlines": "LF in Git checkout",
            "files": inventory()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args()
    expected = document()
    if args.mode == "build":
        MANIFEST.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    else:
        actual = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("public_source_inventory_mismatch")
    print(json.dumps({"status": "pass", "files": len(expected["files"]), "mode": args.mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
