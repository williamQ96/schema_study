from __future__ import annotations

import copy
from pathlib import Path

import pytest

from high_fidelity_schema_study.four_category.checkpoints import build_manifest, verify_manifest
from high_fidelity_schema_study.four_category.common import digest


def checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoint"
    (root / "weights").mkdir(parents=True)
    (root / "config.json").write_bytes(b'{"architectures":["Fixture"]}')
    (root / "tokenizer.json").write_bytes(b'{"version":"1"}')
    (root / "modeling_fixture.py").write_bytes(b"class Fixture: pass\n")
    (root / "weights" / "model-00001.safetensors").write_bytes(b"synthetic weights\x00\x01")
    return root


def test_complete_stable_inventory_and_replay(tmp_path):
    root = checkpoint(tmp_path)
    manifest = build_manifest(root, "fixture-revision")
    assert manifest["schema_version"] == "mercury-checkpoint/v1"
    assert manifest["hash_mode"] == "file_bytes"
    assert manifest["manifest_sha256"] == digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    assert [row["path"] for row in manifest["files"]] == [
        "config.json", "modeling_fixture.py", "tokenizer.json", "weights/model-00001.safetensors"]
    assert manifest == build_manifest(root, "fixture-revision")
    assert verify_manifest(manifest, root, "fixture-revision") == []
    assert "manifest_revision_mismatch" in verify_manifest(manifest, root, "different")


@pytest.mark.parametrize("changed", ["config.json", "weights/model-00001.safetensors"])
def test_changed_bytes_are_detected_even_at_same_length(tmp_path, changed):
    root = checkpoint(tmp_path)
    manifest = build_manifest(root, "r")
    path = root / changed
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert f"checkpoint_file_sha256_mismatch:{changed}" in verify_manifest(manifest, root)


def test_added_removed_files_and_missing_config(tmp_path):
    root = checkpoint(tmp_path)
    manifest = build_manifest(root, "r")
    (root / "new_custom_code.py").write_bytes(b"new")
    (root / "tokenizer.json").unlink()
    errors = verify_manifest(manifest, root)
    assert "checkpoint_file_added:new_custom_code.py" in errors
    assert "checkpoint_file_missing:tokenizer.json" in errors
    (root / "config.json").unlink()
    assert "checkpoint_inventory_failed:ValueError:checkpoint root requires config.json" in verify_manifest(manifest, root)
    with pytest.raises(ValueError, match="config.json"):
        build_manifest(root, "r")


def test_manifest_tampering_and_unsafe_paths_rejected_before_traversal(tmp_path):
    root = checkpoint(tmp_path)
    manifest = build_manifest(root, "r")
    forged = copy.deepcopy(manifest)
    forged["files"][0]["sha256"] = "0" * 64
    assert "manifest_self_hash_mismatch" in verify_manifest(forged, root)
    forged = copy.deepcopy(manifest)
    forged["files"][0]["path"] = "../elsewhere/config.json"
    forged["manifest_sha256"] = digest({k: v for k, v in forged.items() if k != "manifest_sha256"})
    assert "manifest_path_unsafe:0" in verify_manifest(forged, root)
    forged = copy.deepcopy(manifest)
    forged["files"].append(copy.deepcopy(forged["files"][0]))
    forged["manifest_sha256"] = digest({k: v for k, v in forged.items() if k != "manifest_sha256"})
    assert "manifest_duplicate_paths" in verify_manifest(forged, root)


def test_file_symlink_hashes_target_and_directory_symlink_is_rejected(tmp_path):
    root = checkpoint(tmp_path)
    blob = tmp_path / "blob"
    blob.write_bytes(b"external snapshot blob")
    linked = root / "weights" / "linked.safetensors"
    try:
        linked.symlink_to(blob)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    manifest = build_manifest(root, "r")
    assert next(row for row in manifest["files"] if row["path"] == "weights/linked.safetensors")["bytes"] == len(blob.read_bytes())
    assert verify_manifest(manifest, root) == []
    blob.write_bytes(b"changed snapshot blob")
    assert "checkpoint_file_sha256_mismatch:weights/linked.safetensors" in verify_manifest(manifest, root)
    linked.unlink()
    directory_link = root / "linked_directory"
    try:
        directory_link.symlink_to(root / "weights", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this platform")
    with pytest.raises(ValueError, match="symlink is not a regular file"):
        build_manifest(root, "r")


def test_unreadable_or_nonexistent_root_returns_error(tmp_path):
    root = checkpoint(tmp_path)
    manifest = build_manifest(root, "r")
    assert verify_manifest(manifest, tmp_path / "missing") == [
        "checkpoint_inventory_failed:ValueError:checkpoint root must be an existing non-symlink directory"]
