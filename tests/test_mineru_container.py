from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "container/apptainer/run-mineru.sh"


def test_recipe_is_separate_cpu_runtime_with_exact_runner_dependencies():
    requirements = (ROOT / "requirements_mineru.txt").read_text(encoding="utf-8").splitlines()
    pinned = {line for line in requirements if line and not line.startswith("#")}
    assert pinned == {"mineru==4.0.8", "docvortex==0.5.2", "onnxruntime==1.30.0"}
    recipe = (ROOT / "container/apptainer/mineru.def").read_text(encoding="utf-8")
    assert "From: {{ PYTHON_IMAGE }}" in recipe
    assert "requirements_mineru.txt" in recipe and "pip freeze --all" in recipe
    assert "high_fidelity_schema_study.four_category.mineru_runner" in recipe
    assert "requirements-phase1-inference" not in recipe
    assert "--nv" not in recipe
    assert "%test" in recipe and "import onnxruntime" in recipe


def test_wrapper_shell_syntax_when_bash_is_usable():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    path = str(SCRIPT)
    if os.name == "nt":
        # Windows bash.exe is WSL Bash, which does not accept a drive-letter path.
        path = "/mnt/" + path[0].lower() + path[2:].replace("\\", "/")
    result = subprocess.run([bash, "-n", path], capture_output=True, text=True)
    if os.name == "nt" and "no installed distributions" in result.stderr.lower():
        pytest.skip("WSL bash has no installed distribution")
    if os.name == "nt" and "No such file or directory" in result.stderr:
        pytest.skip("WSL distribution cannot access the workspace mount")
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="Linux bash dry-run required")
def test_wrapper_dry_run_mounts_and_rejects_unsupported_paths_and_tier(tmp_path):
    image = tmp_path / "mineru.sif"
    image.write_bytes(b"test image")
    roots = [tmp_path / name for name in ("repo", "pdfs", "outputs", "models", "cache")]
    for root in roots:
        root.mkdir()
    fixed = [str(image), *(str(root) for root in roots), "--"]
    env = dict(os.environ, MINERU_DRY_RUN="1")
    valid = subprocess.run(["bash", str(SCRIPT), *fixed, "inventory", "--model-root", "/models",
                            "--output", "/outputs/model-manifest.json"], env=env, capture_output=True, text=True)
    assert valid.returncode == 0, valid.stderr
    assert "/models:ro" in valid.stdout and "/outputs:rw" in valid.stdout
    assert "/inputs:ro" in valid.stdout and "/cache:rw" in valid.stdout
    assert "--nv" not in valid.stdout and "MINERU_SIF_FILE_BYTES_SHA256=" in valid.stdout
    parse = ["bash", str(SCRIPT), *fixed, "parse", "--pdf", "/inputs/P05.pdf", "--output", "/outputs/P05-attempt",
             "--model-root", "/models", "--model-manifest", "/outputs/model-manifest.json"]
    good = subprocess.run([*parse, "--tier", "basic", "--ocr-mode", "txt"], env=env, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    bad_tier = subprocess.run([*parse, "--tier", "standard"], env=env, capture_output=True, text=True)
    assert bad_tier.returncode == 2
    bad_output = copy_with_output(parse, "/models/wrong")
    denied = subprocess.run(bad_output, env=env, capture_output=True, text=True)
    assert denied.returncode == 2
    assert "Path must be under /outputs" in denied.stderr


def copy_with_output(argv: list[str], output: str) -> list[str]:
    changed = list(argv)
    changed[changed.index("--output") + 1] = output
    return changed
