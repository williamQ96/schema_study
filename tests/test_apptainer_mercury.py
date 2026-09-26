from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import shlex
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="A Bash shell is required for Apptainer script validation")


def _bash_path(path: Path) -> str:
    """Convert Windows paths to the mounted form used by WSL/Git Bash."""
    if os.name != "nt":
        return path.as_posix()
    win = PureWindowsPath(path)
    drive = win.drive.rstrip(":").lower()
    return f"/mnt/{drive}/" + "/".join(win.parts[1:])


def _fake_apptainer(bin_dir: Path, capture: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    executable = bin_dir / "apptainer"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == --version ]]; then printf 'apptainer version test-1\\n'; exit 0; fi\n"
        "if [[ ${1:-} == build ]]; then printf 'mock sif\\n' > \"$4\"; exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE\"\n"
        "printf '%s' \"${APPTAINERENV_TEST_API_TOKEN-__unset__}\" > \"$CAPTURE.secret\"\n"
        "printf '%s' \"${APPTAINERENV_CUDA_VISIBLE_DEVICES-__unset__}\" > \"$CAPTURE.gpus\"\n"
        "printf '%s' \"${APPTAINERENV_MERCURY_PYTHON_MODULE-__unset__}\" > \"$CAPTURE.module\"\n"
        "printf '%s' \"${APPTAINERENV_MERCURY_IMAGE_SHA256-__unset__}\" > \"$CAPTURE.sha256\"\n",
        encoding="utf-8",
        newline="\n",
    )
    executable.chmod(0o755)


def _runtime_dirs(tmp_path: Path) -> tuple[Path, ...]:
    paths = tuple(tmp_path / name for name in ("repo with spaces", "source with spaces", "output with spaces", "models with spaces", "cache with spaces"))
    for path in paths:
        path.mkdir()
    return paths


def _run_launcher(image: Path, dirs: tuple[Path, ...], env: dict[str, str], *cli: str) -> subprocess.CompletedProcess:
    args = [BASH, "container/apptainer/run.sh", _bash_path(image), *(_bash_path(p) for p in dirs)]
    if cli:
        args += ["--", *cli]
    command = shlex.join(args[1:])
    return _run_shell(command, env)


def _run_shell(command: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    exported = {key: value for key, value in env.items() if key in {
        "PATH", "CAPTURE", "CUDA_IMAGE", "CUDA_VISIBLE_DEVICES", "MERCURY_CPU_ONLY",
        "MERCURY_CUDA_DEVICES", "MERCURY_SECRET_ENV_NAMES", "TEST_API_TOKEN",
        "MERCURY_PYTHON_MODULE", "MERCURY_DRY_RUN",
    }}
    prelude = " ".join(f"export {key}={shlex.quote(value)};" for key, value in exported.items())
    return subprocess.run([BASH, "-c", prelude + " " + command], cwd=REPO, capture_output=True, text=True, check=False)


def test_apptainer_shell_syntax_and_definition_contract():
    scripts = ["container/apptainer/build.sh", "container/apptainer/run.sh"]
    for script in scripts:
        syntax = subprocess.run([BASH, "-c", shlex.join(["bash", "-n", script])], cwd=REPO, capture_output=True, text=True, check=False)
        assert syntax.returncode == 0, syntax.stderr

    recipe = (REPO / "container/apptainer/four-category.def").read_text(encoding="utf-8")
    assert "Bootstrap: docker" in recipe
    assert "From: {{ CUDA_IMAGE }}" in recipe
    assert "CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04" in recipe
    assert "requirements-phase1-inference.txt" in recipe
    assert "requirements_four_category_offline.txt" in recipe
    assert "pip freeze --all > /opt/requirements/pip-freeze.txt" in recipe
    assert "HF_HUB_OFFLINE=1" in recipe and "TRANSFORMERS_OFFLINE=1" in recipe
    assert "exec python -m \"$module\" \"$@\"" in recipe
    assert "%test" in recipe and "pip check" in recipe
    assert "COPY ." not in recipe
    launcher = (REPO / "container/apptainer/run.sh").read_text(encoding="utf-8")
    assert "--containall" in launcher
    assert "high_fidelity_schema_study.four_category.cli|high_fidelity_schema_study.four_category.mercury" in launcher
    assert "|four_category.cli|" not in launcher


def test_launcher_binds_paths_with_spaces_and_explicit_secrets_without_logging_values(tmp_path):
    image = tmp_path / "fake image.sif"
    image.write_bytes(b"test sif payload")
    dirs = _runtime_dirs(tmp_path)
    bin_dir = tmp_path / "fake bin"
    capture = tmp_path / "argv.txt"
    _fake_apptainer(bin_dir, capture)
    env = os.environ.copy()
    env.update({
        "PATH": f"{_bash_path(bin_dir)}:/usr/bin:/bin",
        "CAPTURE": _bash_path(capture),
        "MERCURY_CPU_ONLY": "1",
        "MERCURY_SECRET_ENV_NAMES": "TEST_API_TOKEN",
        "TEST_API_TOKEN": "never-print-this-secret",
        "MERCURY_PYTHON_MODULE": "high_fidelity_schema_study.four_category.cli",
        "MERCURY_DRY_RUN": "0",
    })

    completed = _run_launcher(image, dirs, env, "plan", "--name", "a value with spaces")
    assert completed.returncode == 0, completed.stderr
    argv = capture.read_text(encoding="utf-8").splitlines()
    assert "--cleanenv" in argv and "--nv" not in argv
    for host, target, mode in zip(dirs, ("/workspace/high_fidelity_schema_study", "/inputs", "/outputs", "/models", "/cache"), ("ro", "ro", "rw", "ro", "rw")):
        assert f"{_bash_path(host)}:{target}:{mode}" in argv
    assert "--pwd" in argv and "/workspace/high_fidelity_schema_study" in argv
    assert (tmp_path / "argv.txt.secret").read_text(encoding="utf-8") == "never-print-this-secret"
    assert (tmp_path / "argv.txt.gpus").read_text(encoding="utf-8") == ""
    expected_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    assert (tmp_path / "argv.txt.sha256").read_text(encoding="utf-8") == expected_hash
    assert "never-print-this-secret" not in completed.stdout + completed.stderr


def test_gpu_selection_requires_scheduler_or_explicit_nonconflicting_mask(tmp_path):
    image = tmp_path / "runtime.sif"
    image.write_bytes(b"image")
    dirs = _runtime_dirs(tmp_path)
    bin_dir = tmp_path / "bin"
    capture = tmp_path / "args"
    _fake_apptainer(bin_dir, capture)
    base = os.environ.copy()
    base.update({"PATH": f"{_bash_path(bin_dir)}:/usr/bin:/bin", "CAPTURE": _bash_path(capture)})
    base.pop("CUDA_VISIBLE_DEVICES", None)
    base.pop("MERCURY_CUDA_DEVICES", None)

    missing = _run_launcher(image, dirs, base)
    assert missing.returncode != 0 and "requires CUDA_VISIBLE_DEVICES" in missing.stderr

    selected = base.copy()
    selected["MERCURY_CUDA_DEVICES"] = "0,2"
    success = _run_launcher(image, dirs, selected, "run", "--allow-live")
    assert success.returncode == 0, success.stderr
    argv = capture.read_text(encoding="utf-8").splitlines()
    assert "--nv" in argv
    assert (tmp_path / "args.gpus").read_text(encoding="utf-8") == "0,2"
    assert (tmp_path / "args.module").read_text(encoding="utf-8") == "high_fidelity_schema_study.four_category.mercury"

    conflict = selected.copy()
    conflict["CUDA_VISIBLE_DEVICES"] = "1"
    rejected = _run_launcher(image, dirs, conflict)
    assert rejected.returncode != 0 and "conflicts with scheduler-provided" in rejected.stderr


def test_empty_scheduler_gpu_mask_fails_and_cpu_dry_run_is_safe(tmp_path):
    image = tmp_path / "runtime.sif"
    image.write_bytes(b"image")
    dirs = _runtime_dirs(tmp_path)
    bin_dir = tmp_path / "bin"
    capture = tmp_path / "args"
    _fake_apptainer(bin_dir, capture)
    env = os.environ.copy()
    env.update({"PATH": f"{_bash_path(bin_dir)}:/usr/bin:/bin", "CAPTURE": _bash_path(capture), "CUDA_VISIBLE_DEVICES": ""})
    rejected = _run_launcher(image, dirs, env)
    assert rejected.returncode != 0 and "empty GPU selection" in rejected.stderr

    env["MERCURY_CPU_ONLY"] = "1"
    env["MERCURY_DRY_RUN"] = "1"
    env["MERCURY_SECRET_ENV_NAMES"] = "TEST_API_TOKEN"
    env["TEST_API_TOKEN"] = "never-print-this-secret"
    dry = _run_launcher(image, dirs, env, "run", "--allow-live")
    assert dry.returncode == 0, dry.stderr
    assert "MERCURY_APPTAINER_ARGV=" in dry.stdout
    assert "--nv" not in dry.stdout
    assert "never-print-this-secret" not in dry.stdout + dry.stderr
    assert "MERCURY_APPTAINER_MODULE=high_fidelity_schema_study.four_category.mercury" in dry.stdout


def test_build_manifest_records_image_and_inputs_with_mocked_apptainer(tmp_path):
    output_dir = tmp_path / "build output"
    output_dir.mkdir()
    bin_dir = tmp_path / "mock bin"
    capture = tmp_path / "build args"
    _fake_apptainer(bin_dir, capture)
    env = os.environ.copy()
    env.update({"PATH": f"{_bash_path(bin_dir)}:/usr/bin:/bin", "CAPTURE": _bash_path(capture), "CUDA_IMAGE": "example.invalid/cuda:test"})
    out_image = output_dir / "mercury test.sif"
    completed = _run_shell(shlex.join(["container/apptainer/build.sh", _bash_path(out_image)]), env)
    assert completed.returncode == 0, completed.stderr
    manifest_path = Path(str(out_image) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["base_image_uri"] == "example.invalid/cuda:test"
    assert manifest["apptainer_version"] == "apptainer version test-1"
    assert manifest["image_bytes"] == out_image.stat().st_size
    assert manifest["image_sha256"] == hashlib.sha256(out_image.read_bytes()).hexdigest()
    assert set(manifest["requirements_sha256"]) == {"requirements-phase1-inference.txt", "requirements_four_category_offline.txt"}

    refused = _run_shell(shlex.join(["container/apptainer/build.sh", _bash_path(out_image)]), env)
    assert refused.returncode == 2
