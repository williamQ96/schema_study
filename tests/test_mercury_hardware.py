"""Tests for the offline Mercury hardware inventory and readiness assessment."""
from __future__ import annotations

import pytest

from high_fidelity_schema_study.four_category.hardware import (
    _parse_meminfo,
    _parse_nvidia_csv,
    assess_hardware,
    collect_hardware,
)


def _device(index: int, memory: str = "141 GB") -> dict:
    from high_fidelity_schema_study.four_category.hardware import _parse_size_bytes

    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA H200 NVL",
        "memory_total_reported": memory,
        "memory_total_bytes": _parse_size_bytes(memory),
        "driver_version": "580.95.05",
        "compute_capability": "9.0",
    }


def _observation(*, physical_count=4, visible_count=1, mask="0",
                 runtime="13.0", driver="580.95.05", probe="pass",
                 torch_available=True, inventory=True, capacity="141 GB"):
    devices = [_device(i, capacity) for i in range(physical_count)]
    for device in devices:
        device["driver_version"] = driver
    torch_devices = [_device(i, capacity) for i in range(visible_count)]
    for device in torch_devices:
        device["driver_version"] = driver
    return {
        "nvidia_smi": {
            "status": "available" if inventory else "unavailable",
            "devices": devices if inventory else [],
            "topology": {"status": "available", "raw": "GPU0 GPU1"},
        },
        "environment": {"cuda_visible_devices": mask},
        "torch": {
            "import_status": "available",
            "cuda": {
                "available": torch_available,
                "runtime_version": runtime,
                "visible_device_count": visible_count,
                "devices": torch_devices,
                "probe": {"status": probe, "devices": []},
            },
        },
        "memory": {"total_bytes": 1_208_000_000_000},
    }


def test_physical_inventory_can_exceed_masked_torch_view_and_still_pass():
    report = assess_hardware(_observation(), required_gpu_count=1)
    assert report["status"] == "pass"
    assert report["reported"]["physical_gpu_count_nvidia_smi"] == 4
    assert report["reported"]["torch_visible_gpu_count"] == 1
    assert "nvidia_smi_physical_inventory_differs_from_torch_visible_devices" in report["warnings"]
    assert report["ready_for_inference"] is False


def test_mask_and_torch_count_disagreement_blocks_requested_full_visibility():
    report = assess_hardware(_observation(mask="0,1,2,3"), required_gpu_count=4)
    assert report["status"] == "blocked"
    assert "cuda_visible_devices_mask_count_mismatch" in report["blockers"]
    assert "torch_visible_gpu_count_mismatch" in report["blockers"]


def test_cpu_only_doctor_passes_without_torch_cuda_or_smi():
    observation = {
        "nvidia_smi": {"status": "unavailable", "devices": [], "topology": {"status": "not_attempted"}},
        "environment": {"cuda_visible_devices": None},
        "torch": {"import_status": "unavailable", "cuda": {"available": None}},
        "memory": {"total_bytes": 1024},
    }
    report = assess_hardware(observation, required_gpu_count=0, require_cuda=False)
    assert report["status"] == "pass"
    assert report["ready_for_inference"] is False


@pytest.mark.parametrize(("runtime", "driver"), [("13.0", "579.99"), ("12.8", "524.99")])
def test_driver_below_cuda_major_floor_blocks_even_if_probe_passed(runtime, driver):
    report = assess_hardware(_observation(runtime=runtime, driver=driver), required_gpu_count=1)
    assert "nvidia_driver_below_cuda_major_floor" in report["blockers"]


def test_cuda_allocation_sync_failure_blocks_even_with_enumerated_devices():
    report = assess_hardware(_observation(probe="fail"), required_gpu_count=1)
    assert report["status"] == "blocked"
    assert "cuda_allocation_sync_probe_failed" in report["blockers"]


@pytest.mark.parametrize(("runtime", "driver", "blocker"), [
    (None, "580.95.05", "cuda_runtime_driver_compatibility_unverified"),
    ("14.0", "700.1", "cuda_runtime_driver_compatibility_unverified"),
    ("13.0", None, "nvidia_driver_version_unavailable"),
])
def test_unknown_or_unsupported_cuda_compatibility_blocks_gpu_mode(runtime, driver, blocker):
    report = assess_hardware(_observation(runtime=runtime, driver=driver), required_gpu_count=1)
    assert report["status"] == "blocked"
    assert blocker in report["blockers"]


def test_unknown_cuda_compatibility_does_not_block_cpu_only_mode():
    report = assess_hardware(_observation(runtime=None, driver=None), required_gpu_count=0,
                             require_cuda=False)
    assert report["status"] == "pass"


def test_container_os_release_and_mounted_host_os_release_are_distinct(monkeypatch):
    from pathlib import Path

    seen = []

    def fake_release(path=Path("/etc/os-release")):
        normalized = str(path).replace("\\", "/")
        seen.append(normalized)
        label = "host-rhel" if normalized == "/run/mercury/host-os-release" else "container-ubuntu"
        return {"status": "available", "fields": {"ID": label}}

    monkeypatch.setattr("high_fidelity_schema_study.four_category.hardware._read_os_release", fake_release)
    monkeypatch.setattr("high_fidelity_schema_study.four_category.hardware._torch_inventory", lambda: {"import_status": "unavailable", "cuda": {}})
    monkeypatch.setattr("high_fidelity_schema_study.four_category.hardware._package_versions", lambda: {})
    observation = collect_hardware(runner=lambda command, timeout: {
        "returncode": 1, "stdout": "", "stderr": "not found"})
    assert observation["os_release"]["fields"]["ID"] == "container-ubuntu"
    assert observation["host_os_release"]["fields"]["ID"] == "host-rhel"
    assert "/etc/os-release" in seen
    assert "/run/mercury/host-os-release" in seen


def test_meminfo_kb_conversion_is_binary_and_retains_other_values():
    parsed = _parse_meminfo("MemTotal:       1234 kB\nMemFree:        9 kB\nHugePages_Total: 2\n")
    assert parsed["status"] == "available"
    assert parsed["total_bytes"] == 1234 * 1024
    assert parsed["meminfo_bytes"]["MemFree"] == 9 * 1024
    assert parsed["meminfo_bytes"]["HugePages_Total"] == 2


def test_nvidia_csv_preserves_gb_unit_as_decimal_and_parses_compute_capability():
    parsed = _parse_nvidia_csv('0, GPU-a, NVIDIA H200 NVL, 141 GB, 580.95.05, 9.0\n')
    assert len(parsed) == 1
    assert parsed[0]["memory_total_bytes"] == 141_000_000_000
    assert parsed[0]["compute_capability"] == "9.0"


def test_collect_hardware_uses_argument_arrays_and_records_missing_tools(monkeypatch):
    calls = []

    def runner(command, timeout):
        calls.append((command, timeout))
        if command[0] == "nvidia-smi" and command[1].startswith("--query-gpu="):
            return {"returncode": 0, "stdout": "0, GPU-a, NVIDIA H200 NVL, 141 GB, 580.95.05, 9.0\n", "stderr": ""}
        if command[:3] == ["nvidia-smi", "topo", "-m"]:
            return {"returncode": 0, "stdout": "GPU0 GPU1\n", "stderr": ""}
        if command == ["lscpu"]:
            return {"returncode": 0, "stdout": "CPU(s): 128\nModel name: CPU fixture\n", "stderr": ""}
        if command == ["apptainer", "--version"]:
            raise FileNotFoundError("apptainer not installed")
        raise AssertionError(f"unexpected command {command!r}")

    monkeypatch.setattr("high_fidelity_schema_study.four_category.hardware._torch_inventory", lambda: {"import_status": "unavailable", "cuda": {}})
    monkeypatch.setattr("high_fidelity_schema_study.four_category.hardware._package_versions", lambda: {})
    observation = collect_hardware(runner=runner)
    assert observation["nvidia_smi"]["status"] == "available"
    assert observation["nvidia_smi"]["devices"][0]["name"] == "NVIDIA H200 NVL"
    assert observation["nvidia_smi"]["topology"]["raw"] == "GPU0 GPU1\n"
    assert observation["apptainer"]["status"] == "unavailable"
    assert all(isinstance(command, list) and timeout > 0 for command, timeout in calls)
    assert all(not isinstance(command, str) for command, _ in calls)
