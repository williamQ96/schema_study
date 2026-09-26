"""Read-only host inventory and pre-inference hardware readiness checks.

Inventory values are reported evidence, not qualification. The only active
device operation is a one-element CUDA allocation followed by synchronization;
this module never loads a model or downloads weights.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "mercury-hardware-observation/v1"
REPORT_VERSION = "mercury-hardware-assessment/v1"
_TIMEOUT_SECONDS = 10
_GIB = 1024 ** 3


def _read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, Any]:
    try:
        fields: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            fields[key] = value.strip().strip('"').strip("'")
        return {"status": "available", "fields": fields}
    except OSError as exc:
        return {"status": "unavailable", "fields": {}, "error": f"{type(exc).__name__}: {exc}"}


def _parse_meminfo(text: str) -> dict[str, Any]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z_()]+):\s*(\d+)\s*(kB|KB|B)?\s*$", line)
        if not match:
            continue
        key, amount, unit = match.groups()
        multiplier = 1024 if unit and unit.lower() == "kb" else 1
        values[key] = int(amount) * multiplier
    total = values.get("MemTotal")
    if total is None:
        return {"status": "unavailable", "meminfo_bytes": values, "total_bytes": None}
    return {"status": "available", "meminfo_bytes": values, "total_bytes": total}


def _cpuinfo() -> dict[str, Any]:
    path = Path("/proc/cpuinfo")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"status": "unavailable", "logical_processors_reported": None,
                "model_names": [], "error": f"{type(exc).__name__}: {exc}"}
    processors = len(re.findall(r"^processor\s*:\s*\d+\s*$", text, flags=re.MULTILINE))
    models = sorted(set(re.findall(r"^(?:model name|Hardware)\s*:\s*(.+?)\s*$", text, flags=re.MULTILINE)))
    return {"status": "available", "logical_processors_reported": processors or None,
            "model_names": models}


def _default_runner(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _run(command: list[str], runner: Callable | None) -> dict[str, Any]:
    run = runner or _default_runner
    try:
        result = run(command, _TIMEOUT_SECONDS)
        if isinstance(result, dict):
            code, stdout, stderr = result.get("returncode"), result.get("stdout", ""), result.get("stderr", "")
        else:
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        stdout = stdout if isinstance(stdout, str) else str(stdout or "")
        stderr = stderr if isinstance(stderr, str) else str(stderr or "")
        if code == 0:
            return {"status": "available", "returncode": 0, "stdout": stdout, "stderr": stderr}
        return {"status": "error", "returncode": code, "stdout": stdout, "stderr": stderr,
                "error": "command exited unsuccessfully"}
    except FileNotFoundError as exc:
        return {"status": "unavailable", "stdout": "", "stderr": "",
                "error": f"{type(exc).__name__}: {exc}"}
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "stdout": "", "stderr": "",
                "error": f"{type(exc).__name__}: command exceeded {_TIMEOUT_SECONDS}s"}
    except Exception as exc:  # injected wrappers and platform tools may fail in other ways
        return {"status": "error", "stdout": "", "stderr": "",
                "error": f"{type(exc).__name__}: {exc}"}


def _parse_nvidia_csv(text: str) -> list[dict[str, Any]]:
    devices = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row or all(not cell.strip() for cell in row):
            continue
        cells = [cell.strip() for cell in row]
        if len(cells) < 6:
            continue
        idx_text, uuid, name, memory, driver, compute_cap = cells[:6]
        try:
            index = int(idx_text)
        except ValueError:
            index = None
        devices.append({"index": index, "uuid": uuid or None, "name": name or None,
                        "memory_total_reported": memory or None,
                        "memory_total_bytes": _parse_size_bytes(memory),
                        "driver_version": driver or None,
                        "compute_capability": compute_cap or None})
    return devices


def _parse_size_bytes(value: Any) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)?\s*", str(value), re.IGNORECASE)
    if not match:
        # NVIDIA CSV normally provides MiB when unit suffixes are omitted.
        if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", str(value)):
            return round(float(value) * 1024 ** 2)
        return None
    amount, unit = match.groups()
    unit = (unit or "MiB").lower()
    powers = {"b": 1, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3,
              "tib": 1024 ** 4, "kb": 1000, "mb": 1000 ** 2,
              "gb": 1000 ** 3, "tb": 1000 ** 4}
    return round(float(amount) * powers[unit])


def _nvidia_inventory(runner: Callable | None) -> dict[str, Any]:
    fields = "index,uuid,name,memory.total,driver_version,compute_cap"
    query = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"], runner)
    if query["status"] != "available":
        return {"status": query["status"], "devices": [], "error": query.get("error"),
                "stderr": query.get("stderr", ""), "topology": {"status": "not_attempted", "raw": None}}
    devices = _parse_nvidia_csv(query["stdout"])
    if query["stdout"].strip() and not devices:
        status, error = "error", "nvidia-smi returned an unrecognized CSV inventory"
    elif devices:
        status, error = "available", None
    else:
        status, error = "available", None
    topology = _run(["nvidia-smi", "topo", "-m"], runner)
    return {"status": status, "devices": devices, "error": error,
            "topology": {"status": topology["status"], "raw": topology.get("stdout") or None,
                         "error": topology.get("error"), "stderr": topology.get("stderr", "")}}


def _torch_inventory() -> dict[str, Any]:
    """Import torch lazily; perform only small allocation/synchronization probes."""
    try:
        import torch  # intentionally lazy; hardware doctor has no top-level torch import
    except Exception as exc:
        return {"import_status": "unavailable", "version": None, "cuda": {
            "available": None, "runtime_version": None, "visible_device_count": None,
            "devices": [], "probe": {"status": "not_run", "devices": []}},
            "cudnn_version": None, "error": f"{type(exc).__name__}: {exc}"}

    cuda = {"available": None, "runtime_version": getattr(getattr(torch, "version", None), "cuda", None),
            "visible_device_count": None, "devices": [], "probe": {"status": "not_run", "devices": []}}
    result = {"import_status": "available", "version": str(getattr(torch, "__version__", "unknown")),
              "cuda": cuda, "cudnn_version": None}
    try:
        backend = getattr(torch, "backends", None)
        cudnn = getattr(backend, "cudnn", None)
        if cudnn is not None and callable(getattr(cudnn, "version", None)):
            result["cudnn_version"] = cudnn.version()
    except Exception as exc:
        result["cudnn_error"] = f"{type(exc).__name__}: {exc}"
    try:
        cuda_module = torch.cuda
        available = bool(cuda_module.is_available())
        count = int(cuda_module.device_count())
        cuda["available"], cuda["visible_device_count"] = available, count
        devices = []
        probes = []
        for index in range(max(0, count)):
            device: dict[str, Any] = {"index": index}
            try:
                properties = cuda_module.get_device_properties(index)
                device.update({"name": str(properties.name),
                               "memory_total_bytes": int(properties.total_memory),
                               "compute_capability": f"{properties.major}.{properties.minor}"})
            except Exception as exc:
                device["property_error"] = f"{type(exc).__name__}: {exc}"
            try:
                with cuda_module.device(index):
                    device["bf16_supported"] = bool(cuda_module.is_bf16_supported())
            except Exception as exc:
                device["bf16_supported"] = None
                device["bf16_error"] = f"{type(exc).__name__}: {exc}"
            devices.append(device)
            if available:
                probe = {"index": index}
                try:
                    tensor = torch.empty((1,), device=f"cuda:{index}")
                    cuda_module.synchronize(index)
                    del tensor
                    probe.update({"status": "pass"})
                except Exception as exc:
                    probe.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}"})
                probes.append(probe)
        cuda["devices"] = devices
        if not available:
            cuda["probe"] = {"status": "unavailable", "devices": [], "error": "torch.cuda.is_available() is false"}
        elif not count:
            cuda["probe"] = {"status": "fail", "devices": [], "error": "CUDA available but no visible devices"}
        else:
            cuda["probe"] = {"status": "pass" if all(p["status"] == "pass" for p in probes) else "fail",
                             "devices": probes}
    except Exception as exc:
        cuda["probe"] = {"status": "error", "devices": [], "error": f"{type(exc).__name__}: {exc}"}
        cuda["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for package in ("transformers", "accelerate", "bitsandbytes"):
        try:
            versions[package] = {"status": "available", "version": importlib.metadata.version(package)}
        except importlib.metadata.PackageNotFoundError:
            versions[package] = {"status": "unavailable", "version": None}
        except Exception as exc:
            versions[package] = {"status": "error", "version": None,
                                 "error": f"{type(exc).__name__}: {exc}"}
    return versions


def _parse_lscpu(text: str) -> dict[str, Any]:
    parsed = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return {key: parsed.get(key) for key in
            ("Architecture", "CPU(s)", "On-line CPU(s) list", "Thread(s) per core",
             "Core(s) per socket", "Socket(s)", "Model name", "Vendor ID", "Flags")
            if key in parsed}


def collect_hardware(runner: Callable | None = None) -> dict[str, Any]:
    """Collect a read-only host inventory, accepting an injected command runner.

    ``runner`` is called as ``runner(command: list[str], timeout: int)`` and
    returns a CompletedProcess-like object or a mapping with returncode/stdout/
    stderr. Missing commands, timeouts, failed probes, or absent procfs files
    are recorded as data; this function is designed not to crash on them.
    """
    affinity: dict[str, Any]
    try:
        cpu_ids = sorted(os.sched_getaffinity(0))
        affinity = {"status": "available", "cpu_ids": cpu_ids, "count": len(cpu_ids)}
    except (AttributeError, OSError) as exc:
        affinity = {"status": "unavailable", "cpu_ids": None, "count": None,
                    "error": f"{type(exc).__name__}: {exc}"}
    try:
        meminfo_text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        memory = _parse_meminfo(meminfo_text)
    except OSError as exc:
        memory = {"status": "unavailable", "total_bytes": None, "meminfo_bytes": {},
                  "error": f"{type(exc).__name__}: {exc}"}
    lscpu = _run(["lscpu"], runner)
    apptainer = _run(["apptainer", "--version"], runner)
    return {"schema_version": SCHEMA_VERSION,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": {"system": platform.system(), "release": platform.release(),
                         "version": platform.version(), "machine": platform.machine(),
                         "hostname": platform.node(), "platform_string": platform.platform()},
            "os_release": _read_os_release(),
            # This bind-mounted file is explicitly host provenance. The ordinary
            # os-release above describes the active namespace/container.
            "host_os_release": _read_os_release(Path("/run/mercury/host-os-release")),
            "cpu": {**_cpuinfo(), "logical_count_os": os.cpu_count(),
                    "lscpu": {"status": lscpu["status"], "parsed": _parse_lscpu(lscpu.get("stdout", "")),
                              "raw": lscpu.get("stdout") or None, "error": lscpu.get("error"),
                              "stderr": lscpu.get("stderr", "")}},
            "affinity": affinity, "memory": memory,
            "environment": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
            "nvidia_smi": _nvidia_inventory(runner),
            "apptainer": {"status": apptainer["status"],
                          "version": apptainer.get("stdout", "").strip() or None,
                          "host_reported_version": os.environ.get("MERCURY_APPTAINER_VERSION"),
                          "host_version_provenance": "launcher_environment; distinct from in-container executable probe",
                          "error": apptainer.get("error"), "stderr": apptainer.get("stderr", "")},
            "torch": _torch_inventory(), "packages": _package_versions()}


def _visible_mask_count(mask: Any) -> int | None:
    if mask is None:
        return None
    value = str(mask).strip()
    if value.lower() in {"all", "none", "void"}:
        return None
    if value == "" or value == "-1":
        return 0
    return len([item for item in value.split(",") if item.strip()])


def _driver_major(value: Any) -> int | None:
    match = re.match(r"\s*(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def _runtime_major(value: Any) -> int | None:
    match = re.match(r"\s*(\d+)(?:\.|$)", str(value or ""))
    return int(match.group(1)) if match else None


def assess_hardware(observation: dict[str, Any], required_gpu_count: int,
                    require_cuda: bool = True, expected_gpu_substring: str | None = "H200",
                    expected_compute_capability: str | None = "9.0") -> dict[str, Any]:
    """Assess whether observations meet an execution policy, without inference.

    Inventory reports do not prove application qualification. GPU readiness is
    blocked unless torch exposes the requested device count and the small CUDA
    allocation/synchronization probe passes. NVIDIA-SMI physical inventory is
    reported separately and never substituted for torch-visible device count.
    """
    if isinstance(required_gpu_count, bool) or not isinstance(required_gpu_count, int) or required_gpu_count < 0:
        raise ValueError("required_gpu_count must be a nonnegative integer")
    if not isinstance(require_cuda, bool):
        raise ValueError("require_cuda must be boolean")
    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    def record(name: str, status: str, detail: Any = None) -> None:
        item = {"name": name, "status": status}
        if detail is not None:
            item["detail"] = detail
        checks.append(item)

    nvidia = observation.get("nvidia_smi") or {}
    physical = nvidia.get("devices") if isinstance(nvidia.get("devices"), list) else []
    torch_observation = observation.get("torch") or {}
    cuda = torch_observation.get("cuda") or {}
    visible_count = cuda.get("visible_device_count")
    physical_count = len(physical)
    env = observation.get("environment") or {}
    mask = env.get("cuda_visible_devices")
    mask_count = _visible_mask_count(mask)
    gpu_required = require_cuda or required_gpu_count > 0
    target_count = max(required_gpu_count, 1 if require_cuda else 0)

    record("physical_gpu_inventory", "reported" if nvidia.get("status") == "available" else "unknown",
           {"nvidia_smi_status": nvidia.get("status"), "physical_device_count": physical_count,
            "devices": physical, "topology_status": (nvidia.get("topology") or {}).get("status")})
    if nvidia.get("status") != "available":
        warnings.append("nvidia_smi_inventory_unavailable")
    topology_status = (nvidia.get("topology") or {}).get("status")
    if topology_status != "available":
        warnings.append("gpu_interconnect_topology_unknown")

    record("cuda_visibility", "reported", {"cuda_visible_devices": mask,
           "mask_device_count_if_enumerated": mask_count,
           "torch_visible_device_count": visible_count,
           "nvidia_smi_physical_inventory_count": physical_count})
    if physical_count != visible_count and nvidia.get("status") == "available" and visible_count is not None:
        warnings.append("nvidia_smi_physical_inventory_differs_from_torch_visible_devices")
    if mask_count is not None and visible_count is not None and mask_count != visible_count:
        blockers.append("cuda_visible_devices_mask_count_mismatch")
        record("cuda_visible_devices_mask", "blocked", {"mask_count": mask_count,
               "torch_visible_device_count": visible_count})
    else:
        record("cuda_visible_devices_mask", "pass" if mask_count is not None else "unknown",
               {"mask_count": mask_count, "torch_visible_device_count": visible_count})

    if not gpu_required:
        record("cuda_requirement", "pass", "CPU doctor mode; CUDA not required")
        if cuda.get("available") is not True:
            warnings.append("cuda_not_available_cpu_mode")
    else:
        if torch_observation.get("import_status") != "available":
            blockers.append("torch_unavailable_for_cuda_assessment")
            record("torch_cuda", "blocked", torch_observation.get("error"))
        elif cuda.get("available") is not True:
            blockers.append("cuda_unavailable")
            record("torch_cuda", "blocked", {"available": cuda.get("available"), "error": cuda.get("error")})
        elif visible_count != target_count:
            blockers.append("torch_visible_gpu_count_mismatch")
            record("torch_visible_gpu_count", "blocked", {"required": target_count, "observed": visible_count})
        else:
            record("torch_visible_gpu_count", "pass", {"required": target_count, "observed": visible_count})

        probe = cuda.get("probe") or {}
        if probe.get("status") != "pass":
            blockers.append("cuda_allocation_sync_probe_failed")
            record("cuda_allocation_sync_probe", "blocked", probe)
        else:
            record("cuda_allocation_sync_probe", "pass", probe)

        visible_devices = cuda.get("devices") if isinstance(cuda.get("devices"), list) else []
        reported_devices = visible_devices or physical
        names = [str(d.get("name") or "") for d in reported_devices]
        if expected_gpu_substring and reported_devices:
            wrong_names = [name for name in names if expected_gpu_substring.casefold() not in name.casefold()]
            if wrong_names:
                blockers.append("expected_gpu_name_mismatch")
                record("expected_gpu_name", "blocked", {"expected_substring": expected_gpu_substring,
                       "observed_names": names})
            else:
                record("expected_gpu_name", "pass", {"expected_substring": expected_gpu_substring,
                       "observed_names": names})
        elif expected_gpu_substring:
            warnings.append("expected_gpu_name_unverified")
            record("expected_gpu_name", "unknown", {"expected_substring": expected_gpu_substring})

        if expected_compute_capability and reported_devices:
            caps = [str(d.get("compute_capability") or "") for d in reported_devices]
            mismatches = [cap for cap in caps if cap and cap != expected_compute_capability]
            unknown = any(not cap for cap in caps)
            if mismatches:
                blockers.append("expected_compute_capability_mismatch")
                record("expected_compute_capability", "blocked", {"expected": expected_compute_capability,
                       "observed": caps})
            elif unknown:
                warnings.append("compute_capability_unverified")
                record("expected_compute_capability", "unknown", {"expected": expected_compute_capability,
                       "observed": caps})
            else:
                record("expected_compute_capability", "pass", {"expected": expected_compute_capability,
                       "observed": caps})

        runtime_major = _runtime_major(cuda.get("runtime_version"))
        driver_values = [d.get("driver_version") for d in physical if d.get("driver_version")]
        driver = driver_values[0] if driver_values else None
        driver_version = _driver_major(driver)
        floor = {12: 525, 13: 580}.get(runtime_major)
        if floor is None:
            blockers.append("cuda_runtime_driver_compatibility_unverified")
            record("cuda_driver_floor", "blocked", {"cuda_runtime": cuda.get("runtime_version"),
                   "driver": driver, "reason": "runtime unknown or major-version floor unsupported"})
        elif driver_version is None:
            blockers.append("nvidia_driver_version_unavailable")
            record("cuda_driver_floor", "blocked", {"required_major_floor": floor,
                   "cuda_runtime": cuda.get("runtime_version"), "driver": driver,
                   "reason": "driver compatibility cannot be assessed"})
        elif driver_version < floor:
            blockers.append("nvidia_driver_below_cuda_major_floor")
            record("cuda_driver_floor", "blocked", {"required_major_floor": floor,
                   "cuda_runtime": cuda.get("runtime_version"), "driver": driver})
        else:
            record("cuda_driver_floor", "pass", {"required_major_floor": floor,
                   "cuda_runtime": cuda.get("runtime_version"), "driver": driver,
                   "note": "major-version floor is necessary, not sufficient; probe and workload qualification remain required"})

        capacities = [d.get("memory_total_bytes") for d in visible_devices]
        if not capacities:
            capacities = [d.get("memory_total_bytes") for d in physical]
        capacities = [v for v in capacities if isinstance(v, int) and not isinstance(v, bool)]
        if capacities and min(capacities) < 130 * _GIB:
            warnings.append("gpu_memory_below_130_gib_target")
            record("gpu_memory_capacity", "warning", {"minimum_observed_bytes": min(capacities),
                   "warning_threshold_bytes": 130 * _GIB,
                   "policy": "advisory only; reported GPU GB/GBi units are not treated as an exact 141 GiB requirement"})
        elif capacities:
            record("gpu_memory_capacity", "pass", {"minimum_observed_bytes": min(capacities),
                   "warning_threshold_bytes": 130 * _GIB})
        else:
            warnings.append("gpu_memory_capacity_unknown")
            record("gpu_memory_capacity", "unknown")

    observation_memory = (observation.get("memory") or {}).get("total_bytes")
    if observation_memory is None:
        warnings.append("host_memory_unknown")
        record("host_memory", "unknown")
    else:
        record("host_memory", "reported", {"total_bytes": observation_memory})

    return {"schema_version": REPORT_VERSION,
            "status": "blocked" if blockers else "pass",
            "ready_for_inference": False,
            "inference_qualification": "not_performed",
            "model_loaded": False,
            "downloads_performed": False,
            "requested": {"required_gpu_count": required_gpu_count, "require_cuda": require_cuda,
                          "expected_gpu_substring": expected_gpu_substring,
                          "expected_compute_capability": expected_compute_capability},
            "reported": {"physical_gpu_count_nvidia_smi": physical_count,
                         "torch_visible_gpu_count": visible_count,
                         "cuda_visible_devices": mask,
                         "cuda_runtime_version": cuda.get("runtime_version"),
                         "nvidia_driver_version": driver_values[0] if gpu_required and driver_values else None,
                         "gpu_topology_status": topology_status},
            "checks": checks, "blockers": blockers, "warnings": warnings,
            "note": "This is a preflight inventory/probe, not model inference, interconnect qualification, or workload qualification."}


__all__ = ["collect_hardware", "assess_hardware"]
