"""Host-side dispatch and independent Telegram receipt publisher for V11."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from telegram_codex_bridge import atomic, publish, read, sha
from v11_gpu_job import verify_stage


def command(job, gpu=False):
    runtime = read(job / "runtime.json")
    base = Path(runtime["base"])
    cmd = [str(base / "apptainer-1.5.4/bin/apptainer"), "exec"]
    if gpu:
        cmd.append("--nv")
    cmd += ["--cleanenv", "--bind", f"{base}:{base}:ro"]
    for path in [job / "outputs", job / "audit", job / "cache", Path(runtime["leases"])]:
        cmd += ["--bind", f"{path}:{path}:rw"]
    cmd += ["--env", "PYTHONPATH=" + str(job / "source"), "--env", "CUDA_VISIBLE_DEVICES=" + (runtime["gpu_id"] if gpu else ""),
            "--env", "OMP_NUM_THREADS=12", "--env", "HF_HUB_OFFLINE=1",
            "--env", "TORCHINDUCTOR_CACHE_DIR=" + str(job / "cache/inductor"),
            "--env", "TRITON_CACHE_DIR=" + str(job / "cache/triton"), runtime["image"],
            "/opt/phase1-venv/bin/python", "-u", str(job / "v11_gpu_job.py"),
            "execute" if gpu else "preflight", str(job)]
    return cmd


def start_preflight(job):
    verify_stage(job)
    runtime = read(job / "runtime.json")
    assert sha(runtime["image"]) == runtime["image_sha256"]
    with (job / "audit/preflight.log").open("x") as log:
        code = subprocess.call(command(job), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    atomic(job / "audit/preflight-exit.json", {"exit_code": code, "time": time.time()})
    return code


def dispatch(job):
    import mercury_probe_maintenance_v3
    m = mercury_probe_maintenance_v3.maintenance
    verify_stage(job)
    assert read(job / "audit/cpu-preflight.json")["status"] == "pass"
    runtime = read(job / "runtime.json")
    assert sha(runtime["image"]) == runtime["image_sha256"]
    production = Path(runtime["production_job"]).resolve()
    found = []
    for path in Path("/proc").iterdir():
        row = m.proc(int(path.name)) if path.name.isdigit() else None
        if not row or row["uid"] != os.getuid():
            continue
        args = row["argv"]
        if ("high_fidelity_schema_study.four_category.scheduler" in args and "--root" in args
                and Path(args[args.index("--root") + 1]).resolve() == production / "run"):
            found.append(row)
    assert len(found) == 1, "expected_one_active_production_coordinator"
    coord = found[0]
    sup = m.proc(coord["ppid"])
    scripts = [Path(a) for a in sup["argv"] if a.endswith(".py") and Path(a).is_file()]
    assert len(scripts) == 1 and "monitor" in sup["argv"]
    plan = {"job_dir": str(production), "condition": runtime["production_condition"], "maintenance_id": "v11qualification20260928v1",
            "supervisor": {"pid": sup["pid"], "start_ticks": sup["start_ticks"], "argv_sha256": m.digest_argv(sup["argv"]),
                           "script_path": str(scripts[0]), "script_file_sha256": sha(scripts[0])},
            "coordinator": {"pid": coord["pid"], "start_ticks": coord["start_ticks"]},
            "probes": [{"argv": command(job, True), "cwd": str(job), "timeout_s": 10800,
                        "env": {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(Path.home()), "LANG": "C.UTF-8"}}]}
    _, state = m.validate_plan(plan)
    m.write_once(job / "audit/maintenance-plan.json", plan)
    # Publisher starts first; a ready marker is required before disruption.
    with (job / "audit/notifier.log").open("x") as log:
        watcher = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watch", str(job)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not (job / "audit/notifier-ready.json").exists():
        assert watcher.poll() is None
        time.sleep(.1)
    assert (job / "audit/notifier-ready.json").exists()
    with (job / "audit/maintenance.log").open("x") as log:
        process = subprocess.Popen([sys.executable, str(job / "mercury_probe_maintenance_v3.py"), "maintain", str(job / "audit/maintenance-plan.json")],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    m.write_once(job / "audit/dispatch.json", {"time": time.time(), "pid": process.pid,
                   "start_ticks": m.proc(process.pid)["start_ticks"], "watcher_pid": watcher.pid,
                   "maintenance_state": str(state), "maximum_seconds": 10800,
                   "stage_manifest_file_bytes_sha256": sha(job / "stage-manifest.json")})
    print(json.dumps({"dispatched_pid": process.pid, "watcher_pid": watcher.pid, "maximum_seconds": 10800}), flush=True)


def watch(job):
    config = read(Path.home() / ".config/schema-study/v11-event-bridge.json")
    secret = read(Path.home() / ".config/schema-study/mercury-telegram.json")
    assert config["job_id"] == job.name
    outbox = job / "events"
    atomic(job / "audit/notifier-ready.json", {"pid": os.getpid(), "time": time.time()})
    started, last_change, last_signature = time.time(), time.time(), None
    pending, sent = {}, set()
    while time.time() - started < 7 * 86400:
        dispatch_path = job / "audit/dispatch.json"
        if not dispatch_path.exists():
            if time.time() - started > 120:
                return  # Disruption never dispatched.
            time.sleep(2)
            continue
        dispatch_record = read(dispatch_path)
        state = Path(dispatch_record["maintenance_state"])
        pending.setdefault("queue_started", dispatch_path)
        qualification = job / "audit/qualification.json"
        if qualification.exists():
            pending.setdefault("qualification_passed" if read(qualification)["status"] == "pass" else "qualification_failed", qualification)
        job_state = job / "audit/job-state.json"
        current = read(job_state) if job_state.exists() else {"phase": "maintenance_starting"}
        terminal = current["phase"] in {"qualification_failed", "comparison_failed", "comparison_completed"}
        if terminal:
            # A completion wake also tells the analyst whether restoration is done.
            report = {"job_state": current, "qualification": read(qualification) if qualification.exists() else None,
                      "execution": read(job / "audit/execution-summary.json") if (job / "audit/execution-summary.json").exists() else None,
                      "production_resume": read(state / "resume-receipt.json") if (state / "resume-receipt.json").exists() else None}
            atomic(job / "audit/terminal-report.json", report)
            if report["production_resume"] is not None or time.time() - current["time"] > 180:
                pending.setdefault(current["phase"], job / "audit/terminal-report.json")
        if not terminal:
            progresses = []
            for name in ("qualification", "comparison"):
                path = job / "outputs" / name / "progress.json"
                if path.exists():
                    p = read(path)
                    progresses.append([name, p.get("phase"), p.get("request_ordinal"), p.get("prefilled_tokens"), p.get("output_tokens")])
            signature = json.dumps([current["phase"], progresses])
            if signature != last_signature:
                last_signature, last_change = signature, time.time()
            if time.time() - last_change > 1800 or time.time() - dispatch_record["time"] > 11160:
                alarm = job / "audit/stall-report.json"
                if not alarm.exists():
                    atomic(alarm, {"phase": current["phase"], "last_progress_time": last_change, "time": time.time(), "threshold_seconds": 1800})
                pending.setdefault("queue_stalled", alarm)
            if (state / "maintenance-error.json").exists() or (state / "monitor-failure.json").exists():
                path = state / ("maintenance-error.json" if (state / "maintenance-error.json").exists() else "monitor-failure.json")
                pending.setdefault("comparison_failed" if qualification.exists() and read(qualification)["status"] == "pass" else "qualification_failed", path)
                terminal = True
        for kind, path in pending.items():
            if kind in sent:
                continue
            try:
                publish(config, secret, kind, path, outbox)
                sent.add(kind)
            except Exception as exc:
                atomic(job / "audit/notifier-error.json", {"event": kind, "error_type": type(exc).__name__, "time": time.time()})
        atomic(job / "audit/notifier-status.json", {"time": time.time(), "pid": os.getpid(), "sent": sorted(sent), "pending": sorted(set(pending)-sent)})
        if terminal and any(k in sent for k in {"qualification_failed", "comparison_failed", "comparison_completed"}) and set(pending) <= sent:
            return
        time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "dispatch", "watch"])
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    raise SystemExit({"preflight": start_preflight, "dispatch": dispatch, "watch": watch}[args.mode](args.job))
