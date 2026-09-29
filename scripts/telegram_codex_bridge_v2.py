"""Event-driven ephemeral Codex execution; never resume a desktop-owned thread.

V1 remains frozen with the Mercury experiment. This local receiver reuses its
signed receipt format, but owns an independent durable dispatch ledger.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import telegram_codex_bridge as protocol

VERSION = "telegram-codex-bridge/v2"
TERMINAL = {"completed", "legacy_completed", "trigger_failed", "uncertain"}


def command(config, answer):
    # No thread ID, resume, fork, permission override, or prompt from Telegram.
    return [config["codex_exe"], "exec", "--ephemeral", "--json",
            "--output-last-message", str(answer), "-"]


def prompt(config, event):
    receipt = Path(config["state_dir"]) / "inbox" / (event["event_id"] + ".json")
    return (
        "A user-authorized Mercury Telegram completion/failure receipt triggered this independent Codex worker. "
        "Do not resume, fork or write to any desktop thread. Read the operational instructions at "
        + config["instructions_path"] + ". The verified event kind is " + event["kind"]
        + "; signed receipt file: " + str(receipt) + ". "
        "The user authorized collection, verification and analysis of the completed V11 qualification/same-page comparison. "
        "Use the existing operate.py status and collect commands and v11_same_page_analysis.py. "
        "Verify the event report hash against the collected report. Inspect invalid/incomplete results and the production resume receipt. "
        "Write new analysis/report artifacts under data/experiments/v11_gpu_2026_09_28_v1 only; preserve frozen and unrelated files. "
        "Do not modify source, launch more inference, change contracts, restart services, or create an automation. "
        "Do not send Telegram yourself; the bridge delivers your final answer. Never print credentials, signing keys, or secret config. "
        "Conclude in concise Chinese with actual execution/admission counts, qualified versus unqualified semantic claims, "
        "failure causes, and absolute paths to the new report. The previous event-trigger attempt failed before any model call "
        "because the desktop thread had an active writer; this ephemeral execution is the recovery."
    )


def legacy_disposition(receipt, log_text=""):
    if receipt is None:
        return "pending"
    if receipt.get("status") == "finished" and receipt.get("exit_code") == 0:
        return "legacy_completed"
    # This one observed failure is known to precede session creation. Other
    # failures may have performed work, so do not blindly retry them.
    if (receipt.get("status") == "trigger_failed" and receipt.get("exit_code") != 0
            and "already has an active writer" in log_text
            and '"type":"thread.started"' not in log_text.replace(" ", "")):
        return "pending"
    return "uncertain"


def ledger_path(config, event):
    return Path(config["state_dir"]) / "dispatch" / (event["event_id"] + ".json")


def migrate(config, event):
    target = ledger_path(config, event)
    if target.exists():
        return protocol.read(target)
    legacy_path = Path(config["legacy_state_dir"]) / "dispatch" / target.name
    old = protocol.read(legacy_path) if legacy_path.exists() else None
    text = Path(old["log"]).read_text(encoding="utf-8", errors="replace") if old and old.get("log") and Path(old["log"]).is_file() else ""
    row = {"version": VERSION, "event_id": event["event_id"], "kind": event["kind"],
           "status": legacy_disposition(old, text), "attempts": 0, "time": time.time(),
           "legacy_receipt_sha256": protocol.sha(legacy_path) if old else None,
           "legacy_log_sha256": protocol.sha(old["log"]) if old and old.get("log") and Path(old["log"]).is_file() else None}
    protocol.atomic(target, row)
    return row


def launch_receipt(config, event, row, *, popen=subprocess.Popen, fixed_test_prompt=None):
    """Claim before launch, preserving an immutable attempt record on success.

    Tests inject Popen but production always uses the installed Codex CLI.
    """
    state = Path(config["state_dir"])
    stamp = str(time.time_ns())
    answer, log_path = state / ("answer-" + stamp + ".md"), state / ("codex-" + stamp + ".jsonl")
    attempt = {**row, "status": "claimed", "attempts": row.get("attempts", 0) + 1, "time": time.time(),
               "transport": "codex_exec_ephemeral", "answer": str(answer), "log": str(log_path), "attempt_id": stamp}
    protocol.atomic(ledger_path(config, event), attempt)
    env = os.environ.copy()
    env.pop("CODEX_THREAD_ID", None)
    log = log_path.open("wb")
    try:
        child = popen(command(config, answer), cwd=config["repo"], env=env, stdin=subprocess.PIPE,
                      stdout=log, stderr=subprocess.STDOUT,
                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except OSError as exc:
        log.close()
        attempt.update(status="pending" if attempt["attempts"] < 3 else "trigger_failed",
                       error_code="process_not_started", error_type=type(exc).__name__, next_attempt_at=time.time()+60)
        protocol.atomic(ledger_path(config, event), attempt)
        return None, attempt, None
    attempt.update(status="running", pid=child.pid)
    protocol.atomic(ledger_path(config, event), attempt)
    try:
        child.stdin.write((fixed_test_prompt or prompt(config, event)).encode("utf-8"))
        child.stdin.close()
    except (OSError, ValueError) as exc:
        # A process exists: execution may have begun. Do not automatically retry.
        attempt.update(input_error_type=type(exc).__name__)
        protocol.atomic(ledger_path(config, event), attempt)
    return child, attempt, log


def finish(config, event, attempt, returncode):
    answer = Path(attempt["answer"])
    complete = returncode == 0 and answer.is_file() and bool(answer.read_text(encoding="utf-8").strip())
    row = {**attempt, "status": "completed" if complete else "trigger_failed", "exit_code": returncode,
           "finished_at": time.time(), "answer_sha256": protocol.sha(answer) if answer.exists() else None,
           "log_sha256": protocol.sha(attempt["log"])}
    if not complete:
        row["error_code"] = "codex_execution_failed" if returncode else "answer_missing"
    protocol.atomic(ledger_path(config, event), row)
    protocol.atomic(Path(config["state_dir"]) / "attempts" / (attempt["attempt_id"] + ".json"), row)
    return row


def notify(config, row, *, api=protocol.telegram_call):
    """Send failures as well as successful answers; delivery failures are retried."""
    secret = protocol.read(config["telegram_secret_path"])
    if secret.get("enabled") is not True or secret.get("chat_id") != config["chat_id"]:
        raise ValueError("telegram_destination_not_bound")
    bot = api(secret, "getMe", {})
    if bot.get("id") != config["bot_id"] or bot.get("username") != config["bot_username"]:
        raise ValueError("telegram_bot_identity")
    if row["status"] == "completed":
        answer = Path(row["answer"]).read_text(encoding="utf-8").strip()
        text = "Codex 已完成 Telegram 事件的自动检查。\n\n" + answer[:3400]
    else:
        text = ("Codex 自动触发需要检查：" + row["status"] + " / " + row.get("error_code", "execution_state_uncertain")
                + "。事件和日志已保留，不会静默丢弃或重复启动。")
    for value in (secret.get("bot_token"), config.get("hmac_key")):
        if value:
            text = text.replace(value, "[REDACTED]")
    result = api(secret, "sendMessage", {"chat_id": config["chat_id"], "text": text})
    if result.get("chat", {}).get("id") != config["chat_id"] or result.get("from", {}).get("id") != config["bot_id"]:
        raise ValueError("telegram_receipt_mismatch")
    return {"status": "sent", "message_id": result["message_id"], "time": time.time()}


def status_snapshot(config, *, fetch_error=None):
    rows = [protocol.read(p) for p in (Path(config["state_dir"]) / "dispatch").glob("*.json")]
    attention = any(r["status"] in {"trigger_failed", "uncertain"} or r.get("notification_error_type") and not r.get("telegram_result") for r in rows)
    return {"version": VERSION, "pid": os.getpid(), "time": time.time(), "transport": "codex_exec_ephemeral",
            "state": "needs_attention" if attention else "transport_retrying" if fetch_error else "listening",
            "pending_events": sum(r["status"] == "pending" for r in rows),
            "unresolved_events": sum(r["status"] in {"trigger_failed", "uncertain", "claimed", "running"} for r in rows),
            "completed_events": sum(r["status"] in {"completed", "legacy_completed"} for r in rows),
            "pending_notifications": sum(r["status"] in {"completed", "trigger_failed", "uncertain"} and not r.get("telegram_result") for r in rows),
            "fetch_error_type": fetch_error}


def serve(config_path, *, recover_once=False):
    config = protocol.read(config_path)
    sys.path[:0] = config["python_paths"] + [str(Path(config["repo"]).parent)]
    from oaciss_access import mercury
    from high_fidelity_schema_study.four_category.scheduler_io import Lock
    state = Path(config["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    activity = protocol.ThreadActivity(config["rollout_path"])
    with Lock(state / "receiver.lock"):
        # An interrupted receiver cannot assume an old process did no work.
        for path in (state / "dispatch").glob("*.json"):
            row = protocol.read(path)
            if row["status"] in {"running", "claimed"}:
                protocol.atomic(path, {**row, "status": "uncertain", "error_code": "receiver_restarted_during_dispatch"})
        while not (state / "STOP").exists():
            fetch_error = None
            try:
                gateway, compute = mercury()
                try:
                    with compute.open_sftp() as sftp:
                        protocol.receive(config, sftp, state)
                finally:
                    compute.close()
                    gateway.close()
            except Exception as exc:
                fetch_error = type(exc).__name__
            # Already received events remain dispatchable during an SSH outage.
            for path in sorted((state / "inbox").glob("*.json")):
                try:
                    event = protocol.verify(protocol.read(path), config)
                except Exception as exc:
                    protocol.atomic(state / "rejected" / path.name, {"error_type": type(exc).__name__, "time": time.time()})
                    continue
                if event["kind"] not in protocol.WAKE_EVENTS:
                    continue
                row = migrate(config, event)
                eligible = (row["status"] == "pending" and row.get("next_attempt_at", 0) <= time.time()
                            and (state / "ARMED").exists() and (recover_once or activity.idle()))
                if eligible:
                    child, row, log = launch_receipt(config, event, row)
                    if child is not None:
                        try:
                            while child.poll() is None:
                                protocol.atomic(state / "status.json", {**status_snapshot(config, fetch_error=fetch_error),
                                                                         "state": "codex_running", "codex_pid": child.pid})
                                time.sleep(5)
                        finally:
                            log.close()
                        row = finish(config, event, row, child.returncode)
                if row["status"] in {"completed", "trigger_failed", "uncertain"} and not row.get("telegram_result"):
                    try:
                        row["telegram_result"] = notify(config, row)
                    except Exception as exc:
                        row["notification_error_type"] = type(exc).__name__
                    protocol.atomic(ledger_path(config, event), row)
            protocol.atomic(state / "status.json", status_snapshot(config, fetch_error=fetch_error))
            if recover_once:
                return
            time.sleep(config.get("poll_seconds", 30))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--recover-once", action="store_true", help="Operator recovery: process pending receipts once, even while desktop has an active turn")
    args = parser.parse_args()
    serve(args.config, recover_once=args.recover_once)
