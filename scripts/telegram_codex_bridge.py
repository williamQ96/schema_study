"""Allowlisted Telegram delivery receipts -> durable local Codex wake queue.

The transport is an SSH mirror of acknowledged outbound bot messages, not
getUpdates (which does not return this bot's own outgoing private messages).
No message text, callback command, or model output is executed as a prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

EVENTS = {"bridge_test", "queue_started", "qualification_passed", "qualification_failed",
          "comparison_completed", "comparison_failed", "queue_stalled", "resource_alarm"}
WAKE_EVENTS = EVENTS - {"bridge_test", "queue_started", "qualification_passed"}
LABELS = {
    "bridge_test": "Telegram → local receiver connection test",
    "queue_started": "V11 GPU qualification queued; same-page comparison follows only after qualification",
    "qualification_passed": "V11 GPU qualification passed; same-page comparison is starting",
    "qualification_failed": "V11 GPU qualification failed; comparison has NOT started",
    "comparison_completed": "V11 same-page comparison finished; inspect contracts and fidelity evidence",
    "comparison_failed": "V11 comparison stopped with an execution failure; partial results preserved",
    "queue_stalled": "V11 queue exceeded its progress or runtime threshold; inspection required",
    "resource_alarm": "V11 resource qualification requires inspection",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def event_id(job, kind):
    if kind not in EVENTS:
        raise ValueError("event_not_allowlisted")
    return hashlib.sha256(canonical([job, kind])).hexdigest()


def sign(payload, key):
    return {"payload": payload, "hmac_sha256": hmac.new(bytes.fromhex(key), canonical(payload), hashlib.sha256).hexdigest()}


def verify(envelope, config, *, now=None):
    if set(envelope) != {"payload", "hmac_sha256"}:
        raise ValueError("envelope_fields")
    payload = envelope["payload"]
    expected = sign(payload, config["hmac_key"])["hmac_sha256"]
    if not isinstance(envelope["hmac_sha256"], str) or not hmac.compare_digest(expected, envelope["hmac_sha256"]):
        raise ValueError("event_signature")
    if set(payload) != {"version", "job_id", "kind", "event_id", "created_at", "receipt", "report_sha256"}:
        raise ValueError("event_fields")
    if payload["version"] != 1 or payload["job_id"] != config["job_id"] or payload["kind"] not in EVENTS:
        raise ValueError("event_scope")
    if payload["event_id"] != event_id(payload["job_id"], payload["kind"]):
        raise ValueError("event_identity")
    now = time.time() if now is None else now
    if type(payload["created_at"]) not in (int, float) or not 0 <= now - payload["created_at"] <= 7 * 86400:
        raise ValueError("event_age")
    receipt = payload["receipt"]
    if (receipt.get("bot_username") != config["bot_username"] or receipt.get("bot_id") != config["bot_id"]
            or receipt.get("chat_id") != config["chat_id"] or type(receipt.get("message_id")) is not int
            or receipt["message_id"] <= 0):
        raise ValueError("telegram_identity")
    report = payload["report_sha256"]
    if not isinstance(report, str) or len(report) != 64 or any(c not in "0123456789abcdef" for c in report):
        raise ValueError("report_identity")
    return payload


def telegram_call(secret, method, body):
    request = urllib.request.Request("https://api.telegram.org/bot" + secret["bot_token"] + "/" + method,
                                     data=canonical(body), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
    except Exception as exc:
        raise RuntimeError("telegram_" + type(exc).__name__) from None
    if not result.get("ok"):
        raise RuntimeError("telegram_api_rejected")
    return result["result"]


def publish(config, secret, kind, report_path, outbox):
    """One signed receipt per (job,event); preserve before mirror delivery.

    A crash after Telegram accepts but before the receipt is saved can cause
    a duplicate Telegram notice; deterministic event IDs still deduplicate wakes.
    """
    eid = event_id(config["job_id"], kind)
    target = Path(outbox) / (eid + ".json")
    if target.exists():
        verify(read(target), config)
        return target
    if secret.get("enabled") is not True or secret.get("chat_id") != config["chat_id"]:
        raise ValueError("telegram_destination_not_bound")
    bot = telegram_call(secret, "getMe", {})
    if bot.get("username") != config["bot_username"] or bot.get("id") != config["bot_id"]:
        raise ValueError("telegram_bot_mismatch")
    text = LABELS[kind] + "\nJob: " + config["job_id"] + "\n"
    text += "Codex wake: " + ("yes" if kind in WAKE_EVENTS else "no (progress/test)")
    text += "\nStructural validity is not scientific accuracy."
    message = telegram_call(secret, "sendMessage", {"chat_id": config["chat_id"], "text": text})
    if message.get("from", {}).get("id") != config["bot_id"] or message.get("chat", {}).get("id") != config["chat_id"]:
        raise ValueError("telegram_receipt_mismatch")
    payload = {"version": 1, "job_id": config["job_id"], "kind": kind, "event_id": eid,
               "created_at": time.time(), "report_sha256": sha(report_path),
               "receipt": {"bot_username": config["bot_username"], "bot_id": config["bot_id"],
                           "chat_id": config["chat_id"], "message_id": message["message_id"]}}
    atomic(target, sign(payload, config["hmac_key"]))
    return target


class ThreadActivity:
    """Read only lifecycle metadata, incrementally; unknown state fails closed."""
    def __init__(self, path):
        self.path, self.offset, self.active = Path(path), 0, True

    def idle(self):
        if self.path.stat().st_size < self.offset:
            self.offset, self.active = 0, True
        with self.path.open("rb") as f:
            f.seek(self.offset)
            while line := f.readline():
                if not line.endswith(b"\n"):
                    break
                self.offset = f.tell()
                try:
                    row = json.loads(line)
                except ValueError:
                    self.active = True
                    continue
                payload = row.get("payload", {})
                if row.get("type") == "event_msg":
                    if payload.get("type") == "task_started":
                        self.active = True
                    elif payload.get("type") in {"task_complete", "task_aborted", "turn_aborted"}:
                        self.active = False
        return not self.active


def fixed_prompt(config, events):
    # Only verified enums/identities enter this fixed instruction, never text.
    return ("An authorized Mercury Telegram receipt triggered this follow-up. Read "
            + config["instructions_path"] + ". Events: "
            + ", ".join(sorted({e["kind"] for e in events}))
            + ". Verify the local signed receipts and remote report hashes before acting. "
            "Collect V11 qualification/comparison outputs, assess admission failures and same-page fidelity diagnostics. "
            "Preserve frozen inputs and the production condition. Do not automatically launch new experiments, "
            "change contracts, or claim semantic accuracy from structural validity. Report findings in Chinese "
            "and send the authorized concise completion/failure summary to the configured private Telegram chat. "
            "If work is still running normally, leave it running and end this turn.")


def receive(config, sftp, state_dir):
    """SSH host identity is checked by the existing local access helper."""
    state = Path(state_dir)
    for name in sftp.listdir(config["remote_outbox"]):
        if len(name) != 69 or not name.endswith(".json") or any(c not in "0123456789abcdef" for c in name[:-5]):
            continue
        destination = state / "inbox" / name
        if destination.exists():
            continue
        with sftp.open(config["remote_outbox"] + "/" + name, "rb") as f:
            raw = f.read(16385)
        if len(raw) > 16384:
            raise ValueError("oversized_event")
        envelope = json.loads(raw)
        payload = verify(envelope, config)
        if name != payload["event_id"] + ".json":
            raise ValueError("event_filename")
        atomic(destination, envelope)


def serve(config_path):
    config = read(config_path)
    sys.path[:0] = config["python_paths"]
    from oaciss_access import mercury
    sys.path.insert(0, str(Path(config["repo"]).parent))
    from high_fidelity_schema_study.four_category.scheduler_io import Lock
    state = Path(config["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    activity = ThreadActivity(config["rollout_path"])
    with Lock(state / "receiver.lock"):
        while not (state / "STOP").exists():
            status = {"pid": os.getpid(), "time": time.time(), "state": "listening"}
            try:
                gateway, compute = mercury()
                try:
                    with compute.open_sftp() as sftp:
                        receive(config, sftp, state)
                finally:
                    compute.close()
                    gateway.close()
                events = []
                for path in sorted((state / "inbox").glob("*.json")):
                    payload = verify(read(path), config)
                    if payload["kind"] in WAKE_EVENTS and not (state / "dispatch" / path.name).exists():
                        events.append(payload)
                if events and (state / "ARMED").exists() and activity.idle():
                    # Persist a claim BEFORE launching. A crash in this small gap
                    # requires inspection; it never blindly launches twice.
                    for event in events:
                        atomic(state / "dispatch" / (event["event_id"] + ".json"),
                               {"status": "claimed", "time": time.time(), "event_id": event["event_id"]})
                    stamp = str(time.time_ns())
                    log_path = state / ("codex-" + stamp + ".jsonl")
                    command = [config["codex_exe"], "exec", "resume", config["thread_id"], "-", "--json",
                               "--output-last-message", str(state / ("answer-" + stamp + ".md"))]
                    with log_path.open("wb") as log:
                        env = os.environ.copy()
                        env.pop("CODEX_THREAD_ID", None)
                        child = subprocess.Popen(command, cwd=config["repo"], env=env, stdin=subprocess.PIPE,
                                                 stdout=log, stderr=subprocess.STDOUT,
                                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        child.stdin.write(fixed_prompt(config, events).encode())
                        child.stdin.close()
                        for event in events:
                            atomic(state / "dispatch" / (event["event_id"] + ".json"),
                                   {"status": "running", "time": time.time(), "pid": child.pid,
                                    "log": str(log_path), "event_id": event["event_id"]})
                        while child.poll() is None:
                            atomic(state / "status.json", {**status, "time": time.time(), "state": "codex_running", "pid": os.getpid(), "codex_pid": child.pid})
                            time.sleep(10)
                        for event in events:
                            atomic(state / "dispatch" / (event["event_id"] + ".json"),
                                   {"status": "finished" if child.returncode == 0 else "trigger_failed",
                                    "exit_code": child.returncode, "time": time.time(), "log": str(log_path),
                                    "event_id": event["event_id"]})
                status["pending_wake_events"] = len(events)
            except Exception as exc:
                # Exception bodies can contain credential URLs or remote data.
                status.update(state="retrying", error_type=type(exc).__name__)
            atomic(state / "status.json", status)
            time.sleep(config.get("poll_seconds", 30))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    serve(parser.parse_args().config)
