import io
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import telegram_codex_bridge_v2 as bridge


@pytest.fixture
def config(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    return {"state_dir": str(state), "legacy_state_dir": str(tmp_path / "old"), "codex_exe": "codex.exe",
            "repo": str(tmp_path), "thread_id": "desktop-owned", "instructions_path": "fixed.md"}


def event():
    return {"event_id": "ab"*32, "kind": "comparison_completed"}


def test_transport_cannot_resume_or_fork_desktop_thread(config):
    argv = bridge.command(config, Path("answer.md"))
    assert argv[:4] == ["codex.exe", "exec", "--ephemeral", "--json"]
    assert not {"resume", "fork", "desktop-owned", "--dangerously-bypass-approvals-and-sandbox"} & set(argv)


@pytest.mark.parametrize("receipt,log,expected", [
    (None, "", "pending"),
    ({"status": "finished", "exit_code": 0}, "", "legacy_completed"),
    ({"status": "trigger_failed", "exit_code": 1}, "thread already has an active writer", "pending"),
    ({"status": "trigger_failed", "exit_code": 1}, "some other error", "uncertain"),
    ({"status": "running"}, "", "uncertain"),
    ({"status": "trigger_failed", "exit_code": 1}, 'already has an active writer {"type":"thread.started"}', "uncertain"),
])
def test_only_known_no_execution_failure_is_recovered(receipt, log, expected):
    assert bridge.legacy_disposition(receipt, log) == expected


def test_migration_preserves_legacy_evidence(config, tmp_path):
    old = Path(config["legacy_state_dir"]) / "dispatch" / (event()["event_id"] + ".json")
    log = tmp_path / "old.log"
    log.write_text("thread already has an active writer")
    bridge.protocol.atomic(old, {"status": "trigger_failed", "exit_code": 1, "log": str(log)})
    digest = bridge.protocol.sha(old)
    row = bridge.migrate(config, event())
    assert row["status"] == "pending" and row["legacy_receipt_sha256"] == digest
    assert bridge.protocol.sha(old) == digest
    row["status"] = "completed"
    bridge.protocol.atomic(bridge.ledger_path(config, event()), row)
    assert bridge.migrate(config, event())["status"] == "completed"


def test_prelaunch_failure_has_bounded_retries(config):
    def fail(*args, **kwargs):
        raise FileNotFoundError()
    row = {"status": "pending", "attempts": 0}
    for attempt in range(3):
        child, row, log = bridge.launch_receipt(config, event(), row, popen=fail)
        assert child is None and log is None
        assert row["status"] == ("pending" if attempt < 2 else "trigger_failed")


def test_claim_precedes_launch_and_finish_needs_answer(config):
    class Child:
        pid = 123
        stdin = io.BytesIO()
    def fake(argv, **kwargs):
        assert bridge.protocol.read(bridge.ledger_path(config, event()))["status"] == "claimed"
        assert "CODEX_THREAD_ID" not in kwargs["env"]
        return Child()
    child, row, log = bridge.launch_receipt(config, event(), {"attempts": 0}, popen=fake)
    log.close()
    assert row["status"] == "running" and child.pid == 123
    result = bridge.finish(config, event(), row, 0)
    assert result["status"] == "trigger_failed" and result["error_code"] == "answer_missing"
    assert bridge.status_snapshot(config)["state"] == "needs_attention"
    assert bridge.status_snapshot(config)["unresolved_events"] == 1


def test_failed_codex_run_remains_visible(config):
    row = {"status": "trigger_failed", "error_code": "codex_execution_failed"}
    bridge.protocol.atomic(bridge.ledger_path(config, event()), row)
    status = bridge.status_snapshot(config)
    assert status["pending_events"] == 0 and status["unresolved_events"] == 1
    assert status["pending_notifications"] == 1 and status["state"] == "needs_attention"


def test_failure_notice_and_secret_redaction(config, tmp_path):
    secret_path = tmp_path / "secret.json"
    bridge.protocol.atomic(secret_path, {"enabled": True, "chat_id": 456, "bot_token": "SECRET"})
    config.update(telegram_secret_path=str(secret_path), bot_id=123, bot_username="bot", chat_id=456, hmac_key="SIGNING")
    answer = tmp_path / "answer.md"
    answer.write_text("result SECRET SIGNING", encoding="utf-8")
    calls = []
    def api(secret, method, body):
        calls.append((method, body))
        return {"id": 123, "username": "bot"} if method == "getMe" else {"chat": {"id": 456}, "from": {"id": 123}, "message_id": 10}
    receipt = bridge.notify(config, {"status": "completed", "answer": str(answer)}, api=api)
    assert receipt["status"] == "sent"
    text = calls[-1][1]["text"]
    assert "SECRET" not in text and "SIGNING" not in text and "[REDACTED]" in text
    bridge.notify(config, {"status": "trigger_failed", "error_code": "example"}, api=api)
    assert "trigger_failed" in calls[-1][1]["text"]
