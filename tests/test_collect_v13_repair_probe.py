from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_v13_repair_probe as collector
import telegram_codex_bridge as bridge


def _wire(value):
    return bridge.canonical(value)


def _fixture(tmp_path, monkeypatch):
    remote_root = "/storage/jobs/v13-repair-20260928-r2/candidate-02"
    remote = {}
    def add(name, value):
        data = value if isinstance(value, bytes) else _wire(value)
        remote[remote_root + "/" + name] = data
        return collector._hash(data)

    profiles = [{"profile_id": p, "backend": "transformers"} for p in ("qwen", "gemma", "muse")]
    config_remote = {"profiles": profiles, "roles": {"locals": [p["profile_id"] for p in profiles]},
                     "classification": {"profile_id": "qwen"}}
    config_sha = add("config.json", config_remote)
    manifest = {"files": {"four_category/workflow.py": "a" * 64}}
    manifest["source_tree_sha256"] = collector._hash(_wire(manifest["files"]))
    manifest_sha = add("science-manifest.json", manifest)
    papers = []
    for pid in collector.PAPERS:
        content = _wire({"paper_id": pid})
        path = f"mineru/{pid}/input.json"
        add("source_bundle/" + path, content)
        papers.append({"paper_id": pid, "source": {"source_sha256": "d" * 64, "artifacts": {"input": {
            "path": path, "sha256": collector._hash(content), "bytes": len(content)}}}})
    corpus_sha = add("corpus.json", {"papers": papers})
    report_pins = {}
    for profile in profiles:
        pid = profile["profile_id"]
        prefix = f"audit/v13-repair-probe/{pid}/"
        rows = []
        for number in range(1, 10 if pid == "qwen" else 6):
            paper_id = collector.PAPERS[(number - 1) % len(collector.PAPERS)]
            record = {"profile_sha256": collector._hash(_wire(profile)), "profile": profile,
                      "task": {"task_sha256": "b" * 64},
                      "job": {"paper_id": paper_id, "profile_id": pid, "source_sha256": "d" * 64},
                      "paper_input_canonical_sha256": collector._hash(_wire({"paper_id": paper_id}))}
            record["record_sha256"] = collector._hash(_wire(record))
            record_path = prefix + f"case-{number:02d}.record.json"
            record_file_sha = add(record_path, record)
            row = {"kind": "extraction", "paper_id": paper_id, "status": "success", "accepted": True,
                   "record_path": record_path, "record_file_sha256": record_file_sha,
                   "record_sha256": record["record_sha256"], "qualification_failures": [],
                   "reasoning_tokens": 3, "final_tokens": 5, "output_tokens": 8,
                   "object_candidates": 2, "field_candidates": 1}
            add(prefix + f"case-{number:02d}.summary.json", row)
            rows.append(row)
        report_pins[prefix + "report.json"] = add(prefix + "report.json", {"profile_id": pid, "profile_sha256": record["profile_sha256"],
            "config_file_sha256": config_sha, "science_manifest_file_sha256": manifest_sha,
            "source_tree_sha256": manifest["source_tree_sha256"], "corpus_file_sha256": corpus_sha,
            "task_sha256": {"extraction": "b" * 64}, "paper_sources": {p: "d" * 64 for p in collector.PAPERS},
            "status": "pass", "case_summaries": rows})
    terminal = {"status": "pass",
        "config_sha256": collector._hash(_wire(config_remote)),
        "gpu": [{"profile_id": p["profile_id"], "exit_code": 0} for p in profiles]}
    source_hash = add(collector.SOURCE_REPORT, terminal)
    summary_hash = add(collector.REPORT, {"kind": "v13-repair-terminal/v1",
        "source_report": {"path": collector.SOURCE_REPORT, "sha256": source_hash},
        "qualification": terminal, "probe_reports": report_pins})

    class Sftp:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def open(self, path, mode):
            if path not in remote: raise FileNotFoundError(path)
            return BytesIO(remote[path])
    class Client:
        def open_sftp(self): return Sftp()
        def close(self): pass
    class Gateway:
        def close(self): pass
    monkeypatch.setattr(collector, "connect", lambda: (Gateway(), Client()))

    private = {"hmac_key": "ab" * 32, "job_id": "test-job", "bot_username": "notice_bot",
               "bot_id": 123, "chat_id": 456, "state_dir": str(tmp_path / "state"),
               "analysis_root": str(tmp_path / "analysis"), "remote_job_root": remote_root,
               "remote_report": remote_root + "/" + collector.REPORT,
               "config_file_sha256": config_sha, "source_tree_sha256": manifest["source_tree_sha256"]}
    eid = bridge.event_id(private["job_id"], "comparison_completed")
    payload = {"version": 1, "job_id": private["job_id"], "kind": "comparison_completed",
               "event_id": eid, "created_at": time.time(), "report_sha256": summary_hash,
               "receipt": {"bot_username": private["bot_username"], "bot_id": private["bot_id"],
                           "chat_id": private["chat_id"], "message_id": 1}}
    event_path = Path(private["state_dir"]) / "inbox" / (eid + ".json")
    event_path.parent.mkdir(parents=True)
    event_path.write_bytes(_wire(bridge.sign(payload, private["hmac_key"])))
    return private, eid, remote


def test_collect_signed_pinned_probe_into_new_directory(tmp_path, monkeypatch):
    config, event_id, _ = _fixture(tmp_path, monkeypatch)
    first = collector.collect(config, event_id)
    second = collector.collect(config, event_id)
    assert first["collection"] != second["collection"]
    report = json.loads(Path(first["collection"]).read_text())
    assert report["file_count"] == 50
    assert report["production_writes"] == report["model_calls"] == 0
    assert report["semantic_accuracy"] == "not_established"
    assert report["model_counts"]["qwen"]["status_counts"] == {"success": 9}
    assert report["model_counts"]["qwen"]["field_candidates"] == 9
    assert (Path(first["collection"]).parent / "source_bundle/mineru/P01/input.json").is_file()


def test_rejects_path_traversal_and_changed_summary_pin(tmp_path, monkeypatch):
    config, event_id, remote = _fixture(tmp_path, monkeypatch)
    for path in ("../secret", "audit/../secret", "C:/secret", "audit\\secret", "/absolute"):
        with pytest.raises(ValueError, match="unsafe_remote_relative_path"):
            collector._relative(path)
    remote[config["remote_job_root"] + "/" + collector.REPORT] = b"{}"
    with pytest.raises(ValueError, match="remote_file_hash_mismatch"):
        collector.collect(config, event_id)
    assert not Path(config["analysis_root"]).exists()


def test_launcher_failure_is_explicit_without_probe_summary(tmp_path, monkeypatch):
    config, event_id, remote = _fixture(tmp_path, monkeypatch)
    failure = {"status": "failed", "error": "launcher_failed"}
    source_hash = collector._hash(_wire(failure))
    terminal = _wire({"kind": "v13-repair-terminal/v1",
                      "source_report": {"path": collector.FAILURE_REPORT, "sha256": source_hash},
                      "qualification": failure, "probe_reports": {}})
    remote[config["remote_job_root"] + "/" + collector.REPORT] = terminal
    del remote[config["remote_job_root"] + "/" + collector.SOURCE_REPORT]
    remote[config["remote_job_root"] + "/" + collector.FAILURE_REPORT] = _wire(failure)
    event_path = Path(config["state_dir"]) / "inbox" / (event_id + ".json")
    envelope = json.loads(event_path.read_text())
    envelope["payload"]["report_sha256"] = collector._hash(terminal)
    event_path.write_bytes(_wire(bridge.sign(envelope["payload"], config["hmac_key"])))
    with pytest.raises(ValueError, match="qualification_launcher_failed_no_probe_summary"):
        collector.collect(config, event_id)
    assert not Path(config["analysis_root"]).exists()


def test_signed_wrapper_rejects_changed_profile_report(tmp_path, monkeypatch):
    config, event_id, remote = _fixture(tmp_path, monkeypatch)
    path = config["remote_job_root"] + "/audit/v13-repair-probe/qwen/report.json"
    report = json.loads(remote[path])
    report["status"] = "fail"
    remote[path] = _wire(report)
    with pytest.raises(ValueError, match="remote_file_hash_mismatch"):
        collector.collect(config, event_id)
    assert not Path(config["analysis_root"]).exists()
