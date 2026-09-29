"""Collect a signed V13 revision-2 output probe without remote writes or model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

from mercury_scheduler_v2_access import connect
from telegram_codex_bridge import canonical, read, sha, verify

REPORT = "notice/terminal-report.json"
SOURCE_REPORT = "audit/qualification-summary.json"
FAILURE_REPORT = "audit/qualification-launcher-failure.json"
PROBE_ROOT = "audit/v13-repair-probe"
PAPERS = ("P01", "P03", "P06", "P08")
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")
PROFILE = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(path: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path or ":" in path or path.startswith("/"):
        raise ValueError("unsafe_remote_relative_path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(path).as_posix() != path:
        raise ValueError("unsafe_remote_relative_path")
    return path


def _pin(value: str) -> str:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError("invalid_sha256_pin")
    return value


def _json(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid_json:" + label) from exc
    if not isinstance(value, dict):
        raise ValueError("json_object_required:" + label)
    return value


def _summarize(reports: dict[str, dict]) -> dict:
    models = {}
    for profile_id, report in sorted(reports.items()):
        rows = report["case_summaries"]
        status = Counter(str(row.get("status")) for row in rows)
        failures = Counter(str(code) for row in rows for code in row.get("qualification_failures", []))
        reason = [row.get("reasoning_tokens") for row in rows]
        final = [row.get("final_tokens") for row in rows]
        output = [row.get("output_tokens") for row in rows]
        models[profile_id] = {
            "report_status": report["status"], "cases": len(rows),
            "status_counts": dict(sorted(status.items())),
            "accepted": sum(row.get("accepted") is True for row in rows),
            "qualification_failures": dict(sorted(failures.items())),
            "reasoning_tokens_observed": sum(x for x in reason if type(x) is int),
            "final_tokens_observed": sum(x for x in final if type(x) is int),
            "output_tokens_observed": sum(x for x in output if type(x) is int),
            "cases_with_reasoning_tokens": sum(type(x) is int for x in reason),
            "cases_with_final_tokens": sum(type(x) is int for x in final),
            "object_candidates": sum(row.get("object_candidates", 0) for row in rows
                                     if type(row.get("object_candidates")) is int),
            "field_candidates": sum(row.get("field_candidates", 0) for row in rows
                                    if type(row.get("field_candidates")) is int),
        }
    return models


def collect(config: dict, event_id: str) -> dict:
    _pin(event_id)
    state = Path(config["state_dir"])
    event_path = state / "inbox" / (event_id + ".json")
    event_file_sha = sha(event_path)
    event = verify(read(event_path), config)
    if event["event_id"] != event_id or event["kind"] not in {"comparison_completed", "comparison_failed"}:
        raise ValueError("unexpected_probe_completion_event")
    root = config["remote_job_root"].rstrip("/")
    if not root.startswith("/") or ".." in root.split("/") or not re.search(r"/candidate-[0-9]{2}\Z", root):
        raise ValueError("unexpected_remote_job_root")
    configured_report = config.get("remote_report", REPORT)
    if configured_report.startswith(root + "/"):
        configured_report = configured_report[len(root) + 1:]
    if configured_report != REPORT:
        raise ValueError("unexpected_remote_report")
    config_pin, source_pin = _pin(config["config_file_sha256"]), _pin(config["source_tree_sha256"])

    gateway, client = connect()
    downloaded = {}
    total = 0
    try:
        with client.open_sftp() as sftp:
            def get(relative: str, *, limit: int = MAX_FILE, expected: str | None = None) -> bytes:
                nonlocal total
                relative = _relative(relative)
                if relative in downloaded:
                    data = downloaded[relative]
                else:
                    with sftp.open(root + "/" + relative, "rb") as stream:
                        data = stream.read(limit + 1)
                    if len(data) > limit or total + len(data) > MAX_TOTAL:
                        raise ValueError("collection_size_bound_exceeded")
                    total += len(data)
                    downloaded[relative] = data
                if expected is not None and _hash(data) != _pin(expected):
                    raise ValueError("remote_file_hash_mismatch:" + relative)
                return data

            terminal_bytes = get(REPORT, limit=4 * 1024 * 1024, expected=event["report_sha256"])
            terminal = _json(terminal_bytes, REPORT)
            if terminal.get("kind") != "v13-repair-terminal/v1" or set(terminal) != {
                    "kind", "source_report", "qualification", "probe_reports"}:
                raise ValueError("terminal_wrapper_invalid")
            source = terminal["source_report"]
            if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
                raise ValueError("terminal_source_binding_invalid")
            source_path = source["path"]
            if source_path not in {SOURCE_REPORT, FAILURE_REPORT}:
                raise ValueError("terminal_source_path_invalid")
            source_bytes = get(source_path, limit=4 * 1024 * 1024, expected=source["sha256"])
            qualification = _json(source_bytes, source_path)
            if qualification != terminal["qualification"]:
                raise ValueError("terminal_source_content_mismatch")
            if source_path == FAILURE_REPORT or qualification.get("status") == "failed":
                raise ValueError("qualification_launcher_failed_no_probe_summary")
            if qualification.get("status") not in {"pass", "probe_failed"}:
                raise ValueError("qualification_summary_status_invalid")
            if source_path != SOURCE_REPORT:
                raise ValueError("terminal_source_path_invalid")
            config_data = get("config.json", limit=4 * 1024 * 1024, expected=config_pin)
            remote_config = _json(config_data, "config.json")
            manifest_data = get("science-manifest.json", limit=4 * 1024 * 1024)
            manifest = _json(manifest_data, "science-manifest.json")
            if _hash(canonical(manifest.get("files"))) != source_pin or manifest.get("source_tree_sha256") != source_pin:
                raise ValueError("source_tree_pin_mismatch")
            if qualification.get("config_sha256") != _hash(canonical(remote_config)):
                raise ValueError("qualification_config_mismatch")
            corpus_data = get("corpus.json", limit=8 * 1024 * 1024)
            corpus = _json(corpus_data, "corpus.json")
            profiles = {p["profile_id"]: p for p in remote_config["profiles"]}
            expected_profiles = set(remote_config["roles"]["locals"])
            expected_report_paths = {PROBE_ROOT + "/" + pid + "/report.json" for pid in expected_profiles}
            pins = terminal["probe_reports"]
            if (not isinstance(pins, dict) or not set(pins) <= expected_report_paths
                    or qualification["status"] == "pass" and set(pins) != expected_report_paths
                    or any(not HEX.fullmatch(value) for value in pins.values()
                           if isinstance(value, str))
                    or any(not isinstance(value, str) for value in pins.values())):
                raise ValueError("terminal_probe_report_pins_invalid")
            reports = {}
            record_inputs = []
            for profile_id in sorted(pid for pid in expected_profiles
                                     if PROBE_ROOT + "/" + pid + "/report.json" in pins):
                if not PROFILE.fullmatch(profile_id):
                    raise ValueError("unsafe_profile_id")
                prefix = PROBE_ROOT + "/" + profile_id + "/"
                data = get(prefix + "report.json", limit=4 * 1024 * 1024,
                           expected=pins[prefix + "report.json"])
                report = _json(data, prefix + "report.json")
                if (report.get("profile_id") != profile_id or report.get("config_file_sha256") != config_pin
                        or report.get("science_manifest_file_sha256") != _hash(manifest_data)
                        or report.get("source_tree_sha256") != source_pin
                        or report.get("corpus_file_sha256") != _hash(corpus_data)
                        or report.get("profile_sha256") != _hash(canonical(profiles[profile_id]))
                        or report.get("status") not in {"pass", "fail"}):
                    raise ValueError("probe_report_pin_mismatch:" + profile_id)
                rows = report.get("case_summaries")
                expected_cases = 9 if profile_id == remote_config["classification"]["profile_id"] else 5
                if not isinstance(rows, list) or len(rows) != expected_cases:
                    raise ValueError("probe_case_count_invalid:" + profile_id)
                for number, row in enumerate(rows, 1):
                    summary_path = prefix + f"case-{number:02d}.summary.json"
                    summary = _json(get(summary_path, limit=1024 * 1024), summary_path)
                    if summary != row:
                        raise ValueError("probe_case_summary_mismatch:" + profile_id)
                    record_path = row.get("record_path")
                    if record_path is None:
                        if row.get("status") != "infrastructure_failed":
                            raise ValueError("probe_record_missing_without_infrastructure_failure")
                        continue
                    if record_path != prefix + f"case-{number:02d}.record.json":
                        raise ValueError("probe_record_path_invalid:" + profile_id)
                    record = _json(get(record_path, expected=row["record_file_sha256"]), record_path)
                    sealed = record.get("record_sha256")
                    if (sealed != row.get("record_sha256") or
                            sealed != _hash(canonical({k: v for k, v in record.items() if k != "record_sha256"}))
                            or record.get("profile_sha256") != report["profile_sha256"]
                            or record.get("profile", {}).get("profile_id") != profile_id
                            or record.get("task", {}).get("task_sha256") != report.get("task_sha256", {}).get(row.get("kind"))
                            or record.get("job", {}).get("paper_id") != row.get("paper_id")
                            or record.get("job", {}).get("profile_id") != profile_id
                            or record.get("paper_input_canonical_sha256") is None):
                        raise ValueError("probe_record_pin_mismatch:" + profile_id)
                    record_inputs.append((row["paper_id"], record, report))
                reports[profile_id] = report
            if {row["profile_id"] for row in qualification.get("gpu", [])} != expected_profiles:
                raise ValueError("qualification_profile_coverage_mismatch")
            if any(row.get("exit_code") != 0 for row in qualification["gpu"] if qualification["status"] == "pass"):
                raise ValueError("qualification_exit_code_mismatch")
            papers = {row["paper_id"]: row for row in corpus["papers"]}
            if not set(PAPERS) <= papers.keys():
                raise ValueError("probe_paper_scope_missing")
            for paper_id in PAPERS:
                artifact = papers[paper_id]["source"]["artifacts"]["input"]
                path = _relative(artifact["path"])
                if path != f"mineru/{paper_id}/input.json":
                    raise ValueError("unexpected_probe_paper_input_path")
                data = get("source_bundle/" + path, limit=8 * 1024 * 1024, expected=artifact["sha256"])
                if len(data) != artifact["bytes"]:
                    raise ValueError("probe_paper_input_size_mismatch")
            for paper_id, record, report in record_inputs:
                if paper_id not in PAPERS:
                    raise ValueError("probe_record_paper_outside_fixed_scope")
                paper_row = papers[paper_id]
                path = "source_bundle/" + paper_row["source"]["artifacts"]["input"]["path"]
                paper = _json(downloaded[path], path)
                if (record["paper_input_canonical_sha256"] != _hash(canonical(paper))
                        or record["job"].get("source_sha256") != paper_row["source"]["source_sha256"]
                        or report.get("paper_sources", {}).get(paper_id) != paper_row["source"]["source_sha256"]):
                    raise ValueError("probe_record_source_pin_mismatch:" + paper_id)
    finally:
        client.close()
        gateway.close()

    if sha(event_path) != event_file_sha:
        raise ValueError("signed_event_changed_during_collection")
    output = Path(config["analysis_root"]) / ("probe-collection-" + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "event.json").write_bytes(event_path.read_bytes())
    for relative, data in downloaded.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    summary = {"event_id": event_id, "event_kind": event["kind"],
               "qualification_status": qualification["status"],
               "model_counts": _summarize(reports),
               "missing_profile_reports": sorted(expected_profiles - reports.keys()),
               "file_count": len(downloaded), "total_remote_bytes": total,
               "verified_config_file_sha256": config_pin, "verified_source_tree_sha256": source_pin,
               "production_writes": 0, "model_calls": 0,
               "semantic_accuracy": "not_established"}
    (output / "collection.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"collection": str(output / "collection.json"), "qualification_status": qualification["status"],
            "model_counts": summary["model_counts"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    args = parser.parse_args()
    result = collect(read(args.config), args.event_id)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
