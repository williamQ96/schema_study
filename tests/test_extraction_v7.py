from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from high_fidelity_schema_study.four_category.common import ROOT, seal
from high_fidelity_schema_study.four_category.extraction_v7 import (
    build_bundle, completion_status, make_task, materialize_group, parse_response, plan_groups,
    quote_catalog, render_group, response_schema, validate_response, verify_bundle,
    window_catalog,
)

from .test_classification_v2 import paper
from .test_extraction_view import fixture_index


POLICY = {"max_windows": 1, "max_target_chars": 240, "max_mentions": 32, "max_facts": 48}


def empty(group):
    return {"schema_version": "paper-extraction-group-response/v3",
            "coverage": [{"window_id": wid, "state": "reviewed"} for wid in group["window_ids"]],
            "mentions": [], "facts": []}


def mention(source_window=0, label="a descriptive label"):
    return {"kind": "field", "normalized_label": label,
            "source_windows": [source_window], "context": None}


def attribute(primary):
    return {"subject_mention": 0, "primary_window": primary, "support_windows": [],
            "categories": ["structure"], "claim": {"kind": "attribute", "predicate": "description",
            "assertion": {"status": "reported", "value": "paper description", "basis": None}}}


def test_full_paper_precedes_group_schema_and_windows_cover_source():
    p = paper(); index, task = fixture_index(p), make_task()
    groups = plan_groups(p, POLICY)
    windows = window_catalog(p)
    assert [wid for group in groups for wid in group["window_ids"]] == list(range(len(windows)))
    quotes = quote_catalog(p)
    for unit in p["pages"][0]["text_regions"]:
        assert "".join(q["text"] for q in quotes[unit["unit_id"]]) == unit["text"]
    rendered = render_group(task, p, index, groups[0])[1]["content"]
    assert rendered.index("FULL PAPER WITH GLOBALLY NUMBERED") < rendered.index("GROUP RESPONSE SCHEMA")
    assert "unit_index" not in rendered and '"index_entries"' in rendered
    assert '"unit_id":"u0"' not in rendered  # units are compact ordered rows
    schema = response_schema(task, p, index, groups[0])
    assert schema["properties"]["facts"]["items"]["properties"]["primary_window"]["enum"] == groups[0]["window_ids"]
    assert schema["$defs"]["sourceWindows"]["items"]["maximum"] == len(windows) - 1


def test_undersized_group_char_limit_reports_window_error():
    with pytest.raises(ValueError, match=r"window_exceeds_target_chars:\d+"):
        plan_groups(paper(), {**POLICY, "max_target_chars": 1})


def test_pointer_context_materializes_exact_window_spans_without_name_claim():
    from high_fidelity_schema_study.four_category import workflow
    p = paper(); unit = p["pages"][0]["text_regions"][0]
    unit["text"] = "😀" + "x" * 239 + "DATE\nFRUIT"
    unit["document_char_span"]["end"] = unit["document_char_span"]["start"] + len(unit["text"])
    index, task = fixture_index(p), make_task()
    groups = plan_groups(p, POLICY)
    assert len(quote_catalog(p)["u0"]) == 2
    g = groups[0]; value = empty(g)
    value["mentions"] = [{"kind": "dataset", "normalized_label": "model label not in source",
                          "source_windows": [0, 1], "context": None}]
    value["facts"] = [attribute(g["window_ids"][0])]
    assert validate_response(value, p, index, g, task) == []
    # A source-window pointer supplies exact context; label equivalence is not adjudicated.
    assert value["mentions"][0]["normalized_label"] not in unit["text"]
    windows = window_catalog(p)
    assert windows
    assert windows[0]["start"] == 0 and windows[1]["start"] == 240


def test_invalid_pointer_primary_and_claim_combinations():
    p = paper(); index, task = fixture_index(p), make_task(); groups = plan_groups(p, POLICY)
    g = groups[0]; value = empty(g); value["mentions"] = [mention()]
    value["facts"] = [attribute(g["window_ids"][0])]
    assert validate_response(value, p, index, g, task) == []
    bad = copy.deepcopy(value); bad["mentions"][0]["source_windows"] = [999]
    assert validate_response(bad, p, index, g, task)
    bad = copy.deepcopy(value); bad["mentions"][0]["source_windows"] = [0, 0]
    assert validate_response(bad, p, index, g, task)
    bad = copy.deepcopy(value); bad["facts"][0]["primary_window"] = groups[1]["window_ids"][0]
    assert validate_response(bad, p, index, g, task)
    bad = copy.deepcopy(value); bad["facts"][0]["claim"]["target"] = {"kind": "table", "normalized_label": "x", "source_windows": [0]}
    assert validate_response(bad, p, index, g, task)
    link = copy.deepcopy(value)
    link["facts"][0]["claim"] = {"kind": "link", "predicate": "parent",
        "target": {"kind": "table", "normalized_label": "model target", "source_windows": [1]},
        "assertion": {"status": "inferred", "value": "contained in", "basis": "paper context"}}
    assert validate_response(link, p, index, g, task) == []
    bad = copy.deepcopy(link); del bad["facts"][0]["claim"]["target"]
    assert validate_response(bad, p, index, g, task)
    bad = copy.deepcopy(link); bad["facts"][0]["claim"]["assertion"]["basis"] = ""
    assert validate_response(bad, p, index, g, task)


def test_caps_overflow_strict_parser_and_frozen_old_sources():
    p = paper(); index, task = fixture_index(p), make_task(); group = plan_groups(p, POLICY)[0]
    value = empty(group)
    assert completion_status(value, group) == "success"
    value["mentions"] = [mention()] * group["policy"]["max_mentions"]
    assert completion_status(value, group) == "incomplete"
    value = empty(group); value["coverage"][0]["state"] = "overflow"
    assert completion_status(value, group) == "incomplete"
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_response('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        parse_response('```json\n{}\n```')
    frozen = {"extraction_v5.py": "e28eddf0e637bdaabd2e8e89c1ef4b8c00634e1ba2661f670fba817b17286cf4",
              "extraction_v6.py": "08f7f8b0191ca0a70ed57d47b1399ee14ffb94523b89e9b7caa4074fe4272ac0"}
    for name, expected in frozen.items():
        assert hashlib.sha256((ROOT / "four_category" / name).read_bytes()).hexdigest() == expected


def test_bundle_replay_and_resealed_tamper(monkeypatch):
    from high_fidelity_schema_study.four_category import workflow
    p = paper(); index, task = fixture_index(p), make_task(); groups = plan_groups(p, POLICY)
    monkeypatch.setattr(workflow, "replay_run", lambda *args, **kwargs: [])
    records = []
    for group in groups:
        value = empty(group)
        value["mentions"] = [mention()]
        value["facts"] = [attribute(group["window_ids"][0])]
        records.append({"status": "success", "task": task, "extraction_group": group,
                        "job": {"parent_job_id": "outer"}, "profile_sha256": "profile",
                        "index_sha256": index["index_sha256"], "backend_result": {"raw_text": json.dumps(value)},
                        "parsed_response": value, "response_normalization": {"kind": "bare_json"},
                        "run_id": group["group_id"], "record_sha256": group["group_sha256"]})
    bundle = build_bundle(p, index, task, groups, records)
    first = materialize_group(p, index, task, groups[0], records[0])
    assert set(first) == {"group_id", "record_sha256", "mentions", "facts", "coverage"}
    assert first["mentions"] == bundle["mentions"][:len(first["mentions"])]
    assert verify_bundle(bundle, p, index, task, records) == []
    evidence = bundle["mentions"][0]["source_evidence"][0]
    assert evidence["quote_or_cell_text"] == p["pages"][0]["text_regions"][0]["text"]
    assert evidence["unit_id"] == "u0" and evidence["page"] == 1
    assert bundle["mentions"][0]["normalized_label_authority"] == "model_asserted_not_verified"
    changed = copy.deepcopy(bundle); changed["coverage"][0]["state"] = "uncertain"
    changed = seal(changed, "bundle_sha256")
    assert "extraction_bundle_derivation_mismatch" in verify_bundle(changed, p, index, task, records)
    incomplete = copy.deepcopy(records[0]); incomplete["status"] = "incomplete"
    with pytest.raises(ValueError, match="not_successful_or_bound"):
        materialize_group(p, index, task, groups[0], incomplete)
