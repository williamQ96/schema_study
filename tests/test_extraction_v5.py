from __future__ import annotations

import copy
import json

import pytest

from high_fidelity_schema_study.four_category.extraction_v5 import (
    build_bundle, completion_status, group_errors, make_task, parse_response, plan_groups, quote_catalog,
    render_group, response_schema, validate_response,
    verify_bundle,
)
from high_fidelity_schema_study.four_category.common import seal

from .test_classification_v2 import paper
from .test_extraction_view import fixture_index


def empty_payload(group):
    return {"schema_version": "paper-extraction-group-response/v1",
            "coverage": [{"window_id": wid, "state": "reviewed"} for wid in group["window_ids"]],
            "mentions": [], "facts": []}


def test_complete_source_quote_windows_and_bounded_groups():
    p = paper()
    p["pages"][0]["text_regions"][0]["text"] = "é😀 alpha " * 80
    index, task = fixture_index(p), make_task()
    groups = plan_groups(p, {"max_windows": 2, "max_target_chars": 300,
                              "max_mentions": 2, "max_facts": 3})
    assert len(groups) > 1
    quotes = quote_catalog(p)
    for unit in p["pages"][0]["text_regions"]:
        assert "".join(q["text"] for q in quotes[unit["unit_id"]]) == unit["text"]
    assert [wid for group in groups for wid in group["window_ids"]] == [
        uid + "#" + q["quote_id"] for uid in quotes for q in quotes[uid]]
    prompt = render_group(task, p, index, groups[0])[1]["content"]
    assert "é😀 alpha" in prompt and "index_entries" in prompt
    assert all(uid in prompt for uid in quotes)
    schema = response_schema(task, p, index, groups[0])
    assert schema["properties"]["facts"]["maxItems"] == 3


def test_fact_ownership_and_cross_unit_name_anchor():
    p = paper(); index, task = fixture_index(p), make_task()
    groups = plan_groups(p, {"max_windows": 1, "max_target_chars": 300,
                              "max_mentions": 2, "max_facts": 3})
    g = groups[0]
    value = empty_payload(g)
    anchor = {"unit_id": g["window_ids"][0].split("#")[0], "quote_id": "q0000"}
    value["mentions"] = [{"kind": "field", "name": "alpha", "name_anchor": anchor, "context": None}]
    value["facts"] = [{"subject_mention": 0, "predicate": "reported_name", "value": "alpha",
                       "status": "reported", "basis": None, "categories": ["structure"],
                       "primary_anchor": anchor, "support_anchors": [], "target": None}]
    assert validate_response(value, p, index, g, task) == []
    foreign = copy.deepcopy(value)
    foreign["facts"][0]["primary_anchor"] = {"unit_id": groups[1]["window_ids"][0].split("#")[0],
                                             "quote_id": "q0000"}
    assert any("fact_primary_anchor_foreign" in e for e in validate_response(foreign, p, index, g, task))
    assert completion_status(value) == "success"
    value["coverage"][0]["state"] = "uncertain"
    assert completion_status(value) == "success"


def test_duplicate_key_duplicate_fact_and_cap_are_rejected():
    p = paper(); index, task = fixture_index(p), make_task(); group = plan_groups(p)[0]
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_response('{"schema_version":"a","schema_version":"b"}')
    with pytest.raises(ValueError):
        parse_response('```json\n{}\n```')
    value = empty_payload(group)
    anchor = {"unit_id": group["window_ids"][0].split("#")[0], "quote_id": "q0000"}
    value["mentions"] = [{"kind": "field", "name": "alpha", "name_anchor": anchor, "context": None}]
    fact = {"subject_mention": 0, "predicate": "reported_name", "value": "alpha", "status": "reported",
            "basis": None, "categories": ["structure"], "primary_anchor": anchor,
            "support_anchors": [], "target": None}
    value["facts"] = [fact, copy.deepcopy(fact)]
    assert any("duplicate_fact" in e for e in validate_response(value, p, index, group, task))
    value["facts"] = [fact] * 25
    assert any("extraction_schema:" in e for e in validate_response(value, p, index, group, task))
    value["facts"] = [fact, {**fact, "support_anchors": [anchor]}]
    assert validate_response(value, p, index, group, task) == []


def test_missing_target_and_overflow_remain_visible():
    p = paper(); index, task = fixture_index(p), make_task(); group = plan_groups(p)[0]
    value = empty_payload(group)
    value["coverage"].pop()
    assert validate_response(value, p, index, group, task)
    value = empty_payload(group)
    value["coverage"][0]["state"] = "overflow"
    assert validate_response(value, p, index, group, task) == []
    assert completion_status(value) == "incomplete"
    bad_group = copy.deepcopy(group)
    bad_group["index"] = -1
    assert any("extraction_group_plan_mismatch" in e for e in group_errors(bad_group, p))


def test_bundle_keeps_same_name_mentions_separate(monkeypatch):
    from high_fidelity_schema_study.four_category import workflow
    p = paper(); index, task = fixture_index(p), make_task()
    groups = plan_groups(p, {"max_windows": 1, "max_target_chars": 300,
                              "max_mentions": 2, "max_facts": 3})
    monkeypatch.setattr(workflow, "replay_run", lambda *args, **kwargs: [])
    records = []
    for group in groups:
        value = empty_payload(group)
        anchor = {"unit_id": group["window_ids"][0].split("#")[0], "quote_id": "q0000"}
        value["mentions"] = [{"kind": "field", "name": "alpha", "name_anchor":
                              {"unit_id": groups[0]["window_ids"][0].split("#")[0], "quote_id": "q0000"},
                              "context": None}]
        value["facts"] = [{"subject_mention": 0, "predicate": "description", "value": "observed",
                           "status": "reported", "basis": None, "categories": ["structure"],
                           "primary_anchor": anchor, "support_anchors": [], "target": None}]
        records.append({"status": "success", "task": task, "extraction_group": group,
                        "job": {"parent_job_id": "outer"}, "profile_sha256": "profile",
                        "index_sha256": index["index_sha256"],
                        "backend_result": {"raw_text": json.dumps(value)}, "parsed_response": value,
                        "response_normalization": {"kind": "bare_json"},
                        "run_id": group["group_id"], "record_sha256": group["group_sha256"]})
    bundle = build_bundle(p, index, task, groups, records)
    assert bundle["mention_record_count"] == len(groups)
    assert bundle["resolved_object_count"] is None
    assert len({m["mention_id"] for m in bundle["mentions"]}) == len(groups)
    assert bundle["observation_count"] == len(groups)
    assert all(f["primary_evidence"]["quote_or_cell_text"] for f in bundle["facts"])
    other_replicate = copy.deepcopy(records)
    for row in other_replicate:
        row["job"]["parent_job_id"] = "another-outer"
    other_bundle = build_bundle(p, index, task, groups, other_replicate)
    assert other_bundle["mentions"][0]["mention_id"] != bundle["mentions"][0]["mention_id"]
    assert other_bundle["facts"][0]["fact_id"] != bundle["facts"][0]["fact_id"]
    bad = copy.deepcopy(records)
    bad[-1]["profile_sha256"] = "another-profile"
    with pytest.raises(ValueError, match="cross_record_binding"):
        build_bundle(p, index, task, groups, bad)


def test_real_mock_records_replay_and_resealed_bundle_tamper(tmp_path):
    from high_fidelity_schema_study.four_category.grouped_extraction import (
        _extended, execute_grouped, mock_group_transport, verify_grouped,
    )
    from high_fidelity_schema_study.four_category.common import read_json
    from .test_grouped_extraction import setup

    p, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / "grouped"
    body = execute_grouped(job, profile, task, p, index, policy, 1, root,
                           allow_live=False, transport=mock_group_transport)
    errors, bundle = verify_grouped(job, profile, task, p, index, policy, 1, body, root)
    assert errors == [] and bundle is not None
    records = [read_json(_extended(root / ref["path"])) for ref in body["group_record_refs"]]
    assert verify_bundle(bundle, p, index, task, records) == []
    changed = copy.deepcopy(bundle)
    changed["coverage"][0]["state"] = "uncertain"
    changed = seal(changed, "bundle_sha256")
    assert "extraction_bundle_derivation_mismatch" in verify_bundle(changed, p, index, task, records)
