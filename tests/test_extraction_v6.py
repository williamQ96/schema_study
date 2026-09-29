from __future__ import annotations

import copy
import json

import jsonschema
import pytest

from high_fidelity_schema_study.four_category.common import read_json, seal
from high_fidelity_schema_study.four_category.extraction_v6 import (
    build_bundle, completion_status, group_errors, make_task, parse_response,
    plan_groups, quote_catalog, render_group, resolve_source_name, response_schema,
    validate_response, verify_bundle, window_catalog,
)

from .test_classification_v2 import paper
from .test_extraction_view import fixture_index


POLICY = {"max_windows": 1, "max_target_chars": 240, "max_mentions": 32, "max_facts": 48}


def empty(group):
    return {"schema_version": "paper-extraction-group-response/v2",
            "coverage": [{"window_id": wid, "state": "reviewed"} for wid in group["window_ids"]],
            "mentions": [], "facts": []}


def selector(unit, surface, occurrence=0):
    return {"unit_index": unit, "surface": surface, "occurrence": occurrence}


def mention(unit=0, surface="alpha", label="alpha"):
    return {"kind": "field", "normalized_label": label,
            "source_name": selector(unit, surface), "context": None}


def attribute(primary=0, status="reported", value="alpha", basis=None):
    return {"subject_mention": 0, "primary_window": primary, "support_windows": [],
            "categories": ["structure"], "claim": {"kind": "attribute", "predicate": "reported_name",
            "assertion": {"status": status, "value": value, "basis": basis}}}


def test_numbered_full_source_and_assigned_primary_grammar():
    p = paper(); index, task = fixture_index(p), make_task()
    groups = plan_groups(p, POLICY)
    assert [w for group in groups for w in group["window_ids"]] == list(range(len(window_catalog(p))))
    quoted = quote_catalog(p)
    assert "".join(q["text"] for q in quoted["u0"]) == p["pages"][0]["text_regions"][0]["text"]
    prompt = render_group(task, p, index, groups[0])[1]["content"]
    assert "numbered_windows" in prompt and "index_entries" in prompt
    assert "document_char_span" not in prompt
    schema = response_schema(task, p, index, groups[0])
    assert schema["properties"]["facts"]["items"]["properties"]["primary_window"]["enum"] == groups[0]["window_ids"]
    assert schema["$defs"]["sourceName"]["properties"]["unit_index"]["maximum"] == len(p["pages"][0]["text_regions"]) - 1
    assert validate_response(empty(groups[0]), p, index, groups[0], task) == []
    bad = copy.deepcopy(groups[0]); bad["index"] = -1
    assert group_errors(bad, p)


def test_surface_resolution_whitespace_repetition_and_exact_offsets():
    p = paper()
    unit = p["pages"][0]["text_regions"][0]
    unit["text"] = "DATE\u00a0FRUIT\nIMAGE DATASET; DATE FRUIT\nIMAGE DATASET"
    unit["document_char_span"]["end"] = unit["document_char_span"]["start"] + len(unit["text"])
    first = resolve_source_name(selector(0, "DATE FRUIT IMAGE DATASET", 0), p)
    second = resolve_source_name(selector(0, "DATE FRUIT IMAGE DATASET", 1), p)
    assert first["raw_surface"] == "DATE\u00a0FRUIT\nIMAGE DATASET"
    assert second["raw_surface"] == "DATE FRUIT\nIMAGE DATASET"
    assert first["unit_char_span"]["end"] <= second["unit_char_span"]["start"]
    assert unit["text"][first["unit_char_span"]["start"]:first["unit_char_span"]["end"]] == first["raw_surface"]
    assert first["document_char_span"]["start"] == unit["document_char_span"]["start"]
    with pytest.raises(ValueError, match="occurrence_missing"):
        resolve_source_name(selector(0, "DATE FRUIT IMAGE DATASET", 2), p)
    with pytest.raises(ValueError, match="occurrence_missing"):
        resolve_source_name(selector(1, "DATE FRUIT IMAGE DATASET"), p)
    with pytest.raises(ValueError, match="occurrence_missing"):
        resolve_source_name(selector(0, "Date Fruit Image Dataset"), p)
    with pytest.raises(ValueError, match="occurrence_missing"):
        resolve_source_name(selector(0, "DATE FRUIT IMAGE DATASET!"), p)
    with pytest.raises(ValueError, match="empty_after_whitespace"):
        resolve_source_name(selector(0, " \n\u00a0"), p)


def test_source_surface_crosses_quote_window_without_relocation():
    p = paper(); unit = p["pages"][0]["text_regions"][0]
    unit["text"] = "x" * 238 + "DATE\nFRUIT" + "z" * 235
    unit["document_char_span"]["end"] = unit["document_char_span"]["start"] + len(unit["text"])
    assert len(quote_catalog(p)["u0"]) > 1
    evidence = resolve_source_name(selector(0, "DATE FRUIT"), p)
    assert evidence["raw_surface"] == "DATE\nFRUIT"
    assert evidence["unit_char_span"] == {"start": 238, "end": 248}


def test_surface_offsets_preserve_codepoints_and_no_unicode_folding():
    p = paper(); unit = p["pages"][0]["text_regions"][0]
    unit["text"] = "😀DATE\nFRUIT DATE FRUIT"
    unit["document_char_span"]["end"] = unit["document_char_span"]["start"] + len(unit["text"])
    first = resolve_source_name(selector(0, "DATE FRUIT", 0), p)
    second = resolve_source_name(selector(0, "DATE FRUIT", 1), p)
    assert first["unit_char_span"] == {"start": 1, "end": 11}
    assert first["document_char_span"] == {"start": 101, "end": 111}
    assert second["unit_char_span"] == {"start": 12, "end": 22}
    assert first["raw_surface"] == "DATE\nFRUIT" and second["raw_surface"] == "DATE FRUIT"
    unit["text"] = "A\u200bB A\u001cB e\u0301 Ｃ"
    unit["document_char_span"]["end"] = unit["document_char_span"]["start"] + len(unit["text"])
    for surface in ("AB", "é", "C"):
        with pytest.raises(ValueError, match="occurrence_missing"):
            resolve_source_name(selector(0, surface), p)
    assert resolve_source_name(selector(0, "A\u200bB"), p)["raw_surface"] == "A\u200bB"
    assert resolve_source_name(selector(0, "A\u001cB"), p)["raw_surface"] == "A\u001cB"
    assert resolve_source_name(selector(0, "e\u0301"), p)["raw_surface"] == "e\u0301"


def test_branch_status_primary_support_and_cap_saturation():
    p = paper(); index, task = fixture_index(p), make_task(); groups = plan_groups(p, POLICY)
    g = groups[0]; value = empty(g)
    value["mentions"] = [mention()]
    value["facts"] = [attribute(g["window_ids"][0])]
    assert validate_response(value, p, index, g, task) == []
    assert completion_status(value, g) == "success"
    invalid = copy.deepcopy(value)
    invalid["facts"][0]["primary_window"] = groups[1]["window_ids"][0]
    assert validate_response(invalid, p, index, g, task)
    invalid = copy.deepcopy(value)
    invalid["facts"][0]["support_windows"] = [999]
    assert validate_response(invalid, p, index, g, task)
    invalid = copy.deepcopy(value)
    invalid["facts"][0]["claim"]["target"] = {"kind": "table", "normalized_label": "repeat", "source_name": selector(1, "repeat")}
    assert validate_response(invalid, p, index, g, task)
    invalid = copy.deepcopy(value)
    invalid["facts"][0]["claim"]["assertion"] = {"status": "unknown", "value": "alpha", "basis": None}
    assert validate_response(invalid, p, index, g, task)
    invalid = copy.deepcopy(value)
    invalid["facts"][0]["claim"]["assertion"] = {"status": "inferred", "value": "alpha", "basis": ""}
    assert validate_response(invalid, p, index, g, task)
    link = copy.deepcopy(value)
    link["facts"][0]["claim"] = {"kind": "link", "predicate": "parent",
        "target": {"kind": "table", "normalized_label": "repeat", "source_name": selector(1, "repeat")},
        "assertion": {"status": "inferred", "value": "parent", "basis": "paper describes nesting"}}
    assert validate_response(link, p, index, g, task) == []
    no_target = copy.deepcopy(link); del no_target["facts"][0]["claim"]["target"]
    assert validate_response(no_target, p, index, g, task)
    saturated = copy.deepcopy(value); saturated["mentions"] = [mention()] * g["policy"]["max_mentions"]
    assert validate_response(saturated, p, index, g, task) == []
    assert completion_status(saturated, g) == "incomplete"
    overflow = copy.deepcopy(value); overflow["coverage"][0]["state"] = "overflow"
    assert any("fact_on_overflow_window" in e for e in validate_response(overflow, p, index, g, task))
    overflow["facts"] = []
    assert completion_status(overflow, g) == "incomplete"


def test_bundle_replay_identity_and_tamper(monkeypatch):
    from high_fidelity_schema_study.four_category import workflow
    p = paper(); index, task = fixture_index(p), make_task()
    groups = plan_groups(p, POLICY)
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
    assert verify_bundle(bundle, p, index, task, records) == []
    assert bundle["schema_version"] == "paper-derived-observations/v6"
    assert bundle["mentions"][0]["source_name_evidence"]["raw_surface"] == "alpha"
    assert bundle["mentions"][0]["normalized_label_authority"] == "model_asserted_not_verified"
    assert bundle["facts"][0]["primary_evidence"]["unit_id"] == "u0"
    changed = copy.deepcopy(bundle); changed["coverage"][0]["state"] = "uncertain"
    changed = seal(changed, "bundle_sha256")
    assert "extraction_bundle_derivation_mismatch" in verify_bundle(changed, p, index, task, records)


def test_parser_remains_strict_and_v5_task_unchanged():
    from high_fidelity_schema_study.four_category.extraction_v5 import make_task as make_v5_task
    assert make_task()["schema_version"] == "four-category-task/v6"
    assert make_v5_task()["schema_version"] == "four-category-task/v5"
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_response('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        parse_response('```json\n{}\n```')
