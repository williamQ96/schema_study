from __future__ import annotations

import copy
import json

import pytest

from high_fidelity_schema_study.four_category.classification_v2 import parse_response, plan_groups
from high_fidelity_schema_study.four_category.classification_v3 import (
    build_index_v3, make_task, quote_options, render_group, resolve_entries,
    validate_response, verify_index_v3,
)
from high_fidelity_schema_study.four_category.common import seal
from high_fidelity_schema_study.four_category.paper import source_identity

from .test_classification_v2 import paper, record


def payload(p, group):
    return {"schema_version": "paper-category-response/v3", "source_identity": source_identity(p),
            "entries": [{"unit_id": uid, "state": "none", "categories": [], "evidence_quote_ids": [], "rationale": ""}
                        for uid in group["unit_ids"]]}


def test_quote_options_cover_every_codepoint_and_keep_newlines():
    p = paper()
    p["pages"][0]["text_regions"][0]["text"] = "é😀\n" * 150
    group = plan_groups(p)[0]
    options = quote_options(p, group)[0]["options"]
    assert [o["quote_id"] for o in options] == [f"q{i:04d}" for i in range(len(options))]
    assert all(0 < len(o["text"]) <= 240 for o in options)
    assert "".join(o["text"] for o in options) == p["pages"][0]["text_regions"][0]["text"]
    assert "document_char_span" not in render_group(make_task(), p, group)[1]["content"]


def test_repeated_equal_quote_content_has_distinct_positions():
    p = paper()
    p["pages"][0]["text_regions"][0]["text"] = "x" * 480
    p["pages"][0]["text_regions"][0]["document_char_span"] = {"start": 100, "end": 580}
    group = plan_groups(p)[0]
    options = quote_options(p, group)[0]["options"]
    assert options[0]["text"] == options[1]["text"] == "x" * 240
    value = payload(p, group)
    value["entries"][0].update(state="classified", categories=["structure"], evidence_quote_ids=["q0001"])
    assert validate_response(value, p, group) == []
    entry = resolve_entries(value, p, group)[0]
    assert entry["evidence_spans"] == [{"start": 240, "end": 480, "quote": "x"*240}]
    assert entry["document_evidence_spans"][0]["start"] == 340


def test_missing_unknown_and_wrong_unit_ids_rejected():
    p = paper()
    p["pages"][0]["text_regions"][1]["text"] = "y" * 500
    p["pages"][0]["text_regions"][1]["document_char_span"] = {"start": 130, "end": 630}
    group = plan_groups(p)[0]
    value = payload(p, group)
    value["entries"][0].update(state="classified", categories=["value"], evidence_quote_ids=[])
    assert any("requires_categories_and_quote_ids" in e for e in validate_response(value, p, group))
    value["entries"][0]["evidence_quote_ids"] = ["q9999"]
    assert any("quote_id_missing_or_wrong_unit" in e for e in validate_response(value, p, group))
    value["entries"][0]["evidence_quote_ids"] = ["q0001"]
    assert any("quote_id_missing_or_wrong_unit" in e for e in validate_response(value, p, group))
    value["entries"][0]["evidence_quote_ids"] = ["q0000"]
    assert validate_response(value, p, group) == []


def test_empty_unit_and_group_coverage():
    p = paper(); group = plan_groups(p)[0]
    assert quote_options(p, group)[2]["options"] == []
    value = payload(p, group)
    value["entries"][2]["state"] = "uncertain"
    assert "empty_unit_must_be_none:u2" in validate_response(value, p, group)
    value["entries"][2]["state"] = "none"
    value["entries"].pop()
    assert "classification_target_order_or_coverage_mismatch" in validate_response(value, p, group)


def test_multigroup_index_raw_replay_and_resealed_tamper():
    p, task = paper(), make_task()
    groups = plan_groups(p, {"max_units": 2, "max_target_chars": 30})
    values = [payload(p, group) for group in groups]
    values[0]["entries"][0].update(state="classified", categories=["structure"], evidence_quote_ids=["q0000"])
    records = [record(p, task, groups[0], values[0], fenced=True), record(p, task, groups[1], values[1])]
    index = build_index_v3(p, task, groups, records)
    assert verify_index_v3(index, p) == []
    assert [e["unit_id"] for e in index["entries"]] == ["u0", "u1", "u2", "u3"]
    changed = copy.deepcopy(index); changed["entries"][0]["categories"] = ["syntax"]
    changed = seal(changed, "index_sha256")
    assert "index_raw_derivation_mismatch" in verify_index_v3(changed, p)
    changed = copy.deepcopy(index); changed["records"][0]["response_normalization"] = {"kind": "bare_json"}
    changed["records"][0] = seal(changed["records"][0], "record_sha256")
    changed = seal(changed, "index_sha256")
    assert verify_index_v3(changed, p)
    changed = copy.deepcopy(index); changed["records"][0]["backend_result"]["raw_response"]["text"] = '{}'
    changed["records"][0] = seal(changed["records"][0], "record_sha256")
    changed = seal(changed, "index_sha256")
    assert any("raw_text_replay_mismatch" in error for error in verify_index_v3(changed, p))


def test_rationale_limit_and_old_version_intact():
    p = paper(); group = plan_groups(p)[0]
    value = payload(p, group)
    value["entries"][0]["rationale"] = "x" * 300
    assert validate_response(value, p, group) == []
    value["entries"][0]["rationale"] = "x" * 1025
    assert validate_response(value, p, group)
    assert make_task()["schema_version"] == "four-category-task/v4"
