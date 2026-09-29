from __future__ import annotations

import copy
import json

import pytest

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.classification_v2 import (
    build_index_v2, make_task, parse_response, plan_groups, render_group,
    resolve_entries, validate_response, verify_index_v2,
)
from high_fidelity_schema_study.four_category.common import digest, seal
from high_fidelity_schema_study.four_category.paper import source_identity
from high_fidelity_schema_study.four_category.workflow import _attempt


def paper():
    texts = ["é😀 alpha", "repeat repeat", "", "syntax: A; value: B"]
    units = []
    for i, text in enumerate(texts):
        units.append({"unit_id": f"u{i}", "page": 1, "region_kind": "paragraph", "reading_order": i,
                      "bbox": [0, 0, 1, 1], "document_char_span": {"start": 100 + 30*i, "end": 100 + 30*i + len(text)},
                      "document_token_span": {"start": i, "end": i+1}, "text": text})
    return {"schema_version": "paper-evidence-input/v3", "paper_id": "p", "source_pdf_sha256": "a"*64,
            "layout_preprocessing_sha256": "b"*64, "model_input_sha256": "c"*64, "span_conventions": {},
            "pages": [{"page": 1, "layout_mode": "single", "text_regions": units, "tables": []}]}


def response(p, group):
    entries = []
    for uid in group["unit_ids"]:
        entries.append({"unit_id": uid, "state": "none", "categories": [], "evidence_quotes": [], "rationale": ""})
    return {"schema_version": "paper-category-response/v2", "source_identity": source_identity(p), "entries": entries}


def record(p, task, group, payload, *, fenced=False):
    profile = {"backend": "mock", "profile_id": "fixture", "model_id": "fixture", "revision": "fixture",
               "deployment": "mock", "endpoint": None, "context_window": 1000000,
               "capabilities": {"supported_parameters": ["max_output_tokens"], "supports_system_role": True},
               "runtime": {}, "status": "frozen"}
    profhash = profile_hash(profile)
    raw = json.dumps(payload, ensure_ascii=False)
    if fenced:
        raw = "```json\n" + raw + "\n```"
    job = {"kind": "classification", "paper_id": p["paper_id"], "task_sha256": task["task_sha256"],
           "profile_sha256": profhash, "classification_group": group,
           "parameters": {"max_output_tokens": 1000}}
    job["job_id"] = digest(job)
    profile["runtime"]["mock_text"] = raw
    job["profile_sha256"] = profile_hash(profile)
    job["job_id"] = digest({k: v for k, v in job.items() if k != "job_id"})
    return _attempt(job, profile, task, p, None, None, None, 1, allow_live=False, transport=None,
                    counter=None, classification_group=group)


def test_group_bounds_catalog_and_unicode_coordinates():
    p, task = paper(), make_task()
    groups = plan_groups(p, {"max_units": 2, "max_target_chars": 30})
    assert [g["unit_ids"] for g in groups] == [["u0", "u1"], ["u2", "u3"]]
    assert groups[0]["index"] == 0 and groups[1]["total"] == 2
    prompt = render_group(task, p, groups[0])[1]["content"]
    assert '"text":"syntax: A; value: B"' in prompt
    assert "document_char_span" not in prompt and "bbox" not in prompt
    payload = response(p, groups[0])
    payload["entries"][0].update(state="classified", categories=["structure"], evidence_quotes=["😀 alpha"])
    assert validate_response(payload, p, groups[0]) == []
    entry = resolve_entries(payload, p, groups[0])[0]
    assert entry["evidence_spans"] == [{"start": 1, "end": 8, "quote": "😀 alpha"}]
    assert entry["document_evidence_spans"][0]["start"] == 101


def test_quotes_and_coverage_are_strict():
    p = paper(); group = plan_groups(p)[0]
    payload = response(p, group)
    payload["entries"][1].update(state="classified", categories=["value"], evidence_quotes=["repeat"])
    assert any("quote_absent_or_ambiguous" in e for e in validate_response(payload, p, group))
    payload["entries"][1]["evidence_quotes"] = ["missing"]
    assert any("quote_absent_or_ambiguous" in e for e in validate_response(payload, p, group))
    payload["entries"][1]["evidence_quotes"] = ["repeat repeat"]
    assert validate_response(payload, p, group) == []
    bad = copy.deepcopy(payload); bad["entries"].append(copy.deepcopy(bad["entries"][0]))
    assert "duplicate_classification_unit" in validate_response(bad, p, group)
    bad = copy.deepcopy(payload); bad["entries"].pop()
    assert "classification_target_order_or_coverage_mismatch" in validate_response(bad, p, group)
    bad = copy.deepcopy(payload); bad["entries"][2]["state"] = "uncertain"
    assert "empty_unit_must_be_none:u2" in validate_response(bad, p, group)


@pytest.mark.parametrize("raw", ['```json\n{}\n``` extra', '```JSON\n{}\n```', 'before ```json\n{}\n```', '```json\n{}\n```\n```'])
def test_parser_rejects_extra_or_malformed_envelope(raw):
    with pytest.raises(ValueError):
        parse_response(raw)


def test_parser_bare_fence_and_duplicate_keys():
    assert parse_response(' {"a":1} ')[1] == {"kind": "bare_json"}
    assert parse_response('```json\n{"a":1}\n```')[1] == {"kind": "json_code_fence"}
    with pytest.raises(ValueError):
        parse_response('{"a":1,"a":2}')


def test_group_limit_rejects_oversized_unit():
    with pytest.raises(ValueError, match="unit_exceeds_max_target_chars"):
        plan_groups(paper(), {"max_units": 2, "max_target_chars": 5})
    with pytest.raises(ValueError):
        plan_groups(paper(), {"max_units": True, "max_target_chars": 20})


def test_multigroup_index_replays_raw_and_rejects_resealed_tamper():
    p, task = paper(), make_task()
    groups = plan_groups(p, {"max_units": 2, "max_target_chars": 30})
    payloads = [response(p, g) for g in groups]
    payloads[0]["entries"][0].update(state="classified", categories=["structure"], evidence_quotes=["😀 alpha"])
    records = [record(p, task, groups[0], payloads[0], fenced=True), record(p, task, groups[1], payloads[1])]
    index = build_index_v2(p, task, groups, records)
    assert [e["unit_id"] for e in index["entries"]] == ["u0", "u1", "u2", "u3"]
    assert verify_index_v2(index, p) == []
    changed = copy.deepcopy(index); changed["entries"][0]["categories"] = ["syntax"]
    changed = seal(changed, "index_sha256")
    assert "index_raw_derivation_mismatch" in verify_index_v2(changed, p)
    changed = copy.deepcopy(index); changed["records"][0]["response_normalization"] = {"kind": "bare_json"}
    changed["records"][0] = seal(changed["records"][0], "record_sha256")
    changed = seal(changed, "index_sha256")
    assert verify_index_v2(changed, p)
    changed = copy.deepcopy(index); changed["records"][0]["backend_result"]["raw_response"]["text"] = '{}'
    changed["records"][0] = seal(changed["records"][0], "record_sha256")
    changed = seal(changed, "index_sha256")
    assert any("raw_text_replay_mismatch" in error for error in verify_index_v2(changed, p))
    changed = copy.deepcopy(index); changed["records"][0]["backend_result"]["request"]["messages"][0]["content"] = "tampered"
    changed["records"][0] = seal(changed["records"][0], "record_sha256")
    changed = seal(changed, "index_sha256")
    assert any("request_replay_mismatch" in error for error in verify_index_v2(changed, p))
