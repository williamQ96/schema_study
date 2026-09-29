from __future__ import annotations

import copy
import json

import jsonschema

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.classification_v2 import plan_groups
from high_fidelity_schema_study.four_category.classification_navigation_v4 import build_index_v4, verify_index_v4
from high_fidelity_schema_study.four_category.classification_v13r2 import (
    make_task, render_group, response_schema, validate_response,
)
from high_fidelity_schema_study.four_category.common import digest
from high_fidelity_schema_study.four_category.grouped_classification import group_job
from high_fidelity_schema_study.four_category.paper import source_identity
from high_fidelity_schema_study.four_category.workflow import _attempt, production_request, replay_run

from .test_classification_v2 import paper


def _payload(p, group):
    return {"schema_version": "paper-category-response/v3", "source_identity": source_identity(p),
            "entries": [{"unit_id": uid, "state": "none", "categories": [],
                         "evidence_quote_ids": [], "rationale": ""} for uid in group["unit_ids"]]}


def test_runtime_schema_binds_order_state_and_unit_quote_ids():
    p = paper()
    p["pages"][0]["text_regions"][1]["text"] = "y" * 500
    p["pages"][0]["text_regions"][1]["document_char_span"]["end"] = 630
    group = plan_groups(p)[0]
    schema = response_schema(make_task(), p, group)
    validator = jsonschema.Draft202012Validator(schema)
    good = _payload(p, group)
    assert validator.is_valid(good)
    bad = copy.deepcopy(good); bad["entries"][0]["state"] = "value"
    assert not validator.is_valid(bad)
    bad = copy.deepcopy(good); bad["entries"].pop()
    assert not validator.is_valid(bad)
    bad = copy.deepcopy(good); bad["entries"][0], bad["entries"][1] = bad["entries"][1], bad["entries"][0]
    assert not validator.is_valid(bad)
    bad = copy.deepcopy(good); bad["entries"][0].update(state="classified", categories=["structure"], evidence_quote_ids=["q0001"])
    assert not validator.is_valid(bad)
    bad["entries"][0]["evidence_quote_ids"] = ["q0000"]
    assert validator.is_valid(bad)
    bad["entries"][0]["categories"] = []
    assert not validator.is_valid(bad)
    bad = copy.deepcopy(good); bad["entries"][2]["state"] = "uncertain"
    assert not validator.is_valid(bad)
    assert validate_response(good, p, group) == []


def test_group_local_context_and_actual_request_share_schema():
    p, task = paper(), make_task()
    groups = plan_groups(p, {"max_units": 2, "max_target_chars": 30})
    group = groups[0]
    messages, schema = production_request(task, p, classification_group=group)
    assert messages == render_group(task, p, group)
    assert schema == response_schema(task, p, group)
    assert '"unit_id":"u0"' in messages[1]["content"]
    assert '"unit_id":"u2"' not in messages[1]["content"]
    profile = {"backend": "mock", "profile_id": "fixture", "model_id": "fixture", "revision": None,
               "deployment": "mock", "endpoint": None, "context_window": 1000000,
               "capabilities": {"supported_parameters": ["max_output_tokens"], "supports_system_role": True},
               "runtime": {"structured_output": {"engine": "xgrammar", "version": "0.2.8", "channel": "json"}},
               "status": "frozen"}
    payload = _payload(p, group)
    profile["runtime"]["mock_text"] = json.dumps(payload)
    parent = {"kind": "classification", "paper_id": p["paper_id"], "task_sha256": task["task_sha256"],
              "profile_sha256": profile_hash(profile), "classification_group": group,
              "parameters": {"max_output_tokens": 1000}}
    parent["job_id"] = digest(parent)
    record = _attempt(parent, profile, task, p, None, None, None, 1, allow_live=False,
                      transport=None,
                      counter=None, classification_group=group)
    assert record["status"] == "success", record["backend_result"]["errors"]
    assert record["backend_result"]["request"]["structured_output"]["schema"] == schema
    assert replay_run(record, task, p, classification_group=group) == []


def test_new_task_builds_replayable_partial_navigation_index():
    p, task = paper(), make_task()
    policy = {"max_units": 2, "max_target_chars": 30}
    groups = plan_groups(p, policy)
    profile = {"backend": "mock", "profile_id": "fixture", "model_id": "fixture", "revision": None,
               "deployment": "mock", "endpoint": None, "context_window": 1000000,
               "capabilities": {"supported_parameters": ["max_output_tokens"], "supports_system_role": True},
               "runtime": {"structured_output": {"engine": "xgrammar", "version": "0.2.8", "channel": "json"}},
               "status": "frozen"}
    parent = {"kind": "classification", "paper_id": p["paper_id"], "source_sha256": "source-sha",
              "task_sha256": task["task_sha256"], "profile_id": profile["profile_id"],
              "profile_sha256": profile_hash(profile), "parameters": {"max_output_tokens": 1000},
              "grouping": policy, "dependencies": []}
    parent["job_id"] = digest(parent)
    records = []
    for group in groups:
        payload = _payload(p, group)
        from high_fidelity_schema_study.four_category.structured_output import applied_control
        def transport(request, payload=payload):
            return {"text": json.dumps(payload), "model": request["model"], "finish_reason": "stop",
                    "structured_output_applied": applied_control(request, synthetic=True)}
        records.append(_attempt(group_job(parent, group), profile, task, p, None, None, None, 1,
                                allow_live=False, transport=transport, counter=None,
                                classification_group=group))
    index = build_index_v4(p, task, groups, records)
    assert index["availability_counts"]["units_available"] == 4
    assert verify_index_v4(index, p) == []
