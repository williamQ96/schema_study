"""The repaired V13 extraction task follows production, replay and artifact routes."""
from __future__ import annotations

import copy

from high_fidelity_schema_study.four_category import extraction_v13, extraction_v13r2
from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import digest, read_json
from high_fidelity_schema_study.four_category.disabled_navigation import make_disabled_index
from high_fidelity_schema_study.four_category.extraction_protocol import module_for_task
from high_fidelity_schema_study.four_category.grouped_extraction import group_job, mock_group_transport
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category.paper import fact_view
from high_fidelity_schema_study.four_category.tasks import task_errors
from high_fidelity_schema_study.four_category.workflow import (
    _attempt, _grouped_extraction, experiment_errors, production_request,
    replay_run, tasks_for_config,
)


def test_new_protocol_routes_and_old_task_remains_stable(tmp_path):
    config, _ = make_fixture(tmp_path / "sources")
    old = copy.deepcopy(config)
    old["extraction_input_protocol"] = extraction_v13.PROTOCOL
    old_task = tasks_for_config(old)["extraction"]
    config["classification"].update(protocol="classification-anchors/v13r2",
                                    grouping={"max_units": 32, "max_target_chars": 16000})
    config["extraction_input_protocol"] = extraction_v13r2.PROTOCOL
    config["extraction_grouping"] = copy.deepcopy(extraction_v13r2.DEFAULT_POLICY)
    config["execution"]["grouped_mode"] = "all-groups/v1"
    assert experiment_errors(config) == []
    task = tasks_for_config(config)["extraction"]
    assert module_for_task(task) is extraction_v13r2
    assert _grouped_extraction(task)
    assert task_errors(task) == []
    assert old_task == extraction_v13.make_task(old_task["taxonomy"])
    assert old_task["task_sha256"] != task["task_sha256"]


def test_new_request_and_mock_attempt_replay(tmp_path):
    root = tmp_path / "sources"
    config, _ = make_fixture(root)
    paper = read_json(root / "paper_input.json")
    index = make_disabled_index(paper)
    task = extraction_v13r2.make_task()
    group = extraction_v13r2.plan_groups(paper)[0]
    messages, schema = production_request(task, paper, index=index, extraction_group=group)
    assert messages == extraction_v13r2.render_group(task, paper, index, group)
    assert schema == extraction_v13r2.response_schema(task, paper, index, group)
    profile = copy.deepcopy(config["profiles"][0])
    profile["runtime"]["structured_output"] = {"engine": "xgrammar", "version": "0.2.8", "channel": "json"}
    parent = {"kind": "extraction", "paper_id": paper["paper_id"],
              "profile_id": profile["profile_id"], "profile_sha256": profile_hash(profile),
              "task_sha256": task["task_sha256"], "parameters": {"max_output_tokens": 4096}}
    parent["job_id"] = digest(parent)
    record = _attempt(group_job(parent, group, index), profile, task, paper, index, None, None, 1,
                      allow_live=False, transport=mock_group_transport, counter=None,
                      extraction_group=group)
    assert record["backend_result"]["request"]["structured_output"]["schema"] == schema
    assert replay_run(record, task, paper, index=index, extraction_group=group) == []


def test_new_bundle_version_projects_facts_for_downstream_review():
    fact = {"subject_mention_id": "m0", "claim": {"predicate": "datatype",
            "assertion": {"status": "reported", "value": "float", "basis": None}}}
    rows = fact_view({"schema_version": "paper-derived-observations/v13r2", "facts": [fact]})
    assert rows[0]["subject_id"] == "m0"
    assert rows[0]["predicate"] == "datatype"
