"""A verified V13r2 classifier index feeds the V13r2 extraction packet path."""
from __future__ import annotations

import copy
import json

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.classification_v13r2 import make_task as classifier_task
from high_fidelity_schema_study.four_category.complete_groups import execute_complete, verify_complete
from high_fidelity_schema_study.four_category.common import digest, read_json, write_new
from high_fidelity_schema_study.four_category.execution_artifacts import build_artifact
from high_fidelity_schema_study.four_category.extraction_v13r2 import (
    DEFAULT_POLICY, make_task as extraction_task, verify_bundle,
)
from high_fidelity_schema_study.four_category.grouped_classification import mock_group_transport as classifier_mock
from high_fidelity_schema_study.four_category.grouped_extraction import mock_group_transport as extraction_mock
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category.paper import verify_index
from high_fidelity_schema_study.four_category.structured_output import applied_control


def _paper(root):
    make_fixture(root)
    paper = read_json(root / "paper_input.json")
    next_page = copy.deepcopy(paper["pages"][0])
    next_page["page"] = 2
    unit = next_page["text_regions"][0]
    unit["unit_id"] = "v3-p0002-r0001"
    unit["page"] = 2
    unit["document_char_span"] = {"start": 213, "end": 213 + len(unit["text"])}
    unit["document_token_span"] = {"start": 35, "end": 70}
    paper["pages"].append(next_page)
    return paper


def _profile():
    return {"backend": "mock", "profile_id": "synthetic", "model_id": "synthetic",
            "revision": None, "deployment": "mock", "endpoint": None,
            "context_window": 1000000,
            "capabilities": {"supported_parameters": ["max_output_tokens"], "supports_system_role": True},
            "runtime": {"structured_output": {"engine": "xgrammar", "version": "0.2.8", "channel": "json"}},
            "status": "frozen"}


def _job(kind, paper, profile, task):
    job = {"kind": kind, "paper_id": paper["paper_id"], "source_sha256": "synthetic-source",
           "profile_id": profile["profile_id"], "profile_sha256": profile_hash(profile),
           "task_sha256": task["task_sha256"], "parameters": {"max_output_tokens": 4096},
           "group_execution": {"version": "all-groups/v1", "max_attempts": 1,
                               "retry_statuses": ["transport_error"]}}
    job["job_id"] = digest(job)
    return job


def test_classifier_index_is_published_and_consumed_by_complete_and_partial_extraction(tmp_path):
    paper = _paper(tmp_path / "fixture")
    profile = _profile()
    classification = classifier_task()
    extraction = extraction_task()
    classification_policy = {"max_units": 1, "max_target_chars": 1000}
    classification_job = _job("classification", paper, profile, classification)

    def classify(request):
        return {**classifier_mock(request),
                "structured_output_applied": applied_control(request, synthetic=True)}

    classifier_root = tmp_path / "classifier_packet"
    classifier_body = execute_complete(classification_job, profile, classification, paper, None,
                                       classification_policy, 1, classifier_root,
                                       allow_live=False, transport=classify)
    classifier_errors, classifier_records = verify_complete(
        classification_job, profile, classification, paper, None, classification_policy, 1,
        classifier_body, classifier_root)
    assert classifier_errors == []
    assert classifier_body["status"] == "success"
    index = build_artifact(classification_job, profile, classification, paper, None,
                           classification_policy, classifier_records)
    assert index["schema_version"] == "paper-category-index/v4"
    assert index["task"]["admission_rules_version"] == "classification-anchors/v13r2"
    assert index["availability_counts"]["units_available"] == 2
    assert verify_index(index, paper) == []
    published = tmp_path / "published_index.json"
    write_new(published, index)
    verified_index = read_json(published)
    assert verify_index(verified_index, paper) == []

    extraction_job = _job("extraction", paper, profile, extraction)
    policy = copy.deepcopy(DEFAULT_POLICY)
    seen_indexes = []

    def extract(request):
        body = json.loads(request["messages"][1]["content"])
        seen_indexes.append(body["navigation_context"]["index_sha256"])
        assert body["navigation_context"]["scope"] == "target_plus_adjacent_pdf_pages/v1"
        assert {row["availability"] for row in body["navigation_context"]["groups"]} == {"available"}
        assert {row["state"] for row in body["navigation_context"]["groups"]} == {"classified"}
        assert all(row["categories"] == ["structure"] for row in body["navigation_context"]["groups"])
        return extraction_mock(request)

    complete_root = tmp_path / "extraction_complete_packet"
    complete_body = execute_complete(extraction_job, profile, extraction, paper, verified_index,
                                     policy, 1, complete_root, allow_live=False, transport=extract)
    complete_errors, complete_records = verify_complete(
        extraction_job, profile, extraction, paper, verified_index, policy, 1,
        complete_body, complete_root)
    assert complete_errors == []
    assert complete_body["status"] == "success"
    assert seen_indexes == [index["index_sha256"]] * 2
    assert all(record["index_sha256"] == index["index_sha256"] for record in complete_records)
    bundle = build_artifact(extraction_job, profile, extraction, paper, verified_index,
                            policy, complete_records)
    assert bundle["schema_version"] == "paper-derived-observations/v13r2"
    assert verify_bundle(bundle, paper, verified_index, extraction, complete_records) == []

    failed_once = False

    def partly_rejected(request):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            return {"text": "{broken}", "model": request["model"], "finish_reason": "stop",
                    "structured_output_applied": applied_control(request, synthetic=True)}
        return extraction_mock(request)

    partial_root = tmp_path / "extraction_partial_packet"
    partial_body = execute_complete(extraction_job, profile, extraction, paper, verified_index,
                                    policy, 1, partial_root, allow_live=False, transport=partly_rejected)
    partial_errors, partial_records = verify_complete(
        extraction_job, profile, extraction, paper, verified_index, policy, 1,
        partial_body, partial_root)
    assert partial_errors == []
    assert partial_body["status"] == "completed_with_rejections"
    partial = build_artifact(extraction_job, profile, extraction, paper, verified_index,
                             policy, partial_records)
    assert partial["schema_version"] == "paper-partial-observations/v1"
    assert partial["admitted_group_count"] == 1
    assert partial["excluded_groups"][0]["status"] == "contract_invalid"
