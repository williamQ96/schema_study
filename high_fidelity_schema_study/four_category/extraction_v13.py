"""Scheduler-facing, complete-page execution of the V12 extraction contract.

V12's six-page diagnostic remains immutable.  This version binds each complete
PDF page to a resumable group and preserves V12 response semantics.
"""
from __future__ import annotations

import copy
import json

from . import extraction_v10 as v10, extraction_v12 as v12
from .compact_prompt_v13 import compact_messages_ordinal
from .common import canonical_bytes, digest, seal, seal_errors
from .paper import source_identity

VERSION = "four-category-task/v13"
PROTOCOL = "extraction-page-regions/v13"
BUNDLE_VERSION = "paper-derived-observations/v13"
PROMPT_PROJECTION = "paper-extraction-numbered-source/v13-compact"
COMPACT_GUIDANCE = """
The full paper is provided as a compact numbered-window view. Its units are
ordered rows U0..U(n-1); page_runs assign page numbers and source_kind_dictionary
decodes unit roles. Each window retains its exact numeric ID and source text.
String text is literal; [T,n] selects text_dictionary[n], while
[W,window,start,end] is the exact Unicode substring of another numbered
window. The ordered original unit IDs and exact coordinates are bound in the
verified source artifact; do not invent either. Target table unit_indexes point
to rows in units. Feature candidates retain their exact unit IDs and give
unit_index/row_unit_index links. Target anchors contain
[anchor_id,window_id,quote_start,quote_end]; choose an anchor ID whose exact
substring supports the fact. The target_schema is an explanatory guide; the
full response schema is enforced by structured decoding. Preserve all assigned
window coverage and use the named nested-object response shape.
"""
DEFAULT_POLICY = {"max_windows": 512, "max_target_chars": 60000,
                  "max_mentions": 128, "max_facts": 256}


def make_task(taxonomy_value=None):
    base = v10.make_task(taxonomy_value)
    probe = v12.make_task()
    return seal({"kind": "extraction", "schema_version": VERSION,
                 "admission_rules_version": PROTOCOL,
                 "system_prompt": probe["system_prompt"] + COMPACT_GUIDANCE,
                 "output_schema": base["output_schema"], "taxonomy": base["taxonomy"],
                 "v12_contract_sha256": probe["task_sha256"],
                 "grouping": "complete_pdf_pages/v1",
                 "prompt_projection": PROMPT_PROJECTION}, "task_sha256")


def task_errors(task):
    try:
        return [] if task == make_task(task["taxonomy"]) else ["v13_task_mismatch"]
    except (KeyError, TypeError, ValueError):
        return ["invalid_v13_task"]


def _policy(policy):
    value = copy.deepcopy(DEFAULT_POLICY if policy is None else policy)
    if (not isinstance(value, dict) or set(value) != set(DEFAULT_POLICY)
            or any(type(v) is not int or v <= 0 for v in value.values())):
        raise ValueError("invalid_v13_group_policy")
    return value


def _region(group):
    return {k: copy.deepcopy(group[k]) for k in
            ("schema_version", "pages", "source_pdf_sha256",
             "paper_input_canonical_sha256", "window_ids", "limits", "group_sha256")}


def plan_groups(paper_input, policy=None):
    limits = _policy(policy)
    groups = []
    for i, page in enumerate(paper_input["pages"]):
        region = v12.plan_region(paper_input, [page["page"]],
                                 max_windows=limits["max_windows"],
                                 max_target_chars=limits["max_target_chars"],
                                 max_mentions=limits["max_mentions"],
                                 max_facts=limits["max_facts"])
        groups.append({**region, "index": i, "policy": copy.deepcopy(limits),
                       "group_id": digest({"version": PROTOCOL,
                                           "region_sha256": region["group_sha256"],
                                           "index": i})})
    return groups


def _check(task, paper_input, group):
    if task_errors(task):
        raise ValueError("v13_task_invalid")
    groups = plan_groups(paper_input, group["policy"])
    if group not in groups:
        raise ValueError("v13_group_derivation_mismatch")


def response_schema(task, paper_input, index, group):
    _check(task, paper_input, group)
    return v12.response_schema(v12.make_task(), paper_input, index, _region(group))


def render_group(task, paper_input, index, group):
    _check(task, paper_input, group)
    messages = v12.render_group(v12.make_task(), paper_input, index, _region(group))
    body = json.loads(messages[1]["content"])
    body["task_sha256"] = task["task_sha256"]
    messages[1]["content"] = canonical_bytes(body).decode("utf-8")
    messages[0]["content"] = task["system_prompt"]
    return compact_messages_ordinal(messages)


parse_response = v10.parse_response


def validate_response(payload, paper_input, index, group, task=None):
    _check(make_task() if task is None else task, paper_input, group)
    return v12.validate_response(payload, paper_input, index, _region(group))


def completion_status(payload, group, paper_input=None):
    if paper_input is None:
        raise ValueError("v13_paper_required_for_completeness")
    return "incomplete" if v12.completeness(payload, paper_input, _region(group)) else "success"


def materialize_group(paper_input, index, task, group, record):
    from .workflow import replay_run
    if replay_run(record, task, paper_input, index=index, extraction_group=group):
        raise ValueError("v13_record_replay_failed")
    if record["status"] != "success" or record["task"] != task or record["extraction_group"] != group:
        raise ValueError("v13_record_not_admitted")
    payload, normalization = parse_response(record["backend_result"]["raw_text"])
    if record["parsed_response"] != payload or record.get("response_normalization") != normalization:
        raise ValueError("v13_raw_replay_mismatch")
    if validate_response(payload, paper_input, index, group, task):
        raise ValueError("v13_response_invalid")
    if v12.completeness(payload, paper_input, _region(group)):
        raise ValueError("v13_response_incomplete")
    flat = v12.to_v10(copy.deepcopy(payload), paper_input, _region(group))
    prefix = group["group_id"]
    identity_prefix = record["job"]["parent_job_id"] + ":" + prefix
    coverage = [{"group_id": prefix, **copy.deepcopy(row)} for row in flat["coverage"]]
    mentions = [{"mention_id": identity_prefix + ":m" + str(i), "group_id": prefix,
                 **copy.deepcopy(mention),
                 "source_evidence": [v10._window_evidence(w, paper_input) for w in mention["source_windows"]],
                 "normalized_label_authority": "model_asserted_not_verified"}
                for i, mention in enumerate(flat["mentions"])]
    facts = []
    for i, fact in enumerate(flat["facts"]):
        row = copy.deepcopy(fact)
        row["subject_mention_id"] = identity_prefix + ":m" + str(row.pop("subject_mention"))
        facts.append({"fact_id": identity_prefix + ":f" + str(i), "group_id": prefix, **row,
                      "primary_evidence": v10._primary_quote_evidence(fact["primary_window"], fact["primary_quote"], paper_input),
                      "support_evidence": [v10._window_evidence(w, paper_input) for w in fact["support_windows"]],
                      "target_evidence": None if fact["claim"]["kind"] != "link" else
                      [v10._window_evidence(w, paper_input) for w in fact["claim"]["target"]["source_windows"]]})
    return {"group_id": prefix, "record_sha256": record["record_sha256"],
            "mentions": mentions, "facts": facts, "coverage": coverage}


def build_bundle(paper_input, index, task, groups, records):
    if groups != plan_groups(paper_input, groups[0]["policy"] if groups else None) or len(groups) != len(records):
        raise ValueError("v13_group_plan_mismatch")
    parts = [materialize_group(paper_input, index, task, g, r) for g, r in zip(groups, records)]
    bindings = {(r["profile_sha256"], r["job"]["parent_job_id"], r["index_sha256"]) for r in records}
    if len(bindings) > 1:
        raise ValueError("v13_cross_record_binding_mismatch")
    binding = next(iter(bindings), (None, None, None))
    return seal({"schema_version": BUNDLE_VERSION, "source_identity": source_identity(paper_input),
                 "paper_input_canonical_sha256": digest(paper_input), "index_sha256": index["index_sha256"],
                 "task_sha256": task["task_sha256"], "groups": copy.deepcopy(groups),
                 "profile_sha256": binding[0], "parent_job_id": binding[1],
                 "record_refs": [{"group_id": p["group_id"], "run_id": r["run_id"],
                                  "record_sha256": p["record_sha256"]} for p, r in zip(parts, records)],
                 "coverage": [x for p in parts for x in p["coverage"]],
                 "mentions": [x for p in parts for x in p["mentions"]],
                 "facts": [x for p in parts for x in p["facts"]],
                 "observation_count": sum(len(p["facts"]) for p in parts),
                 "mention_record_count": sum(len(p["mentions"]) for p in parts),
                 "resolved_object_count": None, "semantic_correctness": "not_established",
                 "identity_resolution": "unresolved_model_hypotheses_only",
                 "label_evidence_scope": "cited_window_context_not_name_equivalence_or_entailment"},
                "bundle_sha256")


def verify_bundle(bundle, paper_input, index, task, records=None):
    errors = seal_errors(bundle, "bundle_sha256")
    if records is None:
        return errors + ["v13_bundle_records_required_for_replay"]
    try:
        if bundle != build_bundle(paper_input, index, task, bundle["groups"], records):
            errors.append("v13_bundle_derivation_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("v13_bundle_replay:" + str(exc))
    return errors
