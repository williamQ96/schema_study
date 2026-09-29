"""Actual XGrammar matcher checks for the v7 pointer contract, without inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from high_fidelity_schema_study.four_category.structured_output import muse_format
from high_fidelity_schema_study.four_category import extraction_v7
from high_fidelity_schema_study.four_category.common import ROOT


SPECIALS = ["<|message|>", "<|eot|>", "<|eom|>", "<|start|>", "<|tool|>"]
VOCAB = [chr(i) for i in range(32, 127)] + SPECIALS


def schema_for_windows(monkeypatch):
    static = json.loads((ROOT / "templates/paper_extraction_observations_v7.schema.json").read_text())
    monkeypatch.setattr(extraction_v7, "task_errors", lambda task: [])
    monkeypatch.setattr(extraction_v7, "group_errors", lambda group, paper: [])
    monkeypatch.setattr(extraction_v7, "verify_index", lambda index, paper, **kwargs: [])
    monkeypatch.setattr(extraction_v7, "window_catalog", lambda paper: [{}] * 6)
    schema = extraction_v7.response_schema(
        {"output_schema": static, "taxonomy": {}}, {}, {},
        {"window_ids": [4], "policy": {"max_mentions": 3, "max_facts": 4}})
    schema["properties"]["facts"]["minItems"] = 1
    return schema


def valid_response():
    return {"schema_version": "paper-extraction-group-response/v3",
            "coverage": [{"window_id": 4, "state": "reviewed"}],
            "mentions": [{"kind": "variable", "normalized_label": "temperature",
                          "source_windows": [2], "context": None}],
            "facts": [{"subject_mention": 0, "primary_window": 4, "support_windows": [],
                       "categories": ["structure"],
                       "claim": {"kind": "attribute", "predicate": "datatype",
                                 "assertion": {"status": "reported", "value": "float32", "basis": None}}}]}


def matcher_accepts(xgr, grammar, text):
    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    return matcher.accept_string(text) and matcher.is_terminated()


def test_v7_actual_grammar_accepts_valid_branches_and_rejects_bounds(monkeypatch):
    xgr = pytest.importorskip("xgrammar")
    schema = schema_for_windows(monkeypatch)
    grammar = xgr.GrammarCompiler(xgr.TokenizerInfo(VOCAB)).compile_json_schema(
        schema, strict_mode=True, any_order=False)
    base = valid_response()
    def accepts(value):
        return matcher_accepts(xgr, grammar, json.dumps(value, separators=(",", ":")))
    assert accepts(base)
    inferred = copy.deepcopy(base)
    inferred["facts"][0]["claim"]["assertion"] = {"status": "inferred", "value": "float32", "basis": "derived"}
    assert accepts(inferred)
    unknown = copy.deepcopy(base)
    unknown["facts"][0]["claim"]["assertion"] = {"status": "unknown", "value": None, "basis": None}
    assert accepts(unknown)
    link = copy.deepcopy(base)
    link["facts"][0]["claim"] = {"kind": "link", "predicate": "parent",
                                  "target": {"kind": "group", "normalized_label": "root", "source_windows": [1]},
                                  "assertion": {"status": "reported", "value": "member", "basis": None}}
    assert accepts(link)

    for change in (
        lambda x: x["facts"][0].update(primary_window=0),  # Global window, outside this target group.
        lambda x: x["facts"][0].update(primary_window=99),
        lambda x: x["coverage"][0].update(window_id=99),
        lambda x: x["facts"][0].update(support_windows=[99]),
        lambda x: x["mentions"][0].update(source_windows=[99]),
        lambda x: x["mentions"][0].update(source_windows=[]),
        lambda x: x["facts"][0]["claim"].update(predicate="parent"),
    ):
        changed = copy.deepcopy(base); change(changed)
        assert not accepts(changed)
    changed = copy.deepcopy(link)
    changed["facts"][0]["claim"]["target"]["source_windows"] = [99]
    assert not accepts(changed)
    changed = copy.deepcopy(link); del changed["facts"][0]["claim"]["target"]
    assert not accepts(changed)
    for status, value, basis in (("reported", "x", "unsupported"), ("reported", "", None),
                                 ("inferred", "x", None), ("inferred", "", "reason"),
                                 ("unknown", "x", None)):
        changed = copy.deepcopy(base)
        changed["facts"][0]["claim"]["assertion"] = {"status": status, "value": value, "basis": basis}
        assert not accepts(changed), (status, value, basis)
    # uniqueItems and subject-reference validity are independent admission checks.
    duplicate = copy.deepcopy(base)
    duplicate["mentions"][0]["source_windows"] = [2, 2]
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(duplicate))
    dangling = copy.deepcopy(base)
    dangling["facts"][0]["subject_mention"] = 2
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(dangling)) == []


def test_v7_muse_wrapper_accepts_direct_and_reasoning_routes(monkeypatch):
    xgr = pytest.importorskip("xgrammar")
    schema = schema_for_windows(monkeypatch)
    class Tokenizer:
        all_special_tokens = SPECIALS
        def get_vocab(self):
            return {piece: index for index, piece in enumerate(VOCAB)}
    vocab = Tokenizer().get_vocab()
    grammar = xgr.GrammarCompiler(xgr.TokenizerInfo(VOCAB, stop_token_ids=[])).compile_structural_tag(
        muse_format(schema, 16, Tokenizer()))
    body = json.dumps(valid_response(), separators=(",", ":"))
    def accepts(parts):
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        for part in parts:
            for piece in ([part] if part in SPECIALS else list(part)):
                if not matcher.accept_token(vocab[piece]):
                    return False
        return matcher.is_terminated()
    final = [" to=user", "<|message|>", body, "<|eot|>"]
    assert accepts(final)
    assert accepts([" to=self", "<|message|>", "reason", "<|eom|>", "<|start|>", "assistant", *final])
    assert not accepts([" to=tool", "<|message|>", body, "<|eot|>"])
    assert not accepts([" to=self", "<|message|>", "<|tool|>", "<|eom|>", "<|start|>", "assistant", *final])
