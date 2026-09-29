"""Pinned XGrammar checks for the compact v6 fact contract, without inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from high_fidelity_schema_study.four_category.structured_output import muse_format
from high_fidelity_schema_study.four_category import extraction_v6


ROOT = Path(__file__).parents[1]
SPECIALS = ["<|message|>", "<|eot|>", "<|eom|>", "<|start|>", "<|tool|>"]
VOCAB = [chr(i) for i in range(32, 127)] + SPECIALS


def schema_for_windows(monkeypatch):
    static = json.loads((ROOT / "templates/paper_extraction_observations_v6.schema.json").read_text())
    monkeypatch.setattr(extraction_v6, "task_errors", lambda task: [])
    monkeypatch.setattr(extraction_v6, "group_errors", lambda group, paper: [])
    monkeypatch.setattr(extraction_v6, "verify_index", lambda index, paper, **kwargs: [])
    monkeypatch.setattr(extraction_v6, "window_catalog", lambda paper: [{}] * 6)
    monkeypatch.setattr(extraction_v6, "unit_catalog", lambda paper: {str(i): {} for i in range(6)})
    schema = extraction_v6.response_schema(
        {"output_schema": static, "taxonomy": {}}, {}, {},
        {"window_ids": [4], "policy": {"max_mentions": 3, "max_facts": 4}})
    # Bound the test response to one fact while retaining the real dynamic
    # group window, catalog range, and claim constraints.
    schema["properties"]["facts"]["minItems"] = 1
    return schema


def valid_response():
    return {"schema_version": "paper-extraction-group-response/v2",
            "coverage": [{"window_id": 4, "state": "reviewed"}],
            "mentions": [{"kind": "variable", "normalized_label": "temperature",
                          "source_name": {"unit_index": 2, "surface": "temperature", "occurrence": 0},
                          "context": None}],
            "facts": [{"subject_mention": 0, "primary_window": 4, "support_windows": [],
                       "categories": ["structure"],
                       "claim": {"kind": "attribute", "predicate": "datatype",
                                 "assertion": {"status": "reported", "value": "float32", "basis": None}}}]}


def matcher_accepts(xgr, grammar, text):
    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    return matcher.accept_string(text) and matcher.is_terminated()


def test_v6_actual_grammar_rejects_cross_field_and_window_errors(monkeypatch):
    xgr = pytest.importorskip("xgrammar")
    grammar = xgr.GrammarCompiler(xgr.TokenizerInfo(VOCAB)).compile_json_schema(
        schema_for_windows(monkeypatch), strict_mode=True, any_order=False)
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
                                   "target": {"kind": "group", "normalized_label": "root",
                                              "source_name": {"unit_index": 1, "surface": "root", "occurrence": 0}},
                                   "assertion": {"status": "reported", "value": "member", "basis": None}}
    assert accepts(link)

    changed = copy.deepcopy(base)
    changed["facts"][0]["primary_window"] = 99
    assert not accepts(changed)
    changed = copy.deepcopy(base)
    changed["coverage"][0]["window_id"] = 99
    assert not accepts(changed)
    changed = copy.deepcopy(base)
    changed["facts"][0]["support_windows"] = [99]
    assert not accepts(changed)
    changed = copy.deepcopy(base)
    changed["mentions"][0]["source_name"]["unit_index"] = 99
    assert not accepts(changed)
    changed = copy.deepcopy(base)
    changed["facts"][0]["claim"]["predicate"] = "parent"
    assert not accepts(changed)
    changed = copy.deepcopy(link)
    del changed["facts"][0]["claim"]["target"]
    assert not accepts(changed)
    for status, value, basis in (("reported", "x", "unsupported"),
                                 ("reported", "", None),
                                 ("inferred", "x", None),
                                 ("inferred", "", "reason"),
                                 ("unknown", "x", None)):
        changed = copy.deepcopy(base)
        changed["facts"][0]["claim"]["assertion"] = {"status": status, "value": value, "basis": basis}
        assert not accepts(changed), (status, value, basis)


def test_v6_muse_wrapper_accepts_only_final_schema_routes(monkeypatch):
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
