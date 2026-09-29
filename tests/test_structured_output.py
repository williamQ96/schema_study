from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from high_fidelity_schema_study.four_category import structured_output as so
from high_fidelity_schema_study.four_category.backends import backend_record_errors, invoke, request_for, validate_profile


SCHEMA = {"type": "object", "properties": {"category": {"enum": ["A", "B"]},
                                                "evidence": {"type": "string"}},
          "required": ["category", "evidence"], "additionalProperties": False}
MESSAGES = [{"role": "user", "content": "extract"}]


def profile(backend="mock", channel="json"):
    return {"profile_id": "test", "backend": backend, "model_id": "test-model",
            "revision": "commit" if backend == "transformers" else None,
            "deployment": "local" if backend == "transformers" else "mock",
            "endpoint": None, "context_window": 256,
            "capabilities": {"supported_parameters": ["max_output_tokens"], "supports_system_role": True},
            "runtime": {"structured_output": {"engine": "xgrammar", "version": "0.2.8", "channel": channel},
                        **({"response_decoder": "muse-atem/v1"} if channel.startswith("muse-atem/") else {})},
            "status": "frozen"}


def test_opt_in_and_fail_closed():
    p = profile()
    legacy = request_for(p, MESSAGES, {"max_output_tokens": 32})
    assert "structured_output" not in legacy
    req = request_for(p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA)
    assert req["structured_output"]["schema"] == SCHEMA
    assert req["structured_output"]["schema_sha256"] == so.schema_digest(SCHEMA)
    p["runtime"] = {}
    called = []
    result = invoke(p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA,
                    transport=lambda req: called.append(req))
    assert result["status"] == "invalid_request" and called == []
    p = profile()
    p["runtime"]["structured_output"]["version"] = "0.2.7"
    assert invoke(p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA,
                  transport=lambda req: called.append(req))["status"] == "invalid_request"
    p = profile()
    p.update(backend="chat_completions", deployment="remote")
    assert invoke(p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA,
                  allow_live=True, transport=lambda req: called.append(req))["status"] == "invalid_request"
    assert called == []
    p = profile()
    p["runtime"]["structured_output"]["channel"] = {}
    assert invoke(p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA)["status"] == "invalid_request"
    p = profile()
    assert invoke(p, MESSAGES, {"max_output_tokens": 32},
                  response_schema={"type": "not-a-json-schema-type"})["status"] == "invalid_request"


def test_muse_v2_budget_is_explicit_and_v1_is_unchanged():
    old = profile("transformers", "muse-atem/v1")
    old_req = request_for(old, MESSAGES, {"max_output_tokens": 16384}, response_schema=SCHEMA)
    assert set(old_req["structured_output"]) == {"engine", "version", "channel", "schema", "schema_sha256"}
    p = profile("transformers", "muse-atem/v2")
    p["runtime"]["structured_output"].update(reasoning_max_tokens=2048, final_min_tokens=8192,
                                                 max_whitespace_cnt=2)
    req = request_for(p, MESSAGES, {"max_output_tokens": 16384}, response_schema=SCHEMA)
    assert req["generation_parameters"]["max_new_tokens"] == 16384
    assert req["structured_output"]["reasoning_max_tokens"] == 2048
    assert req["structured_output"]["final_min_tokens"] == 8192
    assert so.compiler_schema(req["structured_output"]) == SCHEMA
    assert so.applied_control(req, synthetic=True)["grammar_schema_sha256"] == req["structured_output"]["grammar_schema_sha256"]
    assert validate_profile(p) == []
    with pytest.raises(ValueError, match="exceed max_output_tokens"):
        request_for(p, MESSAGES, {"max_output_tokens": 10240}, response_schema=SCHEMA)
    assert invoke(p, MESSAGES, {"max_output_tokens": 10240}, response_schema=SCHEMA,
                  allow_live=True, transport=lambda _: pytest.fail("must not dispatch"))["status"] == "invalid_request"


@pytest.mark.parametrize("key,value", [("reasoning_max_tokens", True), ("reasoning_max_tokens", float("inf")),
                                        ("reasoning_max_tokens", 0), ("final_min_tokens", False),
                                        ("final_min_tokens", 1.5)])
def test_muse_v2_rejects_invalid_budget_values(key, value):
    p = profile("transformers", "muse-atem/v2")
    p["runtime"]["structured_output"].update(reasoning_max_tokens=2048, final_min_tokens=8192,
                                                 max_whitespace_cnt=2)
    p["runtime"]["structured_output"][key] = value
    assert any(key in error for error in validate_profile(p))


def test_muse_v2_ordered_grammar_identity_and_replay():
    p = profile("transformers", "muse-atem/v2")
    p["runtime"]["structured_output"].update(reasoning_max_tokens=2, final_min_tokens=10,
                                                 max_whitespace_cnt=2)
    req = request_for(p, MESSAGES, {"max_output_tokens": 100}, response_schema=SCHEMA)
    reordered = copy.deepcopy(SCHEMA)
    reordered["properties"] = {"evidence": SCHEMA["properties"]["evidence"],
                               "category": SCHEMA["properties"]["category"]}
    assert so.schema_digest(SCHEMA) == so.schema_digest(reordered)
    assert so.request_control(p, reordered)["grammar_schema_sha256"] != req["structured_output"]["grammar_schema_sha256"]
    result = invoke(p, MESSAGES, {"max_output_tokens": 100}, response_schema=SCHEMA,
                    allow_live=True, transport=lambda request: {
                        "text": "{}", "decoded_with_special_tokens":
                        ' to=user<|message|>{"category":"A","evidence":"x"}<|eot|>',
                        "model": "test-model", "structured_output_applied":
                        so.applied_control(request, synthetic=True)})
    assert result["status"] == "success"
    assert backend_record_errors(result, p, MESSAGES, {"max_output_tokens": 100}, response_schema=SCHEMA) == []
    tampered = copy.deepcopy(result)
    tampered["request"]["structured_output"]["grammar_schema_json"] = so.ordered_schema_json(reordered)
    assert "request_replay_mismatch" in backend_record_errors(tampered, p, MESSAGES,
                                                               {"max_output_tokens": 100}, response_schema=SCHEMA)
    with pytest.raises(ValueError, match="ordered grammar schema"):
        so.compiler_schema(tampered["request"]["structured_output"])


def test_muse_v2_channel_counts_and_final_identity():
    specials = ["<|message|>", "<|eot|>", "<|eom|>", "<|start|>"]
    stream = " to=self<|message|>xx<|eom|><|start|>assistant to=user<|message|>{}<|eot|>"
    vocab = {value: index for index, value in enumerate(specials)}
    vocab.update({value: index + len(specials) for index, value in enumerate(sorted(set("assistant to=selfuser{}x")))})
    class Tokenizer:
        def get_vocab(self):
            return vocab
        def decode(self, ids, skip_special_tokens=False):
            return "".join(next(key for key, value in vocab.items() if value == item) for item in ids)
    tok = Tokenizer()
    ids = []
    cursor = 0
    while cursor < len(stream):
        special = next((item for item in specials if stream.startswith(item, cursor)), None)
        piece = special or stream[cursor]
        ids.append(vocab[piece]); cursor += len(piece)
    usage = so.muse_channel_usage(ids, tok)
    assert usage == {"reasoning_tokens": 2, "final_tokens": 2,
                     "reasoning_terminated": True, "final_terminated": True,
                     "route": "reasoning_then_final"}


def test_json_v2_whitespace_is_opt_in_and_ordered():
    p = profile("transformers", "json/v2")
    p["runtime"]["structured_output"]["max_whitespace_cnt"] = 2
    req = request_for(p, MESSAGES, {"max_output_tokens": 100}, response_schema=SCHEMA)
    assert req["structured_output"]["max_whitespace_cnt"] == 2
    assert so.compiler_schema(req["structured_output"]) == SCHEMA
    assert "max_whitespace_cnt" in so.applied_control(req, synthetic=True)
    p["runtime"]["structured_output"]["max_whitespace_cnt"] = float("nan")
    assert any("max_whitespace_cnt" in error for error in validate_profile(p))


def test_muse_v2_grammar_bounds_only_reasoning_and_final_whitespace():
    specials = ["<|message|>", "<|eot|>", "<|eom|>", "<|start|>"]
    class Tokenizer:
        all_special_tokens = specials
        def get_vocab(self):
            return {value: index for index, value in enumerate(specials)}
    grammar = so.muse_format(SCHEMA, 2048, Tokenizer(), max_whitespace_cnt=2)
    direct, reasoning = grammar["format"]["elements"]
    assert direct["elements"][2]["max_whitespace_cnt"] == 2
    assert reasoning["elements"][2]["max_tokens"] == 2048
    assert reasoning["elements"][-2]["max_whitespace_cnt"] == 2
    old = so.muse_format(SCHEMA, 16384, Tokenizer())
    assert "max_whitespace_cnt" not in old["format"]["elements"][0]["elements"][2]


def test_mock_provenance_and_replay_tamper():
    p = profile()
    p["runtime"]["mock_text"] = '{"category":"A","evidence":"x"}'
    result = invoke(p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA)
    assert result["status"] == "success"
    assert result["structured_output_applied"]["synthetic"] is True
    assert backend_record_errors(result, p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA) == []
    changed = copy.deepcopy(result)
    changed["structured_output_applied"]["schema_sha256"] = "bad"
    assert "structured_output_applied_replay_mismatch" in backend_record_errors(
        changed, p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA)
    changed = copy.deepcopy(result)
    changed["request"]["structured_output"]["schema_sha256"] = "bad"
    assert "request_replay_mismatch" in backend_record_errors(
        changed, p, MESSAGES, {"max_output_tokens": 32}, response_schema=SCHEMA)


def test_injected_transformers_requires_applied_proof():
    p = profile("transformers")
    params = {"max_output_tokens": 32}
    missing = invoke(p, MESSAGES, params, response_schema=SCHEMA, allow_live=True,
                     transport=lambda req: {"text": "{}", "model": "test-model"})
    assert missing["status"] == "contract_invalid"
    applied = invoke(p, MESSAGES, params, response_schema=SCHEMA, allow_live=True,
                     transport=lambda req: {"text": "{}", "model": "test-model",
                                            "structured_output_applied": so.applied_control(req, synthetic=True)})
    assert applied["status"] == "success"
    assert backend_record_errors(applied, p, MESSAGES, params, response_schema=SCHEMA) == []
    def compile_failure(_):
        raise ValueError("structured output grammar compilation failed: unsupported schema")
    failed = invoke(p, MESSAGES, params, response_schema=SCHEMA, allow_live=True,
                    transport=compile_failure)
    assert failed["status"] == "invalid_request"
    assert failed["dispatch_started"] is True


def test_real_xgrammar_json_constraints():
    xgr = pytest.importorskip("xgrammar")
    vocab = [chr(i) for i in range(32, 127)]
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(vocab))
    grammar = compiler.compile_json_schema(SCHEMA, strict_mode=True, any_order=False)
    def accepts(value):
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        return matcher.accept_string(value) and matcher.is_terminated()
    assert accepts('{"category":"A","evidence":"x"}')
    for bad in ('{"category":"C","evidence":"x"}',
                '{"category":"A"}',
                '{"category":"A","category":"B"}',
                '{"category":"A","evidence":"x"'):
        assert not accepts(bad), bad


def test_real_xgrammar_muse_wrapper_compiles():
    xgr = pytest.importorskip("xgrammar")
    specials = ["<|message|>", "<|eot|>", "<|eom|>", "<|start|>", "<|tool|>"]
    vocab = [chr(i) for i in range(32, 127)] + specials
    class Tokenizer:
        all_special_tokens = specials
        def get_vocab(self):
            return {item: index for index, item in enumerate(vocab)}
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(vocab, stop_token_ids=[]))
    grammar = compiler.compile_structural_tag(so.muse_format(SCHEMA, 16, Tokenizer()))
    indices = {value: index for index, value in enumerate(vocab)}
    def accepts(parts):
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        for part in parts:
            tokens = [part] if part in specials else list(part)
            for token in tokens:
                if not matcher.accept_token(indices[token]):
                    return False
        return matcher.is_terminated()
    final = [' to=user', '<|message|>', '{"category":"A","evidence":"x"}', '<|eot|>']
    assert accepts(final)
    assert accepts([' to=self', '<|message|>', 'reason', '<|eom|>', '<|start|>', 'assistant', *final])
    assert not accepts([' to=tool', '<|message|>', '{}', '<|eot|>'])
    assert not accepts([' to=self', '<|message|>', '<|tool|>', '<|eom|>', '<|start|>', 'assistant', *final])


def test_real_xgrammar_dynamic_extraction_schema():
    xgr = pytest.importorskip("xgrammar")
    schema = json.loads((Path(__file__).parents[1] / "templates" /
                         "paper_extraction_observations_v5.schema.json").read_text())
    coverage = schema["properties"]["coverage"]
    coverage["minItems"] = coverage["maxItems"] = 1
    coverage["prefixItems"] = [dict(coverage["items"], properties={
        **coverage["items"]["properties"], "window_id": {"const": "window-1"}})]
    grammar = xgr.GrammarCompiler(xgr.TokenizerInfo([chr(i) for i in range(32, 127)])).compile_json_schema(
        schema, strict_mode=True, any_order=False)
    def accepts(value):
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        return matcher.accept_string(json.dumps(value, separators=(",", ":"))) and matcher.is_terminated()
    value = {"schema_version": "paper-extraction-group-response/v1",
             "coverage": [{"window_id": "window-1", "state": "reviewed"}],
             "mentions": [], "facts": []}
    assert accepts(value)
    value["coverage"][0]["window_id"] = "wrong-window"
    assert not accepts(value)


def test_real_hf_tokenizer_muse_stop_reconstruction():
    xgr = pytest.importorskip("xgrammar")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    specials = ["[UNK]", "<|message|>", "<|eot|>", "<|eom|>", "<|start|>"]
    pieces = [chr(i) for i in range(32, 127)] + specials
    backend = Tokenizer(models.WordLevel({piece: i for i, piece in enumerate(pieces)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        eos_token="<|eot|>", additional_special_tokens=specials[1:])
    original = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(pieces),
                                                   stop_token_ids=[tokenizer.eos_token_id])
    assert original.stop_token_ids == [tokenizer.eos_token_id]
    info = so._tokenizer_info(xgr, tokenizer, len(pieces), {tokenizer.eos_token_id}, muse=True)
    assert info.stop_token_ids == []
    assert info.vocab_type == original.vocab_type
    assert info.add_prefix_space == original.add_prefix_space
    grammar = xgr.GrammarCompiler(info).compile_structural_tag(so.muse_format(SCHEMA, 16, tokenizer))
    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    output = ' to=user<|message|>{"category":"A","evidence":"x"}<|eot|>'
    cursor = 0
    vocab = tokenizer.get_vocab()
    while cursor < len(output):
        special = next((part for part in specials if output.startswith(part, cursor)), None)
        piece = special or output[cursor]
        assert matcher.accept_token(vocab[piece]), piece
        cursor += len(piece)
    assert matcher.is_terminated()
