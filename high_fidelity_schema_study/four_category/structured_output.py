"""Opt-in, generation-time XGrammar constraints for local Transformers.

The native Muse reasoning channel shares the output-token allowance with its
final answer. A length finish can therefore leave a valid grammar prefix but
still fails the ordinary downstream response contract.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json

ENGINE = "xgrammar"
VERSION = "0.2.8"
CHANNELS = {"json", "json/v2", "muse-atem/v1", "muse-atem/v2"}
MUSE_V2 = "muse-atem/v2"
JSON_V2 = "json/v2"
ORDERED_CHANNELS = {MUSE_V2, JSON_V2}
# Conservative allowance for ATEM delimiters and channel headers on both routes.
MUSE_CHANNEL_OVERHEAD_TOKENS = 64


def ordered_schema_json(schema: dict) -> str:
    """The exact property insertion order supplied to any_order=False compilers."""
    return json.dumps(schema, sort_keys=False, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _v2_errors(control: dict, total_tokens: int | None = None) -> list[str]:
    errors = []
    whitespace = control.get("max_whitespace_cnt")
    if isinstance(whitespace, bool) or not isinstance(whitespace, int) or whitespace < 0:
        errors.append("runtime.structured_output.max_whitespace_cnt must be a nonnegative integer")
    if control.get("channel") != MUSE_V2:
        return errors
    for key in ("reasoning_max_tokens", "final_min_tokens"):
        value = control.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"runtime.structured_output.{key} must be a positive integer")
    if not errors and total_tokens is not None:
        if control["reasoning_max_tokens"] + control["final_min_tokens"] + MUSE_CHANNEL_OVERHEAD_TOKENS > total_tokens:
            errors.append("Muse v2 reasoning plus final reserve and channel overhead exceed max_output_tokens")
    return errors


def schema_digest(schema: dict) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def validate_control(profile: dict, schema: dict | None) -> list[str]:
    if schema is None:
        return []
    if not isinstance(schema, dict) or not schema:
        return ["response_schema must be a nonempty JSON Schema object"]
    try:
        schema_digest(schema)
    except (TypeError, ValueError):
        return ["response_schema must be finite JSON"]
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        return ["response_schema is not a valid draft 2020-12 schema: " + str(exc)]
    if profile.get("backend") not in {"transformers", "mock"}:
        return ["structured output is unsupported for this backend"]
    runtime = profile.get("runtime", {})
    control = runtime.get("structured_output") if isinstance(runtime, dict) else None
    if not isinstance(control, dict):
        return ["runtime.structured_output support must be declared"]
    expected = {"engine", "version", "channel"}
    if control.get("channel") == MUSE_V2:
        expected |= {"reasoning_max_tokens", "final_min_tokens"}
    if isinstance(control.get("channel"), str) and control["channel"] in ORDERED_CHANNELS:
        expected.add("max_whitespace_cnt")
    if set(control) != expected:
        return ["runtime.structured_output has missing or unsupported keys"]
    if (control["engine"] != ENGINE or control["version"] != VERSION or
            not isinstance(control["channel"], str) or control["channel"] not in CHANNELS):
        return ["unsupported structured output engine, version, or channel"]
    if control["channel"] == MUSE_V2 and profile.get("backend") != "transformers":
        return ["Muse v2 requires transformers backend"]
    if control["channel"] in ORDERED_CHANNELS:
        errors = _v2_errors(control)
        if errors:
            return errors
    if profile.get("backend") == "transformers" and control["channel"].startswith("muse-atem/") and runtime.get("response_decoder") != "muse-atem/v1":
        return ["Muse structured output requires runtime.response_decoder=muse-atem/v1"]
    if profile.get("backend") == "transformers" and control["channel"] in {"json", JSON_V2} and runtime.get("response_decoder") is not None:
        return ["JSON structured output conflicts with runtime.response_decoder"]
    return []


def request_control(profile: dict, schema: dict) -> dict:
    control = profile["runtime"]["structured_output"]
    requested = {**control, "schema": schema, "schema_sha256": schema_digest(schema)}
    if control["channel"] in ORDERED_CHANNELS:
        ordered = ordered_schema_json(schema)
        requested["grammar_schema_json"] = ordered
        requested["grammar_schema_sha256"] = hashlib.sha256(ordered.encode("utf-8")).hexdigest()
    return requested


def applied_control(req: dict, *, synthetic: bool) -> dict:
    control = req["structured_output"]
    keys = ["engine", "version", "channel", "schema_sha256"]
    if control["channel"] in ORDERED_CHANNELS:
        keys += ["max_whitespace_cnt", "grammar_schema_sha256"]
    if control["channel"] == MUSE_V2:
        keys += ["reasoning_max_tokens", "final_min_tokens"]
    return {key: control[key] for key in keys} | {"synthetic": synthetic}


def validate_request_budget(req: dict) -> list[str]:
    control = req.get("structured_output")
    if not isinstance(control, dict) or control.get("channel") != MUSE_V2:
        return []
    total = req.get("generation_parameters", {}).get("max_new_tokens")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        return ["Muse v2 requires positive max_new_tokens"]
    return _v2_errors(control, total)


def compiler_schema(control: dict) -> dict:
    if control["channel"] not in ORDERED_CHANNELS:
        return control["schema"]
    ordered = control.get("grammar_schema_json")
    if not isinstance(ordered, str) or hashlib.sha256(ordered.encode("utf-8")).hexdigest() != control.get("grammar_schema_sha256"):
        raise ValueError("ordered grammar schema digest mismatch")
    schema = json.loads(ordered)
    if ordered_schema_json(schema) != ordered or schema_digest(schema) != control["schema_sha256"]:
        raise ValueError("ordered grammar schema does not match requested schema")
    return schema


def muse_channel_usage(output_ids, tokenizer) -> dict:
    """Count generated content tokens within ATEM channel boundaries."""
    ids = [int(item) for item in output_ids]
    vocab = tokenizer.get_vocab()
    message = vocab["<|message|>"]
    eom = vocab["<|eom|>"]
    eot = vocab["<|eot|>"]
    first = ids.index(message) if message in ids else None
    reasoning = tokenizer.decode(ids, skip_special_tokens=False).startswith(" to=self<|message|>")
    if reasoning:
        reason_end = ids.index(eom, first + 1) if first is not None and eom in ids[first + 1:] else None
        second = ids.index(message, reason_end + 1) if reason_end is not None and message in ids[reason_end + 1:] else None
        final_start = second + 1 if second is not None else None
        reasoning_tokens = (reason_end if reason_end is not None else len(ids)) - (first + 1) if first is not None else None
    else:
        reason_end = None
        final_start = first + 1 if first is not None else None
        reasoning_tokens = 0
    final_end = ids.index(eot, final_start) if final_start is not None and eot in ids[final_start:] else None
    return {"reasoning_tokens": reasoning_tokens,
            "final_tokens": (final_end if final_end is not None else len(ids)) - final_start if final_start is not None else None,
            "reasoning_terminated": reason_end is not None if reasoning else True,
            "final_terminated": final_end is not None,
            "route": "reasoning_then_final" if reasoning else "direct_final"}


def validate_muse_channel_overhead(tokenizer) -> None:
    """Confirm the fixed request reserve covers the actual tokenizer headers."""
    if not hasattr(tokenizer, "encode"):
        raise ValueError("Muse tokenizer cannot measure channel delimiter allowance")
    # The reasoning route inserts five ATEM special tokens directly by ID.
    text_tokens = sum(len(tokenizer.encode(part, add_special_tokens=False)) for part in
                      (" to=self", "assistant", " to=user"))
    if text_tokens + 5 > MUSE_CHANNEL_OVERHEAD_TOKENS:
        raise ValueError("Muse channel delimiters exceed reserved token allowance")


def muse_format(schema: dict, reasoning_token_budget: int, tokenizer,
                max_whitespace_cnt: int | None = None) -> dict:
    """Exactly the two forms accepted by decode_muse_atem, using token IDs for delimiters."""
    required = ("<|message|>", "<|eot|>", "<|eom|>", "<|start|>")
    vocab = tokenizer.get_vocab()
    missing = [token for token in required if token not in vocab]
    if missing:
        raise ValueError("Muse tokenizer lacks ATEM special tokens: " + ", ".join(missing))
    token = lambda value: {"type": "token", "token": vocab[value]}
    const = lambda value: {"type": "const_string", "value": value}
    json_part = {"type": "json_schema", "json_schema": schema, "any_order": False}
    if max_whitespace_cnt is not None:
        json_part["max_whitespace_cnt"] = max_whitespace_cnt
    final = [const(" to=user"), token("<|message|>"), json_part, token("<|eot|>")]
    # Exclude all special tokens while reasoning, including tool/channel markers.
    excludes = sorted({vocab[value] for value in tokenizer.all_special_tokens if value in vocab})
    reasoning = [const(" to=self"), token("<|message|>"),
                 {"type": "any_tokens", "exclude_tokens": excludes,
                  "max_tokens": reasoning_token_budget},
                 token("<|eom|>"), token("<|start|>"), const("assistant"), *final]
    return {"type": "structural_tag", "format": {"type": "or", "elements": [
        {"type": "sequence", "elements": final},
        {"type": "sequence", "elements": reasoning},
    ]}}


def _tokenizer_info(xgr, tokenizer, vocab_size: int, eos_ids: set[int], *, muse: bool):
    """Keep HF tokenizer metadata while making Muse's final EOS grammatical."""
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                              stop_token_ids=sorted(eos_ids))
    if not muse:
        return info
    # from_huggingface rejects an empty stop list, but the public constructor
    # accepts it. Rebuild with the exact detected vocab type and prefix-space
    # metadata; this lets the grammar require <|eot|> as its final token.
    vocab = tokenizer.get_vocab()
    encoded_vocab = [""] * vocab_size
    for token, token_id in vocab.items():
        if 0 <= token_id < vocab_size:
            encoded_vocab[token_id] = token
    metadata_value = json.loads(info.dump_metadata())
    metadata_value["stop_token_ids"] = []
    rebuilt = xgr.TokenizerInfo.from_vocab_and_metadata(
        encoded_vocab, json.dumps(metadata_value, separators=(",", ":")))
    if rebuilt.stop_token_ids or rebuilt.vocab_size != info.vocab_size or rebuilt.vocab_type != info.vocab_type or rebuilt.add_prefix_space != info.add_prefix_space:
        raise ValueError("Muse tokenizer metadata reconstruction mismatch")
    return rebuilt


def logits_processor(req: dict, tokenizer, model):
    """Compile before generation; return (processor, observed provenance)."""
    try:
        import xgrammar as xgr
    except ImportError as exc:
        raise ValueError("xgrammar==0.2.8 is unavailable") from exc
    observed_version = metadata.version("xgrammar")
    if observed_version != VERSION:
        raise ValueError(f"xgrammar runtime version {observed_version} does not match required {VERSION}")
    control = req["structured_output"]
    if schema_digest(control["schema"]) != control["schema_sha256"]:
        raise ValueError("structured output schema digest mismatch")
    budget_errors = validate_request_budget(req)
    if budget_errors:
        raise ValueError("; ".join(budget_errors))
    schema = compiler_schema(control)
    if control["channel"] == MUSE_V2:
        validate_muse_channel_overhead(tokenizer)
    try:
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise ValueError("model output embedding size unavailable for structured decoding")
        vocab_size = int(output_embeddings.weight.shape[0])
        if vocab_size <= 0:
            raise ValueError("model output embedding size invalid for structured decoding")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if not eos_ids or None in eos_ids or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= vocab_size for item in eos_ids):
            raise ValueError("model generation EOS token IDs unavailable or invalid")
        if control["channel"].startswith("muse-atem/"):
            vocab = tokenizer.get_vocab()
            if vocab.get("<|eot|>") not in eos_ids:
                raise ValueError("Muse final <|eot|> is not a model generation stop token")
            if eos_ids.intersection({vocab.get("<|eom|>"), vocab.get("<|message|>"), vocab.get("<|start|>")}):
                raise ValueError("Muse internal ATEM token is a model generation stop token")
        info = _tokenizer_info(xgr, tokenizer, vocab_size, eos_ids,
                               muse=control["channel"].startswith("muse-atem/"))
        compiler = xgr.GrammarCompiler(info)
        if control["channel"] == "json":
            compiled = compiler.compile_json_schema(schema, strict_mode=True, any_order=False)
        elif control["channel"] == JSON_V2:
            compiled = compiler.compile_json_schema(schema, strict_mode=True, any_order=False,
                                                    max_whitespace_cnt=control["max_whitespace_cnt"])
        elif control["channel"].startswith("muse-atem/"):
            compiled = compiler.compile_structural_tag(muse_format(
                schema, control["reasoning_max_tokens"] if control["channel"] == MUSE_V2
                else req["generation_parameters"]["max_new_tokens"], tokenizer,
                max_whitespace_cnt=control["max_whitespace_cnt"] if control["channel"] == MUSE_V2 else None))
        else:
            raise ValueError("unsupported structured output channel")
    except Exception as exc:
        raise ValueError("structured output grammar compilation failed: " + str(exc)) from exc
    return xgr.contrib.hf.LogitsProcessor(compiled), applied_control(req, synthetic=False)
