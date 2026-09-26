from __future__ import annotations

import math
import copy
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

from high_fidelity_schema_study.four_category.backends import (
    backend_record_errors, invoke, preflight_parameters, profile_hash,
    replay_backend_result, request_for, validate_profile,
)
from high_fidelity_schema_study.four_category import backends as backend_module


MESSAGES = [{"role": "system", "content": "Extract facts."}, {"role": "user", "content": "Paper text"}]


def profile(backend="chat_completions", model="example-a"):
    supported = {
        "chat_completions": ["max_output_tokens", "temperature", "seed"],
        "responses": ["max_output_tokens", "reasoning_effort"],
        "transformers": ["max_output_tokens", "temperature", "top_k", "seed"],
        "mock": ["max_output_tokens"],
    }
    return {
        "profile_id": "role-a", "backend": backend, "model_id": model,
        "revision": "commit-1" if backend == "transformers" else None,
        "deployment": "local" if backend == "transformers" else "mock" if backend == "mock" else "remote",
        "endpoint": "https://example.invalid/v1/chat/completions" if backend == "chat_completions" else "https://example.invalid/v1/responses" if backend == "responses" else None,
        "context_window": 4096,
        "capabilities": {"supported_parameters": supported[backend], "supports_system_role": True},
        "runtime": {}, "status": "frozen",
    }


def test_swap_changes_only_model_and_hash():
    a = profile(model="example-a")
    b = profile(model="example-b")
    requests = []
    for item in (a, b):
        result = invoke(item, MESSAGES, {"max_output_tokens": 64}, allow_live=True,
                        transport=lambda req: requests.append(req) or {"model": req["body"]["model"], "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})
        assert result["status"] == "success"
    assert profile_hash(a) != profile_hash(b)
    assert requests[0]["body"]["messages"] == requests[1]["body"]["messages"] == MESSAGES
    assert {k: v for k, v in requests[0]["body"].items() if k != "model"} == {k: v for k, v in requests[1]["body"].items() if k != "model"}


def test_chat_and_responses_preserve_raw_and_normalize():
    chat = {"model": "a", "choices": [{"message": {"content": "{bad"}, "finish_reason": "stop"}], "usage": {"total_tokens": 9}}
    result = invoke(profile(), MESSAGES, {"max_output_tokens": 8}, allow_live=True, transport=lambda _: chat)
    assert result["raw_response"] is chat and result["raw_text"] == "{bad"
    assert result["request"]["body"]["max_tokens"] == 8
    assert result["sent_parameters"] == {"max_tokens": 8}
    assert result["effective_parameters"] is None
    response = {"model": "a", "status": "completed", "output": [{"content": [{"type": "output_text", "text": "{bad"}]}], "usage": {"input_tokens": 4}}
    result = invoke(profile("responses"), MESSAGES, {"max_output_tokens": 8, "reasoning_effort": "low"}, allow_live=True, transport=lambda _: response)
    assert result["raw_text"] == "{bad" and result["raw_response"] is response
    assert result["request"]["body"]["input"] == MESSAGES
    assert result["request"]["body"]["reasoning"] == {"effort": "low"}
    assert result["sent_parameters"] == {"max_output_tokens": 8, "reasoning": {"effort": "low"}}
    assert result["effective_parameters"] is None


def test_unsupported_controls_fail_before_transport():
    called = []
    for backend, parameters in (("responses", {"seed": 1}), ("chat_completions", {"top_k": 4})):
        result = invoke(profile(backend), MESSAGES, parameters, allow_live=True, transport=lambda req: called.append(req))
        assert result["status"] == "invalid_request"
        assert result["request"] is None
        assert result["dispatch_started"] is False
        assert result["live_request_started"] is False
    assert called == []


def test_no_accidental_live_call_and_profile_preflight():
    called = []
    item = profile()
    result = invoke(item, MESSAGES, {"max_output_tokens": 8}, transport=lambda req: called.append(req))
    assert result["status"] == "unavailable" and not called
    assert result["execution_mode"] == "injected_transport" and result["dispatch_started"] is False
    item["status"] = "qualified"
    result = invoke(item, MESSAGES, {"max_output_tokens": 8}, allow_live=True, transport=lambda req: called.append(req))
    assert result["status"] == "unavailable" and not called
    assert result["dispatch_started"] is False
    assert validate_profile(item, for_execution=True)


def test_timeout_rate_limit_truncation_and_refusal():
    def timeout(_):
        raise TimeoutError("deadline")
    result = invoke(profile(), MESSAGES, {"max_output_tokens": 8}, allow_live=True, transport=timeout)
    assert result["status"] == "transport_error" and result["errors"][0]["type"] == "TimeoutError"

    def rate_limit(_):
        raise HTTPError("https://example.invalid", 429, "rate limit", {}, None)
    result = invoke(profile(), MESSAGES, {"max_output_tokens": 8}, allow_live=True, transport=rate_limit)
    assert result["status"] == "transport_error" and result["errors"][0]["http_status"] == 429
    assert result["dispatch_started"] is True and result["live_request_started"] is False

    def rate_limit_with_body(_):
        headers = Message()
        headers["Retry-After"] = "3"
        headers["x-request-id"] = "req-429"
        raise HTTPError("https://example.invalid", 429, "rate limit", headers,
                        BytesIO(b'{"error":{"code":"rate_limit"}}'))
    result = invoke(profile(), MESSAGES, {"max_output_tokens": 8}, allow_live=True, transport=rate_limit_with_body)
    assert result["raw_response"] == {"error": {"code": "rate_limit"}}
    assert result["response_id"] == "req-429" and result["http_status"] == 429
    assert result["errors"][0]["retry_after"] == "3"
    assert replay_backend_result(profile(), result) == []

    result = invoke(profile(), MESSAGES, {"max_output_tokens": 8}, allow_live=True,
                    transport=lambda _: {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]})
    assert result["status"] == "truncated" and result["raw_text"] == "partial"
    result = invoke(profile("responses"), MESSAGES, {"max_output_tokens": 8}, allow_live=True,
                    transport=lambda _: {"status": "completed", "output": [{"content": [{"type": "refusal", "refusal": "cannot"}]}]})
    assert result["status"] == "refused" and result["raw_text"] is None


def test_transformers_parameter_mapping_with_fake_generation():
    item = profile("transformers")
    item["runtime"] = {"chat_template_kwargs": {"enable_thinking": False}, "quantization": {"load_in_4bit": True}}
    captured = []
    result = invoke(item, MESSAGES, {"max_output_tokens": 16, "top_k": 4, "seed": 7}, allow_live=True,
                    transport=lambda req: captured.append(req) or {"text": "{}", "model": "example-a", "finish_reason": "stop"})
    assert result["status"] == "success"
    assert captured[0]["generation_parameters"] == {"max_new_tokens": 16, "top_k": 4}
    assert captured[0]["seed"] == 7
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert result["sent_parameters"] == {"max_new_tokens": 16, "top_k": 4, "seed": 7}
    assert result["effective_parameters"] is None


def test_mock_runs_offline_and_does_not_repair_json():
    item = profile("mock")
    item["status"] = "draft"
    item["runtime"]["mock_text"] = "{broken"
    result = invoke(item, MESSAGES, {"max_output_tokens": 8}, allow_live=False)
    assert result["status"] == "success" and result["raw_text"] == "{broken"
    assert result["dispatch_started"] is True and result["execution_mode"] == "mock"
    assert preflight_parameters(item, {"seed": 1})


def test_preflight_rejects_bad_types_ranges_and_ignored_controls():
    item = profile("transformers")
    cases = [
        {}, {"max_output_tokens": 0}, {"max_output_tokens": True},
        {"max_output_tokens": 8, "temperature": math.nan},
        {"max_output_tokens": 8, "temperature": math.inf},
        {"max_output_tokens": 8, "top_k": -1},
        {"max_output_tokens": 8, "seed": -1},
        {"max_output_tokens": 8, "do_sample": "false"},
        {"max_output_tokens": 8, "do_sample": False, "top_k": 4},
    ]
    called = []
    for parameters in cases:
        result = invoke(item, MESSAGES, parameters, allow_live=True, transport=lambda req: called.append(req))
        assert result["status"] == "invalid_request", parameters
    item["runtime"]["ignored_parameters"] = ["top_k"]
    assert preflight_parameters(item, {"max_output_tokens": 8, "top_k": 4})
    assert not called


def test_chat_extensions_require_mapping_and_alias_is_explicit():
    item = profile()
    item["capabilities"]["supported_parameters"].extend(["top_k", "repetition_penalty"])
    assert preflight_parameters(item, {"max_output_tokens": 8, "top_k": 10})
    item["runtime"]["parameter_mapping"] = {
        "max_output_tokens": "max_completion_tokens", "top_k": "top_k", "repetition_penalty": "repetition_penalty"
    }
    captured = []
    result = invoke(item, MESSAGES, {"max_output_tokens": 8, "top_k": 10, "repetition_penalty": 1.1},
                    allow_live=True, transport=lambda req: captured.append(req) or {"choices": [{"message": {"content": "ok"}}]})
    assert result["status"] == "success"
    assert result["execution_mode"] == "injected_transport" and result["dispatch_started"] is True
    assert result["sent_parameters"] == {"max_completion_tokens": 8, "top_k": 10, "repetition_penalty": 1.1}
    assert result["effective_parameters"] is None
    assert captured[0]["body"]["max_completion_tokens"] == 8


def test_provider_metadata_and_errors_preserved():
    payload = {"id": "resp-123", "created": 42, "system_fingerprint": "fp", "model": "returned-model",
               "effective_parameters": {"max_tokens": 8},
               "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    result = invoke(profile(), MESSAGES, {"max_output_tokens": 8}, allow_live=True, transport=lambda _: payload)
    assert result["raw_response"] is payload
    assert (result["response_id"], result["created"], result["system_fingerprint"]) == ("resp-123", 42, "fp")
    assert result["returned_model"] == "returned-model"
    assert result["effective_parameters"] == {"max_tokens": 8}
    failure = {"id": "failed-1", "error": {"code": "invalid_request_error", "message": "bad field"}}
    result = invoke(profile(), MESSAGES, {"max_output_tokens": 8}, allow_live=True, transport=lambda _: failure)
    assert result["status"] == "invalid_request" and result["raw_response"] is failure
    assert result["response_id"] == "failed-1" and result["errors"] == [failure["error"]]


def test_transformer_fake_can_report_observed_config_and_runtime():
    payload = {"text": "x", "model": "example-a", "effective_parameters": {"do_sample": True},
               "generation_config": {"do_sample": False}, "runtime_identity": {"transformers": "test"}}
    result = invoke(profile("transformers"), MESSAGES, {"max_output_tokens": 8},
                    allow_live=True, transport=lambda _: payload)
    assert result["effective_parameters"] == {"do_sample": True}
    assert result["generation_config"] == {"do_sample": False}
    assert result["runtime_identity"] == {"transformers": "test"}


def test_replay_rejects_derived_text_status_and_request_edits():
    item = profile()
    params = {"max_output_tokens": 8}
    payload = {"id": "gen-1", "model": "example-a", "choices": [{"message": {"content": "original"}, "finish_reason": "stop"}]}
    result = invoke(item, MESSAGES, params, allow_live=True, transport=lambda _: payload)
    assert replay_backend_result(item, result) == []
    assert backend_record_errors(result, item, MESSAGES, params) == []
    assert request_for(item, MESSAGES, params) == result["request"]
    changed = copy.deepcopy(result)
    changed["raw_text"] = "edited"
    assert "raw_text_replay_mismatch" in replay_backend_result(item, changed)
    changed = copy.deepcopy(result)
    changed["status"] = "truncated"
    assert "status_replay_mismatch" in replay_backend_result(item, changed)
    changed = copy.deepcopy(result)
    changed["request"]["body"]["model"] = "other-model"
    assert "request_replay_mismatch" in backend_record_errors(changed, item, MESSAGES, params)
    changed = copy.deepcopy(result)
    changed["request"]["body"]["messages"][1]["content"] = "edited paper"
    assert "request_replay_mismatch" in backend_record_errors(changed, item, MESSAGES, params)


def test_replay_preserves_failed_attempts_without_raw_success():
    item = profile()
    params = {"max_output_tokens": 8}
    def timeout(_):
        raise TimeoutError("deadline")
    result = invoke(item, MESSAGES, params, allow_live=True, transport=timeout)
    assert backend_record_errors(result, item, MESSAGES, params) == []
    changed = copy.deepcopy(result)
    changed["raw_text"] = "fabricated"
    assert "missing_raw_response_for_generation" in replay_backend_result(item, changed)
    refused = invoke(profile("responses"), MESSAGES, params, allow_live=True,
                     transport=lambda _: {"status": "completed", "output": [{"content": [{"type": "refusal", "refusal": "no"}]}]})
    assert replay_backend_result(profile("responses"), refused) == []


def test_http_client_errors_do_not_become_retryable_transport_errors():
    item = profile()
    params = {"max_output_tokens": 8}
    for code in (400, 401, 403, 404, 405, 422):
        def failed(_):
            body = b'{"error":{"code":"invalid_request_error"}}' if code != 405 else b"Unsupported method"
            raise HTTPError("https://example.invalid", code, "bad request", {}, BytesIO(body))
        result = invoke(item, MESSAGES, params, allow_live=True, transport=failed)
        assert result["status"] == "invalid_request" and result["http_status"] == code
        assert result["raw_text"] is None and result["dispatch_started"] is True
        assert replay_backend_result(item, result) == []
    for code in (408, 429, 503):
        def transient(_):
            raise HTTPError("https://example.invalid", code, "transient", {}, BytesIO(b"retry"))
        result = invoke(item, MESSAGES, params, allow_live=True, transport=transient)
        assert result["status"] == "transport_error"
        assert replay_backend_result(item, result) == []


def test_live_http_request_started_only_at_urlopen(monkeypatch):
    item = profile()
    item["api_key_env"] = "FOUR_CATEGORY_TEST_MISSING_TOKEN"
    monkeypatch.delenv("FOUR_CATEGORY_TEST_MISSING_TOKEN", raising=False)
    missing = invoke(item, MESSAGES, {"max_output_tokens": 8}, allow_live=True)
    assert missing["execution_mode"] == "live"
    assert missing["dispatch_started"] is True and missing["live_request_started"] is False

    del item["api_key_env"]
    class FakeResponse(BytesIO):
        status = 200
        headers = Message()
    def fake_urlopen(request, timeout):
        assert timeout == 120
        return FakeResponse(b'{"model":"example-a","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}')
    monkeypatch.setattr(backend_module.urllib_request, "urlopen", fake_urlopen)
    result = invoke(item, MESSAGES, {"max_output_tokens": 8}, allow_live=True)
    assert result["status"] == "success" and result["http_status"] == 200
    assert result["execution_mode"] == "live" and result["dispatch_started"] is True
    assert result["live_request_started"] is True
    assert replay_backend_result(item, result) == []
