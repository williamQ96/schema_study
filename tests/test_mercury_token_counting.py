from __future__ import annotations

import copy
import io
from urllib.error import HTTPError

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import digest, read_json, seal
from high_fidelity_schema_study.four_category.offline import make_fixture, mock_transport
from high_fidelity_schema_study.four_category.tasks import load_task, render_task
from high_fidelity_schema_study.four_category.token_counting import (
    count_tokens, counter_config_errors, counter_record_errors, counter_request,
)
from high_fidelity_schema_study.four_category import token_counting
from high_fidelity_schema_study.four_category.workflow import _attempt, context_preflight, replay_run


MESSAGES = [{"role": "system", "content": "Extract."}, {"role": "user", "content": "Full paper."}]


def profile(kind="responses_input_tokens"):
    backend = "responses" if kind == "responses_input_tokens" else "chat_completions"
    path = "/v1/responses" if backend == "responses" else "/v1/chat/completions"
    token_path = "/v1/responses/input_tokens" if backend == "responses" else "/tokenize"
    return {"profile_id": "counted", "backend": backend, "model_id": "qualified-model", "revision": None,
            "deployment": "remote" if backend == "responses" else "local",
            "endpoint": "https://example.test" + path, "context_window": 100000,
            "capabilities": {"supported_parameters": ["max_output_tokens"], "supports_system_role": True},
            "runtime": {"token_counter": {"kind": kind, "endpoint": "https://example.test" + token_path,
                                           "timeout_seconds": 3}}, "status": "frozen"}


def test_explicit_responses_counter_and_context_budget_use_raw_evidence():
    item = profile()
    requests = []
    def transport(request):
        requests.append(request)
        return {"object": "response.input_tokens", "model": item["model_id"], "input_tokens": 17}
    assert count_tokens(item, MESSAGES, transport=transport)["status"] == "blocked"
    assert requests == []
    budget = context_preflight(item, MESSAGES, {"max_output_tokens": 20},
                               allow_live=True, token_transport=transport)
    assert budget["status"] == "pass" and budget["input_tokens"] == 17
    assert budget["token_count_record"]["execution_mode"] == "injected_transport"
    assert budget["token_count_record"]["dispatch_started"] is True
    assert budget["token_count_record"]["live_request_started"] is False
    assert requests == [counter_request(item, MESSAGES)]
    assert requests[0]["body"] == {"model": "qualified-model", "input": MESSAGES}
    assert counter_record_errors(budget["token_count_record"], item, MESSAGES) == []
    changed = copy.deepcopy(budget["token_count_record"])
    changed["input_tokens"] = 16
    assert "token_counter_normalization_mismatch" in counter_record_errors(changed, item, MESSAGES)
    changed = copy.deepcopy(budget["token_count_record"])
    changed["request"]["body"]["input"] = []
    assert "token_counter_request_mismatch" in counter_record_errors(changed, item, MESSAGES)


def test_vllm_count_exact_request_and_no_cross_origin_calls():
    item = profile("vllm_tokenize")
    seen = []
    result = count_tokens(item, MESSAGES, allow_live=True, transport=lambda req: (seen.append(req), {"count": 23, "max_model_len": 128000})[1])
    assert result["status"] == "success" and result["input_tokens"] == 23
    assert seen[0]["body"] == {"model": item["model_id"], "messages": MESSAGES, "add_generation_prompt": True}
    assert counter_record_errors(result, item, MESSAGES) == []
    item["runtime"]["token_counter"]["endpoint"] = "https://evil.test/tokenize"
    assert "token_counter_must_share_generation_origin" in counter_config_errors(item)
    result = count_tokens(item, MESSAGES, allow_live=True, transport=lambda req: seen.append(req))
    assert result["status"] == "blocked" and len(seen) == 1


def test_invalid_provider_counts_and_http_error_block_before_generation():
    item = profile()
    for response in ({"object": "response.input_tokens", "input_tokens": True},
                     {"object": "response.input_tokens", "input_tokens": 1.5},
                     {"object": "response.input_tokens", "input_tokens": -1},
                     {"object": "response.input_tokens", "input_tokens": 5, "model": "wrong"},
                     {"object": "response.input_tokens", "error": {"message": "rate limit"}, "input_tokens": 5},
                     {"object": "response", "input_tokens": 5}, {"input_tokens": 5}):
        result = count_tokens(item, MESSAGES, allow_live=True, transport=lambda _req: response)
        assert result["status"] == "blocked" and result["exact"] is False
    def http_error(_req):
        raise HTTPError(item["runtime"]["token_counter"]["endpoint"], 429, "rate limit", {},
                        io.BytesIO(b'{"error":{"message":"rate limit"}}'))
    result = count_tokens(item, MESSAGES, allow_live=True, transport=http_error)
    assert result["status"] == "blocked" and result["http_status"] == 429
    assert result["raw_response"]["error"]["message"] == "rate limit"
    assert counter_record_errors(result, item, MESSAGES) == []


def test_no_redirect_handler_and_live_telemetry_without_network(monkeypatch):
    item = profile()
    calls = []
    class FakeOpener:
        def open(self, request, timeout):
            calls.append((request.full_url, timeout))
            raise HTTPError(request.full_url, 302, "redirect refused", {"Location": "https://evil.test/tokenize"},
                            io.BytesIO(b"redirect"))
    def fake_build_opener(handler):
        assert isinstance(handler, token_counting._NoRedirect)
        assert handler.redirect_request(None, None, 302, "redirect", {}, "https://evil.test/tokenize") is None
        return FakeOpener()
    monkeypatch.setattr(token_counting.urllib_request, "build_opener", fake_build_opener)
    result = count_tokens(item, MESSAGES, allow_live=True)
    assert calls == [(item["runtime"]["token_counter"]["endpoint"], 3)]
    assert result["status"] == "blocked" and result["http_status"] == 302
    assert result["execution_mode"] == "live" and result["live_request_started"] is True
    assert result["raw_response"] == "redirect"
    assert counter_record_errors(result, item, MESSAGES) == []


def test_existing_observation_fallback_only_without_counter():
    item = profile()
    item["runtime"].pop("token_counter")
    item["runtime"]["token_count_observations"] = {
        digest(MESSAGES): {"input_tokens": 9, "counter_id": "previous-qualified-count", "exact": True}}
    assert context_preflight(item, MESSAGES, {"max_output_tokens": 1})["status"] == "pass"
    item["runtime"]["token_counter"] = {"kind": "responses_input_tokens",
                                           "endpoint": "https://example.test/v1/responses/input_tokens"}
    assert context_preflight(item, MESSAGES, {"max_output_tokens": 1})["status"] == "blocked"


def test_replay_rederives_counter_and_rejects_edited_budget(tmp_path):
    source_root = tmp_path / "sources"
    make_fixture(source_root)
    task = load_task("classification")
    paper_input = read_json(source_root / "paper_input.json")
    item = profile()
    messages = render_task(task, paper_input)
    raw_text = mock_transport({"model": "fixture", "messages": messages})["text"]
    def generation(_request):
        return {"model": item["model_id"], "output": [{"content": [{"type": "output_text", "text": raw_text}]}]}
    job = {"job_id": "fixture-job", "kind": "classification", "profile_id": item["profile_id"],
           "profile_sha256": profile_hash(item), "task_sha256": task["task_sha256"],
           "parameters": {"max_output_tokens": 16}}
    record = _attempt(job, item, task, paper_input, None, None, None, 1, allow_live=True,
                      transport=generation, counter=None,
                      token_transport=lambda _request: {"object": "response.input_tokens", "model": item["model_id"], "input_tokens": 10})
    assert record["context_preflight"]["status"] == "pass"
    assert replay_run(record, task, paper_input) == []
    changed = copy.deepcopy(record)
    changed["context_preflight"]["token_count_record"]["raw_response"]["input_tokens"] = 11
    changed = seal(changed, "record_sha256")
    assert "token_counter_normalization_mismatch" in replay_run(changed, task, paper_input)
    changed = copy.deepcopy(record)
    changed["context_preflight"]["input_tokens"] = 11
    changed = seal(changed, "record_sha256")
    assert "token_counter_budget_mismatch" in replay_run(changed, task, paper_input)
    changed = copy.deepcopy(record)
    changed["context_preflight"]["token_count_record"]["errors"] = ["forged"]
    changed = seal(changed, "record_sha256")
    assert "token_counter_errors_mismatch" in replay_run(changed, task, paper_input)
