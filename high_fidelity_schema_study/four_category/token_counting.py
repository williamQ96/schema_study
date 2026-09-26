"""Explicit, auditable HTTP input-token counters for qualified model endpoints."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from urllib import error, request as urllib_request
from urllib.parse import urlsplit

from .common import digest


KINDS = {"responses_input_tokens": "responses", "vllm_tokenize": "chat_completions"}


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    """Treat redirects as HTTP failures; credentials must stay at the declared origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def counter_config_errors(profile: dict) -> list[str]:
    spec = profile.get("runtime", {}).get("token_counter")
    if not isinstance(spec, dict):
        return ["token_counter_must_be_object"]
    if set(spec) - {"kind", "endpoint", "timeout_seconds"}:
        return ["token_counter_unknown_fields"]
    if spec.get("kind") not in KINDS or profile.get("backend") != KINDS.get(spec.get("kind")):
        return ["token_counter_kind_backend_mismatch"]
    endpoint = spec.get("endpoint")
    generation = profile.get("endpoint")
    if not isinstance(endpoint, str) or not isinstance(generation, str):
        return ["token_counter_endpoint_required"]
    try:
        target, origin = urlsplit(endpoint), urlsplit(generation)
        if (target.scheme not in {"http", "https"} or not target.netloc or
                target.username or target.password or origin.username or origin.password or
                (target.scheme.lower(), target.netloc.lower()) != (origin.scheme.lower(), origin.netloc.lower())):
            return ["token_counter_must_share_generation_origin"]
        _ = target.port, origin.port
    except ValueError:
        return ["token_counter_endpoint_invalid"]
    timeout = spec.get("timeout_seconds", 30)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
            not math.isfinite(timeout) or timeout <= 0 or timeout > 120):
        return ["token_counter_timeout_must_be_0_to_120_seconds"]
    return []


def counter_request(profile: dict, messages: list[dict]) -> dict:
    errors = counter_config_errors(profile)
    if errors:
        raise ValueError("; ".join(errors))
    kind = profile["runtime"]["token_counter"]["kind"]
    body = ({"model": profile["model_id"], "input": messages} if kind == "responses_input_tokens"
            else {"model": profile["model_id"], "messages": messages, "add_generation_prompt": True})
    return {"kind": kind, "endpoint": profile["runtime"]["token_counter"]["endpoint"], "body": body}


def _normalize(response: object, model_id: str, kind: str, http_status: int | None) -> tuple[int | None, list[str]]:
    errors = []
    if http_status is not None and not 200 <= http_status < 300:
        errors.append("token_counter_http_status:" + str(http_status))
    if not isinstance(response, dict):
        return None, errors + ["token_counter_response_must_be_object"]
    if response.get("error") is not None:
        errors.append("token_counter_provider_error")
    if response.get("model") is not None and response["model"] != model_id:
        errors.append("token_counter_model_mismatch")
    if kind == "responses_input_tokens" and response.get("object") != "response.input_tokens":
        errors.append("token_counter_response_object_invalid")
    field = "input_tokens" if kind == "responses_input_tokens" else "count"
    count = response.get(field)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        errors.append("token_counter_count_invalid")
        count = None
    return count, errors


def count_tokens(profile: dict, messages: list[dict], *, allow_live: bool = False,
                 transport=None) -> dict:
    """Call only an explicitly configured same-origin counter; no retries/fallback."""
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {"status": "blocked", "request": None, "raw_response": None, "input_tokens": None,
              "counter_id": None, "exact": False, "http_status": None,
              "counted_at_utc": timestamp, "errors": [],
              "execution_mode": "injected_transport" if transport is not None else "live",
              "dispatch_started": False, "live_request_started": False}
    try:
        req = counter_request(profile, messages)
    except (TypeError, ValueError, KeyError) as exc:
        result["errors"] = ["token_counter_invalid_request:" + str(exc)]
        return result
    result["request"] = req
    result["counter_id"] = digest({"kind": req["kind"], "endpoint": req["endpoint"],
                                    "model": profile["model_id"]})
    if not allow_live:
        result["errors"] = ["token_counter_live_call_not_allowed"]
        return result
    try:
        if transport is not None:
            result["dispatch_started"] = True
            payload = transport(req)
        else:
            headers = {"Content-Type": "application/json"}
            name = profile.get("api_key_env")
            if name:
                secret = os.environ.get(name)
                if not secret:
                    raise ValueError("token_counter_credential_unavailable:" + name)
                headers["Authorization"] = "Bearer " + secret
            data = json.dumps(req["body"], ensure_ascii=False, allow_nan=False).encode("utf-8")
            timeout = profile["runtime"]["token_counter"].get("timeout_seconds", 30)
            opener = urllib_request.build_opener(_NoRedirect())
            result["dispatch_started"] = True
            result["live_request_started"] = True
            with opener.open(urllib_request.Request(req["endpoint"], data=data, headers=headers), timeout=timeout) as response:
                result["http_status"] = response.status
                payload = json.load(response)
        result["raw_response"] = payload
        count, errors = _normalize(payload, profile["model_id"], req["kind"], result["http_status"])
        result["errors"] = errors
        if not errors:
            result.update(status="success", input_tokens=count, exact=True)
    except error.HTTPError as exc:
        result["http_status"] = exc.code
        try:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                result["raw_response"] = json.loads(raw)
            except ValueError:
                result["raw_response"] = raw
        except OSError:
            result["raw_response"] = None
        _, result["errors"] = _normalize(result["raw_response"], profile["model_id"], req["kind"], exc.code)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["errors"] = ["token_counter_transport_error:" + type(exc).__name__ + ":" + str(exc)]
    return result


def counter_record_errors(record: dict, profile: dict, messages: list[dict]) -> list[str]:
    """Re-derive the request and normalized count from preserved provider evidence."""
    if not isinstance(record, dict):
        return ["token_counter_record_must_be_object"]
    try:
        expected = counter_request(profile, messages)
    except (TypeError, ValueError, KeyError) as exc:
        return ["token_counter_profile_invalid:" + str(exc)]
    errors = []
    if record.get("request") != expected:
        errors.append("token_counter_request_mismatch")
    expected_id = digest({"kind": expected["kind"], "endpoint": expected["endpoint"],
                          "model": profile["model_id"]})
    if record.get("counter_id") != expected_id:
        errors.append("token_counter_id_mismatch")
    mode = record.get("execution_mode")
    if mode not in {"injected_transport", "live"}:
        errors.append("token_counter_execution_mode_invalid")
    if record.get("live_request_started") is True and (mode != "live" or record.get("dispatch_started") is not True):
        errors.append("token_counter_live_dispatch_mismatch")
    if mode == "injected_transport" and record.get("live_request_started") is not False:
        errors.append("token_counter_injected_claims_live")
    if mode == "live" and record.get("dispatch_started") is True and record.get("live_request_started") is not True:
        errors.append("token_counter_live_dispatch_mismatch")
    if record.get("status") == "success" and record.get("dispatch_started") is not True:
        errors.append("token_counter_success_without_dispatch")
    if record.get("raw_response") is not None and record.get("dispatch_started") is not True:
        errors.append("token_counter_response_without_dispatch")
    count, normalization_errors = _normalize(record.get("raw_response"), profile["model_id"],
                                              expected["kind"], record.get("http_status"))
    if record.get("status") == "success":
        if normalization_errors or record.get("exact") is not True or record.get("input_tokens") != count:
            errors.append("token_counter_normalization_mismatch")
        if record.get("errors") != []:
            errors.append("token_counter_errors_mismatch")
    else:
        if record.get("exact") is True or record.get("input_tokens") is not None:
            errors.append("token_counter_failed_record_claims_count")
        if record.get("raw_response") is not None and not normalization_errors:
            errors.append("token_counter_failed_record_has_valid_response")
        if record.get("raw_response") is not None and record.get("errors") != normalization_errors:
            errors.append("token_counter_errors_mismatch")
    if record.get("status") not in {"success", "blocked"}:
        errors.append("token_counter_status_invalid")
    return errors


__all__ = ["counter_config_errors", "counter_request", "count_tokens", "counter_record_errors"]
