"""Profile-driven generation adapters for the additive four-category study.

No adapter silently substitutes a model, drops a requested control, or repairs text.
The injected ``transport`` receives a credential-free request dictionary and may
return a provider payload. For Transformers it may return a synthetic generation
payload, allowing the full parameter mapping to be tested without model weights.
"""

from __future__ import annotations

import hashlib
import gc
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import re
from typing import Any, Callable
from urllib import error, request as urllib_request


BACKENDS = {"transformers", "chat_completions", "responses", "mock"}
STATUSES = {"draft", "qualified", "frozen"}
PARAMETERS = {
    "transformers": {"max_output_tokens", "temperature", "top_p", "top_k", "do_sample", "repetition_penalty", "seed"},
    "chat_completions": {"max_output_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "seed", "frequency_penalty", "presence_penalty", "stop"},
    "responses": {"max_output_tokens", "temperature", "top_p", "reasoning_effort", "truncation"},
    "mock": {"max_output_tokens", "temperature", "top_p", "top_k", "do_sample", "repetition_penalty", "seed", "reasoning_effort", "stop"},
}
TRANSFORMERS_SETTINGS = {
    "trust_remote_code", "local_files_only", "device_map", "max_memory", "torch_dtype", "dtype",
    "attn_implementation", "offload_folder", "offload_state_dict", "low_cpu_mem_usage",
}
QUANTIZATION_SETTINGS = {
    "load_in_4bit", "load_in_8bit", "bnb_4bit_quant_type", "bnb_4bit_use_double_quant",
    "bnb_4bit_compute_dtype", "bnb_4bit_quant_storage", "llm_int8_threshold",
    "llm_int8_skip_modules", "llm_int8_enable_fp32_cpu_offload",
}
DTYPES = {"auto", "float16", "bfloat16", "float32"}
MEMORY = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)?(?:GiB|MiB|GB|MB)$")


def _transformers_runtime_errors(runtime: dict) -> list[str]:
    errors = []
    settings = runtime.get("settings", {})
    if not isinstance(settings, dict):
        return ["runtime.settings must be an object"]
    for key in sorted(set(settings) - TRANSFORMERS_SETTINGS):
        errors.append(f"unsupported Transformers setting: {key}")
    for key in ("trust_remote_code", "local_files_only", "offload_state_dict", "low_cpu_mem_usage"):
        if key in settings and not isinstance(settings[key], bool):
            errors.append(f"runtime.settings.{key} must be boolean")
    if settings.get("local_files_only") is False:
        errors.append("Transformers profiles require local_files_only=true")
    if "torch_dtype" in settings and "dtype" in settings:
        errors.append("choose one of torch_dtype or dtype")
    for key in ("torch_dtype", "dtype"):
        if key in settings and (not isinstance(settings[key], str) or settings[key] not in DTYPES):
            errors.append(f"runtime.settings.{key} must be one of {sorted(DTYPES)}")
    if "attn_implementation" in settings and (not isinstance(settings["attn_implementation"], str) or not settings["attn_implementation"]):
        errors.append("runtime.settings.attn_implementation must be a nonempty string")
    if "offload_folder" in settings and (not isinstance(settings["offload_folder"], str) or not settings["offload_folder"].strip()):
        errors.append("runtime.settings.offload_folder must be a nonempty path")
    device_map = settings.get("device_map")
    if device_map is not None:
        if isinstance(device_map, str):
            if device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
                errors.append("runtime.settings.device_map has unsupported strategy")
        elif isinstance(device_map, dict):
            if not device_map or any(not isinstance(k, str) or isinstance(v, bool) or not (
                isinstance(v, int) and v >= 0 or isinstance(v, str) and v in {"cpu", "disk"}
            ) for k, v in device_map.items()):
                errors.append("runtime.settings.device_map must map module names to visible GPU ordinals, cpu, or disk")
        else:
            errors.append("runtime.settings.device_map must be a strategy or module map")
    max_memory = settings.get("max_memory")
    if max_memory is not None:
        if not isinstance(max_memory, dict) or not max_memory:
            errors.append("runtime.settings.max_memory must be a nonempty object")
        else:
            for key, value in max_memory.items():
                if not isinstance(key, str) or (key != "cpu" and not re.fullmatch(r"0|[1-9][0-9]*", key)):
                    errors.append(f"invalid max_memory device key: {key}")
                if isinstance(value, bool) or not (isinstance(value, int) and value > 0 or isinstance(value, str) and MEMORY.fullmatch(value)):
                    errors.append(f"invalid max_memory capacity for {key}")
        if device_map is None:
            errors.append("runtime.settings.max_memory requires device_map")
    if "offload_folder" in settings and device_map is None:
        errors.append("runtime.settings.offload_folder requires device_map")
    quant = runtime.get("quantization")
    if quant is not None:
        if not isinstance(quant, dict):
            return errors + ["runtime.quantization must be an object or null"]
        for key in sorted(set(quant) - QUANTIZATION_SETTINGS):
            errors.append(f"unsupported quantization setting: {key}")
        if quant and not (quant.get("load_in_4bit") or quant.get("load_in_8bit")):
            errors.append("quantization requires load_in_4bit or load_in_8bit")
        if quant.get("load_in_4bit") and quant.get("load_in_8bit"):
            errors.append("4-bit and 8-bit quantization cannot both be enabled")
        for key in ("load_in_4bit", "load_in_8bit", "bnb_4bit_use_double_quant", "llm_int8_enable_fp32_cpu_offload"):
            if key in quant and not isinstance(quant[key], bool):
                errors.append(f"runtime.quantization.{key} must be boolean")
        if "bnb_4bit_quant_type" in quant and (not isinstance(quant["bnb_4bit_quant_type"], str) or quant["bnb_4bit_quant_type"] not in {"nf4", "fp4"}):
            errors.append("bnb_4bit_quant_type must be nf4 or fp4")
        for key in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
            if key in quant and (not isinstance(quant[key], str) or quant[key] not in DTYPES - {"auto"}):
                errors.append(f"runtime.quantization.{key} must be a concrete dtype")
        if "llm_int8_threshold" in quant and (isinstance(quant["llm_int8_threshold"], bool) or not isinstance(quant["llm_int8_threshold"], (int, float)) or not math.isfinite(quant["llm_int8_threshold"])):
            errors.append("llm_int8_threshold must be a finite number")
        if "llm_int8_skip_modules" in quant and (not isinstance(quant["llm_int8_skip_modules"], list) or any(not isinstance(x, str) for x in quant["llm_int8_skip_modules"])):
            errors.append("llm_int8_skip_modules must be a string list")
    return errors


def validate_profile(profile: dict, for_execution: bool = False) -> list[str]:
    """Return actionable profile errors; execution requires a frozen profile."""
    if not isinstance(profile, dict):
        return ["profile must be an object"]
    errors = []
    for key in ("profile_id", "backend", "model_id"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            errors.append(f"{key} must be a nonempty string")
    backend = profile.get("backend")
    if backend not in BACKENDS:
        errors.append(f"backend must be one of {sorted(BACKENDS)}")
    deployment = profile.get("deployment")
    if deployment not in {"local", "remote", "mock"}:
        errors.append("deployment must be local, remote, or mock")
    if backend == "mock" and deployment != "mock":
        errors.append("mock backend requires mock deployment")
    if backend == "transformers" and deployment != "local":
        errors.append("transformers backend requires local deployment")
    if backend in {"chat_completions", "responses"} and deployment == "mock":
        errors.append("HTTP backends require local or remote deployment")
    revision = profile.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        errors.append("revision must be a nonempty string or null")
    if backend == "transformers" and revision is None:
        errors.append("transformers profile requires a revision")
    window = profile.get("context_window")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        errors.append("context_window must be a positive integer")
    capabilities = profile.get("capabilities")
    if not isinstance(capabilities, dict):
        errors.append("capabilities must be an object")
    else:
        supported = capabilities.get("supported_parameters")
        if not isinstance(supported, list) or any(not isinstance(x, str) for x in supported) or len(supported) != len(set(supported)):
            errors.append("capabilities.supported_parameters must be a list of unique strings")
        elif backend in PARAMETERS and set(supported) - PARAMETERS[backend]:
            errors.append(f"unsupported declared parameters for {backend}: {sorted(set(supported) - PARAMETERS[backend])}")
        if not isinstance(capabilities.get("supports_system_role"), bool):
            errors.append("capabilities.supports_system_role must be boolean")
    if not isinstance(profile.get("runtime"), dict):
        errors.append("runtime must be an object")
    else:
        runtime = profile["runtime"]
        mapping = runtime.get("parameter_mapping", {})
        if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in mapping.items()):
            errors.append("runtime.parameter_mapping must map parameter names to nonempty wire names")
        elif backend == "chat_completions":
            allowed = {"max_output_tokens": {"max_tokens", "max_completion_tokens"},
                       "top_k": {"top_k"}, "repetition_penalty": {"repetition_penalty"}}
            for key, value in mapping.items():
                if key not in allowed or value not in allowed[key]:
                    errors.append(f"unsupported chat parameter mapping: {key} -> {value}")
        elif mapping:
            errors.append("runtime.parameter_mapping is supported only for chat_completions")
        ignored = runtime.get("ignored_parameters", [])
        if not isinstance(ignored, list) or any(not isinstance(x, str) for x in ignored):
            errors.append("runtime.ignored_parameters must be a list of strings")
        if backend == "transformers":
            errors.extend(_transformers_runtime_errors(runtime))
    if profile.get("status") not in STATUSES:
        errors.append("status must be draft, qualified, or frozen")
    if for_execution and backend != "mock" and profile.get("status") != "frozen":
        errors.append("live execution requires a frozen profile")
    if for_execution and backend != "mock" and str(profile.get("model_id", "")).upper() in {"UNSELECTED", "UNKNOWN", "TODO", "TBD"}:
        errors.append("live model identity remains unselected")
    endpoint = profile.get("endpoint")
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://"))):
        errors.append("endpoint must be an HTTP(S) URL or null")
    secret_name = profile.get("api_key_env")
    if secret_name is not None and (not isinstance(secret_name, str) or not secret_name.isidentifier()):
        errors.append("api_key_env must be an environment variable name")
    if "api_key" in profile:
        errors.append("store an api_key_env name, never an api_key value")
    return errors


def profile_hash(profile: dict) -> str:
    """Hash the complete declarative profile using canonical UTF-8 JSON."""
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def preflight_parameters(profile: dict, parameters: dict) -> list[str]:
    if not isinstance(parameters, dict):
        return ["parameters must be an object"]
    backend = profile.get("backend") if isinstance(profile, dict) else None
    capabilities = profile.get("capabilities") if isinstance(profile, dict) else None
    declared = capabilities.get("supported_parameters", []) if isinstance(capabilities, dict) else []
    declared = declared if isinstance(declared, list) else []
    runtime = profile.get("runtime", {}) if isinstance(profile, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    mapping = runtime.get("parameter_mapping", {})
    mapping = mapping if isinstance(mapping, dict) else {}
    errors = []
    if "max_output_tokens" not in parameters:
        errors.append("max_output_tokens is required")
    for key, value in parameters.items():
        if key not in PARAMETERS.get(backend, set()):
            errors.append(f"{key} is unsupported by {backend}")
        elif key not in declared:
            errors.append(f"{key} is not declared in profile capabilities")
        if backend == "chat_completions" and key in {"top_k", "repetition_penalty"} and key not in mapping:
            errors.append(f"{key} requires explicit runtime.parameter_mapping")
        if key in (runtime.get("ignored_parameters") if isinstance(runtime.get("ignored_parameters"), list) else []):
            errors.append(f"{key} is declared ignored by the runtime")
        if key in {"max_output_tokens", "top_k", "seed"} and (isinstance(value, bool) or not isinstance(value, int) or (key != "seed" and value <= 0)):
            errors.append(f"{key} must be an integer" + (" > 0" if key != "seed" else ""))
        if key == "seed" and isinstance(value, int) and value < 0:
            errors.append("seed must be nonnegative")
        if key in {"temperature", "top_p", "repetition_penalty", "frequency_penalty", "presence_penalty"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{key} must be a finite number")
            elif key == "temperature" and value < 0:
                errors.append("temperature must be >= 0")
            elif key == "top_p" and not 0 < value <= 1:
                errors.append("top_p must be in (0, 1]")
            elif key == "repetition_penalty" and value <= 0:
                errors.append("repetition_penalty must be > 0")
            elif key in {"frequency_penalty", "presence_penalty"} and not -2 <= value <= 2:
                errors.append(f"{key} must be in [-2, 2]")
        if key == "do_sample" and not isinstance(value, bool):
            errors.append("do_sample must be boolean")
        if key == "stop" and not (isinstance(value, str) and value or isinstance(value, list) and value and all(isinstance(x, str) and x for x in value)):
            errors.append("stop must be a nonempty string or list of nonempty strings")
        if key == "reasoning_effort" and (not isinstance(value, str) or value not in {"none", "minimal", "low", "medium", "high"}):
            errors.append("reasoning_effort is invalid")
        if key == "truncation" and value != "disabled":
            errors.append("truncation must be disabled to preserve the identical full-paper condition")
    if backend == "transformers" and parameters.get("do_sample") is False and any(x in parameters for x in ("temperature", "top_p", "top_k")):
        errors.append("sampling controls cannot be effective when do_sample is false")
    return errors


def _base(profile: dict, parameters: dict) -> dict:
    return {
        "status": "unavailable", "raw_text": None, "raw_response": None,
        "dispatch_started": False, "live_request_started": False,
        "execution_mode": None,
        "request": None, "requested_parameters": dict(parameters) if isinstance(parameters, dict) else parameters,
        "sent_parameters": None, "effective_parameters": None, "requested_model": profile.get("model_id") if isinstance(profile, dict) else None,
        "returned_model": None, "usage": None, "finish_reason": None, "errors": [],
        "response_id": None, "created": None, "system_fingerprint": None,
        "http_status": None, "provider_version": None, "provider_status": None,
        "runtime_identity": None, "generation_config": None,
    }


def _request(profile: dict, messages: list[dict], parameters: dict) -> tuple[dict, dict]:
    backend = profile["backend"]
    mapping = profile["runtime"].get("parameter_mapping", {})
    mapped = {}
    for key, value in parameters.items():
        if key == "seed" and backend == "transformers":
            continue
        target = "max_new_tokens" if key == "max_output_tokens" and backend == "transformers" else (
            mapping.get(key, "max_tokens") if key == "max_output_tokens" and backend == "chat_completions" else mapping.get(key, key)
        )
        mapped[target] = value
    if backend == "transformers":
        sent = dict(mapped)
        if "seed" in parameters:
            sent["seed"] = parameters["seed"]
        return {"backend": backend, "model": profile["model_id"], "revision": profile["revision"], "messages": messages,
                "generation_parameters": mapped, "seed": parameters.get("seed"),
                "chat_template_kwargs": profile["runtime"].get("chat_template_kwargs", {}),
                "quantization": profile["runtime"].get("quantization"),
                "settings": profile["runtime"].get("settings", {})}, sent
    if backend == "mock":
        return {"backend": backend, "model": profile["model_id"], "messages": messages, "parameters": parameters}, dict(parameters)
    if backend == "chat_completions":
        body = {"model": profile["model_id"], "messages": messages, **mapped}
        return {"backend": backend, "endpoint": profile.get("endpoint"), "body": body}, mapped
    body = {"model": profile["model_id"], "input": messages, **mapped}
    if "reasoning_effort" in body:
        body["reasoning"] = {"effort": body.pop("reasoning_effort")}
    return {"backend": backend, "endpoint": profile.get("endpoint"), "body": body}, {k: v for k, v in body.items() if k not in {"model", "input"}}


def request_for(profile: dict, messages: list[dict], parameters: dict) -> dict:
    """Rebuild the credential-free wire request from independent inputs."""
    errors = validate_profile(profile) + preflight_parameters(profile, parameters)
    if errors:
        raise ValueError("; ".join(errors))
    if not isinstance(messages, list) or any(not isinstance(x, dict) or "role" not in x or "content" not in x for x in messages):
        raise ValueError("messages must be role/content objects")
    return _request(profile, messages, parameters)[0]


def _http_transport(req: dict, profile: dict, record: dict) -> tuple[dict, dict]:
    endpoint = req["endpoint"]
    if not endpoint:
        raise ValueError("endpoint is required for live HTTP execution")
    headers = {"Content-Type": "application/json"}
    env_name = profile.get("api_key_env")
    if env_name:
        token = os.environ.get(env_name)
        if not token:
            raise ValueError(f"missing credential environment variable: {env_name}")
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(req["body"], ensure_ascii=False).encode("utf-8")
    timeout = profile["runtime"].get("timeout_seconds", 120)
    record["live_request_started"] = True
    with urllib_request.urlopen(urllib_request.Request(endpoint, data=payload, headers=headers), timeout=timeout) as response:
        body = json.load(response)
        return body, {"http_status": response.status,
                      "response_id": response.headers.get("x-request-id"),
                      "provider_version": response.headers.get("openai-version") or response.headers.get("x-api-version")}


def transformers_load_kwargs(profile: dict, visible_cuda_count: int, *, torch_module: Any,
                             quantization_config_cls: Any) -> dict:
    """Resolve a validated profile to kwargs accepted by from_pretrained.

    GPU ordinals are relative to the container's already fixed visible device
    set. This function never changes CUDA_VISIBLE_DEVICES or device affinity.
    """
    errors = _transformers_runtime_errors(profile["runtime"])
    if errors:
        raise ValueError("; ".join(errors))
    settings = profile["runtime"].get("settings", {})
    kwargs = {"revision": profile["revision"], "local_files_only": True,
              "trust_remote_code": settings.get("trust_remote_code", False)}
    device_map = settings.get("device_map")
    if isinstance(device_map, dict):
        for target in device_map.values():
            if isinstance(target, int) and target >= visible_cuda_count:
                raise ValueError(f"device_map GPU ordinal {target} is not among {visible_cuda_count} visible CUDA devices")
        kwargs["device_map"] = dict(device_map)
    elif device_map is not None:
        kwargs["device_map"] = device_map
    max_memory = settings.get("max_memory")
    if max_memory is not None:
        converted = {}
        for key, capacity in max_memory.items():
            if key != "cpu" and int(key) >= visible_cuda_count:
                raise ValueError(f"max_memory GPU ordinal {key} is not among {visible_cuda_count} visible CUDA devices")
            converted[key if key == "cpu" else int(key)] = capacity
        kwargs["max_memory"] = converted
    for key in ("torch_dtype", "dtype"):
        if key in settings:
            value = settings[key]
            kwargs[key] = value if value == "auto" else getattr(torch_module, value)
    for key in ("attn_implementation", "offload_folder", "offload_state_dict", "low_cpu_mem_usage"):
        if key in settings:
            kwargs[key] = settings[key]
    quant = profile["runtime"].get("quantization")
    if quant:
        resolved = dict(quant)
        for key in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
            if key in resolved:
                resolved[key] = getattr(torch_module, resolved[key])
        kwargs["quantization_config"] = quantization_config_cls(**resolved)
    return kwargs


def _installed_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _cuda_identity(torch_module: Any) -> dict:
    cuda = torch_module.cuda
    count = cuda.device_count()
    devices = []
    for ordinal in range(count):
        properties = cuda.get_device_properties(ordinal)
        devices.append({"ordinal": ordinal, "name": cuda.get_device_name(ordinal),
                        "uuid": getattr(properties, "uuid", None),
                        "total_memory_bytes": getattr(properties, "total_memory", None)})
    return {"cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_runtime": getattr(torch_module.version, "cuda", None),
            "visible_cuda_devices": devices}


def _transformers_transport(req: dict, profile: dict) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model = tokenizer = encoded = generated = output_ids = None
    try:
        load_kwargs = transformers_load_kwargs(profile, torch.cuda.device_count(), torch_module=torch,
                                               quantization_config_cls=BitsAndBytesConfig)
        tokenizer_kwargs = {key: load_kwargs[key] for key in ("revision", "local_files_only", "trust_remote_code")}
        tokenizer = AutoTokenizer.from_pretrained(profile["model_id"], **tokenizer_kwargs)
        try:
            model = AutoModelForCausalLM.from_pretrained(profile["model_id"], **load_kwargs)
        except (ValueError, TypeError, ImportError) as exc:
            raise ValueError("declared architecture cannot be loaded by generic AutoModelForCausalLM: " + str(exc)) from exc
        config = model.generation_config.to_dict()
        sampling = req["generation_parameters"].get("do_sample", config.get("do_sample", False))
        if sampling is False and any(key in req["generation_parameters"] for key in ("temperature", "top_p", "top_k")):
            raise ValueError("sampling controls would be ignored by the loaded model generation config")
        encoded = tokenizer.apply_chat_template(req["messages"], tokenize=True, add_generation_prompt=True,
                                                return_tensors="pt", **req["chat_template_kwargs"])
        input_tokens = int(encoded.shape[-1])
        limit = req["generation_parameters"].get("max_new_tokens", 0)
        if input_tokens + limit > profile["context_window"]:
            raise ValueError("input plus output allowance exceeds context_window")
        if req["seed"] is not None:
            transformers.set_seed(req["seed"])
        encoded = encoded.to(model.device)
        with torch.inference_mode():
            generated = model.generate(encoded, **req["generation_parameters"])
        output_ids = generated[0][input_tokens:]
        observed = {key: req["generation_parameters"].get(key, config.get(key))
                    for key in ("max_new_tokens", "temperature", "top_p", "top_k", "do_sample", "repetition_penalty")
                    if key in req["generation_parameters"] or key in config}
        observed["seed"] = req["seed"]
        placement = getattr(model, "hf_device_map", None)
        if isinstance(placement, dict):
            placement = {str(key): value if isinstance(value, (str, int)) else str(value)
                         for key, value in placement.items()}
        return {"text": tokenizer.decode(output_ids, skip_special_tokens=True),
                "decoded_with_special_tokens": tokenizer.decode(output_ids, skip_special_tokens=False),
                "output_token_ids": output_ids.tolist(), "text_decode_policy": "tokenizer_skip_special_tokens; full decoding and IDs retained",
                "model": profile["model_id"],
                "usage": {"input_tokens": input_tokens, "output_tokens": int(output_ids.shape[-1])},
                "finish_reason": "length" if limit and int(output_ids.shape[-1]) >= limit else "stop",
                "effective_parameters": observed, "generation_config": config,
                "runtime_identity": {"python": platform.python_version(), "torch": torch.__version__,
                                     "transformers": transformers.__version__,
                                     "accelerate": _installed_version("accelerate"),
                                     "bitsandbytes": _installed_version("bitsandbytes"),
                                     "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
                                     "config_commit_hash": getattr(model.config, "_commit_hash", None),
                                     "requested_revision": profile["revision"], "hf_device_map": placement,
                                     "primary_device": str(model.device),
                                     "placement_semantics": "device_map dispatch; tensor parallelism not claimed",
                                     **_cuda_identity(torch)}}
    finally:
        # A batch may switch among very large profiles. Do not retain model or
        # generated tensors across calls; offload files are user-managed.
        del output_ids, generated, encoded, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _extract_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part.get("text") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
        return "".join(texts) if texts else None
    return None


def _normalize(record: dict, backend: str, payload: Any) -> dict:
    record["raw_response"] = payload
    if not isinstance(payload, dict):
        record["status"] = "transport_error"
        record["errors"] = ["provider payload must be an object"]
        return record
    record["response_id"] = payload.get("id")
    record["created"] = payload.get("created") or payload.get("created_at")
    record["system_fingerprint"] = payload.get("system_fingerprint")
    record["provider_status"] = payload.get("status")
    record["http_status"] = payload.get("http_status") if isinstance(payload.get("http_status"), int) else None
    record["provider_version"] = payload.get("provider_version")
    record["runtime_identity"] = payload.get("runtime_identity")
    record["generation_config"] = payload.get("generation_config")
    record["effective_parameters"] = payload.get("effective_parameters")
    if payload.get("error") is not None:
        provider_error = payload["error"]
        code = provider_error.get("code") if isinstance(provider_error, dict) else None
        record["errors"] = [provider_error]
        record["status"] = "invalid_request" if code in {400, 422, "invalid_request_error", "invalid_request"} else "transport_error"
        return record
    if backend in {"mock", "transformers"}:
        record["raw_text"] = payload.get("text") if isinstance(payload.get("text"), str) else None
        record["returned_model"] = payload.get("model")
        record["usage"] = payload.get("usage")
        record["finish_reason"] = payload.get("finish_reason")
        record["status"] = payload.get("status") if payload.get("status") in {"refused", "truncated", "unavailable", "transport_error", "invalid_request"} else "success"
    elif backend == "chat_completions":
        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        record["raw_text"] = _extract_text(message.get("content"))
        record["returned_model"] = payload.get("model")
        record["usage"] = payload.get("usage")
        record["finish_reason"] = choice.get("finish_reason")
        record["status"] = "refused" if message.get("refusal") or record["finish_reason"] == "content_filter" else "truncated" if record["finish_reason"] == "length" else "unavailable" if not choice else "success"
    else:
        output = payload.get("output") or []
        chunks = []
        refused = False
        for item in output:
            if not isinstance(item, dict):
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict):
                    if part.get("type") == "refusal":
                        refused = True
                    if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        record["raw_text"] = "".join(chunks) if chunks else payload.get("output_text") if isinstance(payload.get("output_text"), str) else None
        record["returned_model"] = payload.get("model")
        record["usage"] = payload.get("usage")
        details = payload.get("incomplete_details") or {}
        record["finish_reason"] = details.get("reason") if isinstance(details, dict) else None
        record["status"] = "refused" if refused or payload.get("status") == "refused" else "truncated" if payload.get("status") == "incomplete" else "unavailable" if payload.get("status") in {"failed", "cancelled", "queued", "in_progress"} else "success"
    return record


def _http_error_status(code: int) -> str:
    """Classify nonretryable client errors separately from transient failures."""
    if 400 <= code < 500 and code not in {408, 409, 425, 429}:
        return "invalid_request"
    return "transport_error"


def invoke(profile: dict, messages: list[dict], parameters: dict, *, allow_live: bool = False,
           transport: Callable[[dict], dict] | None = None) -> dict:
    """Preflight, invoke, and preserve the provider's unmodified response."""
    record = _base(profile, parameters)
    record["execution_mode"] = "injected_transport" if transport is not None else (
        "mock" if isinstance(profile, dict) and profile.get("backend") == "mock" else "live"
    )
    errors = validate_profile(profile)
    if not isinstance(messages, list) or any(not isinstance(x, dict) or x.get("role") not in {"system", "developer", "user", "assistant", "tool"} or "content" not in x for x in messages):
        errors.append("messages must be a list of role/content objects")
    elif isinstance(profile, dict) and isinstance(profile.get("capabilities"), dict) and not profile["capabilities"].get("supports_system_role") and any(x["role"] == "system" for x in messages):
        errors.append("profile does not support system role")
    errors.extend(preflight_parameters(profile, parameters))
    if errors:
        record.update(status="invalid_request", errors=errors)
        return record
    req, sent = _request(profile, messages, parameters)
    record.update(request=req, sent_parameters=sent)
    backend = profile["backend"]
    if backend != "mock" and (not allow_live or profile["status"] != "frozen"):
        record.update(status="unavailable", errors=["live backend requires allow_live=True and a frozen profile"])
        return record
    try:
        if transport is not None:
            record["dispatch_started"] = True
            payload = transport(req)
        elif backend == "mock":
            record["dispatch_started"] = True
            payload = {"text": profile["runtime"].get("mock_text", ""), "model": profile["model_id"], "usage": None, "finish_reason": "stop"}
        elif backend == "transformers":
            record["dispatch_started"] = True
            payload = _transformers_transport(req, profile)
        else:
            record["dispatch_started"] = True
            payload, http_meta = _http_transport(req, profile, record)
            record.update(http_meta)
        normalized = _normalize(record, backend, payload)
        if backend in {"chat_completions", "responses"} and transport is None:
            for key, value in http_meta.items():
                if normalized.get(key) is None:
                    normalized[key] = value
        return normalized
    except error.HTTPError as exc:
        metadata = {"http_status": exc.code, "reason": str(exc.reason)}
        for key, name in (("Retry-After", "retry_after"), ("x-request-id", "request_id")):
            if exc.headers and exc.headers.get(key):
                metadata[name] = exc.headers.get(key)
        body = exc.read()
        if body:
            try:
                record["raw_response"] = json.loads(body)
            except (ValueError, UnicodeError):
                record["raw_response"] = body.decode("utf-8", errors="replace")
        record["response_id"] = metadata.get("request_id")
        record["http_status"] = exc.code
        record.update(status=_http_error_status(exc.code), errors=[metadata])
    except (TimeoutError, error.URLError, OSError) as exc:
        record.update(status="transport_error", errors=[{"type": type(exc).__name__, "message": str(exc)}])
    except ValueError as exc:
        record.update(status="invalid_request", errors=[str(exc)])
    except Exception as exc:
        record.update(status="transport_error", errors=[{"type": type(exc).__name__, "message": str(exc)}])
    return record


def replay_backend_result(profile: dict, result: dict) -> list[str]:
    """Check normalized fields against the immutable raw provider payload.

    A failed attempt can have no provider response (for example a timeout). Such
    records retain their failure status but must not claim generated text.
    """
    if not isinstance(result, dict):
        return ["backend_result must be an object"]
    errors = []
    if result.get("requested_model") != profile.get("model_id"):
        errors.append("requested_model_mismatch")
    if result.get("execution_mode") not in {"mock", "injected_transport", "live"}:
        errors.append("execution_mode_invalid")
    if result.get("live_request_started") and (not result.get("dispatch_started") or result.get("execution_mode") != "live" or profile.get("backend") not in {"chat_completions", "responses"}):
        errors.append("live_request_dispatch_mismatch")
    raw = result.get("raw_response")
    if raw is None:
        if result.get("status") == "success" or result.get("raw_text") is not None:
            errors.append("missing_raw_response_for_generation")
        return errors
    if not result.get("dispatch_started"):
        errors.append("raw_response_without_dispatch")
    http_status = result.get("http_status")
    http_error = isinstance(http_status, int) and not isinstance(http_status, bool) and http_status >= 400
    if http_error and result.get("errors") and isinstance(result["errors"][0], dict) and result["errors"][0].get("http_status") == http_status:
        if result.get("status") != _http_error_status(http_status):
            errors.append("status_replay_mismatch")
        if result.get("raw_text") is not None:
            errors.append("raw_text_replay_mismatch")
        if result.get("response_id") != result["errors"][0].get("request_id"):
            errors.append("response_id_replay_mismatch")
        return errors
    observed = _normalize(_base(profile, result.get("requested_parameters") or {}), profile.get("backend"), raw)
    for key in ("status", "raw_text", "returned_model", "usage", "finish_reason",
                "created", "system_fingerprint", "runtime_identity", "generation_config", "effective_parameters", "provider_status"):
        if result.get(key) != observed.get(key):
            errors.append(f"{key}_replay_mismatch")
    # HTTP transport headers may supply a request ID/version outside the body.
    if observed.get("response_id") is not None and result.get("response_id") != observed["response_id"]:
        errors.append("response_id_replay_mismatch")
    if observed.get("http_status") is not None and result.get("http_status") != observed["http_status"]:
        errors.append("http_status_replay_mismatch")
    if observed.get("provider_version") is not None and result.get("provider_version") != observed["provider_version"]:
        errors.append("provider_version_replay_mismatch")
    return errors


def backend_record_errors(result: dict, profile: dict, messages: list[dict], parameters: dict) -> list[str]:
    """Replay normalization and independently verify request provenance."""
    errors = replay_backend_result(profile, result)
    if not isinstance(result, dict):
        return errors
    if result.get("requested_parameters") is not None and result["requested_parameters"] != parameters:
        errors.append("requested_parameters_mismatch")
    if result.get("request") is not None:
        try:
            expected, sent = _request(profile, messages, parameters)
            if result["request"] != expected:
                errors.append("request_replay_mismatch")
            if result.get("sent_parameters") != sent:
                errors.append("sent_parameters_replay_mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append("request_replay_error:" + str(exc))
    elif result.get("status") == "success":
        errors.append("successful_generation_missing_request")
    return errors
