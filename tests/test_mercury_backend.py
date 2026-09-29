"""Offline checks for explicit Mercury Transformers runtime contracts."""

from __future__ import annotations

from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest

from high_fidelity_schema_study.four_category.backends import (
    invoke, transformers_load_kwargs, validate_profile,
)
from high_fidelity_schema_study.four_category.workflow import context_preflight


MESSAGES = [{"role": "system", "content": "Extract."}, {"role": "user", "content": "Full paper."}]


def profile() -> dict:
    return {"profile_id": "mercury-local-a", "backend": "transformers",
            "model_id": "/models/example", "revision": "frozen-revision", "deployment": "local",
            "context_window": 4096,
            "capabilities": {"supported_parameters": ["max_output_tokens", "do_sample", "seed"],
                             "supports_system_role": True},
            "runtime": {"settings": {"device_map": "auto", "max_memory": {"0": "118GiB", "1": "118GiB",
                                                                     "2": "118GiB", "3": "118GiB", "cpu": "850GiB"},
                                      "dtype": "bfloat16", "attn_implementation": "sdpa",
                                      "offload_folder": "/scratch/offload", "offload_state_dict": True,
                                      "low_cpu_mem_usage": True, "trust_remote_code": False}},
            "status": "frozen"}


def test_mercury_load_settings_are_forwarded_without_gpu_env_mutation(monkeypatch):
    item = profile()
    item["runtime"]["quantization"] = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                       "bnb_4bit_compute_dtype": "bfloat16"}
    dtype = object()
    fake_torch = SimpleNamespace(bfloat16=dtype)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    kwargs = transformers_load_kwargs(item, 4, torch_module=fake_torch,
                                      quantization_config_cls=lambda **kw: kw)
    assert kwargs["revision"] == "frozen-revision" and kwargs["local_files_only"] is True
    assert kwargs["max_memory"] == {0: "118GiB", 1: "118GiB", 2: "118GiB", 3: "118GiB", "cpu": "850GiB"}
    assert kwargs["device_map"] == "auto" and kwargs["dtype"] is dtype
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["offload_folder"] == "/scratch/offload"
    assert kwargs["offload_state_dict"] is True and kwargs["low_cpu_mem_usage"] is True
    assert kwargs["quantization_config"]["bnb_4bit_compute_dtype"] is dtype
    assert set(kwargs) == {"revision", "local_files_only", "trust_remote_code", "device_map",
                           "max_memory", "dtype", "attn_implementation", "offload_folder",
                           "offload_state_dict", "low_cpu_mem_usage", "quantization_config"}
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b,GPU-c,GPU-d"


def test_invalid_settings_and_quantization_fail_before_transport():
    invalid = [
        {"mystery_loader_option": True},
        {"max_memory": {"0": "100GiB"}},
        {"device_map": "tensor_parallel"},
        {"device_map": {"": "cuda:0"}},
        {"device_map": "auto", "max_memory": {"00": "100GiB"}},
        {"device_map": "auto", "max_memory": {"0": "lots"}},
        {"dtype": "float128"},
        {"offload_state_dict": "yes"},
        {"local_files_only": False},
    ]
    calls = []
    for settings in invalid:
        item = profile()
        item["runtime"]["settings"] = settings
        result = invoke(item, MESSAGES, {"max_output_tokens": 8}, allow_live=True,
                        transport=lambda request: calls.append(request))
        assert result["status"] == "invalid_request" and result["dispatch_started"] is False, settings
    for quantization in ({"load_in_4bit": True, "unknown": 1},
                         {"load_in_4bit": True, "load_in_8bit": True},
                         {"load_in_4bit": "true"},
                         {"load_in_4bit": True, "bnb_4bit_compute_dtype": "float128"}):
        item = profile()
        item["runtime"]["quantization"] = quantization
        assert validate_profile(item)
    assert not calls


def test_visible_ordinal_checked_before_weight_load():
    item = profile()
    item["runtime"]["settings"]["max_memory"] = {"0": "118GiB", "4": "118GiB"}
    with pytest.raises(ValueError, match="not among 4 visible"):
        transformers_load_kwargs(item, 4, torch_module=SimpleNamespace(bfloat16=object()),
                                  quantization_config_cls=lambda **kw: kw)
    item["runtime"]["settings"] = {"device_map": {"": 3, "lm_head": "cpu"}}
    kwargs = transformers_load_kwargs(item, 4, torch_module=SimpleNamespace(),
                                      quantization_config_cls=lambda **kw: kw)
    assert kwargs["device_map"] == {"": 3, "lm_head": "cpu"}


def test_fake_generation_records_observed_placement_and_releases_cache(monkeypatch):
    class Tokens:
        shape = (1, 3)
        def to(self, device):
            assert device == "cuda:0"
            return self

    class OutputIds:
        shape = (2,)
        def tolist(self):
            return [10, 11]

    class Generated:
        def __getitem__(self, item):
            return self if item == 0 else OutputIds()

    class Model:
        device = "cuda:0"
        hf_device_map = {"": 0, "lm_head": 1}
        config = SimpleNamespace(_commit_hash="observed-commit")
        generation_config = SimpleNamespace(to_dict=lambda: {"do_sample": True, "temperature": 1.0})
        def generate(self, encoded, **kwargs):
            assert kwargs == {"max_new_tokens": 2, "do_sample": True}
            if loaded.get("fail_generation"):
                raise RuntimeError("synthetic generation failure")
            return Generated()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == MESSAGES
            return Tokens()
        def decode(self, ids, *, skip_special_tokens):
            return "{}" if skip_special_tokens else "<s>{}"

    loaded = {}
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded["tokenizer"] = kwargs
            return Tokenizer()
    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded["model"] = kwargs
            return Model()
    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    def empty_cache():
        loaded["empty_cache_calls"] = loaded.get("empty_cache_calls", 0) + 1
    cuda = SimpleNamespace(device_count=lambda: 2, is_available=lambda: True,
                           get_device_name=lambda i: ["H200-A", "H200-B"][i],
                           get_device_properties=lambda i: SimpleNamespace(uuid=f"GPU-{i}", total_memory=141_000_000_000),
                           empty_cache=empty_cache)
    fake_torch = SimpleNamespace(__version__="test-torch", version=SimpleNamespace(cuda="12.x"),
                                 bfloat16=object(), cuda=cuda, inference_mode=nullcontext)
    fake_transformers = SimpleNamespace(__version__="test-transformers", AutoTokenizer=AutoTokenizer,
                                        AutoModelForCausalLM=AutoModelForCausalLM,
                                        BitsAndBytesConfig=BitsAndBytesConfig,
                                        set_seed=lambda seed: loaded.setdefault("seed", seed))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    item = profile()
    item["runtime"]["settings"] = {"device_map": "auto", "dtype": "bfloat16", "trust_remote_code": True}
    result = invoke(item, MESSAGES, {"max_output_tokens": 2, "do_sample": True, "seed": 7}, allow_live=True)
    # Exhausting the exact output-token budget is not a confirmed clean stop.
    assert result["status"] == "truncated", result["errors"]
    assert loaded["tokenizer"]["trust_remote_code"] is True and loaded["model"]["dtype"] is fake_torch.bfloat16
    assert loaded["seed"] == 7 and loaded["empty_cache_calls"] == 1
    identity = result["runtime_identity"]
    assert identity["hf_device_map"] == {"": 0, "lm_head": 1}
    assert identity["primary_device"] == "cuda:0"
    assert [d["name"] for d in identity["visible_cuda_devices"]] == ["H200-A", "H200-B"]
    assert [d["uuid"] for d in identity["visible_cuda_devices"]] == ["GPU-0", "GPU-1"]
    assert identity["placement_semantics"].startswith("device_map dispatch")
    loaded["fail_generation"] = True
    failed = invoke(item, MESSAGES, {"max_output_tokens": 2, "do_sample": True}, allow_live=True)
    assert failed["status"] == "transport_error"
    assert loaded["empty_cache_calls"] == 2


def test_context_preflight_tokenizer_uses_same_trust_remote_code(monkeypatch):
    observed = {}
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(apply_chat_template=lambda messages, **kw: [1, 2, 3])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    item = profile()
    item["runtime"]["settings"]["trust_remote_code"] = True
    result = context_preflight(item, MESSAGES, {"max_output_tokens": 8})
    assert result["status"] == "pass"
    assert observed == {"revision": "frozen-revision", "local_files_only": True, "trust_remote_code": True}
