"""One model resident per worker process; one request at a time; no RNG sharing."""
from __future__ import annotations
import gc
import platform
import time
from . import backends, structured_output


def execution_kwargs(profile, generation_config):
    """Explicit resource controls are profile-bound, separate from sampling."""
    controls = profile['runtime'].get('generation_execution', {})
    if not isinstance(controls, dict) or set(controls) - {'prefill_chunk_size', 'kernel_options'}:
        raise ValueError('unsupported_resident_execution_control')
    if 'prefill_chunk_size' in controls:
        size = controls['prefill_chunk_size']
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= profile['context_window']:
            raise ValueError('invalid_prefill_chunk_size')
        if 'prefill_chunk_size' not in generation_config:
            raise ValueError('runtime_does_not_support_chunked_prefill')
    if 'kernel_options' in controls:
        options = controls['kernel_options']
        if (profile['runtime'].get('settings', {}).get('attn_implementation') != 'flex_attention'
                or not isinstance(options, dict) or not options
                or set(options) - {'BLOCK_M', 'BLOCK_N', 'num_stages'}
                or any(isinstance(v, bool) or not isinstance(v, int) or
                       v not in ((1, 2, 3, 4) if k == 'num_stages' else (16, 32, 64, 128))
                       for k, v in options.items())):
            raise ValueError('invalid_flex_kernel_options')
    return dict(controls)


def install_forward_controls(model, profile):
    """Flex kernel options belong to forward(), not GenerationConfig's parser."""
    controls = execution_kwargs(profile, model.generation_config.to_dict())
    options = controls.get('kernel_options')
    if options is not None:
        def forward_options(module, args, kwargs):
            if 'kernel_options' in kwargs and kwargs['kernel_options'] != options:
                raise ValueError('forward_kernel_options_conflict')
            return args, {**kwargs, 'kernel_options': dict(options)}
        model.register_forward_pre_hook(forward_options, with_kwargs=True)


def generation_call_kwargs(controls):
    return {k: v for k, v in controls.items() if k != 'kernel_options'}


class ResidentCache:
    def __init__(self, loader, closer, event=lambda *a, **k: None):
        self.loader, self.closer, self.event = loader, closer, event
        self.key, self.value = None, None
        self.loads = self.reuses = 0

    def acquire(self, profile):
        key = backends.profile_hash(profile)
        if key == self.key:
            self.reuses += 1
            self.event('model_reused', profile_id=profile['profile_id'])
            return self.value
        self.close()
        started = time.monotonic()
        value = self.loader(profile)
        self.key, self.value = key, value
        self.loads += 1
        self.event('model_loaded', profile_id=profile['profile_id'], duration_s=time.monotonic()-started)
        return value

    def close(self):
        if self.value is not None:
            value, self.value = self.value, None
            self.key = None
            self.closer(value)


class TransformersSession:
    def __init__(self, progress, event):
        self.progress, self.event = progress, event
        self.tokenizer_key = self.tokenizer = None
        self.cache = ResidentCache(self.load, self.unload, event)

    def count(self, profile, messages):
        import transformers
        key = backends.profile_hash(profile)
        if key != self.tokenizer_key:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(profile['model_id'], revision=profile['revision'],
                local_files_only=True, trust_remote_code=profile['runtime'].get('settings', {}).get('trust_remote_code', False))
            self.tokenizer_key = key
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False,
                    **profile['runtime'].get('chat_template_kwargs', {}))
        return {'input_tokens': len(tokens), 'exact': True, 'counter_id': 'resident-local-chat-tokenizer/v1'}

    def load(self, profile):
        import torch
        import transformers
        self.progress('loading')
        from .compile_policy import configure
        configure(profile, torch)
        kwargs = backends.transformers_load_kwargs(profile, torch.cuda.device_count(), torch_module=torch,
                                                    quantization_config_cls=transformers.BitsAndBytesConfig)
        name = profile['runtime'].get('auto_model_class', 'AutoModelForCausalLM')
        if name not in {'AutoModelForCausalLM', 'AutoModelForImageTextToText'}:
            raise ValueError('unqualified_model_loader')
        model = getattr(transformers, name).from_pretrained(profile['model_id'], **kwargs)
        model.eval()
        from .chunked_prefill import install
        install(model, profile)
        install_forward_controls(model, profile)
        from .sliding_cache import install as install_cache_compaction
        install_cache_compaction(model, profile)
        return [model]

    def unload(self, value):
        import torch
        value.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, req, profile):
        import torch
        import transformers
        model = self.cache.acquire(profile)[0]
        tokenizer = self.tokenizer
        config = model.generation_config.to_dict()
        execution = execution_kwargs(profile, config)
        params = req['generation_parameters']
        if not params.get('do_sample', config.get('do_sample', False)) and any(k in params for k in ('temperature', 'top_p', 'top_k')):
            raise ValueError('sampling_controls_ignored_by_generation_config')
        encoded = tokenizer.apply_chat_template(req['messages'], tokenize=True, add_generation_prompt=True,
                    return_tensors='pt', return_dict=False, **req['chat_template_kwargs'])
        input_tokens = int(encoded.shape[-1])
        limit = params.get('max_new_tokens', 0)
        if input_tokens + limit > profile['context_window']:
            raise ValueError('context_overflow')
        if req['seed'] is None:
            raise ValueError('resident_requests_require_explicit_seed')
        transformers.set_seed(req['seed'])
        generate_kwargs = {}
        applied = None
        if 'structured_output' in req:
            processor, applied = structured_output.logits_processor(req, tokenizer, model)
            generate_kwargs['logits_processor'] = [processor]
        callback = self.progress
        first_token_at = [None]
        event = self.event
        class ProgressOnly(transformers.StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if first_token_at[0] is None:
                    first_token_at[0] = time.monotonic()
                    event('prefill_finished', duration_s=first_token_at[0]-started)
                callback('decoding', output_tokens=int(input_ids.shape[-1])-input_tokens)
                return False
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        self.progress('prefill', input_tokens=input_tokens)
        started = time.monotonic()
        prefilled = [0]
        def on_forward(module, args, kwargs, output):
            ids = kwargs.get('input_ids')
            if first_token_at[0] is None and ids is not None:
                prefilled[0] += int(ids.shape[-1])
                callback('prefill', prefilled_tokens=prefilled[0])
        handle = model.register_forward_hook(on_forward, with_kwargs=True)
        try:
            with torch.inference_mode():
                generated = model.generate(encoded.to(model.device), **params, **generation_call_kwargs(execution),
                                           stopping_criteria=transformers.StoppingCriteriaList([ProgressOnly()]),
                                           **generate_kwargs)
            output = generated[0][input_tokens:]
            observed = {k: params.get(k, config.get(k)) for k in
                        ('max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 'repetition_penalty')
                        if k in params or k in config}
            observed['seed'] = req['seed']
            placement = {str(k): str(v) for k, v in getattr(model, 'hf_device_map', {}).items()}
            metrics = {'duration_s': time.monotonic()-started, 'input_tokens': input_tokens, 'output_tokens': int(output.shape[-1]),
                       'time_to_first_token_s': None if first_token_at[0] is None else first_token_at[0]-started,
                       'peak_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
                       'peak_reserved_bytes': [torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())]}
            if req.get('structured_output', {}).get('channel') == structured_output.MUSE_V2:
                metrics['muse_channels'] = structured_output.muse_channel_usage(output.tolist(), tokenizer)
            self.event('generation_finished', **metrics)
            from .compile_policy import observe
            return {'text': tokenizer.decode(output, skip_special_tokens=True),
                    'decoded_with_special_tokens': tokenizer.decode(output, skip_special_tokens=False),
                    'output_token_ids': output.tolist(), 'model': profile['model_id'],
                    'usage': {'input_tokens': input_tokens, 'output_tokens': int(output.shape[-1])},
                    'finish_reason': 'length' if limit and int(output.shape[-1]) >= limit else 'stop',
                    'status': 'truncated' if limit and int(output.shape[-1]) >= limit else 'success',
                    'generation_config': config, 'effective_parameters': observed,
                    'structured_output_applied': applied,
                    'runtime_identity': {'python': platform.python_version(), 'torch': torch.__version__,
                        'transformers': transformers.__version__, 'model_class': type(model).__name__,
                        'requested_revision': profile['revision'], 'hf_device_map': placement,
                        'generation_execution': execution,
                        'chunked_prefill_adapter': profile['runtime'].get('chunked_prefill_adapter'),
                        'sliding_cache_compaction': profile['runtime'].get('sliding_cache_compaction'),
                        'torch_compile_limits': observe(torch),
                        'structured_output_applied': applied,
                        'resident_worker_version': 'resident-worker/v1', 'request_metrics': metrics,
                        **backends._cuda_identity(torch)}}
        except Exception:
            # Do not carry suspect state across requests after an OOM/backend failure.
            self.cache.close()
            raise
        finally:
            handle.remove()
