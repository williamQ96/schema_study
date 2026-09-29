"""Measured, synthetic full-window admission probe; never a research result.

Run once per frozen profile and placement in the same image as resident workers.
The caller must impose a process timeout and retain failed probe artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import threading
import time

from .backends import _cuda_identity, profile_hash
from .common import digest, read_json, write_new
from .mercury import code_identity
from .resident_worker import TransformersSession, execution_kwargs, generation_call_kwargs
from .scheduler_io import atomic_json
from .scheduler_worker import sample_resources


def qualify(profile, output):
    import resource
    import torch
    import transformers
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def progress(phase, **facts):
        atomic_json(output / 'progress.json', {'phase': phase, 'time': time.time(), **facts})
    session = TransformersSession(progress, lambda *a, **k: None)
    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    peaks = {g: 0 for g in gpus}
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            for row in sample_resources({'gpu_ids': gpus}, True)['gpu_metrics']:
                peaks[row['uuid']] = max(peaks[row['uuid']], row['used_bytes'])
            stop.wait(1)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    result = {'schema_version': 'resident-context-probe/v1', 'status': 'failed',
        'purpose': 'synthetic_capacity_qualification_not_research_output',
        'profile_sha256': profile_hash(profile), 'source_tree_sha256': digest(code_identity()),
        'gpu_count': len(gpus), 'gpu_ids': gpus, 'sequence_tokens': profile['context_window'],
        'long_prefill_and_decode_passed': False}
    try:
        transformers.set_seed(1729)
        session.count(profile, [{'role': 'user', 'content': 'Synthetic capacity probe.'}])
        model = session.cache.acquire(profile)[0]
        execution = execution_kwargs(profile, model.generation_config.to_dict())
        result['generation_execution'] = execution
        from .compile_policy import observe
        result['torch_compile_limits'] = observe(torch)
        if profile['runtime'].get('sliding_cache_compaction'):
            sample = session.tokenizer.encode('Dataset field value. ' * 1800, add_special_tokens=False)[:8192]
            short = torch.tensor([sample], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                model._schema_compaction_enabled = False
                plain = model.generate(short, do_sample=False, max_new_tokens=8, min_new_tokens=8,
                                       **generation_call_kwargs(execution))
                model._schema_compaction_enabled = True
                compacted = model.generate(short, do_sample=False, max_new_tokens=8, min_new_tokens=8,
                                           **generation_call_kwargs(execution))
            result['cache_compaction_check'] = {'input_tokens': len(sample),
                'plain_output_ids': plain[0, len(sample):].tolist(),
                'compacted_output_ids': compacted[0, len(sample):].tolist(),
                'equal': bool(torch.equal(plain, compacted))}
            if not result['cache_compaction_check']['equal']:
                raise ValueError('cache_compaction_output_mismatch')
            del short, plain, compacted
            torch.cuda.empty_cache()
        if 'prefill_chunk_size' in execution:
            # Detect cache/position mistakes in the installed model's native chunking path.
            sample = session.tokenizer.encode('Dataset field value. ' * 1800, add_special_tokens=False)[:8192]
            short = torch.tensor([sample], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                plain = model.generate(short, do_sample=False, max_new_tokens=8, min_new_tokens=8)
                chunked = model.generate(short, do_sample=False, max_new_tokens=8, min_new_tokens=8, **execution)
            result['short_chunking_check'] = {'input_tokens': len(sample),
                'plain_output_ids': plain[0, len(sample):].tolist(),
                'chunked_output_ids': chunked[0, len(sample):].tolist(),
                'equal': bool(torch.equal(plain, chunked))}
            if not result['short_chunking_check']['equal']:
                raise ValueError('short_chunking_output_mismatch')
            del short, plain, chunked
            torch.cuda.empty_cache()
        seed_ids = session.tokenizer.encode('Dataset field value. ', add_special_tokens=False)
        n = profile['context_window'] - 8
        ids = (seed_ids * (n // len(seed_ids) + 1))[:n]
        result['input_token_ids_sha256'] = hashlib.sha256(__import__('json').dumps(ids).encode()).hexdigest()
        encoded = torch.tensor([ids], dtype=torch.long, device=model.device)
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        first = [None]
        class Progress(transformers.StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                first[0] = first[0] or time.monotonic()
                progress('decoding', output_tokens=int(input_ids.shape[-1])-n)
                return False
        progress('prefill', input_tokens=n)
        prefill_start = time.monotonic()
        prefilled = [0]
        def on_forward(module, args, kwargs, output):
            ids = kwargs.get('input_ids')
            if first[0] is None and ids is not None:
                prefilled[0] += int(ids.shape[-1])
                progress('prefill', input_tokens=n, prefilled_tokens=prefilled[0])
        handle = model.register_forward_hook(on_forward, with_kwargs=True)
        with torch.inference_mode():
            generated = model.generate(encoded, do_sample=False, max_new_tokens=8, min_new_tokens=8,
                **generation_call_kwargs(execution),
                stopping_criteria=transformers.StoppingCriteriaList([Progress()]))
        handle.remove()
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        result.update(status='pass', long_prefill_and_decode_passed=True,
            input_tokens=n, output_tokens=int(generated.shape[-1])-n,
            time_to_first_token_s=first[0]-prefill_start,
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(len(gpus))],
            peak_reserved_bytes=[torch.cuda.max_memory_reserved(i) for i in range(len(gpus))],
            runtime_identity=_cuda_identity(torch),
            hf_device_map={str(k): str(v) for k,v in getattr(model, 'hf_device_map', {}).items()})
        if result['output_tokens'] != 8 or any(v in ('cpu', 'disk') for v in result['hf_device_map'].values()):
            result.update(status='failed', long_prefill_and_decode_passed=False, error='decode_or_gpu_placement_incomplete')
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:2000])
    finally:
        stop.set(); thread.join(timeout=5)
        result.update(duration_s=time.monotonic()-started,
            cpu_peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            peak_bytes_per_gpu=[max(peaks[g], torch.cuda.max_memory_reserved(i)) for i,g in enumerate(gpus)])
        write_new(output / 'result.json', result)
        progress('completed', status=result['status'])
        session.cache.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = qualify(read_json(args.profile), args.output)
    raise SystemExit(0 if result['status'] == 'pass' else 1)
