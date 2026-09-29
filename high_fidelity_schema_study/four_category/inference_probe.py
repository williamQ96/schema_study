"""Isolated sequential Mercury runtime probe; never a production study result."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import re
import threading
import time

from . import backends
from .common import ROOT, digest, file_digest, now, read_json, write_new
from .scheduler_io import Lock, atomic_json


ALLOWED_RUNTIME_CHANGES = {'settings', 'generation_execution', 'sliding_cache_compaction',
                           'chunked_prefill_adapter', 'torch_compile'}
ALLOWED_SETTING_CHANGES = {'device_map', 'max_memory', 'offload_folder',
                           'offload_state_dict', 'low_cpu_mem_usage', 'attn_implementation'}
SAFE_COUNTERS = {'input_tokens', 'prefilled_tokens', 'output_tokens', 'duration_s',
                 'time_to_first_token_s', 'peak_allocated_bytes', 'peak_reserved_bytes'}
SAFE_EVENTS = {'model_loaded', 'model_reused', 'prefill_finished', 'generation_finished'}


class ProgressRecorder:
    """Bounded phase telemetry with no prompt or output text fields."""

    def __init__(self, output: Path, interval_s: float = 5.0):
        self.output = output
        self.interval_s = interval_s
        self.last_write = None
        self.last_phase = None
        self.latest = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        (output / 'events.jsonl').open('x', encoding='utf-8').close()

    def start(self):
        def heartbeat():
            while not self.stop.wait(min(1.0, self.interval_s / 2)):
                with self.lock:
                    tick = time.monotonic()
                    if self.latest is not None and self.last_write is not None and tick - self.last_write >= self.interval_s:
                        row = {**self.latest, 'kind': 'heartbeat', 'utc': now(), 'monotonic_s': tick}
                        atomic_json(self.output / 'progress.json', row)
                        self._append(row)
                        self.last_write = tick
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    @staticmethod
    def _safe(facts):
        return {key: value for key, value in facts.items()
                if key in SAFE_COUNTERS and (isinstance(value, (int, float)) and not isinstance(value, bool)
                                             or isinstance(value, list) and all(
                                                 isinstance(v, (int, float)) and not isinstance(v, bool) for v in value))}

    def _append(self, row):
        with (self.output / 'events.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def phase(self, phase: str, ordinal: int | None, *, force: bool = False, **facts):
        with self.lock:
            tick = time.monotonic()
            row = {'kind': 'phase', 'phase': phase, 'request_ordinal': ordinal,
                   'utc': now(), 'monotonic_s': tick, **self._safe(facts)}
            self.latest = row
            if phase != self.last_phase or self.last_write is None or tick - self.last_write >= self.interval_s or force:
                self._append(row)
                self.last_phase = phase
            if force or self.last_write is None or tick - self.last_write >= self.interval_s:
                atomic_json(self.output / 'progress.json', row)
                self.last_write = tick

    def event(self, name: str, ordinal: int | None, **facts):
        if name not in SAFE_EVENTS:
            return None
        row = {'kind': 'event', 'event': name, 'request_ordinal': ordinal,
               'utc': now(), 'monotonic_s': time.monotonic(), **self._safe(facts)}
        with self.lock:
            self._append(row)
        return row


def executing_code_identity() -> dict:
    """Hash active source bytes, including the probe and prompt/schema templates."""
    paths = set(ROOT.glob('*.py'))
    for directory in ('four_category', 'extractors'):
        paths.update((ROOT / directory).rglob('*.py'))
    paths.update((ROOT / 'templates').glob('four_category*'))
    paths.update((ROOT / 'templates').glob('paper_to_schema_*'))
    paths.update((ROOT / 'templates').glob('paper_extraction_*'))
    paths.update((ROOT / 'templates').glob('paper_category_*'))
    paths.update((ROOT / 'templates').glob('paper_derived_schema_*'))
    paths.update((ROOT / 'templates').glob('paper_evidence_*'))
    paths.update((ROOT / 'templates').glob('paper_structure_*'))
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(paths) if p.is_file()}


def verify_candidate_checkpoint(profile: dict) -> dict:
    """Rehash manifest and complete local checkpoint before any model access."""
    from .checkpoints import verify_manifest

    binding = profile['runtime'].get('checkpoint_manifest')
    audit = {'schema_version': 'mercury-probe-checkpoint-verification/v1',
             'scope': 'candidate_model_id_complete_checkpoint_bytes',
             'checked_at': now(), 'model_id': profile['model_id'],
             'revision': profile['revision'], 'status': 'failed', 'errors': []}
    if (not isinstance(binding, dict) or not isinstance(binding.get('path'), str)
            or not isinstance(binding.get('file_bytes_sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', binding['file_bytes_sha256'])):
        audit['errors'].append('checkpoint_manifest_binding_required')
        return audit
    audit['manifest_path'] = binding['path']
    audit['bound_manifest_file_sha256'] = binding['file_bytes_sha256']
    try:
        path = Path(binding['path'])
        observed = file_digest(path)
        audit['observed_manifest_file_sha256'] = observed
        if observed != binding['file_bytes_sha256']:
            audit['errors'].append('checkpoint_manifest_file_identity_mismatch')
            return audit
        manifest = read_json(path)
        audit['manifest_sha256'] = manifest.get('manifest_sha256')
        audit['manifest_file_count'] = len(manifest.get('files', [])) if isinstance(manifest.get('files'), list) else None
        audit['errors'].extend(verify_manifest(manifest, Path(profile['model_id']),
                                               expected_revision=profile['revision']))
        if file_digest(path) != observed:
            audit['errors'].append('checkpoint_manifest_changed_during_verification')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        audit['errors'].append('checkpoint_verification:' + type(exc).__name__ + ':' + str(exc)[:500])
    if not audit['errors']:
        audit['status'] = 'pass'
    return audit


def validate_input(saved: dict, candidate: dict, *, allow_model_change: bool = False) -> None:
    source = saved.get('source_profile')
    if not isinstance(source, dict) or not isinstance(candidate, dict):
        raise ValueError('source_profile_and_candidate_required')
    if not isinstance(saved.get('source_task'), dict) or not saved['source_task']:
        raise ValueError('source_task_identity_required')
    for label, profile in (('source', source), ('candidate', candidate)):
        errors = backends.validate_profile(profile)
        if errors:
            raise ValueError(label + '_profile_invalid:' + ';'.join(errors))
        if profile['backend'] != 'transformers':
            raise ValueError(label + '_must_use_transformers')
    changed = {k for k in source.keys() | candidate.keys() if source.get(k) != candidate.get(k)}
    if (source['model_id'], source['revision']) != (candidate['model_id'], candidate['revision']) and not allow_model_change:
        raise ValueError('model_checkpoint_change_requires_explicit_flag')
    allowed = {'profile_id', 'context_window', 'runtime', 'status'}
    if allow_model_change:
        allowed |= {'model_id', 'revision'}
    if changed - allowed:
        raise ValueError('candidate_changes_nonruntime_fields:' + ','.join(sorted(changed - allowed)))
    old, new = source['runtime'], candidate['runtime']
    runtime_changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    if runtime_changed - ALLOWED_RUNTIME_CHANGES:
        raise ValueError('candidate_changes_request_semantics:' + ','.join(sorted(runtime_changed - ALLOWED_RUNTIME_CHANGES)))
    old_settings, new_settings = old.get('settings', {}), new.get('settings', {})
    settings_changed = {k for k in old_settings.keys() | new_settings.keys()
                        if old_settings.get(k) != new_settings.get(k)}
    if settings_changed - ALLOWED_SETTING_CHANGES:
        raise ValueError('candidate_changes_model_precision_or_loading:' + ','.join(sorted(settings_changed - ALLOWED_SETTING_CHANGES)))
    messages, params = saved.get('messages'), saved.get('parameters')
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get('role') not in {'system', 'developer', 'user', 'assistant', 'tool'}
        or 'content' not in m for m in messages
    ):
        raise ValueError('messages_invalid')
    errors = backends.preflight_parameters(candidate, params)
    if errors:
        raise ValueError('parameters_invalid:' + ';'.join(errors))
    if 'seed' not in params:
        raise ValueError('explicit_seed_required')
    if saved.get('response_schema') is not None:
        from .structured_output import validate_control
        errors = validate_control(candidate, saved['response_schema'])
        if errors:
            raise ValueError('structured_output_invalid:' + ';'.join(errors))


def _execute_one(session, saved, candidate, identity):
    started = time.monotonic()
    result = {'schema_version': 'mercury-runtime-probe-result/v1', 'status': 'failed',
              'identity_sha256': digest(identity), 'generation': None}
    try:
        req, sent = backends._request(candidate, copy.deepcopy(saved['messages']),
                                      copy.deepcopy(saved['parameters']), saved.get('response_schema'))
        count = session.count(candidate, req['messages'])
        result['count'] = count
        if count['input_tokens'] + req['generation_parameters']['max_new_tokens'] > candidate['context_window']:
            raise ValueError('context_overflow_before_model_load')
        payload = session(req, candidate)
        metrics = payload.get('runtime_identity', {}).get('request_metrics', {})
        result.update(status='completed', generation=payload, request=req, sent_parameters=sent,
                      normal_stop=payload.get('finish_reason') == 'stop',
                      truncated=payload.get('finish_reason') == 'length',
                      generation_duration_s=metrics.get('duration_s'),
                      time_to_first_token_s=metrics.get('time_to_first_token_s'),
                      output_tokens=payload.get('usage', {}).get('output_tokens'),
                      peak_allocated_bytes=metrics.get('peak_allocated_bytes'),
                      peak_reserved_bytes=metrics.get('peak_reserved_bytes'))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:2000])
    result['total_wall_s'] = time.monotonic() - started
    return result


def run_probe(saved_path: Path | list[Path], candidate_path: Path, output: Path, leases: Path,
              gpu_ids: list[str], *, allow_live: bool = False, mode: str = 'sequential',
              allow_model_change: bool = False, session_factory=None) -> dict:
    """Run one or more saved requests under one GPU lease and resident session.

    A single request preserves the original result layout; multiple requests
    get numbered, immutable request directories and a series summary.
    """
    if mode != 'sequential':
        raise ValueError('batch_mode_unqualified: multimodal positions, cache, left padding, per-row processors and RNG require parity proof')
    if not allow_live:
        raise ValueError('live_probe_requires_allow_live')
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or any(not g or '/' in g or '\\' in g for g in gpu_ids):
        raise ValueError('explicit_unique_gpu_ids_required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None or set(visible.split(',')) != set(gpu_ids):
        raise ValueError('cuda_visible_devices_must_equal_gpu_ids')
    paths = [Path(p) for p in (saved_path if isinstance(saved_path, list) else [saved_path])]
    if not paths:
        raise ValueError('at_least_one_saved_request_required')
    candidate_path, output, leases = map(Path, (candidate_path, output, leases))
    candidate = read_json(candidate_path)
    saved_records = [read_json(path) for path in paths]
    for saved in saved_records:
        validate_input(saved, candidate, allow_model_change=allow_model_change)
    if output.exists():
        raise FileExistsError('immutable_probe_attempt_exists:' + str(output))
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'candidate_profile.json', candidate)
    active_code = executing_code_identity()
    write_new(output / 'executing_code_identity.json', active_code)
    executing_code_sha256 = digest(active_code)
    series = len(paths) > 1
    entries = []
    for number, (path, saved) in enumerate(zip(paths, saved_records), 1):
        target = output / 'requests' / f'{number:04d}' if series else output
        if series:
            target.mkdir(parents=True, exist_ok=False)
        write_new(target / 'saved_request.json', saved)
        identity = {'schema_version': 'mercury-runtime-probe/v1',
                    'purpose': 'isolated_runtime_probe_not_formal_research_result',
                    'created_at': now(), 'mode': mode, 'series_index': number if series else None,
                    'source_request_path': str(path.resolve()),
                    'source_request_file_sha256': file_digest(path),
                    'candidate_profile_path': str(candidate_path.resolve()),
                    'candidate_profile_file_sha256': file_digest(candidate_path),
                    'executing_code_identity_file': 'executing_code_identity.json',
                    'executing_code_identity_sha256': executing_code_sha256,
                    'checkpoint_verification_file': 'checkpoint_verification.json',
                    'source_task': saved['source_task'],
                    'source_profile_sha256': backends.profile_hash(saved['source_profile']),
                    'candidate_profile_sha256': backends.profile_hash(candidate),
                    'request_sha256': digest({'messages': saved['messages'], 'parameters': saved['parameters'],
                                              'response_schema': saved.get('response_schema')}),
                    'gpu_ids': gpu_ids, 'cuda_visible_devices': visible,
                    'allow_model_change': allow_model_change}
        write_new(target / 'identity.json', identity)
        entries.append((target, saved, identity))
    results = []
    real_session = session_factory is None
    run_error = None
    cleanup_error = None
    final_cuda = {}
    lease_acquired = False
    run_started = time.monotonic()
    recorder = ProgressRecorder(output)
    recorder.phase('loading', None)
    recorder.start()
    checkpoint_audit = None
    try:
        if real_session:
            recorder.phase('verifying_checkpoint', None)
            checkpoint_audit = verify_candidate_checkpoint(candidate)
        else:
            checkpoint_audit = {'schema_version': 'mercury-probe-checkpoint-verification/v1',
                                'scope': 'injected_test_session_no_weight_access',
                                'checked_at': now(), 'status': 'not_performed'}
        write_new(output / 'checkpoint_verification.json', checkpoint_audit)
        if checkpoint_audit['status'] == 'failed':
            raise ValueError('checkpoint_verification_failed:' + ';'.join(checkpoint_audit['errors']))
        with ExitStack() as stack:
            for gpu in sorted(gpu_ids):
                stack.enter_context(Lock(leases / (digest(gpu) + '.lock')))
            lease_acquired = True
            session = None
            events = []
            current_ordinal = None
            try:
                if session_factory is None:
                    import torch
                    if torch.cuda.device_count() != len(gpu_ids):
                        raise RuntimeError('visible_gpu_count_mismatch')
                    from .resident_worker import TransformersSession
                    session_factory = TransformersSession
                def progress(phase, **facts):
                    recorder.phase(phase, current_ordinal, **facts)

                def event(name, **facts):
                    row = recorder.event(name, current_ordinal, **facts)
                    if row is not None:
                        events.append(row)

                session = session_factory(progress, event)
                for ordinal, (target, saved, identity) in enumerate(entries, 1):
                    current_ordinal = ordinal
                    events = []
                    recorder.phase('tokenizing', ordinal)
                    result = _execute_one(session, saved, candidate, identity)
                    result['events'] = events
                    results.append(result)
                    if series:
                        write_new(target / 'result.json', result)
                    if result['status'] != 'completed':
                        break  # A failed call may leave the model/cache suspect.
                    recorder.phase('request_completed', ordinal,
                                   input_tokens=result.get('count', {}).get('input_tokens'),
                                   output_tokens=result.get('output_tokens'))
            finally:
                # Both cleanup and CUDA observation happen before ExitStack
                # releases the GPU leases.
                recorder.phase('cleaning_up', current_ordinal)
                if session is not None:
                    try:
                        session.cache.close()
                    except Exception as exc:
                        cleanup_error = {'cleanup_error_type': type(exc).__name__,
                                         'cleanup_error': str(exc)[:2000]}
                if real_session:
                    try:
                        import torch
                        final_cuda = {
                            'final_peak_allocated_bytes': [torch.cuda.max_memory_allocated(i)
                                                           for i in range(torch.cuda.device_count())],
                            'final_peak_reserved_bytes': [torch.cuda.max_memory_reserved(i)
                                                          for i in range(torch.cuda.device_count())],
                            'cuda_runtime_identity': backends._cuda_identity(torch)}
                    except Exception as exc:
                        final_cuda = {'cuda_metrics_error_type': type(exc).__name__}
    except Exception as exc:
        run_error = {'error_type': type(exc).__name__, 'error': str(exc)[:2000]}
    recorder.close()
    recorder.phase('completed' if not run_error and not cleanup_error and all(
        r['status'] == 'completed' for r in results) else 'failed',
        len(results) if results else None, force=True)
    for target, _, identity in entries[len(results):]:
        result = {'schema_version': 'mercury-runtime-probe-result/v1',
                  'status': 'failed' if run_error else 'skipped',
                  'identity_sha256': digest(identity), 'generation': None,
                  **(run_error or {'error_type': 'PriorRequestFailed',
                                   'error': 'series stopped after prior request failure'}),
                  'total_wall_s': 0.0}
        results.append(result)
        if series:
            write_new(target / 'result.json', result)
    summary = {'gpu_lease_acquired': lease_acquired,
               'checkpoint_verification_file': 'checkpoint_verification.json',
               'checkpoint_verification_status': checkpoint_audit['status'] if checkpoint_audit else 'unavailable',
               'total_wall_s': time.monotonic() - run_started,
               **final_cuda, **(run_error or {}), **(cleanup_error or {})}
    if series:
        summary.update(schema_version='mercury-runtime-probe-series/v1',
                       status='completed' if not run_error and not cleanup_error and all(
                           r['status'] == 'completed' for r in results) else 'failed',
                       request_results=[{'index': i, 'identity_sha256': r['identity_sha256'],
                                         'status': r['status'], 'result_path': f'requests/{i:04d}/result.json'}
                                        for i, r in enumerate(results, 1)],
                       resident_session_reused=True)
        write_new(output / 'result.json', summary)
        return summary
    result = results[0]
    result.update(final_cuda)
    result['checkpoint_verification_status'] = summary['checkpoint_verification_status']
    result['total_wall_s'] = summary['total_wall_s']
    if cleanup_error:
        result.update(status='failed', **cleanup_error)
    write_new(output / 'result.json', result)
    write_new(output / 'run_summary.json', summary)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-request', type=Path, action='append', required=True,
                        help='repeat for a sequential resident series')
    parser.add_argument('--candidate-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new, exclusive attempt directory')
    parser.add_argument('--leases', type=Path, required=True, help='same lease directory used by scheduler')
    parser.add_argument('--gpu-id', action='append', required=True, dest='gpu_ids')
    parser.add_argument('--mode', choices=('sequential', 'batch'), default='sequential')
    parser.add_argument('--allow-live', action='store_true')
    parser.add_argument('--allow-model-change', action='store_true')
    args = parser.parse_args(argv)
    request = args.saved_request if len(args.saved_request) > 1 else args.saved_request[0]
    result = run_probe(request, args.candidate_profile, args.output, args.leases,
                       args.gpu_ids, allow_live=args.allow_live, mode=args.mode,
                       allow_model_change=args.allow_model_change)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
