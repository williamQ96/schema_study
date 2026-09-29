"""Long-lived mailbox worker, usable inside Apptainer or locally with mocks."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import threading
import time
import subprocess

from .common import digest, read_json, seal
from .resident_worker import ResidentCache, TransformersSession
from .scheduler_io import atomic_json, read_optional, Lock, Telemetry
from .scheduler_tasks import execute
from .mercury import code_identity
from .backends import profile_hash


def sample_resources(worker, live):
    metrics = {'process_cpu_s': sum(os.times()[:2]), 'gpu_metrics': [], 'rss_bytes': None}
    if os.name == 'posix':
        try:
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    metrics['rss_bytes'] = int(line.split()[1])*1024
        except (OSError, ValueError):
            pass
    if not live or not worker['gpu_ids']:
        return metrics
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        for line in result.stdout.splitlines():
            uuid, used, total, util = [s.strip() for s in line.split(',')]
            if uuid in worker['gpu_ids']:
                metrics['gpu_metrics'].append({'uuid': uuid, 'used_bytes': int(used)*1024**2,
                                               'total_bytes': int(total)*1024**2, 'utilization_pct': float(util)})
    except (OSError, ValueError, subprocess.SubprocessError):
        metrics['gpu_observation'] = 'unavailable'
    return metrics


def admit_gpu(condition, worker, profile, session):
    """Fresh capacity veto in addition to exclusive leases and measured budgets."""
    metrics = sample_resources(worker, True)['gpu_metrics']
    observed = {m['uuid']: m for m in metrics}
    if set(observed) != set(worker['gpu_ids']):
        raise RuntimeError('allocated_gpu_observation_unavailable')
    import torch
    if torch.cuda.device_count() != len(worker['gpu_ids']):
        raise RuntimeError('visible_gpu_count_mismatch')
    peak = condition['qualification_record']['profiles'][profile_hash(profile)]['peak_bytes_per_gpu']
    reserve = worker['gpu_reserve_bytes']
    for i, gpu in enumerate(worker['gpu_ids']):
        row = observed[gpu]
        reclaimable = torch.cuda.memory_reserved(i)
        if (worker['gpu_budget_bytes'][i] > row['total_bytes'] or
                peak[i] + reserve > row['total_bytes'] - row['used_bytes'] + reclaimable):
            raise RuntimeError('fresh_gpu_headroom_insufficient')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--worker-id', required=True)
    p.add_argument('--leases', type=Path, required=True)
    args = p.parse_args()
    condition = read_json(args.root / 'condition.json')
    if condition['source_file_bytes_sha256'] != code_identity():
        raise ValueError('worker_source_identity_mismatch')
    worker = next(w for w in condition['policy']['workers'] if w['worker_id'] == args.worker_id)
    base = args.root / 'workers' / worker['worker_id']
    with ExitStack() as stack:
        stack.enter_context(Lock(base / 'worker.lock'))
        for gpu in sorted(worker.get('gpu_ids', [])):
            stack.enter_context(Lock(args.leases / (digest(gpu) + '.lock')))
        telemetry = Telemetry(base / 'events.jsonl', condition['condition'])
        state = {'worker_id': worker['worker_id'], 'condition': condition['condition'], 'pid': os.getpid(),
                 'job_id': None, 'phase': 'starting', 'progress_at': time.time(), 'heartbeat_at': time.time(),
                 'profile_id': None, 'gpu_ids': worker.get('gpu_ids', []),
                 'gpu_reserve_bytes': worker.get('gpu_reserve_bytes', 0)}
        state_lock, stop = threading.Lock(), threading.Event()
        heartbeat_error = [None]
        def progress(phase, **facts):
            if heartbeat_error[0] is not None:
                raise RuntimeError('heartbeat_writer_failed:' + heartbeat_error[0])
            with state_lock:
                state.update(phase=phase, progress_at=time.time(), **facts)
        def event(name, **facts):
            telemetry.emit(name, worker_id=worker['worker_id'], job_id=state.get('job_id'), **facts)
        def heartbeat():
            try:
                while not stop.is_set():
                    metrics = sample_resources(worker, condition['live'])
                    with state_lock:
                        state['heartbeat_at'] = time.time()
                        state.update(metrics)
                        payload = dict(state)
                    atomic_json(base / 'heartbeat.json', payload)
                    telemetry.emit('worker_sample', **payload)
                    stop.wait(condition['policy']['heartbeat_s'])
            except Exception as exc:
                heartbeat_error[0] = type(exc).__name__
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        session = TransformersSession(progress, event)
        mock_cache = ResidentCache(lambda profile: object(), lambda value: None, event)
        consumed = set()
        try:
            progress('idle')
            while not (base / 'stop.json').exists():
                if heartbeat_error[0] is not None:
                    raise RuntimeError('heartbeat_writer_failed:' + heartbeat_error[0])
                envelope = read_optional(base / 'inbox.json')
                if not envelope or envelope['request_id'] in consumed:
                    time.sleep(condition['policy']['poll_s'])
                    continue
                if envelope['condition'] != condition['condition'] or envelope['worker_id'] != worker['worker_id']:
                    raise ValueError('mailbox_identity_mismatch')
                consumed.add(envelope['request_id'])
                job = envelope['job']
                with state_lock:
                    state.update(job_id=job['job_id'], profile_id=job.get('profile_id'), request_id=envelope['request_id'])
                progress('verifying')
                event('task_started', attempt=envelope['attempt'])
                started = time.monotonic()
                try:
                    if condition['live'] and job['kind'] != 'dataset_parse':
                        profile = next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id'])
                        admit_gpu(condition, worker, profile, session)
                    if not condition['live'] and job['kind'] != 'dataset_parse':
                        profile = next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id'])
                        mock_cache.acquire(profile)
                    body = execute(condition, envelope, args.sources, session, progress, args.root)
                except Exception as exc:
                    body = {'status': 'infrastructure_failed', 'error_type': type(exc).__name__}
                result = seal({k: envelope[k] for k in ('condition', 'job_id', 'worker_id', 'attempt')} |
                              {'body': body, 'duration_s': time.monotonic()-started}, 'result_sha256')
                result_path = args.root / 'results' / job['job_id'] / ('attempt-' + str(envelope['attempt']) + '.json')
                if result_path.exists():
                    raise ValueError('immutable_attempt_already_exists')
                atomic_json(result_path, result)
                event('task_finished', attempt=envelope['attempt'], status=body['status'], duration_s=result['duration_s'])
                with state_lock:
                    state.update(job_id=None, request_id=None)
                progress('idle')
        finally:
            session.cache.close()
            mock_cache.close()
            stop.set()
            thread.join(timeout=5)
            with state_lock:
                state.update(job_id=None, phase='stopped', heartbeat_at=time.time())
            atomic_json(base / 'heartbeat.json', state)


if __name__ == '__main__':
    main()
