"""Resident one-group mailbox worker; only this process owns its model and RNG."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import threading
import time

from .io import Lock, atomic_json, digest, file_sha, optional, read, write_once
from .science import execute_group


def run(args):
    sys.path.insert(0, str(args.science_source))
    from high_fidelity_schema_study.four_category.mercury import code_identity
    from high_fidelity_schema_study.four_category.scheduler_worker import sample_resources, admit_gpu
    from high_fidelity_schema_study.four_category.resident_worker import TransformersSession
    from high_fidelity_schema_study.four_category.common import digest as scientific_digest
    config = read(args.root / 'deployment.json')
    condition = read(args.root / 'condition.json')
    if file_sha(args.root / 'condition.json') != config['condition_file_sha256']:
        raise ValueError('scientific_condition_bytes_changed')
    if code_identity() != condition['source_file_bytes_sha256']:
        raise ValueError('frozen_scientific_source_changed')
    for name, sha in config['operational_files'].items():
        if file_sha(Path(__file__).resolve().parents[1] / name) != sha:
            raise ValueError('operational_worker_source_changed:' + name)
    worker = next(w for w in condition['policy']['workers'] if w['worker_id'] == args.worker_id)
    if condition['live']:
        condition['qualification_record'] = read(args.qualification)
    base = args.local / 'workers' / args.worker_id
    base.mkdir(parents=True, exist_ok=True)
    shared = args.root / 'worker_receipts' / args.worker_id / args.incarnation
    shared.mkdir(parents=True, exist_ok=True)
    guard, stopped = threading.Lock(), threading.Event()
    state = dict(worker_id=args.worker_id, incarnation=args.incarnation, pid=os.getpid(),
                 condition=condition['condition'], phase='starting', heartbeat_at=time.time(),
                 progress_at=time.time(), job_id=None, group_id=None, assignment_id=None,
                 gpu_ids=worker['gpu_ids'], gpu_reserve_bytes=worker.get('gpu_reserve_bytes', 0))
    heartbeat_error = []

    def progress(phase, **facts):
        if heartbeat_error:
            raise RuntimeError('heartbeat_writer_failed')
        with guard:
            state.update(phase=phase, progress_at=time.time(), **facts)

    def event(name, **facts):
        path = base / 'events.jsonl'
        import json
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=time.time(), event=name, worker_id=args.worker_id,
                                        incarnation=args.incarnation, assignment_id=state['assignment_id'], **facts)) + '\n')

    def heartbeat():
        while not stopped.is_set():
            try:
                metrics = sample_resources(worker, condition['live'])
                with guard:
                    state.update(heartbeat_at=time.time(), **metrics)
                    atomic_json(base / 'heartbeat.json', state)
                    probe = optional(base / 'probe.json')
                    if probe and probe.get('incarnation') == args.incarnation:
                        atomic_json(base / 'probe-reply.json', dict(nonce=probe['nonce'],
                            incarnation=args.incarnation, time=time.time(),
                            assignment_id=state['assignment_id'], progress_at=state['progress_at']))
                stopped.wait(5)
            except Exception as exc:
                heartbeat_error.append(type(exc).__name__)
                return

    with ExitStack() as stack:
        from high_fidelity_schema_study.four_category.replay_cache import replay_validation_scope
        stack.enter_context(replay_validation_scope())
        stack.enter_context(Lock(base / 'worker.lock'))
        for gpu in sorted(worker['gpu_ids']):
            # Keep BOTH namespaces until all legacy/probe clients migrate.
            stack.enter_context(Lock(args.legacy_leases / (scientific_digest(gpu) + '.lock')))
            stack.enter_context(Lock(args.leases / (scientific_digest(gpu) + '.lock')))
        session = TransformersSession(progress, event)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        write_once(shared / 'started.json', dict(time=time.time(), incarnation=args.incarnation,
                                                worker_id=args.worker_id, pid=os.getpid(),
                                                source_identity=config['execution_identity']))
        consumed = set()
        try:
            while True:
                if heartbeat_error:
                    raise RuntimeError('heartbeat_writer_failed')
                control = optional(base / 'control.json') or {}
                if control.get('action') in {'drain', 'release', 'stop'}:
                    progress('releasing')
                    session.cache.close()
                    write_once(shared / 'released.json', dict(time=time.time(), command=control,
                                                              incarnation=args.incarnation))
                    break
                if control.get('action') == 'pause':
                    progress('paused')
                    time.sleep(.5)
                    continue
                progress('idle')
                envelope = optional(base / 'inbox.json')
                if not envelope or envelope['assignment_id'] in consumed:
                    time.sleep(.25)
                    continue
                if (envelope['incarnation'] != args.incarnation or
                        envelope['worker_id'] != args.worker_id or
                        envelope['condition'] != condition['condition']):
                    raise ValueError('mailbox_identity_mismatch')
                assignment_id = envelope['assignment_id']
                consumed.add(assignment_id)
                job = next(j for j in condition['jobs'] if j['job_id'] == envelope['job_id'])
                if worker.get('allowed_profile_ids') and job['profile_id'] not in worker['allowed_profile_ids']:
                    raise ValueError('worker_profile_not_allowed')
                with guard:
                    state.update(job_id=job['job_id'], group_id=envelope['group_id'],
                                 assignment_id=assignment_id, profile_id=job['profile_id'])
                started = time.time()
                write_once(shared / ('dispatch-' + assignment_id + '.json'), envelope)
                try:
                    if condition['live']:
                        profile = next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id'])
                        admit_gpu(condition, worker, profile, session)
                    result = execute_group(condition, job, envelope['group_id'], args.sources,
                                           args.root, session, progress)
                    payload = dict(status='returned', outcome=result)
                except Exception as exc:
                    # Runtime exceptions are operational records, not synthetic model answers.
                    payload = dict(status='worker_error', error_type=type(exc).__name__, error=str(exc)[:1000])
                receipt = {**envelope, 'version': 'scheduler-v2-group-receipt/v1',
                           'worker_started_at': started, 'finished_at': time.time(), **payload}
                path = shared / ('result-' + assignment_id + '.json')
                sha = write_once(path, receipt)
                atomic_json(base / 'outbox.json', dict(assignment_id=assignment_id,
                    path=path.relative_to(args.root).as_posix(), file_bytes_sha256=sha))
                with guard:
                    state.update(job_id=None, group_id=None, assignment_id=None)
        finally:
            stopped.set()
            thread.join(timeout=6)
            session.cache.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'local', 'sources', 'science-source', 'legacy-leases', 'leases', 'qualification'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--worker-id', required=True)
    p.add_argument('--incarnation', required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
