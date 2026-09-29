"""Single-host group coordinator with asynchronous replay and resident workers."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .io import Lock, atomic_json, digest, file_sha, local_storage_preflight, optional, read, write_once
from .migrate import initialize_verifier
from .policy import choose
from .processes import identity, same, terminate, worker_identity, resources_free
from .state import Ledger, TERMINAL


def checked_ref(root, ref):
    rel = Path(ref['path'])
    if rel.is_absolute() or '..' in rel.parts or '\\' in ref['path']:
        raise ValueError('unsafe_artifact_reference')
    path = Path(root) / rel
    if any(p.is_symlink() for p in (path, *path.parents) if p != Path(root).parent) or file_sha(path) != ref['file_bytes_sha256']:
        raise ValueError('artifact_reference_changed')
    return read(path)


def checked_worker_receipt(root, assignment, ref):
    expected_path = (Path('worker_receipts') / assignment['worker_id'] /
                     assignment['incarnation'] / ('result-' + assignment['assignment_id'] + '.json')).as_posix()
    if ref['path'] != expected_path:
        raise ValueError('worker_receipt_path_mismatch')
    receipt = checked_ref(root, ref)
    if any(receipt.get(k) != assignment[k] for k in
           ('assignment_id', 'worker_id', 'incarnation', 'job_id', 'group_id')):
        raise ValueError('worker_receipt_identity_mismatch')
    if receipt.get('status') != 'returned' or not isinstance(receipt.get('outcome'), dict):
        raise ValueError('worker_receipt_not_returned')
    dispatch = read(Path(root) / 'dispatches' / (assignment['assignment_id'] + '.json'))
    if any(dispatch.get(k) != assignment[k] for k in
           ('assignment_id', 'worker_id', 'incarnation', 'job_id', 'group_id', 'started_at')) or any(
           receipt.get(k) != dispatch.get(k) for k in
           ('assignment_id', 'worker_id', 'incarnation', 'job_id', 'group_id', 'epoch', 'started_at', 'condition')):
        raise ValueError('worker_dispatch_identity_mismatch')
    return receipt


def commit_verified_result(ledger, root, assignment, outcome, now):
    receipt = checked_worker_receipt(root, assignment, assignment['raw_ref'])
    if receipt['outcome'] != outcome:
        raise ValueError('verified_outcome_differs_from_worker_receipt')
    return ledger.commit(assignment['assignment_id'], outcome, assignment['raw_ref'], now)


def probe_stalled_worker(base, process, confirmed, heartbeat, now, heartbeat_limit=90):
    """Terminate only after two stale windows and a failed independent probe."""
    marker_path = Path(base) / 'stall-probe-state.json'
    marker = optional(marker_path)
    if not confirmed or not same(confirmed) or heartbeat.get('incarnation', process['incarnation']) != process['incarnation']:
        return False
    if now - heartbeat.get('heartbeat_at', process.get('time', now)) <= heartbeat_limit:
        if marker:
            atomic_json(marker_path, {'state': 'healthy', 'incarnation': process['incarnation']})
        return False
    if not marker or marker.get('incarnation') != process['incarnation'] or marker.get('state') != 'probing':
        marker = {'state': 'probing', 'incarnation': process['incarnation'],
                  'first_stale_at': now, 'probe_sent_at': now, 'nonce': uuid.uuid4().hex}
        atomic_json(marker_path, marker)
        atomic_json(Path(base) / 'probe.json', {'incarnation': process['incarnation'],
                                               'nonce': marker['nonce'], 'time': now})
        return False
    reply = optional(Path(base) / 'probe-reply.json')
    if (reply and reply.get('incarnation') == process['incarnation']
            and reply.get('nonce') == marker['nonce'] and reply.get('time', 0) >= marker['probe_sent_at']):
        atomic_json(marker_path, {'state': 'probe_answered', 'incarnation': process['incarnation'],
                                  'time': now})
        return False
    if (now - marker['first_stale_at'] < 2 * heartbeat_limit or
            now - marker['probe_sent_at'] < heartbeat_limit):
        return False
    if not same(confirmed):
        return False
    atomic_json(marker_path, {'state': 'terminating', 'incarnation': process['incarnation'],
                              'time': now, 'nonce': marker['nonce']})
    terminate(confirmed)
    return True


def validation_task(config, kind, job_id, group_id=None):
    from .science import verify_group, finalize
    condition = read(Path(config['root']) / 'condition.json')
    job = next(j for j in condition['jobs'] if j['job_id'] == job_id)
    if kind == 'group':
        return verify_group(condition, job, group_id, config['sources'], config['root'], config['execution_identity'])
    return finalize(condition, job, config['sources'], config['root'], config['execution_identity'])


def worker_command(config, condition, worker, incarnation):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('APPTAINERENV_', 'SINGULARITYENV_'))}
    if not condition['live']:
        source_parent = str(Path(config['science_source']).resolve().parent)
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [source_parent, str(Path(config['runtime_source']).resolve()),
                                                        env.get('PYTHONPATH')]))
        return ([sys.executable, '-u', '-m', 'scheduler_v2.worker',
                 '--root', str(config['root']), '--local', str(config['local_root']),
                 '--sources', str(config['sources']), '--science-source', str(config['science_source']),
                 '--leases', str(config['local_leases']), '--legacy-leases', str(config['legacy_leases']),
                 '--qualification', str(config['qualification']), '--worker-id', worker['worker_id'],
                 '--incarnation', incarnation], env)
    env.update(APPTAINERENV_CUDA_VISIBLE_DEVICES=','.join(worker['gpu_ids']),
               APPTAINERENV_OMP_NUM_THREADS=str(worker['cpu_threads']),
               APPTAINERENV_PYTHONPATH='/opt/scheduler-v2:/scientific',
               APPTAINERENV_HF_HUB_OFFLINE='1', APPTAINERENV_TRANSFORMERS_OFFLINE='1',
               APPTAINERENV_HF_HOME='/control/workers/' + worker['worker_id'] + '/cache/hf',
               APPTAINERENV_XDG_CACHE_HOME='/control/workers/' + worker['worker_id'] + '/cache',
               APPTAINERENV_TMPDIR='/control/workers/' + worker['worker_id'] + '/tmp')
    command = [config.get('apptainer', 'apptainer'), 'exec', '--cleanenv', '--containall', '--nv']
    mounts = [(config['runtime_source'], '/opt/scheduler-v2', 'ro'),
              (config['science_source'], '/scientific', 'ro'), (config['root'], '/results', 'rw'),
              (config['local_root'], '/control', 'rw'), (config['sources'], '/inputs', 'ro'),
              (config['local_leases'], '/leases', 'rw'), (config['legacy_leases'], '/legacy-leases', 'rw'),
              (config['qualification'], '/qualification.json', 'ro')]
    for profile in condition['config']['profiles']:
        if profile['backend'] == 'transformers':
            mounts.append((profile['model_id'], profile['model_id'], 'ro'))
    for source, destination, mode in mounts:
        if any(c in str(source) for c in (':', ',', '\n', '\r')):
            raise ValueError('unsafe_container_bind')
        command += ['--bind', str(source) + ':' + destination + ':' + mode]
    command += [config['image'], '/opt/phase1-venv/bin/python', '-u', '-m', 'scheduler_v2.worker',
                '--root', '/results', '--local', '/control', '--sources', '/inputs',
                '--science-source', '/scientific', '--leases', '/leases', '--legacy-leases', '/legacy-leases',
                '--qualification', '/qualification.json', '--worker-id', worker['worker_id'],
                '--incarnation', incarnation]
    return command, env


def launch_worker(config, condition, worker):
    incarnation = uuid.uuid4().hex
    base = Path(config['local_root']) / 'workers' / worker['worker_id']
    for name in ('tmp', 'cache'):
        (base / name).mkdir(parents=True, exist_ok=True)
    command, env = worker_command(config, condition, worker, incarnation)
    # Publish intent before launching. Recovery searches for this incarnation
    # before allowing a replacement, including a crash before the PID receipt.
    intent = dict(incarnation=incarnation, worker_id=worker['worker_id'], time=time.time(), command=command)
    atomic_json(base / 'launch-intent.json', intent)
    write_once(Path(config['root']) / 'launches' / (incarnation + '.json'), intent)
    atomic_json(base / 'process.json', dict(**intent, launcher=None))
    atomic_json(base / 'control.json', {'action': 'resume', 'time': time.time()})
    with (base / 'process.log').open('a') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=log, start_new_session=True)
    receipt = dict(**intent, launcher=identity(process.pid), spawn_pid=process.pid)
    atomic_json(base / 'process.json', receipt)
    return receipt


def launch_with_backoff(config, condition, worker, now):
    """Bound repeated launch failures across coordinator restarts."""
    path = Path(config['local_root']) / 'workers' / worker['worker_id'] / 'launch-backoff.json'
    state = optional(path) or {}
    if now < state.get('next_at', 0):
        return None, 'worker_launch_backoff'
    try:
        receipt = launch_worker(config, condition, worker)
    except (OSError, ValueError, RuntimeError) as exc:
        count = state.get('count', 0) + 1 if now - state.get('first_at', now) < 3600 else 1
        first = state.get('first_at', now) if count > 1 else now
        atomic_json(path, {'count': count, 'first_at': first,
                           'next_at': now + min(300, 5 * 2 ** min(count, 6)),
                           'error': type(exc).__name__})
        return None, 'worker_launch_failed:' + type(exc).__name__
    return receipt, None


def note_early_worker_exit(config, worker, incarnation, now):
    """Rate-limit a worker that launches successfully then dies before work."""
    path = Path(config['local_root']) / 'workers' / worker['worker_id'] / 'launch-backoff.json'
    state = optional(path) or {}
    if state.get('last_failed_incarnation') == incarnation:
        return
    count = state.get('count', 0) + 1 if now - state.get('first_at', now) < 3600 else 1
    first = state.get('first_at', now) if count > 1 else now
    atomic_json(path, {'count': count, 'first_at': first,
                       'next_at': now + min(300, 5 * 2 ** min(count, 6)),
                       'last_failed_incarnation': incarnation, 'error': 'worker_early_exit'})


def maintenance_deadline(controls, worker_id, reservations):
    """Return the earliest durable active maintenance deadline for a worker."""
    deadlines = [entry['deadline'] for entry in controls.get('maintenance', [])
                 if entry['worker_id'] is None or entry['worker_id'] == worker_id]
    deadlines.extend(reservation['expires_at'] for reservation in reservations)
    return min(deadlines) if deadlines else None


def make_ready(ledger, snapshot, now, ready_times=None):
    by_id = {j['job_id']: j for j in snapshot['jobs']}
    classifier_jobs = []
    seen_papers = set()
    for job in snapshot['jobs']:
        if job['job']['kind'] == 'classification' and job['status'] == 'pending':
            paper = job['job'].get('paper_id')
            if paper not in seen_papers:
                classifier_jobs.append(job)
                seen_papers.add(paper)
                if len(classifier_jobs) == 8:
                    break
    allowed_classifiers = {j['job_id'] for j in classifier_jobs}
    for job in snapshot['jobs']:
        if job['status'] != 'pending' or not job['groups']:
            continue
        dependencies = [by_id[d] for d in job['job'].get('dependencies', [])]
        failed = next((d for d in dependencies if d['status'] in TERMINAL
                       and d['status'] != 'success' and not d['index_ref']), None)
        if failed:
            ledger.block_dependency(job['job_id'], failed['job_id'], now)
            continue
        if job['job']['kind'] == 'classification' and job['job_id'] not in allowed_classifiers:
            continue
        if not all(d['status'] in TERMINAL and (d['status'] == 'success' or d['index_ref'])
                   for d in dependencies):
            continue
        if any(g['status'] == 'assigned' for g in job['groups']):
            continue
        pending = next((g for g in job['groups'] if g['status'] == 'pending'), None)
        if pending and pending['ready_at'] is None:
            inherited = (ready_times or {}).get(job['job_id'])
            ledger.mark_ready(job['job_id'], pending['group_id'], min(now, inherited) if inherited is not None else now)


def summary(snapshot, workers, condition, config, now, errors=None):
    stages = {}
    coverage = {}
    for kind in ('classification', 'local_extraction', 'dataset_parse', 'soft_reference'):
        jobs = [j for j in snapshot['jobs'] if j['job']['kind'] == kind]
        def status(j):
            if j['status'] != 'pending':
                return j['status']
            if any(g['status'] == 'assigned' for g in j['groups']):
                return 'running'
            if any(g['status'] == 'pending_verification' for g in j['groups']):
                return 'awaiting_validation'
            return 'pending'
        stages[kind] = dict(Counter(status(j) for j in jobs))
        groups = [g for j in jobs for g in j['groups']]
        coverage[kind] = dict(planned=len(groups), returned=sum(bool(g['generation_returned']) for g in groups),
                              admitted=sum(g['status'] == 'success' for g in groups),
                              validated=sum(g['status'] in TERMINAL for g in groups))
    return dict(schema_version='mercury-scheduler-snapshot/v2', time=now, condition=condition['condition'],
                deployment_id=config['deployment_id'], epoch=snapshot['epoch'], workers=workers,
                terminal=all(j['status'] in TERMINAL for j in snapshot['jobs']),
                stage_counts=stages, group_coverage=coverage, controls=snapshot['controls'],
                errors=dict(errors or {}),
                report_path=config['root'], semantic_accuracy=None,
                interpretation='Contract admission is not schema fidelity.')


def run(config_path):
    config = read(config_path)
    root, local = Path(config['root']), local_storage_preflight(config['local_root'])
    condition = read(root / 'condition.json')
    if file_sha(root / 'condition.json') != config['condition_file_sha256']:
        raise ValueError('condition_identity_changed')
    for name, sha in config['operational_files'].items():
        if file_sha(Path(config['runtime_source']) / name) != sha:
            raise ValueError('operational_source_changed:' + name)
    sys.path.insert(0, config['science_source'])
    initialize_verifier(config['science_source'], local / 'verification.key')
    imported_manifest = checked_ref(root, read(root / 'import-latest.json'))
    inherited_ready = {jid: row.get('ready_at') for jid, row in imported_manifest.get('ancestor_ledger_rows', {}).items()}
    plans = read(root / 'group_plans.json')
    workers = [w for w in condition['policy']['workers'] if w['gpu_ids']]
    with ExitStack() as locks:
        locks.enter_context(Lock(local / 'coordinator.lock'))
        # Shared transition lock prevents legacy takeover tooling from creating a second authority.
        locks.enter_context(Lock(Path(config['authority_lock'])))
        ledger = locks.enter_context(Ledger(local, root, config['deployment_id']))
        ledger.initialize(condition['jobs'], plans, imported_manifest['imported'])
        epoch = ledger.new_epoch()
        atomic_json(local / 'coordinator-process.json', identity(os.getpid()))
        # Linux fork would inherit coordinator/authority file descriptors and
        # keep their locks alive after a coordinator crash.
        pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_verifier,
                                   initargs=(config['science_source'], local / 'verification.key'))
        pending, parents, errors, idle_since = {}, {}, {}, {}
        validation_timing_path = local / 'validation-timing.json'
        validation_timing = optional(validation_timing_path) or {}
        startup = time.time()
        try:
            while True:
                now = time.time()
                snapshot = ledger.snapshot()
                for path in sorted((local / 'commands').glob('*.json')):
                    result_path = root / 'control_receipts' / path.name
                    if result_path.exists():
                        continue
                    command = read(path)
                    action = command['action'].replace('-', '_')
                    payload = dict(command.get('payload', {}))
                    try:
                        if action == 'cancel_reservation':
                            current_controls = ledger.snapshot()['controls']
                            reservation = next(r for r in current_controls['reservations'] if r['key'] == payload['reservation_key'])
                            owner = next(w for w in workers if set(w['gpu_ids']) & set(reservation['gpu_ids']))
                            if not resources_free(owner, config['local_leases'], config['legacy_leases']):
                                raise ValueError('reservation_resources_not_released')
                            payload['resource_release_ack'] = True
                        ledger.control(action, command['command_id'], payload, now)
                        receipt = dict(command=command, status='accepted', time=now)
                    except Exception as exc:
                        receipt = dict(command=command, status='rejected', error=str(exc), time=now)
                    write_once(result_path, receipt)
                for future in list(pending):
                    if not future.done():
                        continue
                    assignment = pending.pop(future)
                    try:
                        outcome = future.result()
                        if not commit_verified_result(ledger, root, assignment, outcome, now):
                            raise ValueError('late_or_conflicting_group_commit')
                        validation_timing.pop(assignment['assignment_id'], None)
                        atomic_json(validation_timing_path, validation_timing)
                    except Exception as exc:
                        errors[assignment['worker_id']] = 'validation_failed:' + str(exc)
                        ledger.control('pause', 'validation-' + assignment['assignment_id'], {'worker_id': assignment['worker_id']}, now)
                        write_once(root / 'failures' / (assignment['assignment_id'] + '.json'), dict(time=now, error=str(exc)))
                for future in list(parents):
                    if future.done():
                        jid = parents.pop(future)
                        try:
                            result = future.result()
                            ledger.set_parent(jid, result['status'], result['result_ref'], result['index_ref'], now)
                        except Exception as exc:
                            errors[jid] = 'parent_validation_failed:' + str(exc)
                            write_once(root / 'failures' / (jid + '.json'), dict(error=str(exc)))
                snapshot = ledger.snapshot()
                make_ready(ledger, snapshot, now, inherited_ready)
                snapshot = ledger.snapshot()
                seen_workers = []
                dispatchable = []
                operational = {}
                for worker in workers:
                    wid = worker['worker_id']; base = local / 'workers' / wid
                    process = optional(base / 'process.json') or optional(base / 'launch-intent.json')
                    active = [a for a in snapshot['assignments'] if a['worker_id'] == wid and a['status'] in {'assigned', 'pending_verification'}]
                    controls = snapshot['controls']
                    reservations = [r for r in controls['reservations'] if set(r['gpu_ids']) & set(worker['gpu_ids'])]
                    paused = controls['paused'] or wid in controls.get('paused_workers', [])
                    draining = controls['draining'] or wid in controls.get('draining_workers', []) or bool(reservations)
                    confirmed = worker_identity(process['incarnation']) if process else None
                    heartbeat = optional(base / 'heartbeat.json') or {}
                    if process and confirmed and probe_stalled_worker(
                            base, process, confirmed, heartbeat, now,
                            condition['policy'].get('health', {}).get('heartbeat_s', 90)):
                        confirmed = worker_identity(process['incarnation'])
                    if process and confirmed:
                        ledger.register_worker(wid, process['incarnation'], worker['gpu_ids'])
                        if (now - process['time'] > 30 and heartbeat.get('incarnation') == process['incarnation']
                                and now - heartbeat.get('heartbeat_at', 0) < 15
                                and (optional(base / 'launch-backoff.json') or {}).get('count', 0)):
                            atomic_json(base / 'launch-backoff.json', {'count': 0, 'next_at': 0})
                        for a in active:
                            if a['epoch'] != epoch:
                                proof = ({'raw_ref_verified': True, 'raw_ref_sha256': digest(a['raw_ref'])}
                                         if a['status'] == 'pending_verification' else {'process_incarnation_confirmed': True})
                                if a['raw_ref']:
                                    checked_worker_receipt(root, a, a['raw_ref'])
                                ledger.adopt(a['assignment_id'], wid, a['incarnation'], epoch, proof)
                    # Consume closed immutable receipts before making any recovery decision.
                    for a in active:
                        if a['status'] != 'assigned':
                            continue
                        receipt_path = root / 'worker_receipts' / wid / a['incarnation'] / ('result-' + a['assignment_id'] + '.json')
                        if not receipt_path.exists():
                            continue
                        raw_receipt = read(receipt_path)
                        if raw_receipt.get('status') == 'worker_error':
                            if any(raw_receipt.get(k) != a[k] for k in ('assignment_id', 'worker_id', 'incarnation', 'job_id', 'group_id')):
                                raise ValueError('worker_error_receipt_identity_mismatch')
                            if confirmed:
                                operational[wid] = 'worker_error:' + raw_receipt.get('error_type', 'unknown')
                                atomic_json(base / 'control.json', {'action': 'drain', 'reason': 'worker_error', 'time': now})
                            continue
                        ref = {'path': receipt_path.relative_to(root).as_posix(),
                               'file_bytes_sha256': file_sha(receipt_path)}
                        receipt = checked_worker_receipt(root, a, ref)
                        if a['epoch'] != epoch:
                            ledger.adopt_closed_raw(a['assignment_id'], epoch, now)
                        elif receipt['status'] == 'returned':
                            ledger.acknowledge(a['assignment_id'], ref, wid, a['incarnation'], epoch, now)
                    snapshot = ledger.snapshot()
                    active = [a for a in snapshot['assignments'] if a['worker_id'] == wid and a['status'] in {'assigned', 'pending_verification'}]
                    if process and not confirmed and now - process['time'] > 15:
                        if resources_free(worker, config['local_leases'], config['legacy_leases']):
                            if not active:
                                note_early_worker_exit(config, worker, process['incarnation'], now)
                            evidence = {a['assignment_id']: {'owner_released': True, 'incarnation': a['incarnation']}
                                        for a in active if a['status'] == 'assigned'}
                            if evidence:
                                recovered = ledger.recover(now, evidence)
                                if len(recovered) != len(evidence):
                                    operational[wid] = 'automatic_recovery_budget_exhausted'
                            snapshot = ledger.snapshot()
                            active = [a for a in snapshot['assignments'] if a['worker_id'] == wid and a['status'] in {'assigned', 'pending_verification'}]
                            if not draining and not paused and wid not in errors and wid not in operational and not active:
                                process, launch_error = launch_with_backoff(config, condition, worker, now)
                                if process:
                                    ledger.register_worker(wid, process['incarnation'], worker['gpu_ids'])
                                else:
                                    operational[wid] = launch_error
                        elif not draining:
                            operational[wid] = 'resource_ownership_unproven'
                    elif not process and not paused and not draining:
                        if resources_free(worker, config['local_leases'], config['legacy_leases']):
                            process, launch_error = launch_with_backoff(config, condition, worker, now)
                            if process:
                                ledger.register_worker(wid, process['incarnation'], worker['gpu_ids'])
                            else:
                                operational[wid] = launch_error
                        else:
                            operational[wid] = 'resource_ownership_unproven'
                    state = ledger.snapshot()
                    active = [a for a in state['assignments'] if a['worker_id'] == wid and a['status'] in {'assigned','pending_verification'}]
                    for a in active:
                        if a['status'] == 'pending_verification' and not any(x['assignment_id'] == a['assignment_id'] for x in pending.values()):
                            checked_worker_receipt(root, a, a['raw_ref'])
                            if a['epoch'] != epoch:
                                ledger.adopt(a['assignment_id'], wid, a['incarnation'], epoch,
                                             {'raw_ref_verified': True, 'raw_ref_sha256': digest(a['raw_ref'])})
                            future = pool.submit(validation_task, config, 'group', a['job_id'], a['group_id'])
                            pending[future] = a
                            validation_timing.setdefault(a['assignment_id'], now)
                            atomic_json(validation_timing_path, validation_timing)
                    hb = optional(base / 'heartbeat.json') or dict(worker_id=wid, phase='starting', heartbeat_at=startup, progress_at=startup)
                    if process and hb.get('incarnation') not in {None, process['incarnation']}:
                        hb = dict(worker_id=wid, phase='starting', heartbeat_at=process['time'], progress_at=process['time'])
                    if not hb.get('job_id'):
                        idle_since.setdefault(wid, now)
                    else:
                        idle_since.pop(wid, None)
                    compatible = [j for j in state['jobs'] if j['status']=='pending' and j['job'].get('profile_id') in worker.get('allowed_profile_ids', [])]
                    ready = [j for j in compatible if any(g['status']=='pending' and g['ready_at'] is not None for g in j['groups'])]
                    dependency = [j for j in compatible if not any(g['ready_at'] is not None for g in j['groups'])]
                    waits = lambda kind: max((now-g['ready_at'] for j in ready if j['job']['kind']==kind for g in j['groups']
                                              if g['status']=='pending' and g['ready_at'] is not None), default=0)
                    view = dict(worker, **{k:v for k,v in hb.items() if k not in worker},
                                idle_since=idle_since.get(wid), dependency_idle_since=idle_since.get(wid) if dependency and not ready else None,
                                compatible_ready_jobs=len(ready), dependency_wait_jobs=len(dependency),
                                oldest_classification_wait_s=waits('classification'), oldest_extraction_wait_s=waits('local_extraction'),
                                validation_pending=sum(a['status']=='pending_verification' for a in active),
                                validation_progress_at=min((validation_timing.get(a['assignment_id'], a['started_at'])
                                                            for a in active if a['status']=='pending_verification'), default=now),
                                recovery_failed=wid in errors or wid in operational,
                                error=errors.get(wid) or operational.get(wid))
                    if operational.get(wid) == 'resource_ownership_unproven':
                        view.update(ownership_conflict=True, recovery_failed=False,
                                    waiting_reason='resource_ownership_unproven')
                        if not confirmed:
                            view.update(phase='waiting_resource', heartbeat_at=None, progress_at=None)
                    if paused or draining:
                        action = 'drain' if draining else 'pause'
                        atomic_json(base / 'control.json', dict(action=action, time=now))
                        view['maintenance'] = dict(state='released' if not confirmed else 'paused',
                            confirmed=not confirmed or hb.get('phase')=='paused',
                            deadline=maintenance_deadline(controls, wid, reservations))
                        if confirmed and reservations:
                            view['reservation_wait_overdue_s'] = max(0, now-min(r['grace_until'] for r in reservations))
                    elif not errors.get(wid) and process and confirmed and not any(a['status']=='assigned' for a in active) and view['validation_pending'] < 2:
                        atomic_json(base / 'control.json', dict(action='resume', time=now))
                        dispatchable.append((worker, process, base))
                    seen_workers.append(view)
                # Receipts may have moved an assigned group to pending verification
                # during this pass. Admit its next ordered group before choosing for
                # any worker, so a classifier can stay on the same paper immediately.
                # One refresh suffices: all receipts are handled above, and claims
                # below cannot create new readiness.
                make_ready(ledger, ledger.snapshot(), now, inherited_ready)
                for worker, process, base in dispatchable:
                    state = ledger.snapshot()
                    state['workers'] = seen_workers
                    selection = choose(state, worker, now)
                    if selection:
                        assignment = ledger.claim(selection['job_id'], selection['group_id'],
                                                  worker['worker_id'], process['incarnation'], epoch, now)
                        if assignment:
                            envelope = dict(assignment, condition=condition['condition'])
                            write_once(root / 'dispatches' / (assignment['assignment_id'] + '.json'), envelope)
                            atomic_json(base / 'inbox.json', envelope)
                snapshot = ledger.snapshot()
                for job in snapshot['jobs']:
                    if (job['status']=='pending' and job['groups'] and all(g['status'] in TERMINAL for g in job['groups'])
                            and job['job_id'] not in parents.values() and job['job_id'] not in errors):
                        parents[pool.submit(validation_task, config, 'parent', job['job_id'])] = job['job_id']
                status = summary(snapshot, seen_workers, condition, config, now, errors)
                atomic_json(local / 'snapshot.json', status)
                atomic_json(root / 'snapshot.json', status)
                if status['terminal']:
                    write_once(root / 'completed.json', status)
                    for worker in workers:
                        atomic_json(local / 'workers' / worker['worker_id'] / 'control.json', dict(action='drain', time=now))
                    return
                time.sleep(1)
        finally:
            # Coordinator restart adopts resident workers; never broadcast cleanup.
            pool.shutdown(wait=False, cancel_futures=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    run(p.parse_args().config)


if __name__ == '__main__':
    main()
