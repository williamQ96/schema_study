"""Recoverable single-host matrix scheduler with resident subprocess workers.

Local mocks by default. Apptainer execution requires a new qualified condition.
The health watchdog is an independent process, not part of the dispatch loop.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

from .backends import profile_hash, validate_profile, preflight_parameters
from .common import ROOT, contained, digest, file_digest, read_json, seal, write_new
from .mercury import code_identity, checkpoint_binding_errors
from .queue_health import thresholds
from .scheduler_io import Lock, Telemetry, atomic_json, read_optional
from .scheduler_tasks import verify_result
from .scheduler_observability import summarize_stages, worker_queue_context
from .workflow import plan_jobs, experiment_errors

TERMINAL = {'success', 'deferred', 'contract_invalid', 'invalid_request', 'truncated', 'incomplete',
            'infrastructure_failed', 'transport_error', 'blocked_dependency', 'refused', 'unavailable', 'dataset_failed',
            'completed_with_rejections', 'generation_incomplete'}


def default_policy():
    return {'schema_version': 'matrix-scheduler-policy/v1', 'max_active_matrices': 2,
            'poll_s': 1, 'heartbeat_s': 5, 'max_attempts': 2, 'model_affinity_max_wait_s': 300,
            'host_cpu_threads': 128, 'host_ram_bytes': 1024*1024**3,
            'workers': [{'worker_id': 'gpu-0', 'gpu_ids': ['0', '1'], 'cpu_threads': 8, 'ram_bytes': 128*1024**3},
                        {'worker_id': 'gpu-1', 'gpu_ids': ['2', '3'], 'cpu_threads': 8, 'ram_bytes': 128*1024**3},
                        {'worker_id': 'dataset-0', 'gpu_ids': [], 'cpu_threads': 2, 'ram_bytes': 8*1024**3}],
            'defer_soft_reference': True, 'health': thresholds(), 'qualification': None}


def validate_policy(policy):
    if policy.get('schema_version') != 'matrix-scheduler-policy/v1' or policy.get('defer_soft_reference') is not True:
        raise ValueError('unsupported_policy_or_reference_scope')
    for key in ('max_active_matrices', 'max_attempts', 'host_cpu_threads', 'host_ram_bytes'):
        if isinstance(policy.get(key), bool) or not isinstance(policy.get(key), int) or policy[key] < 1:
            raise ValueError('positive_integer_required:' + key)
    import math
    for key in ('poll_s', 'heartbeat_s', 'model_affinity_max_wait_s'):
        if isinstance(policy.get(key), bool) or not isinstance(policy.get(key), (int, float)) or not math.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError('positive_finite_duration_required:' + key)
    workers = policy.get('workers', [])
    ids, devices = [], []
    if not workers:
        raise ValueError('workers_required')
    for w in workers:
        if not isinstance(w.get('worker_id'), str) or not re.fullmatch('[a-zA-Z0-9_-]+', w['worker_id']):
            raise ValueError('invalid_worker_id')
        ids.append(w['worker_id'])
        if not isinstance(w.get('gpu_ids'), list) or any(not isinstance(d, str) or not re.fullmatch(r'[0-9]+|GPU-[A-Za-z0-9-]+', d) for d in w['gpu_ids']):
            raise ValueError('invalid_gpu_allocation')
        devices.extend(w['gpu_ids'])
        allowed = w.get('allowed_profile_ids')
        if allowed is not None and (not w['gpu_ids'] or not isinstance(allowed, list) or not allowed
                or any(not isinstance(p, str) or not p for p in allowed) or len(allowed) != len(set(allowed))):
            raise ValueError('invalid_worker_profile_allowlist')
        for k in ('ram_bytes', 'cpu_threads'):
            if isinstance(w.get(k), bool) or not isinstance(w.get(k), int) or w[k] <= 0:
                raise ValueError('worker_budget_required')
    if len(set(ids)) != len(ids) or len(set(devices)) != len(devices):
        raise ValueError('overlapping_worker_or_gpu_lease')
    if not devices or not any(not w['gpu_ids'] for w in workers):
        raise ValueError('gpu_and_cpu_workers_required')
    if sum(w['cpu_threads'] for w in workers) > policy['host_cpu_threads'] or sum(w['ram_bytes'] for w in workers) > policy['host_ram_bytes']:
        raise ValueError('host_budget_exceeded')
    thresholds(policy.get('health'))


def compile_condition(config, corpus, policy, *, live=False):
    validate_policy(policy)
    if len(config['replicates']) != 3 or len({r['seed'] for r in config['replicates']}) != 3:
        raise ValueError('three_distinct_replicates_required')
    if experiment_errors(config):
        raise ValueError('invalid_experiment_configuration')
    if config['classification']['profile_id'] == config['roles']['soft_reference']:
        raise ValueError('classifier_cannot_use_deferred_reference')
    active = set(config['roles']['locals']) | {config['classification']['profile_id']}
    covered = set()
    for worker in policy['workers']:
        if worker['gpu_ids']:
            allowed = set(worker.get('allowed_profile_ids', active))
            if not allowed <= active:
                raise ValueError('unknown_worker_profile')
            covered |= allowed
    if covered != active:
        raise ValueError('active_profile_has_no_compatible_worker')
    profiles = [p for p in config['profiles'] if p['profile_id'] in active]
    if any(validate_profile(p, for_execution=live) for p in profiles):
        raise ValueError('active_profile_contract_invalid')
    if not live and any(p['backend'] != 'mock' for p in profiles):
        raise ValueError('local_scheduler_accepts_only_mock_profiles')
    if live and any(p['backend'] != 'transformers' or p['status'] != 'frozen' for p in profiles):
        raise ValueError('resident_live_workers_require_frozen_transformers_profiles')
    if live and (config['status'] != 'frozen' or not config['inference_enabled'] or
                 config['classification']['profile_sha256'] != profile_hash(next(p for p in profiles if p['profile_id'] == config['classification']['profile_id']))):
        raise ValueError('frozen_experiment_and_classifier_pin_required')
    plan = plan_jobs(config, corpus)
    by_id = {p['profile_id']: p for p in profiles}
    if any(preflight_parameters(by_id[j['profile_id']], j['parameters']) for j in plan['jobs'] if j['kind'] != 'soft_reference'):
        raise ValueError('unsupported_task_parameters')
    jobs = list(plan['jobs'])
    for ds in corpus['datasets']:
        body = {'kind': 'dataset_parse', 'dataset': ds, 'dependencies': [],
                'parser_config': config['dataset_parser'], 'source_identity': digest(ds)}
        body['job_id'] = digest(body)
        jobs.append(body)
    sources = {'source_file_bytes_sha256': code_identity()}
    if policy.get('execution_source') is not None:
        execution_root = execution_source_root(policy)
        sources = {'source_file_bytes_sha256': code_identity(execution_root),
                   'coordinator_source_file_bytes_sha256': code_identity()}
    return seal({'schema_version': 'matrix-scheduler-condition/v1', 'config': config, 'corpus': corpus,
                 'policy': policy, 'live': live, 'jobs': jobs,
                 'logical_plan_sha256': plan['plan_sha256'], **sources}, 'condition')


def execution_source_root(policy):
    """Validate an explicitly frozen worker tree; coordinator identity stays separate."""
    binding = policy.get('execution_source')
    if binding is None:
        return ROOT
    if not isinstance(binding, dict) or set(binding) != {'path', 'source_file_bytes_sha256', 'source_tree_sha256'}:
        raise ValueError('execution_source_binding_invalid')
    path = Path(binding['path'])
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise ValueError('execution_source_path_invalid')
    actual = code_identity(path)
    if (not actual or actual != binding['source_file_bytes_sha256']
            or digest(actual) != binding['source_tree_sha256']):
        raise ValueError('execution_source_bytes_mismatch')
    return path


class Store:
    def __init__(self, root, condition):
        self.root, self.condition = Path(root), condition
        self.db = sqlite3.connect(self.root / 'scheduler.sqlite', timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, body TEXT, status TEXT, worker TEXT,
          attempt INTEGER DEFAULT 0, ready_at REAL, started_at REAL, finished_at REAL, result TEXT);
        CREATE TABLE IF NOT EXISTS papers(id TEXT PRIMARY KEY, admitted INTEGER DEFAULT 0);
        ''')
        existing = {r['id']: json.loads(r['body']) for r in self.db.execute('SELECT id,body FROM jobs')}
        expected = {j['job_id']: j for j in condition['jobs']}
        if existing and existing != expected:
            self.db.close()
            raise ValueError('ledger_job_identity_mismatch')
        for job in condition['jobs']:
            status = 'deferred' if job['kind'] == 'soft_reference' else 'pending'
            self.db.execute('INSERT OR IGNORE INTO jobs(id,body,status) VALUES(?,?,?)', (job['job_id'], json.dumps(job), status))
            if job.get('paper_id'):
                self.db.execute('INSERT OR IGNORE INTO papers(id) VALUES(?)', (job['paper_id'],))
        self.db.commit()

    def rows(self):
        return [dict(r) | {'job': json.loads(r['body'])} for r in self.db.execute('SELECT * FROM jobs ORDER BY rowid')]

    def refresh(self, at):
        rows = self.rows()
        states = {r['id']: r['status'] for r in rows}
        by_id = {r['id']: r for r in rows}
        def dependency_ready(identifier):
            row = by_id[identifier]
            if states[identifier] == 'success':
                return True
            # A finalized, replay-verified navigation artifact is usable even
            # when some predictions are unavailable. This is not legacy success.
            return (row['job']['kind'] == 'classification'
                    and row['job'].get('group_execution', {}).get('version') == 'all-groups/v1'
                    and states[identifier] in {'completed_with_rejections', 'generation_incomplete'}
                    and (self.root / 'indexes' / (digest(row['job']['paper_id']) + '.json')).is_file())
        for r in rows:
            if r['status'] == 'pending' and any(states[d] in TERMINAL and not dependency_ready(d) for d in r['job']['dependencies']):
                self.db.execute("UPDATE jobs SET status='blocked_dependency',finished_at=? WHERE id=?", (at, r['id']))
        rows = self.rows()
        outstanding = {p['paper_id'] for p in self.condition['corpus']['papers']
                       if any(r['job'].get('paper_id') == p['paper_id'] and r['status'] not in TERMINAL for r in rows)}
        admitted = {r[0] for r in self.db.execute('SELECT id FROM papers WHERE admitted=1')} & outstanding
        for pid in sorted(outstanding - admitted):
            if len(admitted) >= self.condition['policy']['max_active_matrices']:
                break
            self.db.execute('UPDATE papers SET admitted=1 WHERE id=?', (pid,))
            admitted.add(pid)
        for r in rows:
            if r['status'] == 'pending' and (r['job'].get('paper_id') in admitted or r['job']['kind'] == 'dataset_parse') and all(dependency_ready(d) for d in r['job']['dependencies']):
                self.db.execute('UPDATE jobs SET ready_at=COALESCE(ready_at,?) WHERE id=?', (at, r['id']))
        self.db.commit()

    def eligible(self, worker):
        return [r for r in self.rows() if r['status'] == 'pending' and r['ready_at'] is not None
                and (r['job']['kind'] != 'dataset_parse') == bool(worker['gpu_ids'])
                and (not worker.get('allowed_profile_ids') or r['job'].get('profile_id') in worker['allowed_profile_ids'])]

    def choose(self, worker, profile_id, at):
        ready = self.eligible(worker)
        aging = self.condition['policy']['model_affinity_max_wait_s']
        ready.sort(key=lambda r: (not (at-r['ready_at'] >= aging),
                                  r['ready_at'] if at-r['ready_at'] >= aging else 0,
                                  r['job'].get('profile_id') != profile_id, r['ready_at'], r['id']))
        return ready[0] if ready else None

    def claim(self, row, worker, at):
        with self.db:
            cursor = self.db.execute("UPDATE jobs SET status='running',worker=?,attempt=attempt+1,started_at=? WHERE id=? AND status='pending'",
                                     (worker, at, row['id']))
        return cursor.rowcount == 1

    def summary(self):
        rows = self.rows()
        matrices = []
        for paper in self.condition['corpus']['papers']:
            subset = [r for r in rows if r['job'].get('paper_id') == paper['paper_id']]
            cells = [dict(model=r['job']['profile_id'], replicate=r['job']['replicate_id'], status=r['status'], result=r['result'])
                     for r in subset if r['job']['kind'] == 'local_extraction']
            matrices.append({'paper_id': paper['paper_id'], 'cells': cells,
                             'terminal': all(r['status'] in TERMINAL for r in subset),
                             'successful_cells': sum(c['status'] == 'success' for c in cells)})
        result = {'schema_version': 'matrix-scheduler-summary/v1', 'condition': self.condition['condition'],
                'execution_mode': 'live' if self.condition['live'] else 'synthetic_mock',
                'terminal': all(r['status'] in TERMINAL for r in rows), 'matrices': matrices,
                'slot_counts': dict(Counter(r['status'] for r in rows)),
                'stage_counts': summarize_stages(rows),
                'dataset_jobs': [{'dataset_id': r['job']['dataset']['dataset_id'], 'status': r['status'], 'result': r['result']}
                                 for r in rows if r['job']['kind'] == 'dataset_parse'],
                'semantic_accuracy': None, 'packet_status': 'not_built_by_scheduler'}
        if self.condition['config']['execution'].get('grouped_mode') == 'all-groups/v1':
            model_rows = [r for r in rows if r['job']['kind'] in {'classification', 'local_extraction'}]
            completed = []
            for row in model_rows:
                if row['result']:
                    body = read_json(contained(self.root, row['result']))['body']
                    if body.get('schema_version') == 'group-execution-result/v2':
                        completed.append({'job_id': row['id'], 'kind': row['job']['kind'],
                                          'paper_id': row['job']['paper_id'],
                                          'generation_status': body['generation_status'],
                                          'admission_status': body['admission_status'],
                                          'counts': body['counts']})
            local = [r for r in completed if r['kind'] == 'local_extraction']
            classifier = [r for r in completed if r['kind'] == 'classification']
            planned_local = sum(r['job']['kind'] == 'local_extraction' for r in rows)
            result['group_execution'] = {
                'version': 'all-groups/v1', 'completed_jobs': completed,
                'planned_local_cells': planned_local,
                'returned_local_cells': sum(r['generation_status'] == 'complete' for r in local),
                'fully_admitted_local_cells': sum(r['admission_status'] == 'full' for r in local),
                'planned_classifier_jobs': sum(r['job']['kind'] == 'classification' for r in rows),
                'returned_classifier_jobs': sum(r['generation_status'] == 'complete' for r in classifier),
                'classification_group_counts': {key: sum(r['counts'][key] for r in classifier)
                    for key in ('planned_groups', 'accounted_groups', 'returned_groups', 'admitted_groups', 'missing_generation_groups')},
                'extraction_group_counts': {key: sum(r['counts'][key] for r in local)
                    for key in ('planned_groups', 'accounted_groups', 'returned_groups', 'admitted_groups', 'missing_generation_groups')},
                'group_count_scope': 'Finalized jobs only; pending/running job groups are not included.',
                'full_local_inference_coverage': bool(self.condition['live']) and len(local) == planned_local
                    and all(r['generation_status'] == 'complete' for r in local),
                'interpretation': 'Returned responses include authentic rejected answers; admission is separate.'}
        return result


def envelope_for(row, condition):
    job = row['job']
    index = 'indexes/' + digest(job['paper_id']) + '.json' if job['kind'] == 'local_extraction' else None
    body = {'condition': condition['condition'], 'job_id': row['id'], 'job': job,
            'attempt': row['attempt'], 'worker_id': row['worker'], 'index_path': index}
    body['request_id'] = digest(body)
    return body


def collect(store, sources, telemetry, *, revalidate=False):
    for row in store.rows():
        if revalidate and row['status'] == 'success' and not row['result']:
            raise ValueError('successful_job_missing_result')
        if row['status'] != 'running' and not (revalidate and row['result']):
            continue
        path = store.root / 'results' / row['id'] / ('attempt-' + str(row['attempt']) + '.json')
        result = read_optional(path)
        if result is None:
            if revalidate and row['result']:
                raise ValueError('committed_result_missing')
            continue
        envelope = envelope_for(row, store.condition)
        errors, index = verify_result(store.condition, envelope, result, sources, store.root)
        if errors:
            raise ValueError('result_replay_failed:' + ','.join(errors[:4]))
        if index:
            target = (store.root / 'indexes' / (digest(row['job']['paper_id']) + '.json')
                      if row['job']['kind'] == 'classification' else
                      store.root / 'extractions' / (row['id'] + '.json'))
            if target.exists() and read_json(target) != index:
                raise ValueError('frozen_index_conflict')
            if not target.exists():
                atomic_json(target, index)
        status = result['body']['status']
        if status not in TERMINAL:
            raise ValueError('unexpected_result_status')
        if revalidate and row['status'] != 'running':
            if row['status'] != status:
                raise ValueError('committed_status_mismatch')
            continue
        retry = status in {'transport_error', 'infrastructure_failed'} and row['attempt'] < store.condition['policy']['max_attempts']
        with store.db:
            store.db.execute('UPDATE jobs SET status=?,finished_at=?,result=? WHERE id=?',
                ('pending' if retry else status, time.time(), None if retry else path.relative_to(store.root).as_posix(), row['id']))
        duration_basis = result.get('duration_basis', 'worker_monotonic_execution')
        timing = ({'duration_s': result['duration_s']} if duration_basis == 'worker_monotonic_execution'
                  else {'duration_s_unmeasured_placeholder': result['duration_s']})
        telemetry.emit('task_committed', job_id=row['id'], worker_id=row['worker'], attempt=row['attempt'],
                       status=status, retry=retry, duration_basis=duration_basis, **timing)


def _recover_interrupted_attempts(store, telemetry):
    """Close claimed attempts only after the caller proves all worker leases free.

    The original request is mandatory. A missing or changed request is a ledger
    integrity error, never a reason to fabricate model output or skip an attempt.
    """
    for row in store.rows():
        if row['status'] != 'running':
            continue
        request_path = store.root / 'requests' / row['id'] / ('attempt-' + str(row['attempt']) + '.json')
        if not request_path.is_file() or read_json(request_path) != envelope_for(row, store.condition):
            raise ValueError('interrupted_attempt_request_missing_or_changed:' + row['id'])
        result_path = store.root / 'results' / row['id'] / ('attempt-' + str(row['attempt']) + '.json')
        if result_path.exists():
            raise ValueError('interrupted_attempt_result_already_exists:' + row['id'])
        request = read_json(request_path)
        result = seal({key: request[key] for key in ('condition', 'job_id', 'worker_id', 'attempt')} |
                      {'body': {'status': 'infrastructure_failed',
                                'error_type': 'InterruptedAttemptAfterLeaseRelease'},
                       'duration_s': 0.0, 'duration_basis': 'unmeasured_interrupted_attempt',
                       'recovery_basis': 'all_worker_and_gpu_leases_released'}, 'result_sha256')
        atomic_json(result_path, result)
        telemetry.emit('interrupted_attempt_recovered', job_id=row['id'], attempt=row['attempt'],
                       next_status='pending' if row['attempt'] < store.condition['policy']['max_attempts']
                       else 'infrastructure_failed')


def live_gate(condition, image, sources):
    if not condition['live']:
        return
    if not image or not image.is_file():
        raise ValueError('actual_sif_required')
    policy = condition['policy']
    execution_root = execution_source_root(policy)
    if code_identity(execution_root) != condition['source_file_bytes_sha256']:
        raise ValueError('execution_source_condition_mismatch')
    qualified = policy.get('qualification')
    if not isinstance(qualified, dict) or not qualified.get('path') or not qualified.get('file_bytes_sha256'):
        raise ValueError('measured_qualification_binding_required')
    path = Path(qualified['path'])
    if file_digest(path) != qualified['file_bytes_sha256']:
        raise ValueError('qualification_file_identity_mismatch')
    record = read_json(path)
    if (record.get('status') != 'pass' or record.get('image_file_bytes_sha256') != file_digest(image)
            or record.get('source_tree_sha256') != digest(condition['source_file_bytes_sha256'])):
        raise ValueError('qualification_runtime_identity_mismatch')
    active = set(condition['config']['roles']['locals']) | {condition['config']['classification']['profile_id']}
    selected = [p for p in condition['config']['profiles'] if p['profile_id'] in active]
    if policy.get('execution_source') is not None:
        if condition.get('coordinator_source_file_bytes_sha256') != code_identity():
            raise ValueError('coordinator_source_identity_mismatch')
        for profile in selected:
            deployment = profile.get('runtime', {}).get('deployment_identity', {})
            if (deployment.get('source_tree_canonical_sha256') != digest(condition['source_file_bytes_sha256'])
                    or deployment.get('image_file_bytes_sha256') != record['image_file_bytes_sha256']):
                raise ValueError('profile_execution_identity_mismatch')
    if checkpoint_binding_errors({'profiles': selected}, verify_bytes=True):
        raise ValueError('checkpoint_bytes_unverified')
    allocated = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    requested = [g for w in policy['workers'] for g in w['gpu_ids']]
    if not set(requested) <= set(allocated) or any(not g.startswith('GPU-') for g in requested):
        raise ValueError('live_workers_require_explicit_allocated_gpu_UUIDs')
    for worker in policy['workers']:
        if not worker['gpu_ids']:
            continue
        width = len(worker['gpu_ids'])
        for profile in selected:
            if worker.get('allowed_profile_ids') and profile['profile_id'] not in worker['allowed_profile_ids']:
                continue
            q = record.get('profiles', {}).get(profile_hash(profile), {})
            peaks = q.get('peak_bytes_per_gpu', [])
            budgets = worker.get('gpu_budget_bytes', [])
            margin = worker.get('gpu_reserve_bytes')
            if (q.get('gpu_count') != width or q.get('sequence_tokens', 0) < profile['context_window']
                    or len(peaks) != width or len(budgets) != width or not isinstance(margin, int) or margin <= 0
                    or any(not isinstance(v, int) or v <= 0 for v in peaks + budgets)
                    or any(p + margin > b for p, b in zip(peaks, budgets))
                    or not isinstance(q.get('cpu_peak_bytes'), int) or q['cpu_peak_bytes'] < 0
                    or q['cpu_peak_bytes'] > worker['ram_bytes']
                    or q.get('long_prefill_and_decode_passed') is not True):
                raise ValueError('profile_peak_context_or_headroom_unqualified')


def worker_command(root, sources, leases, worker, image=None):
    module = 'high_fidelity_schema_study.four_category.scheduler_worker'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('APPTAINERENV_', 'SINGULARITYENV_'))}
    env['CUDA_VISIBLE_DEVICES'] = ','.join(worker['gpu_ids'])
    env['OMP_NUM_THREADS'] = str(worker['cpu_threads'])
    env['PYTHONPATH'] = str(ROOT.parent)
    if image is None:
        return [sys.executable, '-u', '-m', module, '--root', str(root), '--sources', str(sources),
                '--leases', str(leases), '--worker-id', worker['worker_id']], env
    condition = read_json(root / 'condition.json')
    execution_root = execution_source_root(condition['policy'])
    for path in (execution_root, root, sources, leases):
        if any(x in str(path) for x in (':', ',', '\n', '\r')):
            raise ValueError('unsafe_apptainer_bind_path')
    env.update(APPTAINERENV_CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],
               APPTAINERENV_OMP_NUM_THREADS=env['OMP_NUM_THREADS'], APPTAINERENV_PYTHONPATH='/workspace',
               APPTAINERENV_HF_HUB_OFFLINE='1', APPTAINERENV_TRANSFORMERS_OFFLINE='1')
    env.update(APPTAINERENV_HF_HOME='/scheduler/workers/'+worker['worker_id']+'/cache/hf',
               APPTAINERENV_XDG_CACHE_HOME='/scheduler/workers/'+worker['worker_id']+'/cache',
               APPTAINERENV_TMPDIR='/scheduler/workers/'+worker['worker_id']+'/tmp')
    # Preserve model path semantics through an explicit read-only mount for each checkpoint.
    command = ['apptainer', 'exec', '--cleanenv', '--containall']
    if worker['gpu_ids']:
        command += ['--nv']
    command += ['--bind', str(execution_root)+':/workspace/high_fidelity_schema_study:ro', '--bind', str(root)+':/scheduler:rw',
                '--bind', str(sources)+':/inputs:ro', '--bind', str(leases)+':/leases:rw']
    for profile in condition['config']['profiles']:
        if profile['backend'] == 'transformers':
            modelpath = Path(profile['model_id'])
            if not modelpath.is_absolute() or any(c in str(modelpath) for c in (':', ',', '\n', '\r')):
                raise ValueError('absolute_safe_checkpoint_path_required')
            command += ['--bind', str(modelpath)+':'+str(modelpath)+':ro']
    command += [str(image), '/opt/phase1-venv/bin/python', '-u', '-m', module, '--root', '/scheduler',
                '--sources', '/inputs', '--leases', '/leases', '--worker-id', worker['worker_id']]
    return command, env


def run_scheduler(config, corpus, policy, *, root, sources, leases, image=None, allow_live=False,
                  start_watchdog=True, telegram_secret=None):
    root, sources, leases = Path(root).resolve(), Path(sources).resolve(), Path(leases).resolve()
    root.mkdir(parents=True, exist_ok=True)
    leases.mkdir(parents=True, exist_ok=True)
    if root == sources or root.is_relative_to(sources) or sources.is_relative_to(root):
        raise ValueError('scheduler_state_and_sources_must_be_disjoint')
    condition = compile_condition(config, corpus, policy, live=allow_live)
    live_gate(condition, image, sources)
    if allow_live:
        condition = seal({**condition, 'qualification_record': read_json(Path(policy['qualification']['path']))}, 'condition')
    processes, logs, watchdog = {}, [], None
    watchdog_restarts = 0
    with Lock(root / 'coordinator.lock'):
        condition_path = root / 'condition.json'
        if condition_path.exists() and read_json(condition_path) != condition:
            raise ValueError('resume_condition_changed_create_new_run')
        if not condition_path.exists():
            write_new(condition_path, condition)
        # Refuse to recover until old workers released their OS leases. No stale
        # heartbeat, PID lookup or timeout authorises running a duplicate request.
        with ExitStack() as locks:
            for worker in policy['workers']:
                locks.enter_context(Lock(root / 'workers' / worker['worker_id'] / 'worker.lock'))
            for gpu in sorted(g for w in policy['workers'] for g in w['gpu_ids']):
                locks.enter_context(Lock(leases / (digest(gpu)+'.lock')))
        if policy.get('group_cache_import'):
            from .cache_lineage import verify_cache_lineage
            errors, _ = verify_cache_lineage(root, condition, source_root=sources)
            if errors:
                raise ValueError('group_cache_lineage_invalid:' + ','.join(errors[:5]))
        store = Store(root, condition)
        telemetry = Telemetry(root / 'telemetry/events.jsonl', condition['condition'])
        telemetry.emit('coordinator_started', pid=os.getpid(), live=allow_live)
        collect(store, sources, telemetry, revalidate=True)
        # Workers and GPU leases were proved free above. Close every still-running
        # outer attempt before normal commit/retry, preserving all group caches.
        _recover_interrupted_attempts(store, telemetry)
        collect(store, sources, telemetry)
        try:
            atomic_json(root / 'snapshot.json', {'condition': condition['condition'], 'time': time.time(),
                        'terminal': False, 'workers': [], 'ready_jobs': 0})
            if start_watchdog:
                watchdog_cmd = [sys.executable, '-u', '-m', 'high_fidelity_schema_study.four_category.queue_health', '--root', str(root)]
                if telegram_secret:
                    watchdog_cmd += ['--enable-telegram', '--secret', str(telegram_secret)]
                watchdog_env = dict(os.environ, PYTHONPATH=str(ROOT.parent))
                watchdog_log = (root / 'watchdog.log').open('a', encoding='utf-8'); logs.append(watchdog_log)
                try:
                    with Lock(root / 'health/watchdog.lock'):
                        pass
                except RuntimeError:
                    telemetry.emit('existing_watchdog_observed')
                else:
                    watchdog = subprocess.Popen(watchdog_cmd, env=watchdog_env, stdin=subprocess.DEVNULL, stdout=watchdog_log, stderr=watchdog_log)
            for worker in policy['workers']:
                base = root / 'workers' / worker['worker_id']
                (base / 'tmp').mkdir(exist_ok=True)
                (base / 'cache').mkdir(exist_ok=True)
                # These are ephemeral control files, never scientific attempts.
                for name in ('stop.json', 'inbox.json', 'heartbeat.json'):
                    (base / name).unlink(missing_ok=True)
                command, env = worker_command(root, sources, leases, worker, image if allow_live else None)
                log = (base / 'process.log').open('a', encoding='utf-8')
                logs.append(log)
                processes[worker['worker_id']] = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                                                  stdout=log, stderr=log)
            while True:
                at = time.time()
                collect(store, sources, telemetry)
                store.refresh(at)
                rows = store.rows()
                running = {r['worker']: r for r in rows if r['status'] == 'running'}
                queue_context = worker_queue_context(rows, policy['workers'], at)
                workers_state, eligible_idle = [], []
                for worker in policy['workers']:
                    wid = worker['worker_id']
                    heartbeat = read_optional(root / 'workers' / wid / 'heartbeat.json')
                    process = processes[wid]
                    if process.poll() is not None:
                        raise RuntimeError('worker_process_exited:' + wid)
                    if wid not in running and heartbeat and heartbeat.get('phase') == 'idle':
                        choice = store.choose(worker, heartbeat.get('profile_id'), at)
                        if choice and store.claim(choice, wid, at):
                            row = next(r for r in store.rows() if r['id'] == choice['id'])
                            envelope = envelope_for(row, condition)
                            write_new(root / 'requests' / row['id'] / ('attempt-'+str(row['attempt'])+'.json'), envelope)
                            atomic_json(root / 'workers' / wid / 'inbox.json', envelope)
                            telemetry.emit('task_dispatched', job_id=row['id'], worker_id=wid, profile_id=row['job'].get('profile_id'),
                                           attempt=row['attempt'], ready_wait_s=at-row['ready_at'])
                        elif choice:
                            eligible_idle.append(wid)
                    if heartbeat:
                        workers_state.append({**heartbeat, **queue_context.get(wid, {})})
                summary = store.summary()
                ready = [r for r in store.rows() if r['status'] == 'pending' and r['ready_at'] is not None]
                snapshot = {'condition': condition['condition'], 'time': at, 'terminal': summary['terminal'],
                            'workers': workers_state, 'ready_jobs': len(ready), 'eligible_idle_workers': eligible_idle,
                            'oldest_ready_wait_s': max((at-r['ready_at'] for r in ready), default=0),
                            'slot_counts': summary['slot_counts'], 'completed_matrices': sum(m['terminal'] for m in summary['matrices'])}
                snapshot['stage_counts'] = summary['stage_counts']
                if 'group_execution' in summary:
                    snapshot['inference_coverage'] = {k: v for k, v in summary['group_execution'].items()
                                                      if k != 'completed_jobs'}
                snapshot['watchdog_status'] = 'external_or_disabled' if watchdog is None else 'running' if watchdog.poll() is None else 'failed'
                if start_watchdog and watchdog is None and not summary['terminal']:
                    try:
                        with Lock(root / 'health/watchdog.lock'):
                            pass
                    except RuntimeError:
                        pass
                    else:
                        watchdog = subprocess.Popen(watchdog_cmd, env=watchdog_env, stdin=subprocess.DEVNULL, stdout=watchdog_log, stderr=watchdog_log)
                if watchdog is not None and watchdog.poll() is not None and not summary['terminal'] and watchdog_restarts < 2:
                    watchdog_restarts += 1
                    telemetry.emit('watchdog_restarting', previous_exit=watchdog.returncode, restart=watchdog_restarts)
                    watchdog = subprocess.Popen(watchdog_cmd, env=watchdog_env, stdin=subprocess.DEVNULL, stdout=watchdog_log, stderr=watchdog_log)
                atomic_json(root / 'snapshot.json', snapshot)
                atomic_json(root / 'summary.json', summary)
                telemetry.emit('scheduler_sample', ready_jobs=len(ready), slot_counts=summary['slot_counts'],
                               completed_matrices=snapshot['completed_matrices'], worker_count=len(workers_state))
                if summary['terminal']:
                    telemetry.emit('coordinator_completed', slot_counts=summary['slot_counts'])
                    return summary
                time.sleep(policy['poll_s'])
        except BaseException as exc:
            telemetry.emit('coordinator_failed', error_type=type(exc).__name__)
            previous = read_optional(root / 'snapshot.json') or {'workers': []}
            atomic_json(root / 'snapshot.json', {**previous, 'condition': condition['condition'],
                        'time': time.time(), 'terminal': False, 'coordinator_failed': True})
            # Preserve a nonterminal snapshot; an independent watchdog reports
            # the ensuing missing coordinator heartbeat even after this exits.
            raise
        finally:
            for wid in processes:
                atomic_json(root / 'workers' / wid / 'stop.json', {'requested_at': time.time()})
            for process in processes.values():
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=5)
            store.db.close()
            for log in logs:
                log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'corpus', 'policy', 'root', 'sources', 'leases'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--allow-live', action='store_true')
    p.add_argument('--telegram-secret', type=Path)
    args = p.parse_args()
    result = run_scheduler(read_json(args.config), read_json(args.corpus), read_json(args.policy),
                           root=args.root, sources=args.sources, leases=args.leases, image=args.image,
                           allow_live=args.allow_live, telegram_secret=args.telegram_secret)
    print(json.dumps({'terminal': result['terminal'], 'slot_counts': result['slot_counts']}))


if __name__ == '__main__':
    main()
