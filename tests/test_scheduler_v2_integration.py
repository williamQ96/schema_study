"""Offline group continuation across old scheduler, migration and V2 worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid

import pytest

from high_fidelity_schema_study.four_category.scheduler import run_scheduler
from scheduler_v2 import finalizer, migrate, runtime
from scheduler_v2.io import atomic_json, file_sha, read, write_once
from scheduler_v2.state import Ledger, TERMINAL
from .test_complete_scheduler import configure
from .test_matrix_scheduler import setup


@pytest.fixture
def short_root():
    # Frozen group paths contain two SHA-256 directory names; keep Windows paths short.
    with tempfile.TemporaryDirectory(prefix='v2-', ignore_cleanup_errors=True) as directory:
        yield Path(directory)


def _fixture(tmp_path, *, interrupt=True):
    sources, config, corpus, policy = setup(tmp_path)
    configure(config)
    old = tmp_path / 'old'
    old_summary = run_scheduler(config, corpus, policy, root=old, sources=sources,
                                leases=tmp_path / 'old-leases', start_watchdog=False)
    assert old_summary['terminal']
    condition = read(old / 'condition.json')
    target = next(j for j in condition['jobs'] if j['kind'] == 'local_extraction')
    group_dir = old / 'extraction_groups' / target['job_id']
    group = sorted(p for p in group_dir.iterdir() if p.is_dir())[0]
    removed = {}
    if interrupt:
        removed = {p.name: p.read_bytes() for p in group.iterdir()}
        for path in group.iterdir():
            path.unlink()
        group.rmdir()
        result = old / 'results' / target['job_id'] / 'attempt-1.json'
        result.unlink()
        artifact = old / 'extractions' / (target['job_id'] + '.json')
        artifact.unlink(missing_ok=True)
        with sqlite3.connect(old / 'scheduler.sqlite') as db:
            db.execute("UPDATE jobs SET status='pending',result=NULL,worker=NULL,ready_at=NULL WHERE id=?",
                       (target['job_id'],))
    for path in old.rglob('*'):
        if path.is_file():
            os.utime(path, (time.time() - 10, time.time() - 10))
    root = tmp_path / 'v2'
    package = Path(__file__).resolve().parents[1]
    deployment = dict(root=str(root), local_root=str(tmp_path / 'local'),
                      ancestor_root=str(old), sources=str(sources),
                      science_source=str(package), runtime_source=str(package),
                      deployment_id='mock-integration', execution_identity='mock-adapter-v1',
                      condition_file_sha256=file_sha(old / 'condition.json'),
                      local_leases=str(tmp_path / 'local-leases'),
                      legacy_leases=str(tmp_path / 'legacy-leases'),
                      qualification=str(tmp_path / 'qualification.json'))
    deployment['operational_files'] = {name: file_sha(package / name) for name in
        ('scheduler_v2/io.py', 'scheduler_v2/worker.py', 'scheduler_v2/science.py')}
    config_path = tmp_path / 'deployment-config.json'
    atomic_json(config_path, deployment)
    migrate.prepare(config_path)
    return deployment, config_path, condition, target, group.name, removed


def test_mock_worker_continues_one_missing_group_and_packet_replays(short_root):
    tmp_path = short_root
    deployment, config_path, condition, target, group_id, removed = _fixture(tmp_path)
    root, local = Path(deployment['root']), Path(deployment['local_root'])
    imported = runtime.checked_ref(root, read(root / 'import-latest.json'))['imported']
    assert imported[target['job_id']]['parent_status'] == 'pending'
    assert group_id not in imported[target['job_id']]['groups']
    preserved = {ref['path']: ref['file_bytes_sha256']
                 for outcome in imported[target['job_id']]['groups'].values()
                 for ref in outcome['attempt_refs']}
    plans = read(root / 'group_plans.json')
    worker = next(w for w in condition['policy']['workers'] if w['gpu_ids'] and
                  (not w.get('allowed_profile_ids') or target['profile_id'] in w['allowed_profile_ids']))
    incarnation = uuid.uuid4().hex
    command, env = runtime.worker_command(deployment, condition, worker, incarnation)
    assert command[:4] == [runtime.sys.executable, '-u', '-m', 'scheduler_v2.worker']
    with Ledger(local, root, deployment['deployment_id']) as ledger:
        ledger.initialize(condition['jobs'], plans, imported)
        epoch = ledger.new_epoch()
        ledger.register_worker(worker['worker_id'], incarnation, worker['gpu_ids'])
        ledger.mark_ready(target['job_id'], group_id, 1)
        assignment = ledger.claim(target['job_id'], group_id, worker['worker_id'], incarnation, epoch, 2)
        assert assignment
        envelope = {**assignment, 'condition': condition['condition']}
        write_once(root / 'dispatches' / (assignment['assignment_id'] + '.json'), envelope)
        base = local / 'workers' / worker['worker_id']
        base.mkdir(parents=True, exist_ok=True)
        atomic_json(base / 'inbox.json', envelope)
        with (tmp_path / 'worker.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            try:
                receipt_path = (root / 'worker_receipts' / worker['worker_id'] / incarnation /
                                ('result-' + assignment['assignment_id'] + '.json'))
                until = time.time() + 45
                while not receipt_path.exists() and process.poll() is None and time.time() < until:
                    time.sleep(.1)
                assert receipt_path.exists(), (tmp_path / 'worker.log').read_text()
                receipt = read(receipt_path)
                assert receipt['status'] == 'returned', receipt
                ref = {'path': receipt_path.relative_to(root).as_posix(),
                       'file_bytes_sha256': file_sha(receipt_path)}
                runtime.checked_worker_receipt(root, assignment, ref)
                assert ledger.acknowledge(assignment['assignment_id'], ref, worker['worker_id'], incarnation, epoch)
            finally:
                atomic_json(base / 'control.json', {'action': 'drain'})
                process.wait(timeout=20)
        outcome = runtime.validation_task(deployment, 'group', target['job_id'], group_id)
        current = next(a for a in ledger.snapshot()['assignments'] if a['assignment_id'] == assignment['assignment_id'])
        assert runtime.commit_verified_result(ledger, root, current, outcome, 3)
        parent = runtime.validation_task(deployment, 'parent', target['job_id'])
        assert ledger.set_parent(target['job_id'], parent['status'], parent['result_ref'], parent['index_ref'], 4)
        assert all(j['status'] in TERMINAL for j in ledger.snapshot()['jobs'])
    finalizer.run(config_path)
    packet = root.parent / 'packets' / condition['corpus']['papers'][0]['paper_id']
    assert finalizer.verify_packet(packet, deployment['sources'],
                                   expected_condition=condition['condition'])['derivation_validity'] == 'pass'
    assert all(file_sha(root / path) == sha for path, sha in preserved.items())
    assert removed
    manifest = read(packet / 'manifest.json')
    parent_path = 'v2_results/' + target['job_id'] + '.json'
    tampered = read(packet / parent_path)
    tampered['body']['status'] = 'contract_invalid'
    atomic_json(packet / parent_path, tampered)
    manifest['files'][parent_path] = file_sha(packet / parent_path)
    atomic_json(packet / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='packet_parent_identity_invalid'):
        finalizer.verify_packet(packet, deployment['sources'], expected_condition=condition['condition'])


def test_packet_explicitly_replays_blocked_dependency_proof(tmp_path, monkeypatch):
    from high_fidelity_schema_study.four_category import common, scheduled_packet
    monkeypatch.setattr(common, 'seal_errors', lambda *args: [])
    jobs = [{'job_id': 'soft', 'kind': 'soft_reference', 'dependencies': []},
            {'job_id': 'extract', 'kind': 'local_extraction', 'dependencies': ['soft']}]
    monkeypatch.setattr(scheduled_packet, '_jobs', lambda condition: jobs)
    monkeypatch.setattr(scheduled_packet, '_selection', lambda condition, paper_id:
                        {'job_ids': ['soft', 'extract']})
    packet = tmp_path / 'packet'
    event_path = packet / 'scheduler_v2' / 'd' / 'events' / '00000001.json'
    atomic_json(packet / 'condition.json', {'condition': 'c'})
    atomic_json(event_path, {'seq': 1, 'prev_sha': None, 'kind': 'blocked_dependency',
                             'job_id': 'extract', 'dependency_id': 'soft'})
    files = {'condition.json': file_sha(packet / 'condition.json'),
             event_path.relative_to(packet).as_posix(): file_sha(event_path)}
    manifest = {'schema_version': 'scheduled-paper-packet/v3', 'paper_id': 'p',
                'deployment_id': 'd', 'job_ids': ['soft', 'extract'], 'files': files,
                'terminal_states': {'soft': {'status': 'deferred', 'result_ref': None,
                                             'index_ref': None, 'group_statuses': []},
                                    'extract': {'status': 'blocked_dependency', 'result_ref': None,
                                                'index_ref': None,
                                                'group_statuses': ['blocked_dependency']}}}
    atomic_json(packet / 'manifest.json', manifest)
    assert finalizer.verify_packet(packet, tmp_path, expected_condition='c')['derivation_validity'] == 'pass'
    event = read(event_path)
    event['dependency_id'] = 'other'
    atomic_json(event_path, event)
    manifest['files'][event_path.relative_to(packet).as_posix()] = file_sha(event_path)
    atomic_json(packet / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='packet_blocked_dependency_proof_missing'):
        finalizer.verify_packet(packet, tmp_path, expected_condition='c')


def test_full_coordinator_loop_mock_completes_and_finalizes(short_root, monkeypatch):
    deployment, config_path, condition, target, group_id, _ = _fixture(short_root)
    deployment['authority_lock'] = str(short_root / 'authority.lock')
    atomic_json(config_path, deployment)
    root, local = Path(deployment['root']), Path(deployment['local_root'])
    original_launch = runtime.launch_worker
    launched = {}
    def launch(config, scientific_condition, worker):
        receipt = original_launch(config, scientific_condition, worker)
        launched[receipt['incarnation']] = receipt['spawn_pid']
        return receipt
    monkeypatch.setattr(runtime, 'launch_worker', launch)
    monkeypatch.setattr(runtime, 'worker_identity', lambda incarnation:
                        {'pid': launched[incarnation]} if incarnation in launched else None)
    monkeypatch.setattr(runtime, 'same', lambda expected: True)
    monkeypatch.setattr(runtime, 'resources_free', lambda *args, **kwargs: True)
    result = {}
    def coordinator():
        try:
            runtime.run(config_path)
            result['status'] = 'returned'
        except BaseException as exc:
            result['error'] = exc
    thread = threading.Thread(target=coordinator, daemon=True)
    thread.start()
    thread.join(timeout=45)
    if thread.is_alive():
        for worker in condition['policy']['workers']:
            atomic_json(local / 'workers' / worker['worker_id'] / 'control.json', {'action': 'drain'})
        pytest.fail('mock_coordinator_did_not_terminate_in_45s')
    assert result == {'status': 'returned'}, result
    assert read(root / 'completed.json')['terminal']
    with Ledger(local, root, deployment['deployment_id']) as ledger:
        assert all(j['status'] in TERMINAL for j in ledger.snapshot()['jobs'])
    finalizer.run(config_path)
    packet = root.parent / 'packets' / condition['corpus']['papers'][0]['paper_id']
    assert finalizer.verify_packet(packet, deployment['sources'],
                                   expected_condition=condition['condition'])['derivation_validity'] == 'pass'


def test_v2_ancestor_multi_hop_replay_retains_lineage(short_root):
    first, _, condition, target, _, _ = _fixture(short_root, interrupt=False)
    first_root = Path(first['root'])
    imported = runtime.checked_ref(first_root, read(first_root / 'import-latest.json'))['imported']
    with Ledger(first['local_root'], first_root, first['deployment_id']) as ledger:
        ledger.initialize(condition['jobs'], read(first_root / 'group_plans.json'), imported)
        assert all(j['status'] in TERMINAL for j in ledger.snapshot()['jobs'])
    write_once(first_root / 'dispatches' / 'proof.json', {'assignment_id': 'proof'})
    write_once(first_root / 'worker_receipts' / 'gpu' / 'inc' / 'result-proof.json',
               {'assignment_id': 'proof', 'status': 'returned'})
    second = dict(first, root=str(short_root / 'hop2'), local_root=str(short_root / 'hop2-local'),
                  ancestor_root=str(first_root), deployment_id='mock-hop2', ancestor_quiescent=True)
    config_path = short_root / 'hop2-config.json'
    atomic_json(config_path, second)
    migrate.prepare(config_path)
    second_root = Path(second['root'])
    manifest = runtime.checked_ref(second_root, read(second_root / 'import-latest.json'))
    assert manifest['ancestor_ledger_rows'][target['job_id']]['ancestor_deployment'] == first['deployment_id']
    assert manifest['imported'][target['job_id']]['parent_status'] == 'success'
    assert list((second_root / 'ancestors').rglob('node.json'))
    assert len(list((second_root / 'ancestors').rglob('result-proof.json'))) == 1
    assert len(list((second_root / 'ancestors').rglob('proof.json'))) == 1
    with Ledger(second['local_root'], second_root, second['deployment_id']) as ledger:
        ledger.initialize(condition['jobs'], read(second_root / 'group_plans.json'), manifest['imported'])
        assert all(j['status'] in TERMINAL for j in ledger.snapshot()['jobs'])
