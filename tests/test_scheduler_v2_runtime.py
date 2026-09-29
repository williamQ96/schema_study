from __future__ import annotations

import pytest

from scheduler_v2 import runtime
from scheduler_v2.health import evaluate
from scheduler_v2.io import write_once
from scheduler_v2.state import Ledger


def assignment(tmp_path):
    root = tmp_path / 'shared'
    ledger = Ledger(tmp_path / 'local', root, 'd')
    ledger.initialize([{'job_id': 'j', 'kind': 'classification', 'paper_id': 'p'}], {'j': ['g']})
    epoch = ledger.new_epoch()
    ledger.register_worker('gpu', 'inc', ['0'])
    ledger.mark_ready('j', 'g', 1)
    a = ledger.claim('j', 'g', 'gpu', 'inc', epoch, 2)
    write_once(root / 'dispatches' / (a['assignment_id'] + '.json'),
               {**a, 'condition': 'c'})
    return ledger, root, a


def test_verified_outcome_must_match_worker_receipt(tmp_path):
    ledger, root, a = assignment(tmp_path)
    try:
        outcome = {'status': 'success', 'generation_returned': True}
        path = root / 'worker_receipts' / 'gpu' / 'inc' / ('result-' + a['assignment_id'] + '.json')
        sha = write_once(path, {**a, 'condition': 'c', 'status': 'returned', 'outcome': outcome})
        ref = {'path': path.relative_to(root).as_posix(), 'file_bytes_sha256': sha}
        assert ledger.acknowledge(a['assignment_id'], ref, 'gpu', 'inc', a['epoch'], 3)
        current = ledger.snapshot()['assignments'][0]
        with pytest.raises(ValueError, match='verified_outcome_differs'):
            runtime.commit_verified_result(ledger, root, current,
                                           {'status': 'refused', 'generation_returned': True}, 4)
        assert runtime.commit_verified_result(ledger, root, current, outcome, 4)
    finally:
        ledger.close()


def test_receipt_identity_and_immutable_reference(tmp_path):
    ledger, root, a = assignment(tmp_path)
    try:
        path = root / 'worker_receipts' / 'gpu' / 'inc' / ('result-' + a['assignment_id'] + '.json')
        sha = write_once(path, {**a, 'group_id': 'wrong', 'status': 'returned', 'outcome': {}})
        ref = {'path': path.relative_to(root).as_posix(), 'file_bytes_sha256': sha}
        with pytest.raises(ValueError, match='worker_receipt_identity_mismatch'):
            runtime.checked_worker_receipt(root, a, ref)
        with pytest.raises(ValueError, match='artifact_reference_changed'):
            runtime.checked_ref(root, {**ref, 'file_bytes_sha256': '0' * 64})
    finally:
        ledger.close()


def test_make_ready_blocks_irrecoverable_dependency(tmp_path):
    root = tmp_path / 'shared'
    with Ledger(tmp_path / 'local', root, 'd') as ledger:
        jobs = [{'job_id': 'c', 'kind': 'classification', 'paper_id': 'p'},
                {'job_id': 'e', 'kind': 'local_extraction', 'paper_id': 'p',
                 'dependencies': ['c']}]
        ledger.initialize(jobs, {'c': ['g'], 'e': ['g']},
                          {'c': {'parent_status': 'contract_invalid'}})
        runtime.make_ready(ledger, ledger.snapshot(), 10)
        snap = ledger.snapshot()
        assert snap['jobs'][1]['status'] == 'blocked_dependency'
        assert snap['jobs'][1]['groups'][0]['status'] == 'blocked_dependency'


def test_returned_classifier_group_reopens_same_paper_before_next_dispatch(tmp_path):
    root = tmp_path / 'shared'
    worker = {'worker_id': 'gpu', 'gpu_ids': ['0'], 'allowed_profile_ids': ['classifier']}
    with Ledger(tmp_path / 'local', root, 'd') as ledger:
        ledger.initialize([
            {'job_id': 'p06', 'kind': 'classification', 'paper_id': 'P06',
             'profile_id': 'classifier', 'dependencies': []},
            {'job_id': 'p07', 'kind': 'classification', 'paper_id': 'P07',
             'profile_id': 'classifier', 'dependencies': []}],
            {'p06': ['p06-g1', 'p06-g2', 'p06-g3'], 'p07': ['p07-g1']})
        epoch = ledger.new_epoch()
        ledger.register_worker('gpu', 'inc', ['0'])
        runtime.make_ready(ledger, ledger.snapshot(), 10)
        first = ledger.claim('p06', 'p06-g1', 'gpu', 'inc', epoch, 11)
        write_once(root / 'dispatches' / (first['assignment_id'] + '.json'),
                   {**first, 'condition': 'c'})
        path = root / 'worker_receipts' / 'gpu' / 'inc' / ('result-' + first['assignment_id'] + '.json')
        sha = write_once(path, {**first, 'condition': 'c', 'status': 'returned',
                                'outcome': {'status': 'success', 'generation_returned': True}})
        ref = {'path': path.relative_to(root).as_posix(), 'file_bytes_sha256': sha}
        runtime.checked_worker_receipt(root, first, ref)
        assert ledger.acknowledge(first['assignment_id'], ref, 'gpu', 'inc', epoch, 12)
        # The old dispatch order would choose P07 here because P06-g2 was not ready.
        assert runtime.choose(ledger.snapshot(), worker, 12)['job_id'] == 'p07'
        runtime.make_ready(ledger, ledger.snapshot(), 12)
        choice = runtime.choose(ledger.snapshot(), worker, 12)
        assert choice == {'job_id': 'p06', 'group_id': 'p06-g2', 'lane': 'classification'}
        second = ledger.claim(choice['job_id'], choice['group_id'], 'gpu', 'inc', epoch, 12)
        assert second
        runtime.make_ready(ledger, ledger.snapshot(), 12)
        assert next(j for j in ledger.snapshot()['jobs'] if j['job_id'] == 'p06')['groups'][2]['ready_at'] is None
        assert ledger.acknowledge(second['assignment_id'], {'path': 'raw2', 'file_bytes_sha256': 'pinned'},
                                  'gpu', 'inc', epoch, 13)
        runtime.make_ready(ledger, ledger.snapshot(), 13)
        assert ledger.claim('p06', 'p06-g3', 'gpu', 'inc', epoch, 13) is None


def test_stall_probe_requires_two_windows_and_failed_reply(tmp_path, monkeypatch):
    base = tmp_path / 'worker'
    process = {'incarnation': 'one'}
    confirmed = {'pid': 123}
    heartbeat = {'incarnation': 'one', 'heartbeat_at': 1}
    killed = []
    monkeypatch.setattr(runtime, 'same', lambda p: True)
    monkeypatch.setattr(runtime, 'terminate', lambda p: killed.append(p))
    assert not runtime.probe_stalled_worker(base, process, confirmed, heartbeat, 100, 10)
    assert not runtime.probe_stalled_worker(base, process, confirmed, heartbeat, 119, 10)
    assert runtime.probe_stalled_worker(base, process, confirmed, heartbeat, 121, 10)
    assert killed == [confirmed]


def test_live_probe_reply_vetoes_stall_termination(tmp_path, monkeypatch):
    base = tmp_path / 'worker'
    process = {'incarnation': 'one'}
    heartbeat = {'incarnation': 'one', 'heartbeat_at': 1}
    monkeypatch.setattr(runtime, 'same', lambda p: True)
    monkeypatch.setattr(runtime, 'terminate', lambda p: pytest.fail('must not terminate'))
    assert not runtime.probe_stalled_worker(base, process, {'pid': 123}, heartbeat, 100, 10)
    probe = runtime.optional(base / 'probe.json')
    runtime.atomic_json(base / 'probe-reply.json', {'incarnation': 'one',
                        'nonce': probe['nonce'], 'time': 105})
    assert not runtime.probe_stalled_worker(base, process, {'pid': 123}, heartbeat, 121, 10)


def test_launch_backoff_persists_across_calls(tmp_path, monkeypatch):
    attempts = []
    def fail(*args):
        attempts.append(1)
        raise OSError('launch failed')
    monkeypatch.setattr(runtime, 'launch_worker', fail)
    config = {'local_root': str(tmp_path)}
    worker = {'worker_id': 'gpu'}
    assert runtime.launch_with_backoff(config, {}, worker, 100)[0] is None
    assert runtime.launch_with_backoff(config, {}, worker, 101)[1] == 'worker_launch_backoff'
    assert len(attempts) == 1
    runtime.launch_with_backoff(config, {}, worker, 111)
    assert len(attempts) == 2


def test_maintenance_deadline_stable_and_existing_health_alert(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        ledger.control('pause', 'global', now=100)
        ledger.control('drain', 'target', {'worker_id': 'gpu0', 'deadline_s': 120}, now=110)
        controls = ledger.snapshot()['controls']
        assert runtime.maintenance_deadline(controls, 'gpu0', []) == 230
        assert runtime.maintenance_deadline(controls, 'gpu1', []) == 1000
        assert runtime.maintenance_deadline(controls, 'gpu0', [{'expires_at': 210}]) == 210
        assert runtime.maintenance_deadline(controls, 'gpu0', []) == 230
        observation = {'terminal': False, 'workers': [{'worker_id': 'gpu0',
                       'maintenance': {'state': 'released', 'confirmed': True,
                                       'deadline': runtime.maintenance_deadline(controls, 'gpu0', [])}}]}
        assert not any(issue['kind'] == 'reservation_expired' for issue in evaluate(observation, 229))
        assert any(issue['kind'] == 'reservation_expired' for issue in evaluate(observation, 230))
