import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scheduler_v2.io import atomic_json, read
from scheduler_v2.watchdog import cycle, merge_worker_heartbeats
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import production_health_events


def snapshot(**overrides):
    value = {'deployment_id': 'dep-1', 'condition': 'a' * 64, 'time': 10000,
             'terminal': False, 'stage_counts': {}, 'workers': []}
    value.update(overrides)
    return value


def worker(**overrides):
    value = {'worker_id': 'gpu0', 'phase': 'idle', 'heartbeat_at': 10000,
             'progress_at': 10000, 'job_id': None, 'group_id': None,
             'compatible_ready_jobs': 0, 'dependency_wait_jobs': 0,
             'dependency_idle_since': None, 'oldest_classification_wait_s': 2000,
             'oldest_extraction_wait_s': 0, 'validation_pending': 0,
             'validation_progress_at': 10000,
             'maintenance': {'state': 'active', 'confirmed': False, 'deadline': None},
             'recovery_failed': False, 'ownership_conflict': False, 'reservation_expired': False}
    value.update(overrides)
    return value


def roots(tmp_path):
    shared, local = tmp_path / 'shared', tmp_path / 'local'
    shared.mkdir(); local.mkdir()
    return shared, local


def config(condition):
    return {'condition': condition}


def test_sent_event_is_durable_deduplicated_and_bridge_compatible(tmp_path):
    shared, local = roots(tmp_path)
    atomic_json(local / 'snapshot.json', snapshot(workers=[worker()]))
    sent = []
    def sender(text):
        sent.append(text)
        return 82
    cycle(shared, local, at=10000, sender=sender)
    first_state = json.loads((shared / 'health' / 'state.json').read_text())
    row = next(iter(first_state['outbox'].values()))
    assert row['sent'] and row['message_id'] == 82
    assert 'classification queue' in sent[0].lower()
    assert cycle(shared, local, at=10001, sender=sender)
    assert len(sent) == 1
    assert production_health_events.candidates(first_state, config('a' * 64))[0]['row']['event'] == row['event']
    status = read(shared / 'health' / 'watchdog_status.json')
    assert status['condition'] == 'a' * 64 and status['pending'] == 0
    assert read(local / 'watchdogheartbeat.json')['status'] == 'ok'


def test_transport_failure_retries_with_bounded_schedule_and_no_error_body(tmp_path):
    shared, local = roots(tmp_path)
    atomic_json(local / 'snapshot.json', snapshot(workers=[worker()]))
    calls = []
    def sender(_):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('secret-token-raw-response')
        return 7
    cycle(shared, local, at=10000, sender=sender)
    state = read(shared / 'health' / 'state.json')
    row = next(iter(state['outbox'].values()))
    assert row['attempts'] == 1 and row['next_retry'] == 10010
    assert row['last_error_type'] == 'RuntimeError'
    assert 'secret-token' not in json.dumps(state)
    cycle(shared, local, at=10009, sender=sender)
    assert len(calls) == 1
    cycle(shared, local, at=10010, sender=sender)
    assert len(calls) == 2
    assert next(iter(read(shared / 'health' / 'state.json')['outbox'].values()))['sent']


def test_worker_heartbeat_merge_is_whitelisted_and_keeps_coordinator_maintenance(tmp_path):
    _, local = roots(tmp_path)
    base_worker = worker(maintenance={'state': 'paused', 'confirmed': True, 'deadline': 20000},
                         dependency_wait_jobs=3, dependency_idle_since=9000)
    hb = {'phase': 'prefill', 'heartbeat_at': 9999, 'progress_at': 9998,
          'job_id': 'job-live', 'group_id': 'group-live', 'maintenance': {'state': 'active'},
          'dependency_wait_jobs': 0, 'worker_id': 'spoofed-id', 'condition': 'spoofed-condition',
          'arbitrary_payload': 'must-ignore'}
    atomic_json(local / 'workers' / 'gpu0' / 'heartbeat.json', hb)
    merged = merge_worker_heartbeats(snapshot(workers=[base_worker]), local)['workers'][0]
    assert merged['phase'] == 'prefill' and merged['heartbeat_at'] == 9999
    assert merged['worker_id'] == 'gpu0' and merged.get('condition') != 'spoofed-condition'
    assert merged['job_id'] == 'job-live' and merged['group_id'] == 'group-live'
    assert merged['maintenance'] == base_worker['maintenance']
    assert merged['dependency_wait_jobs'] == 3 and merged['dependency_idle_since'] == 9000
    assert 'arbitrary_payload' not in merged


def test_stale_local_snapshot_uses_newer_durable_snapshot(tmp_path):
    shared, local = roots(tmp_path)
    atomic_json(local / 'snapshot.json', snapshot(time=100))
    shared_health = shared / 'health'
    atomic_json(shared_health / 'state.json', {'condition': 'a' * 64, 'incidents': {}, 'outbox': {},
                                               'last_snapshot': snapshot(time=150)})
    observed = cycle(shared, local, at=1000, snapshot_stale_s=90)
    assert observed['time'] == 150


def test_lost_snapshot_falls_back_and_opens_delayed_issue(tmp_path):
    shared, local = roots(tmp_path)
    old = snapshot(time=1, workers=[worker(oldest_classification_wait_s=2400)])
    atomic_json(shared / 'health' / 'state.json', {'condition': 'a' * 64, 'incidents': {},
                                                   'outbox': {}, 'last_snapshot': old})
    observed = cycle(shared, local, at=2000)
    assert observed['time'] == 1
    state = read(shared / 'health' / 'state.json')
    assert any(row['event']['kind'] == 'classification_queue_overdue'
               for row in state['outbox'].values())
