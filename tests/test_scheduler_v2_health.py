import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scheduler_v2.health import evaluate, render, transitions


def snap(**overrides):
    value = {'deployment_id': 'deploy-1', 'condition': 'c' * 64, 'time': 100,
             'terminal': False, 'stage_counts': {}, 'workers': []}
    value.update(overrides)
    return value


def worker(**overrides):
    value = {'worker_id': 'gpu0', 'phase': 'decode', 'heartbeat_at': 100,
             'progress_at': 100, 'job_id': 'job1', 'group_id': 'group1',
             'compatible_ready_jobs': 0, 'dependency_wait_jobs': 0,
             'dependency_idle_since': None, 'oldest_classification_wait_s': 0,
             'oldest_extraction_wait_s': 0, 'validation_pending': 0,
             'validation_progress_at': 100, 'maintenance': None,
             'recovery_failed': False, 'ownership_conflict': False,
             'reservation_expired': False}
    value.update(overrides)
    return value


def test_confirmed_maintenance_suppresses_expected_liveness_until_deadline():
    w = worker(heartbeat_at=0, progress_at=0,
               maintenance={'state': 'paused', 'confirmed': True, 'deadline': 200})
    assert evaluate(snap(workers=[w]), 150) == []
    issues = evaluate(snap(time=200, workers=[w]), 200)
    assert {i['kind'] for i in issues} == {'reservation_expired', 'worker_heartbeat_lost'}
    assert next(i for i in issues if i['kind'] == 'reservation_expired')['severity'] == 'critical'


def test_unconfirmed_maintenance_does_not_suppress_real_heartbeat():
    w = worker(heartbeat_at=0, maintenance={'state': 'paused', 'confirmed': False, 'deadline': 500})
    assert 'worker_heartbeat_lost' in {i['kind'] for i in evaluate(snap(workers=[w]), 100)}


def test_stale_coordinator_is_reported_even_when_worker_heartbeat_is_fresh():
    w = worker(heartbeat_at=199, progress_at=199)
    issues = evaluate(snap(time=100, workers=[w]), 200)
    assert {i['kind'] for i in issues} == {'scheduler_heartbeat_lost'}
    assert issues[0]['scope'] == 'coordinator' and issues[0]['age_s'] == 100


def test_explicit_coordinator_failure_and_nonfinite_policy_are_handled():
    w = worker(heartbeat_at=199, progress_at=199)
    issues = evaluate(snap(coordinator_failed=True, workers=[w]), 200)
    assert {i['kind'] for i in issues} == {'coordinator_failed'}
    for invalid in (math.nan, math.inf, -math.inf):
        try:
            evaluate(snap(), 100, {'heartbeat_s': invalid})
        except ValueError as exc:
            assert str(exc) == 'invalid_health_policy'
        else:
            raise AssertionError(f'accepted non-finite policy: {invalid}')


def test_queue_thresholds_and_late_watchdog_report_duration():
    w = worker(heartbeat_at=999, progress_at=999, phase='idle',
               oldest_classification_wait_s=1801, oldest_extraction_wait_s=7201)
    got = evaluate(snap(time=1000, workers=[w]), 1000)
    assert {i['kind'] for i in got} == {'classification_queue_overdue', 'extraction_queue_overdue'}
    state, events = transitions({}, snap(time=1000, workers=[w]), 1000)
    assert len(events) == 2
    assert all(e['time'] == 1000 and e['transition'] == 'open' for e in events)
    assert 'Duration:' in render(events[0])


def test_open_escalation_reminder_recovery_and_durable_dedup():
    s = snap(time=301, workers=[worker(dependency_wait_jobs=1, dependency_idle_since=0,
                             heartbeat_at=301, progress_at=301)])
    state, opened = transitions({}, s, 301)
    opening_id = next(e['event_id'] for e in opened if e['kind'] == 'dependency_starvation')
    state = json.loads(json.dumps(state))  # durable restart round trip
    _, same = transitions(state, s, 302)
    assert same == []
    escalation_worker = worker(dependency_wait_jobs=1, dependency_idle_since=0, heartbeat_at=700, progress_at=700)
    state, escalated = transitions(state, snap(time=700, workers=[escalation_worker]), 700)
    assert len(escalated) == 1 and escalated[0]['transition'] == 'reminder'
    assert escalated[0]['severity'] == 'critical'
    assert escalated[0]['opening_event_id'] == opening_id
    reminder_worker = worker(dependency_wait_jobs=1, dependency_idle_since=0, heartbeat_at=2500, progress_at=2500)
    state, reminder = transitions(state, snap(time=2500, workers=[reminder_worker]), 2500)
    assert len(reminder) == 1 and reminder[0]['transition'] == 'reminder'
    state, recovered = transitions(state, snap(time=2501), 2501)
    assert len(recovered) == 1 and all(e['transition'] == 'recovered' for e in recovered)
    assert all(e['opening_event_id'] for e in recovered)
    _, again = transitions(state, snap(time=2502), 2502)
    assert again == []


def test_idle_maintenance_does_not_replace_worker_heartbeat_or_progress():
    w = worker(heartbeat_at=10, progress_at=10,
               maintenance={'state': 'reserved', 'confirmed': True, 'deadline': 500})
    # Heartbeat data stays in the snapshot and is only suppressed as an expected alert.
    assert w['heartbeat_at'] == 10
    assert not evaluate(snap(workers=[w]), 100)


def test_resource_validation_and_terminal_contract_are_operational_only():
    w = worker(ownership_conflict=True, validation_pending=4, validation_progress_at=0)
    issues = evaluate(snap(terminal=True, workers=[w]), 500)
    assert {i['kind'] for i in issues} == {'resource_ownership_conflict', 'validation_backlog'}
    text = render({'kind': 'validation_backlog', 'scope': 'gpu0', 'facts': {'age_s': 400}}, 'zh')
    assert '验证' in text and '400s' in text
def test_confirmed_maintenance_suppresses_queue_and_dependency_waits():
    from scheduler_v2.health import evaluate
    snapshot = {'time': 10000, 'workers': [{'worker_id': 'a', 'heartbeat_at': 0,
        'maintenance': {'state': 'released', 'confirmed': True, 'deadline': 11000},
        'dependency_wait_jobs': 9, 'dependency_idle_since': 0,
        'oldest_classification_wait_s': 9999, 'oldest_extraction_wait_s': 9999}]}
    assert evaluate(snapshot, 10000) == []
    assert 'reservation_expired' in {i['kind'] for i in evaluate(snapshot, 11001)}


def test_completion_waits_for_packet_attempt_and_emits_once():
    from scheduler_v2.health import transitions
    snapshot = {'condition': 'c', 'time': 1, 'terminal': True, 'workers': [], 'stage_counts': {}}
    state, events = transitions({}, snapshot, 1)
    assert events == []
    snapshot['finalization_status'] = {'status': 'pass'}
    state, events = transitions(state, snapshot, 2)
    assert [e['kind'] for e in events] == ['run_completed']
    from scheduler_v2.health import render
    assert 'completed' in render(events[0])
    assert transitions(state, snapshot, 3)[1] == []
