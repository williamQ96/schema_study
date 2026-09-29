"""Operational health evaluation and durable incident transitions for scheduler V2."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone

DEFAULT_POLICY = {
    'heartbeat_s': 90, 'decode_no_progress_s': 300,
    'load_no_progress_s': 1800, 'prefill_no_progress_s': 1800,
    'dependency_starvation_s': 300, 'classification_ready_s': 1800,
    'extraction_ready_s': 7200, 'validation_backlog_s': 300,
    'reminder_s': 1800,
}
RANK = {'info': 0, 'warning': 1, 'critical': 2}
KINDS = {'scheduler_heartbeat_lost', 'coordinator_failed', 'worker_heartbeat_lost', 'worker_no_progress', 'dependency_starvation',
         'classification_queue_overdue', 'extraction_queue_overdue', 'validation_backlog',
         'planned_resume_failure', 'resource_ownership_conflict', 'reservation_expired',
         'reservation_wait_delayed', 'run_completed', 'run_completed_with_failures'}


def _seconds(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return None
    return None


def _age(now, since):
    value = _seconds(since)
    return max(0.0, now - value) if value is not None else None


def _policy(policy):
    values = {**DEFAULT_POLICY, **(policy or {})}
    if set(values) != set(DEFAULT_POLICY) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for v in values.values()
    ):
        raise ValueError('invalid_health_policy')
    return values


def evaluate(snapshot, now, policy=None):
    """Return operational issue observations; this makes no semantic accuracy claim."""
    if not isinstance(snapshot, dict):
        raise ValueError('snapshot_must_be_object')
    limits = _policy(policy)
    at = _seconds(now)
    if at is None:
        raise ValueError('invalid_now')
    issues = []

    def add(kind, scope, severity, age, threshold, **facts):
        issues.append({'kind': kind, 'scope': str(scope), 'severity': severity,
                       'age_s': age, 'threshold_s': threshold, 'facts': facts})

    if not snapshot.get('terminal'):
        if snapshot.get('coordinator_failed'):
            add('coordinator_failed', 'coordinator', 'critical', None, None)
        else:
            coordinator_age = _age(at, snapshot.get('time'))
            if coordinator_age is not None and coordinator_age > limits['heartbeat_s']:
                add('scheduler_heartbeat_lost', 'coordinator', 'critical', coordinator_age,
                    limits['heartbeat_s'])

    queue_workers = []
    for worker in snapshot.get('workers', []):
        wid = worker.get('worker_id', 'unknown')
        maintenance = worker.get('maintenance') or {}
        maint_state = maintenance.get('state')
        deadline = _seconds(maintenance.get('deadline'))
        confirmed_maintenance = bool(maintenance.get('confirmed')) and maint_state in {'paused', 'released', 'reserved'}
        if worker.get('reservation_expired') or (confirmed_maintenance and deadline is not None and at >= deadline):
            add('reservation_expired', wid, 'critical', at - deadline if deadline is not None else None,
                0, maintenance_state=maint_state)
        if worker.get('recovery_failed'):
            add('planned_resume_failure', wid, 'critical', None, None)
        if worker.get('ownership_conflict'):
            add('resource_ownership_conflict', wid, 'critical', None, None)

        before_maintenance_deadline = (confirmed_maintenance and deadline is not None and at < deadline)
        if not before_maintenance_deadline:
            queue_workers.append(worker)
        if worker.get('reservation_wait_overdue_s', 0) > 0:
            add('reservation_wait_delayed', wid, 'warning', worker['reservation_wait_overdue_s'] + 900,
                900, action='Wait for the safe group boundary; no forced preemption.')
        if not before_maintenance_deadline and not snapshot.get('terminal'):
            hb_age = _age(at, worker.get('heartbeat_at'))
            if hb_age is not None and hb_age > limits['heartbeat_s']:
                # Stale heartbeat is an observation; it never authorizes termination.
                add('worker_heartbeat_lost', wid, 'critical', hb_age, limits['heartbeat_s'],
                    phase=worker.get('phase'), job_id=worker.get('job_id'))
            phase = worker.get('phase')
            progress_limit = {'decode': limits['decode_no_progress_s'], 'decoding': limits['decode_no_progress_s'],
                              'load': limits['load_no_progress_s'], 'loading': limits['load_no_progress_s'],
                              'prefill': limits['prefill_no_progress_s']}.get(phase)
            progress_age = _age(at, worker.get('progress_at'))
            if progress_limit is not None and progress_age is not None and progress_age > progress_limit:
                add('worker_no_progress', wid, 'critical', progress_age, progress_limit,
                    phase=phase, job_id=worker.get('job_id'))

        dep_count = worker.get('dependency_wait_jobs', 0) or 0
        dep_age = _age(at, worker.get('dependency_idle_since'))
        if not before_maintenance_deadline and dep_count > 0 and dep_age is not None and dep_age > limits['dependency_starvation_s']:
            add('dependency_starvation', wid, 'critical' if dep_age >= 2 * limits['dependency_starvation_s'] else 'warning', dep_age, limits['dependency_starvation_s'],
                dependency_wait_jobs=dep_count, compatible_ready_jobs=worker.get('compatible_ready_jobs', 0))
        if worker.get('validation_pending'):
            validation_age = _age(at, worker.get('validation_progress_at'))
            if validation_age is not None and validation_age > limits['validation_backlog_s']:
                add('validation_backlog', wid, 'warning', validation_age, limits['validation_backlog_s'],
                    validation_pending=worker.get('validation_pending'))

    for kind, field, threshold_key in (
        ('classification_queue_overdue', 'oldest_classification_wait_s', 'classification_ready_s'),
        ('extraction_queue_overdue', 'oldest_extraction_wait_s', 'extraction_ready_s'),
    ):
        waits = [w.get(field, 0) or 0 for w in queue_workers]
        age = max(waits, default=0)
        if age > limits[threshold_key]:
            add(kind, 'queue', 'critical' if age >= 2 * limits[threshold_key] else 'warning', age, limits[threshold_key])
    return issues


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def transitions(state, snapshot, now, policy=None):
    """Update JSON-serializable incident ledger and emit open/escalate/reminder/recovery events."""
    if not isinstance(state, dict) or not isinstance(snapshot, dict):
        raise ValueError('state_and_snapshot_must_be_objects')
    at = _seconds(now)
    if at is None:
        raise ValueError('invalid_now')
    limits = _policy(policy)
    condition = snapshot.get('condition')
    incidents = state.setdefault('incidents', {})
    if state.get('condition') not in (None, condition):
        raise ValueError('condition_mismatch')
    state['condition'] = condition
    found = {i['kind'] + ':' + i['scope']: i for i in evaluate(snapshot, at, limits)}
    emitted = []

    def event_for(key, item, transition, severity, episode, opening_id=None):
        event = {'kind': item['kind'], 'scope': item['scope'], 'severity': severity,
                 'transition': transition, 'time': at, 'condition': condition,
                 'episode': episode, 'facts': {'age_s': item.get('age_s'),
                 'threshold_s': item.get('threshold_s'), **item.get('facts', {})}}
        if opening_id:
            event['opening_event_id'] = opening_id
        event['event_id'] = _hash(event)
        return event

    for key, issue in found.items():
        incident = incidents.get(key)
        if incident is None or not incident.get('active'):
            episode = (incident or {}).get('episode', 0) + 1
            event = event_for(key, issue, 'open', issue['severity'], episode)
            incidents[key] = {'active': True, 'episode': episode, 'opened_at': at,
                              'opening_event_id': event['event_id'], 'severity': issue['severity'],
                              'last_event_at': at}
            emitted.append(event)
        elif RANK[issue['severity']] > RANK.get(incident.get('severity'), 0):
            # The signed production bridge accepts reminder transitions; a higher
            # severity carries the escalation without changing that wire contract.
            event = event_for(key, issue, 'reminder', issue['severity'], incident['episode'],
                              incident.get('opening_event_id'))
            incident.update(severity=issue['severity'], last_event_at=at)
            emitted.append(event)
        elif at - incident.get('last_event_at', incident.get('opened_at', at)) >= limits['reminder_s']:
            # Reminder IDs change with time; bridge consumers must not interpret as a new invocation.
            event = event_for(key, issue, 'reminder', incident['severity'], incident['episode'],
                              incident.get('opening_event_id'))
            incident['last_event_at'] = at
            emitted.append(event)
    for key, incident in list(incidents.items()):
        if key in found or not incident.get('active'):
            continue
        kind, scope = key.split(':', 1)
        recovery_issue = {'kind': kind, 'scope': scope, 'facts': {}}
        event = event_for(key, recovery_issue, 'recovered', 'info', incident['episode'],
                          incident.get('opening_event_id'))
        incident['active'] = False
        incident['recovered_at'] = at
        incident['last_event_at'] = at
        emitted.append(event)
    finalization = snapshot.get('finalization_status') or {}
    if snapshot.get('terminal') and finalization.get('status') in {'pass', 'fail'} and not state.get('completion_event_id'):
        rejected = any(status not in {'success', 'deferred'} and count
                       for counts in snapshot.get('stage_counts', {}).values()
                       for status, count in counts.items())
        kind = 'run_completed_with_failures' if rejected or finalization['status'] == 'fail' else 'run_completed'
        issue = {'kind': kind, 'scope': 'run', 'facts': {'stage_counts': snapshot.get('stage_counts'),
                 'group_coverage': snapshot.get('group_coverage'), 'packet_status': finalization['status'],
                 'semantic_accuracy': None}}
        event = event_for(kind + ':run', issue, 'open', 'warning' if rejected else 'info', 1)
        state['completion_event_id'] = event['event_id']
        emitted.append(event)
    return state, emitted


_LABELS = {
    'en': {'scheduler_heartbeat_lost': 'Scheduler heartbeat is overdue', 'coordinator_failed': 'Scheduler coordinator reported a failure',
           'worker_heartbeat_lost': 'Worker heartbeat is overdue', 'worker_no_progress': 'Worker progress is overdue',
           'dependency_starvation': 'Worker is waiting on dependencies', 'classification_queue_overdue': 'Classification queue is overdue',
           'extraction_queue_overdue': 'Extraction queue is overdue', 'validation_backlog': 'Validation backlog is not advancing',
           'planned_resume_failure': 'Planned worker resume failed', 'resource_ownership_conflict': 'Resource ownership conflict detected',
           'reservation_expired': 'Maintenance reservation expired',
           'reservation_wait_delayed': 'Waiting for the reserved GPU to drain',
           'run_completed': 'Inference and packet validation completed',
           'run_completed_with_failures': 'Run completed with rejected results or packet failures'},
    'zh': {'scheduler_heartbeat_lost': '调度器心跳超时', 'coordinator_failed': '调度器协调器报告故障',
           'worker_heartbeat_lost': 'Worker 心跳超时', 'worker_no_progress': 'Worker 进度超时',
           'dependency_starvation': 'Worker 正在等待依赖任务', 'classification_queue_overdue': '分类队列等待超时',
           'extraction_queue_overdue': '提取队列等待超时', 'validation_backlog': '验证积压没有进展',
           'planned_resume_failure': '计划中的 Worker 恢复失败', 'resource_ownership_conflict': '检测到资源所有权冲突',
           'reservation_expired': '维护预留已过期',
           'reservation_wait_delayed': 'GPU 预约仍在等待安全释放',
           'run_completed': '推理与 packet 验证完成',
           'run_completed_with_failures': '本轮结束，含拒绝结果或 packet 验证失败'},
}


def render(event, language='en'):
    if language not in _LABELS:
        raise ValueError('unsupported_language')
    kind = event.get('kind')
    if kind not in KINDS:
        raise ValueError('unknown_health_kind')
    facts = event.get('facts') or {}
    age = facts.get('age_s')
    duration = f" Duration: {int(age)}s." if isinstance(age, (int, float)) else ''
    action_en = 'Inspect the scheduler report and worker state.'
    action_zh = '请检查调度器报告和 Worker 状态。'
    path = event.get('report_path') or 'scheduler_v2 health report'
    if language == 'zh':
        return f"{_LABELS[language][kind]}：{event.get('scope', 'unknown')}。{duration}请检查报告：{path}。{action_zh}"
    return f"{_LABELS[language][kind]}: {event.get('scope', 'unknown')}.{duration} Inspect report: {path}. {action_en}"
