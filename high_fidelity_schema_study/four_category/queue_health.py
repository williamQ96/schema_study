"""Independent queue-health watchdog with durable incidents and Telegram outbox.

Dry delivery is the default. A stalled scheduler cannot stall this process.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import urllib.request

from .common import digest
from .scheduler_io import Lock, atomic_json, read_optional, Telemetry

DEFAULTS = {'scheduler_heartbeat_s': 90, 'worker_heartbeat_s': 90,
            'load_no_progress_s': 1800, 'prefill_no_progress_s': 1800,
            'decode_no_progress_s': 300, 'other_no_progress_s': 1800,
            'imbalance_hold_s': 300, 'ready_wait_s': 1800,
            'queue_hold_s': 120, 'recovery_hold_s': 60, 'cooldown_s': 1800,
            'poll_s': 10, 'snapshot_stale_s': 90}
DEFAULTS.update(utilization_gap_pct=60, utilization_skew_hold_s=600, headroom_hold_s=60)
DEFAULTS.update(classification_min_completed=2, classification_failure_fraction=.5)


def thresholds(overrides=None):
    values = {**DEFAULTS, **(overrides or {})}
    if (set(values) != set(DEFAULTS) or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            or (v < 0 if name == 'classification_failure_fraction' else v <= 0)
            for name, v in values.items())):
        raise ValueError('health_thresholds_must_be_known_positive_finite_seconds')
    if values['utilization_gap_pct'] > 100:
        raise ValueError('utilization_gap_must_not_exceed_100_percent')
    if values['classification_failure_fraction'] > 1:
        raise ValueError('classification_failure_fraction_must_not_exceed_1')
    if (isinstance(values['classification_min_completed'], bool)
            or not isinstance(values['classification_min_completed'], int)):
        raise ValueError('classification_min_completed_must_be_integer')
    return values


from .scheduler_observability import summarize_stages


def _systemic_classifier(stage_counts, limits, inference_coverage=None):
    group_counts = (inference_coverage or {}).get('classification_group_counts')
    if isinstance(group_counts, dict):
        completed = group_counts.get('accounted_groups', 0)
        failures = completed - group_counts.get('admitted_groups', 0)
        return completed >= limits['classification_min_completed'] and failures / max(1, completed) >= limits['classification_failure_fraction'], completed, failures
    counts = stage_counts.get('classification', {})
    completed_statuses = {'success', 'contract_invalid', 'invalid_request', 'truncated',
                          'transport_error', 'infrastructure_failed', 'refused', 'unavailable',
                          'completed_with_rejections', 'generation_incomplete'}
    completed = sum(count for status, count in counts.items() if status in completed_statuses)
    failures = sum(count for status, count in counts.items() if status in completed_statuses - {'success'})
    return completed >= limits['classification_min_completed'] and failures / completed >= limits['classification_failure_fraction'], completed, failures


def observations(snapshot, at, limits):
    """No-progress and liveness are separate. Waiting on dependencies is not ready."""
    if not isinstance(snapshot, dict):
        return []
    issues = []
    stage_counts = snapshot.get('stage_counts')
    if isinstance(stage_counts, dict):
        rejected, completed, failures = _systemic_classifier(stage_counts, limits, snapshot.get('inference_coverage'))
        if rejected:
            issues.append(('systemic_classification_rejection', 'classification', 'critical', 0,
                           {'stage_counts': stage_counts, 'completed': completed, 'failures': failures}))
    if snapshot.get('terminal'):
        return issues
    if snapshot.get('coordinator_failed'):
        issues.append(('coordinator_failed', 'scheduler', 'critical', 0, {}))
    elif at - snapshot['time'] > limits['scheduler_heartbeat_s']:
        issues.append(('scheduler_heartbeat_lost', 'scheduler', 'critical', 0,
                       {'age_s': at - snapshot['time'], 'threshold_s': limits['scheduler_heartbeat_s']}))
    fresh = at - snapshot['time'] <= limits['snapshot_stale_s']
    utilisation = []
    for worker in snapshot.get('workers', []):
        dependency_idle = (fresh and not worker.get('job_id')
                           and worker.get('compatible_ready_jobs') == 0
                           and worker.get('dependency_wait_jobs', 0) > 0)
        metrics = worker.get('gpu_metrics', [])
        if metrics and at - worker['heartbeat_at'] <= limits['worker_heartbeat_s']:
            if not dependency_idle:
                utilisation.append((worker['worker_id'], sum(m['utilization_pct'] for m in metrics)/len(metrics)))
            margin = worker.get('gpu_reserve_bytes', 0)
            if margin and any(m['total_bytes']-m['used_bytes'] < margin for m in metrics):
                issues.append(('gpu_headroom_low', worker['worker_id'], 'warning', limits['headroom_hold_s'],
                               {'threshold_s': limits['headroom_hold_s'], 'reserve_bytes': margin}))
        if not worker.get('job_id'):
            continue
        if at - worker['heartbeat_at'] > limits['worker_heartbeat_s']:
            issues.append(('worker_heartbeat_lost', worker['worker_id'], 'critical', 0,
                           {'age_s': at-worker['heartbeat_at'], 'threshold_s': limits['worker_heartbeat_s']}))
            continue
        phase = worker.get('phase')
        key = {'loading': 'load_no_progress_s', 'prefill': 'prefill_no_progress_s',
               'decoding': 'decode_no_progress_s'}.get(phase, 'other_no_progress_s')
        age = at - worker['progress_at']
        if age > limits[key]:
            issues.append(('worker_no_progress', worker['worker_id'], 'critical', 0,
                           {'age_s': age, 'threshold_s': limits[key], 'phase': phase, 'job_id': worker['job_id']}))
    if fresh and snapshot.get('ready_jobs', 0):
        if len(utilisation) >= 2:
            gap = max(v for _, v in utilisation) - min(v for _, v in utilisation)
            if gap >= limits['utilization_gap_pct']:
                issues.append(('sustained_utilization_skew', 'queue', 'warning', limits['utilization_skew_hold_s'],
                               {'gap_pct': gap, 'threshold_pct': limits['utilization_gap_pct'],
                                'threshold_s': limits['utilization_skew_hold_s']}))
        # Only idle workers that can actually accept a ready task count. Unequal
        # utilisation while every compatible worker is busy is not an imbalance.
        idle = list(snapshot.get('eligible_idle_workers', []))
        workers_by_id = {w.get('worker_id'): w for w in snapshot.get('workers', [])}
        idle = [wid for wid in idle if not (workers_by_id.get(wid, {}).get('compatible_ready_jobs') == 0
                                             and workers_by_id.get(wid, {}).get('dependency_wait_jobs', 0) > 0)]
        if idle and any(w.get('job_id') for w in snapshot.get('workers', [])):
            dependency_wait = sum(w.get('dependency_wait_jobs', 0) for w in snapshot.get('workers', []))
            issues.append(('allocation_imbalance', 'queue', 'warning', limits['imbalance_hold_s'],
                           {'idle_workers': idle, 'ready_jobs': snapshot['ready_jobs'],
                            'dependency_wait_jobs': dependency_wait,
                            'threshold_s': limits['imbalance_hold_s']}))
        if snapshot.get('oldest_ready_wait_s', 0) > limits['ready_wait_s']:
            issues.append(('ready_queue_overdue', 'queue', 'warning', limits['queue_hold_s'],
                           {'age_s': snapshot['oldest_ready_wait_s'], 'threshold_s': limits['ready_wait_s'],
                            'ready_jobs': snapshot['ready_jobs']}))
    return issues


def evaluate(state, snapshot, at, limits):
    """Durable debounce, episode identity, cooldown and recovery hysteresis."""
    found = {kind + ':' + scope: (kind, scope, severity, hold, facts)
             for kind, scope, severity, hold, facts in observations(snapshot, at, limits)}
    events = []
    incidents = state.setdefault('incidents', {})
    for key, (kind, scope, severity, hold, facts) in found.items():
        item = incidents.setdefault(key, {'since': at, 'active': False, 'last_sent': None, 'episode': 0})
        item.pop('clear_since', None)
        if at - item['since'] >= hold and (not item['active'] or at - item['last_sent'] >= limits['cooldown_s']):
            if not item['active']:
                item['episode'] += 1
            event = {'kind': kind, 'scope': scope, 'severity': severity, 'transition': 'open' if not item['active'] else 'reminder',
                     'facts': facts, 'time': at, 'condition': snapshot['condition'], 'episode': item['episode']}
            event['event_id'] = digest(event)
            events.append(event)
            item.update(active=True, last_sent=at)
    for key in list(incidents):
        if key in found:
            continue
        item = incidents[key]
        if not item['active']:
            del incidents[key]
            continue
        item.setdefault('clear_since', at)
        if at - item['clear_since'] < limits['recovery_hold_s']:
            continue
        if item['active']:
            kind, scope = key.split(':', 1)
            event = {'kind': kind, 'scope': scope, 'severity': 'info', 'transition': 'recovered',
                     'facts': {}, 'time': at, 'condition': snapshot['condition'], 'episode': item['episode']}
            event['event_id'] = digest(event)
            events.append(event)
        del incidents[key]
    return events


def render_alert(event, reports, language='en'):
    labels = {
        'en': {'scheduler_heartbeat_lost': 'Scheduler heartbeat missing', 'worker_heartbeat_lost': 'Worker heartbeat missing',
               'coordinator_failed': 'Scheduler stopped after an internal failure; inspect the event log',
               'run_completed': 'Queue finished; all slots have final states',
               'run_completed_with_failures': 'Queue finished with incomplete or rejected results',
               'gpu_headroom_low': 'GPU safety headroom is below its configured reserve',
               'sustained_utilization_skew': 'GPU worker utilization is persistently uneven; inspect workload and I/O',
               'worker_no_progress': 'Task has stopped reporting progress', 'allocation_imbalance': 'Capacity is idle while compatible tasks wait',
               'ready_queue_overdue': 'Ready tasks have waited beyond the configured limit',
               'systemic_classification_rejection': 'Completed classifier outcomes are predominantly rejected or failed; inspect operational records'},
        'zh': {'scheduler_heartbeat_lost': '调度器心跳中断', 'worker_heartbeat_lost': 'Worker 心跳中断',
               'coordinator_failed': '调度器遇到内部错误后停止，请检查事件日志',
               'run_completed': '队列已结束，全部槽位已有最终状态',
               'run_completed_with_failures': '队列已结束，部分结果未完成或未通过检查',
               'gpu_headroom_low': 'GPU 剩余显存低于配置的安全余量',
               'sustained_utilization_skew': 'GPU worker 利用率持续不均，请检查任务差异和 I/O',
               'worker_no_progress': '任务长时间没有报告进展', 'allocation_imbalance': '有兼容任务等待，但可用 worker 持续空闲',
               'ready_queue_overdue': '可执行任务等待超过阈值',
               'systemic_classification_rejection': '已完成的分类器结果多数被拒绝或失败，请检查运行记录'}}
    zh = language == 'zh'
    title = labels[language][event['kind']]
    transition = {'open': '告警' if zh else 'ALERT', 'reminder': '持续告警' if zh else 'REMINDER', 'recovered': '恢复' if zh else 'RECOVERED'}[event['transition']]
    if event['kind'].startswith('run_completed'):
        transition = '任务结束' if zh else 'FINISHED'
    lines = ['Mercury | ' + transition, title, ('范围：' if zh else 'Scope: ') + event['scope']]
    facts = event['facts']
    if 'age_s' in facts:
        lines.append(('已持续：' if zh else 'Duration: ') + str(round(facts['age_s'])) + ' s')
    if 'threshold_s' in facts:
        lines.append(('阈值：' if zh else 'Threshold: ') + str(round(facts['threshold_s'])) + ' s')
    if 'ready_jobs' in facts:
        lines.append(('等待任务：' if zh else 'Ready jobs: ') + str(facts['ready_jobs']))
    if facts.get('dependency_wait_jobs', 0):
        lines.append(('依赖等待任务：' if zh else 'Jobs waiting on dependencies: ') + str(facts['dependency_wait_jobs']))
    if 'gap_pct' in facts:
        lines.append(('利用率差距：' if zh else 'Utilization gap: ') + str(round(facts['gap_pct'])) + ' percentage points')
    if 'successful_slots' in facts:
        lines.append(('通过输出检查的槽位：' if zh else 'Slots passing output checks: ') + str(facts['successful_slots']))
        lines.append(('失败或受阻槽位：' if zh else 'Failed/blocked slots: ') + str(facts['unsuccessful_slots']))
        lines.append(('暂缓槽位：' if zh else 'Deferred slots: ') + str(facts['deferred_slots']))
    if 'stage_counts' in facts:
        names = {'classification': 'Classifier', 'local_extraction': 'Extraction',
                 'dataset_parse': 'Dataset', 'soft_reference': 'Deferred reference'}
        names_zh = {'classification': '分类器', 'local_extraction': '提取',
                    'dataset_parse': '数据集', 'soft_reference': '暂缓参考'}
        for stage, counts in facts['stage_counts'].items():
            label = (names_zh if zh else names).get(stage, stage)
            summary = ', '.join(f'{status}={count}' for status, count in sorted(counts.items())) or '0'
            lines.append(label + '：' + summary if zh else label + ': ' + summary)
    coverage = facts.get('inference_coverage')
    if isinstance(coverage, dict):
        lines.append(('已有完整响应的本地槽位：' if zh else 'Local cells with all responses returned: ')
                     + str(coverage.get('returned_local_cells', 0)) + '/' + str(coverage.get('planned_local_cells', 0)))
        lines.append(('全部响应通过合约的本地槽位：' if zh else 'Local cells with every response admitted: ')
                     + str(coverage.get('fully_admitted_local_cells', 0)))
        missing = coverage.get('extraction_group_counts', {}).get('missing_generation_groups', 0)
        lines.append(('已结束槽位中仍缺失的生成分组：' if zh else 'Missing generation groups in finalized cells: ') + str(missing))
    from datetime import datetime, timezone
    lines.append(datetime.fromtimestamp(event['time'], timezone.utc).isoformat())
    lines.append(('运行标识：' if zh else 'Run: ') + event['condition'][:16])
    lines += [('报告：' if zh else 'Reports: ') + str(reports),
              '这是运行健康告警，不是模型准确率结论。不会因此自动改变模型或参数。' if zh else
              'Operational health alert, not an accuracy conclusion; no model or parameter changes are triggered.']
    return '\n'.join(lines)[:3900]


def telegram(secret, text):
    """No credential-bearing URL or exception body is emitted."""
    if not secret.get('enabled') or not isinstance(secret.get('chat_id'), int):
        raise ValueError('telegram_destination_not_bound')
    def call(method, body):
        request = urllib.request.Request('https://api.telegram.org/bot' + secret['bot_token'] + '/' + method,
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)
        except Exception as exc:
            raise RuntimeError('telegram_delivery_' + type(exc).__name__) from None
    bot = call('getMe', {})
    if not bot.get('ok') or bot['result'].get('username') != secret.get('bot_username'):
        raise RuntimeError('telegram_bot_identity_mismatch')
    result = call('sendMessage', {'chat_id': secret['chat_id'], 'text': text})
    if not result.get('ok') or result['result']['chat']['id'] != secret['chat_id']:
        raise RuntimeError('telegram_receipt_rejected')
    return result['result']['message_id']


def cycle(root, limits, *, at=None, sender=None, language='en'):
    root, at = Path(root), time.time() if at is None else at
    snapshot = read_optional(root / 'snapshot.json')
    prior = read_optional(root / 'health/state.json') or {}
    if snapshot is None:
        snapshot = prior.get('last_snapshot')
    if snapshot is None:
        return None
    # Heartbeats are read independently, even if the scheduler is frozen.
    snapshot = dict(snapshot)
    snapshot['workers'] = [dict(w, **(read_optional(root / 'workers' / w['worker_id'] / 'heartbeat.json') or {}))
                           for w in snapshot.get('workers', [])]
    path = root / 'health' / 'state.json'
    state = read_optional(path) or {'condition': snapshot['condition'], 'outbox': {}, 'incidents': {}}
    if state['condition'] != snapshot['condition']:
        raise ValueError('watchdog_condition_mismatch')
    if state.get('thresholds_sha256') != digest(limits):
        state.setdefault('threshold_history', []).append({'time': at, 'values': limits, 'sha256': digest(limits)})
        state['thresholds_sha256'] = digest(limits)
    state['last_snapshot'] = snapshot
    for event in evaluate(state, snapshot, at, limits):
        state['outbox'][event['event_id']] = {'event': event, 'attempts': 0, 'next_retry': at, 'sent': False}
        Telemetry(root / 'health/events.jsonl', snapshot['condition']).emit('health_transition', **event)
    if snapshot.get('terminal') and snapshot.get('slot_counts') and not state.get('terminal_event_created'):
        counts = snapshot['slot_counts']
        failed = sum(v for k, v in counts.items() if k not in {'success', 'deferred'})
        event = {'kind': 'run_completed_with_failures' if failed else 'run_completed', 'scope': 'queue',
                 'severity': 'warning' if failed else 'info', 'transition': 'open', 'time': at,
                 'condition': snapshot['condition'], 'episode': 1,
                 'facts': {'successful_slots': counts.get('success', 0), 'deferred_slots': counts.get('deferred', 0),
                           'unsuccessful_slots': failed}}
        if isinstance(snapshot.get('stage_counts'), dict):
            event['facts']['stage_counts'] = snapshot['stage_counts']
            event['facts'].pop('successful_slots', None)
            event['facts'].pop('unsuccessful_slots', None)
            event['facts'].pop('deferred_slots', None)
        if isinstance(snapshot.get('inference_coverage'), dict):
            event['facts']['inference_coverage'] = snapshot['inference_coverage']
        event['event_id'] = digest(event)
        state['outbox'][event['event_id']] = {'event': event, 'attempts': 0, 'next_retry': at, 'sent': False}
        state['terminal_event_created'] = True
    atomic_json(path, state)  # Persist before attempting external delivery.
    for row in state['outbox'].values():
        if row['sent'] or row['next_retry'] > at or sender is None:
            continue
        try:
            message_id = sender(render_alert(row['event'], root, language))
            row.update(sent=True, message_id=message_id, delivered_at=at)
        except Exception as exc:
            row['attempts'] += 1
            row['last_error_type'] = type(exc).__name__
            row['next_retry'] = at + min(300, 5 * 2 ** min(row['attempts'], 6))
        atomic_json(path, state)
    return snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--secret', type=Path)
    p.add_argument('--enable-telegram', action='store_true')
    p.add_argument('--language', choices=['en', 'zh'], default='en')
    p.add_argument('--thresholds', type=Path, help='Independent versioned override; does not alter scientific tasks')
    args = p.parse_args()
    condition = json.loads((args.root / 'condition.json').read_text(encoding='utf-8'))
    limits = thresholds(json.loads(args.thresholds.read_text(encoding='utf-8')) if args.thresholds else condition['policy'].get('health'))
    sender = None
    if args.enable_telegram:
        if not args.secret or args.secret.resolve().is_relative_to(args.root.resolve()):
            raise ValueError('external_secret_path_required')
        secret = json.loads(args.secret.read_text(encoding='utf-8-sig'))
        sender = lambda text: telegram(secret, text)
    with Lock(args.root / 'health/watchdog.lock'):
        while True:
            snap = cycle(args.root, limits, sender=sender, language=args.language)
            state = read_optional(args.root / 'health/state.json') or {}
            if snap and snap.get('terminal') and not state.get('incidents'):
                if sender is None or all(r['sent'] for r in state.get('outbox', {}).values()):
                    break
            time.sleep(limits['poll_s'])


if __name__ == '__main__':
    main()
