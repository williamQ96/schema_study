"""Independent V2 health observer with a durable, bridge-compatible outbox."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .health import DEFAULT_POLICY, render, transitions
from .io import Lock, atomic_json, digest, optional, read

WORKER_FIELDS = frozenset({
    'phase', 'heartbeat_at', 'progress_at', 'job_id', 'group_id',
    'compatible_ready_jobs', 'dependency_wait_jobs', 'dependency_idle_since',
    'oldest_classification_wait_s', 'oldest_extraction_wait_s',
    'validation_pending', 'validation_progress_at', 'maintenance',
    'recovery_failed', 'ownership_conflict', 'reservation_expired',
})
# Coordinator-owned signals must be preserved even if a worker heartbeat contains
# similarly named fields; local heartbeats own liveness and execution phase only.
WORKER_OWNED_FIELDS = frozenset({'phase', 'heartbeat_at', 'progress_at', 'job_id', 'group_id'})


def _read_object(path):
    try:
        value = read(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def merge_worker_heartbeats(snapshot, local_root):
    """Merge whitelisted per-worker liveness while retaining coordinator policy facts."""
    merged = dict(snapshot)
    workers = []
    for source in snapshot.get('workers', []):
        worker = dict(source)
        worker_id = worker.get('worker_id')
        if isinstance(worker_id, str):
            heartbeat = _read_object(Path(local_root) / 'workers' / worker_id / 'heartbeat.json') or {}
            for key in WORKER_FIELDS & WORKER_OWNED_FIELDS:
                if key in heartbeat:
                    worker[key] = heartbeat[key]
        workers.append(worker)
    merged['workers'] = workers
    return merged


def _load_snapshot(local_root, previous, at, stale_after_s=90):
    current = _read_object(Path(local_root) / 'snapshot.json')
    last = previous.get('last_snapshot') if isinstance(previous.get('last_snapshot'), dict) else None
    if current is None:
        return last
    current_time = current.get('time')
    try:
        stale = current_time is None or at - float(current_time) > stale_after_s
    except (TypeError, ValueError):
        stale = True
    # A stale local coordinator file may still contain the freshest queue view.
    # Prefer the persisted snapshot only when its coordinator timestamp is newer.
    if stale and last is not None:
        try:
            if float(last.get('time', -1)) > float(current_time if current_time is not None else -1):
                return last
        except (TypeError, ValueError):
            return last
    return current


def _read_secret(path, shared_root, local_root, science_source):
    resolved = Path(path).resolve()
    roots = [Path(p).resolve() for p in (shared_root, local_root, science_source)]
    if any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError('external_secret_path_required')
    secret = read(resolved)
    if not isinstance(secret, dict) or not secret.get('enabled'):
        raise ValueError('telegram_secret_not_enabled')
    return secret


def _sender(secret, science_source, callable_sender=None):
    if callable_sender is not None:
        return callable_sender
    # The Telegram implementation is loaded from the frozen science source tree,
    # which validates both bot username and destination chat before acknowledging.
    sys.path.insert(0, str(Path(science_source).resolve()))
    from high_fidelity_schema_study.four_category.queue_health import telegram
    return lambda text: telegram(secret, text)


def cycle(shared_root, local_root, *, at=None, policy=None, sender=None,
          snapshot_stale_s=90, bridge_config=None):
    """Evaluate one local snapshot, checkpoint incidents, then independently send due rows."""
    shared_root, local_root = Path(shared_root), Path(local_root)
    now = time.time() if at is None else float(at)
    state_path = shared_root / 'health' / 'state.json'
    persisted = optional(state_path)
    if persisted is not None and not isinstance(persisted, dict):
        raise ValueError('watchdog_state_must_be_object')
    state = persisted or {'outbox': {}, 'incidents': {}, 'threshold_history': []}
    snapshot = _load_snapshot(local_root, state, now, snapshot_stale_s)
    if snapshot is None:
        atomic_json(local_root / 'watchdogheartbeat.json', {'time': now, 'status': 'waiting_for_snapshot'})
        return None
    snapshot = merge_worker_heartbeats(snapshot, local_root)
    snapshot['finalization_status'] = optional(shared_root / 'finalization-status.json')
    if state.get('condition') not in (None, snapshot.get('condition')):
        raise ValueError('watchdog_condition_mismatch')

    thresholds = {**DEFAULT_POLICY, **(policy or {})}
    policy_hash = digest(thresholds)
    if state.get('thresholds_sha256') != policy_hash:
        state.setdefault('threshold_history', []).append({'time': now, 'values': thresholds,
                                                          'sha256': policy_hash})
        state['thresholds_sha256'] = policy_hash
    state, events = transitions(state, snapshot, now, thresholds)
    state['last_snapshot'] = snapshot
    outbox = state.setdefault('outbox', {})
    for event in events:
        outbox.setdefault(event['event_id'], {'event': event, 'attempts': 0,
                                              'next_retry': now, 'sent': False})
    # Durable before any potentially blocking NFS or network work.
    atomic_json(state_path, state)
    atomic_json(local_root / 'watchdogheartbeat.json', {'time': now, 'status': 'ok',
                                                         'events_created': len(events),
                                                         'condition': snapshot.get('condition')})

    sent_this_cycle = 0
    for row in outbox.values():
        if row.get('sent') or row.get('next_retry', 0) > now or sender is None:
            continue
        try:
            display_event = dict(row['event'])
            display_event['report_path'] = str(bridge_config or (shared_root / 'health' / 'state.json'))
            message_id = sender(render(display_event))
            if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
                raise ValueError('telegram_receipt_invalid')
            row.update(sent=True, message_id=message_id, delivered_at=now)
            sent_this_cycle += 1
        except Exception as exc:
            attempts = int(row.get('attempts', 0)) + 1
            row['attempts'] = attempts
            # Record only an exception class, never a credential, URL, response, or body.
            row['last_error_type'] = type(exc).__name__
            row['next_retry'] = now + min(300, 5 * (2 ** min(attempts, 6)))
        atomic_json(state_path, state)

    status = {'time': now, 'condition': snapshot.get('condition'),
              'events_created': len(events), 'events_sent': sent_this_cycle,
              'pending': sum(not row.get('sent') for row in outbox.values())}
    atomic_json(shared_root / 'health' / 'watchdog_status.json', status)
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Shared results directory')
    parser.add_argument('--local', type=Path, required=True, help='Host-local control directory')
    parser.add_argument('--science-source', type=Path, required=True)
    parser.add_argument('--telegram-secret', type=Path)
    parser.add_argument('--bridge-config', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--snapshot-stale-s', type=float, default=90)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.snapshot_stale_s <= 0:
        raise ValueError('watchdog_intervals_must_be_positive')
    sender = None
    if args.telegram_secret:
        secret = _read_secret(args.telegram_secret, args.root, args.local, args.science_source)
        sender = _sender(secret, args.science_source)
    policy = read(args.policy) if args.policy else None
    with Lock(args.local / 'watchdog.lock'):
        while True:
            cycle(args.root, args.local, sender=sender, policy=policy,
                  snapshot_stale_s=args.snapshot_stale_s, bridge_config=args.bridge_config)
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
