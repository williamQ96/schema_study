"""Fixed-role command and inspection interface for Mercury scheduler V2."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import time
import uuid
from pathlib import Path

from .io import digest, file_sha, optional, read, write_once
from .processes import same


def _config(path):
    value = read(path)
    for name in ('root', 'local_root', 'deployment_id'):
        if not value.get(name):
            raise ValueError('deployment_config_missing:' + name)
    return value


def create_command(config, action, payload, command_id=None):
    command_id = command_id or uuid.uuid4().hex
    if not isinstance(command_id, str) or not command_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in command_id):
        raise ValueError('invalid_command_id')
    command = {'command_id': command_id, 'action': action, 'payload': payload}
    path = Path(config['local_root']) / 'commands' / (command_id + '.json')
    write_once(path, command)
    return {'command': command, 'path': str(path)}


def _condition(config):
    path = Path(config['root']) / 'condition.json'
    if file_sha(path) != config.get('condition_file_sha256'):
        raise ValueError('condition_identity_changed')
    return read(path)


def _worker_payload(config, worker_id):
    if worker_id is None:
        return {}
    worker_ids = {w['worker_id'] for w in _condition(config)['policy']['workers']}
    if worker_id not in worker_ids:
        raise ValueError('unknown_worker:' + worker_id)
    return {'worker_id': worker_id}


def status(config):
    snapshot = optional(Path(config['local_root']) / 'snapshot.json')
    if snapshot is None:
        return {'status': 'waiting_for_snapshot', 'snapshot': None}
    reasons = Counter()
    worker_rows = []
    for worker in snapshot.get('workers', []):
        row = dict(worker)
        reason = row.get('waiting_reason')
        if not reason:
            if row.get('error') or row.get('recovery_failed'):
                reason = 'worker_error_or_recovery'
            elif row.get('maintenance'):
                reason = 'maintenance_' + str(row['maintenance'].get('state', 'active'))
            elif row.get('validation_pending'):
                reason = 'awaiting_validation'
            elif row.get('dependency_wait_jobs') and not row.get('compatible_ready_jobs'):
                reason = 'dependency_wait'
            elif not row.get('compatible_ready_jobs'):
                reason = 'no_compatible_ready_job'
            else:
                reason = 'dispatchable_or_running'
        row['waiting_reason'] = str(reason)
        worker_rows.append(row)
        reasons[str(reason)] += 1
    return {'status': 'ok', 'time': snapshot.get('time'), 'condition': snapshot.get('condition'),
            'deployment_id': snapshot.get('deployment_id'), 'epoch': snapshot.get('epoch'),
            'stage_counts': snapshot.get('stage_counts', {}), 'group_coverage': snapshot.get('group_coverage', {}),
            'waiting_reasons': dict(reasons), 'workers': worker_rows,
            'controls': snapshot.get('controls', {}), 'terminal': snapshot.get('terminal', False),
            'report_path': snapshot.get('report_path')}


def reconcile(config):
    """Read-only ledger/process summary; never removes locks or repairs state."""
    local, root = Path(config['local_root']), Path(config['root'])
    snapshot = optional(local / 'snapshot.json')
    coordinator = optional(local / 'coordinator-process.json')
    if coordinator and isinstance(coordinator, dict) and coordinator.get('pid'):
        coordinator['alive_same_identity'] = same(coordinator)
    assignments = [a for a in (snapshot or {}).get('assignments', []) if a.get('status') in ('assigned', 'pending_verification')]
    if snapshot is not None and 'assignments' not in snapshot:
        # Summary snapshots intentionally omit the assignment table. Reconstruct
        # only active ownership from the immutable event journal, without opening
        # SQLite or mutating/replaying scheduler state.
        active = {}
        journal = root / 'scheduler_v2' / config['deployment_id'] / 'events'
        for path in sorted(journal.glob('*.json')):
            event = read(path)
            kind = event.get('kind')
            if kind == 'claim':
                active[event['assignment']['assignment_id']] = {**event['assignment'], 'status': 'assigned'}
            elif kind in ('acknowledge', 'closed_raw') and event.get('assignment_id') in active:
                active[event['assignment_id']]['status'] = 'pending_verification'
            elif kind in ('commit', 'recover'):
                active.pop(event['assignment_id'], None)
        assignments = list(active.values())
    return {'status': 'read_only', 'coordinator': coordinator,
            'unfinished_assignments': assignments,
            'unreceipted_commands': [p.name for p in (local / 'commands').glob('*.json')
                                     if not (root / 'control_receipts' / p.name).exists()],
            'journal_path': str(root / 'scheduler_v2' / config['deployment_id'] / 'events'),
            'action_taken': False}


def verify(config):
    """Check frozen deployment pins and code hashes only; performs no inference."""
    errors = []
    root = Path(config['root'])
    condition_path = root / 'condition.json'
    try:
        condition_valid = condition_path.is_file() and file_sha(condition_path) == config.get('condition_file_sha256')
    except OSError:
        condition_valid = False
    condition = None
    if condition_valid:
        try:
            condition = read(condition_path)
        except Exception:
            condition_valid = False
    if not condition_valid:
        errors.append('condition_identity_mismatch')
    else:
        try:
            actual_identity = _science_code_identity(Path(config['science_source']))
            if condition.get('source_file_bytes_sha256') != actual_identity:
                errors.append('scientific_source_identity_mismatch')
        except Exception as exc:
            errors.append('scientific_source_identity_unavailable:' + str(exc))
    operational = config.get('operational_files')
    if not isinstance(operational, dict) or not operational:
        errors.append('operational_code_hash_pins_missing')
    else:
        for name, expected in operational.items():
            path = Path(config['runtime_source']) / name
            if not path.is_file() or file_sha(path) != expected:
                errors.append('operational_code_hash_mismatch:' + name)
    latest_path = root / 'import-latest.json'
    try:
        latest = read(latest_path)
        rel = Path(latest['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in latest['path']:
            raise ValueError('unsafe_import_reference')
        manifest_path = root / rel
        if manifest_path.is_symlink() or file_sha(manifest_path) != latest['file_bytes_sha256']:
            raise ValueError('import_manifest_reference_changed')
        manifest = read(manifest_path)
        for key in ('artifact_manifest_sha256', 'group_plan_sha256', 'execution_identity'):
            if not manifest.get(key):
                errors.append('historical_manifest_missing:' + key)
        configured_identity = config.get('execution_identity')
        if not isinstance(configured_identity, dict) or manifest.get('execution_identity') != configured_identity:
            errors.append('historical_execution_identity_mismatch')
        members_path = root / 'imports' / ('members-' + str(manifest.get('artifact_manifest_sha256')) + '.json')
        if not members_path.is_file() or digest(read(members_path)) != manifest.get('artifact_manifest_sha256'):
            errors.append('historical_artifact_manifest_mismatch')
        plans_path = root / 'group_plans.json'
        if not plans_path.is_file() or digest(read(plans_path)) != manifest.get('group_plan_sha256'):
            errors.append('historical_group_plan_mismatch')
    except Exception as exc:
        errors.append('import_manifest_invalid:' + str(exc))
    return {'status': 'pass' if not errors else 'failed', 'errors': errors,
            'model_calls': 0, 'condition_file_sha256': config.get('condition_file_sha256'),
            'operational_files': operational or {}}


def _science_code_identity(root):
    """Mirror Mercury's byte identity inventory for the configured frozen tree."""
    root = Path(root)
    if (root / 'high_fidelity_schema_study').is_dir():
        root = root / 'high_fidelity_schema_study'
    paths = set(root.glob('*.py'))
    for directory in ('four_category', 'extractors'):
        paths.update((root / directory).rglob('*.py'))
    paths.update((root / 'templates').glob('four_category*'))
    names = (
        'paper_to_schema_system_v4.txt', 'paper_to_schema_user_v4.txt', 'paper_to_schema_user_v5.txt',
        'paper_to_schema_system_v6.txt', 'paper_to_schema_user_v6.txt', 'paper_extraction_observations_v5.schema.json',
        'paper_to_schema_system_v7.txt', 'paper_to_schema_user_v7.txt', 'paper_extraction_observations_v6.schema.json',
        'paper_to_schema_system_v8.txt', 'paper_to_schema_user_v8.txt', 'paper_extraction_observations_v7.schema.json',
        'paper_to_schema_system_v9.txt', 'paper_to_schema_user_v9.txt', 'paper_extraction_observations_v9.schema.json',
        'paper_to_schema_system_v10.txt', 'paper_to_schema_user_v10.txt', 'paper_extraction_observations_v10.schema.json',
        'paper_category_response_v1.schema.json', 'paper_category_response_v2.schema.json',
        'paper_category_response_v3.schema.json', 'paper_derived_schema_v4.schema.json',
        'paper_evidence_input_v3.schema.json', 'paper_evidence_layout_v3.schema.json',
        'paper_evidence_input_v4.schema.json', 'paper_evidence_layout_v4.schema.json',
        'paper_evidence_input_v5.schema.json', 'paper_evidence_layout_v5.schema.json',
        'paper_evidence_input_v6.schema.json', 'paper_evidence_layout_v6.schema.json')
    paths.update(root / 'templates' / name for name in names)
    return {path.relative_to(root).as_posix(): file_sha(path) for path in sorted(paths) if path.is_file()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    subs = parser.add_subparsers(dest='action', required=True)
    for name in ('start', 'status', 'pause', 'resume', 'drain', 'reserve', 'cancel-reservation', 'reconcile', 'verify'):
        sub = subs.add_parser(name)
        if name == 'start':
            sub.add_argument('--telegram-secret', type=Path)
        elif name in ('pause', 'resume', 'drain'):
            sub.add_argument('--worker')
            sub.add_argument('--command-id')
        elif name == 'reserve':
            sub.add_argument('--gpu', required=True)
            sub.add_argument('--duration', type=int, default=10800)
            sub.add_argument('--grace', type=int, default=900)
            sub.add_argument('--reservation-key', required=True)
            sub.add_argument('--command-id')
        elif name == 'cancel-reservation':
            sub.add_argument('--reservation-key', required=True)
            sub.add_argument('--command-id')
    args = parser.parse_args(argv)
    config = _config(args.config)
    if args.action == 'start':
        from .supervisor import start_background
        result = start_background(args.config, telegram_secret=args.telegram_secret)
    elif args.action == 'status':
        result = status(config)
    elif args.action == 'reconcile':
        result = reconcile(config)
    elif args.action == 'verify':
        result = verify(config)
    else:
        now = time.time()
        if args.action in ('pause', 'resume', 'drain'):
            action = args.action
            payload = _worker_payload(config, args.worker)
        elif args.action == 'reserve':
            if not 1 <= args.duration <= 10800 or not 0 <= args.grace <= 900:
                raise ValueError('reservation_duration_or_grace_out_of_range')
            if not args.reservation_key or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.reservation_key):
                raise ValueError('reservation_key_required')
            if args.command_id is not None and args.command_id != args.reservation_key:
                raise ValueError('reservation_key_is_command_id')
            try:
                existing = read(Path(config['local_root']) / 'commands' / (args.reservation_key + '.json'))
            except FileNotFoundError:
                existing = None
            if existing is not None:
                payload = existing.get('payload', {})
                if (existing.get('action') != 'reserve' or payload.get('reservation_key') != args.reservation_key
                        or payload.get('gpu_ids') != [args.gpu] or payload.get('duration_s') != args.duration
                        or payload.get('grace_s') != args.grace):
                    raise ValueError('reservation_key_replay_conflict')
                result = create_command(config, 'reserve', payload, args.reservation_key)
                print(json.dumps(result, sort_keys=True, ensure_ascii=False))
                return
            condition = _condition(config)
            matches = [w for w in condition['policy']['workers'] if args.gpu in w['gpu_ids']]
            if not matches:
                raise ValueError('gpu_not_in_deployment:' + args.gpu)
            action = 'reserve'
            payload = {'gpu_ids': [args.gpu], 'expires_at': now + args.duration,
                       'grace_until': now + args.grace, 'reservation_key': args.reservation_key,
                       'duration_s': args.duration, 'grace_s': args.grace}
            args.command_id = args.reservation_key
        else:
            action = 'cancel_reservation'
            payload = {'reservation_key': args.reservation_key}
        result = create_command(config, action, payload, getattr(args, 'command_id', None))
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
