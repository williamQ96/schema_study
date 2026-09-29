"""Additive preparation and replay of a frozen V1 run, with no GPU operations."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

from .io import atomic_json, digest, file_sha, local_storage_preflight, read, write_once

_REPLAY_SCOPE = None


def initialize_verifier(science_source, private_key):
    global _REPLAY_SCOPE
    sys.path.insert(0, str(science_source))
    os.environ['MERCURY_VERIFICATION_KEY'] = str(private_key)
    if _REPLAY_SCOPE is None:
        # Frozen verifier's bounded, content-rehashed cache. It never caches failures.
        from high_fidelity_schema_study.four_category.replay_cache import replay_validation_scope
        _REPLAY_SCOPE = replay_validation_scope()
        _REPLAY_SCOPE.__enter__()


def verify_import(root, sources, job_id, execution_identity):
    from .science import verify_group, finalize
    root = Path(root)
    condition = read(root / 'condition.json')
    job = next(j for j in condition['jobs'] if j['job_id'] == job_id)
    plans = read(root / 'group_plans.json')
    folder = 'classification_groups' if job['kind'] == 'classification' else 'extraction_groups'
    imported = dict(parent_status='pending', groups={})
    for gid in plans[job_id]:
        group_dir = root / folder / job_id / gid
        if not group_dir.exists():
            continue
        try:
            imported['groups'][gid] = verify_group(condition, job, gid, sources, root, execution_identity)
        except ValueError as exc:
            if str(exc) != 'group_not_terminal':
                raise
    if len(imported['groups']) == len(plans[job_id]):
        result = finalize(condition, job, sources, root, execution_identity)
        imported.update(parent_status=result['status'], result_ref=result['result_ref'], index_ref=result['index_ref'])
    return job_id, imported


def mirror_immutable(old, root, *, quiescent=False):
    """Hardlink only closed valid JSON; future creation never replaces existing files."""
    copied = {}
    for name in ('requests', 'results', 'classification_groups', 'extraction_groups', 'indexes', 'extractions', 'lineage'):
        for p in (old / name).rglob('*.json'):
            if p.is_symlink():
                raise ValueError('legacy_artifact_symlink')
            if not quiescent and p.stat().st_mtime > time.time() - 2:
                continue
            try:
                read(p)
            except json.JSONDecodeError:
                if quiescent:
                    raise
                continue  # live writer; final handover requires a complete parse
            target = root / p.relative_to(old)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(p, target)
                except OSError:
                    shutil.copyfile(p, target)
            original_sha = file_sha(p)
            if file_sha(target) != original_sha:
                raise ValueError('legacy_copy_conflict:' + str(p))
            copied[p.relative_to(old).as_posix()] = original_sha
    return copied


def ancestor_rows(old, root, *, quiescent=False):
    """Keep each V2 hop rather than relaxing the original V1 lineage verifier."""
    prior_path = old / 'deployment.json'
    if not prior_path.exists():
        db_path, query = old / 'scheduler.sqlite', 'SELECT * FROM jobs'
        db = sqlite3.connect('file:' + str(db_path) + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            return {r['id']: dict(r) for r in db.execute(query)}
        finally:
            db.close()
    previous = read(prior_path)
    if not quiescent:
        raise ValueError('v2_ancestor_must_be_quiescent_before_graph_capture')
    if Path(previous['root']).resolve() != old.resolve() or old.resolve() == root.resolve():
        raise ValueError('ancestor_deployment_identity_invalid')
    condition = read(old / 'condition.json')
    if condition != read(root / 'condition.json'):
        raise ValueError('ancestor_scientific_condition_changed')
    node = digest([previous['deployment_id'], file_sha(prior_path)])
    target = root / 'ancestors' / node
    if (old / 'ancestors' / node).exists():
        raise ValueError('ancestor_cycle')
    members = [old / 'condition.json', prior_path, old / 'import-latest.json']
    for name in ('imports', 'ancestors', 'scheduler_v2', 'v2_results',
                 'dispatches', 'worker_receipts', 'verification', 'launches',
                 'control_receipts', 'failures'):
        members.extend((old / name).rglob('*.json'))
    pins = {}
    for source in members:
        if not source.is_file() or source.is_symlink():
            raise ValueError('ancestor_member_invalid')
        relative = source.relative_to(old)
        if len(relative.parts) > 40:
            raise ValueError('ancestor_chain_too_deep')
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        if file_sha(source) != file_sha(destination):
            raise ValueError('ancestor_member_changed')
        pins[relative.as_posix()] = file_sha(source)
    write_once(target / 'node.json', dict(ancestor_deployment=previous['deployment_id'],
                scientific_condition=condition['condition'], files=pins,
                original_root=str(old), record_paths='raw group paths also preserved at new run root'))
    db = sqlite3.connect('file:' + str(Path(previous['local_root']) / 'scheduler_v2.sqlite') + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        result = {}
        for r in db.execute('SELECT * FROM jobs'):
            value = dict(r)
            ref = json.loads(value['result_ref']) if value.get('result_ref') else None
            result[value['job_id']] = dict(id=value['job_id'], body=value['body'], status=value['status'],
                result=ref['path'] if ref else None,
                attempt=read(old/ref['path']).get('attempt', 1) if ref else 0,
                ancestor_deployment=previous['deployment_id'])
        return result
    finally:
        db.close()


def prepare(config_path, *, sync_only=False):
    config = read(config_path)
    root, old = Path(config['root']), Path(config['ancestor_root'])
    local = local_storage_preflight(config['local_root'])
    root.mkdir(parents=True, exist_ok=True)
    for name in ('condition.json',):
        if not (root / name).exists():
            shutil.copyfile(old / name, root / name)
        if file_sha(root / name) != file_sha(old / name):
            raise ValueError('condition_copy_changed')
    write_once(root / ('preparation-' + digest(config) + '.json' if config.get('preparation_only') else 'deployment.json'), config)
    private = local / 'verification.key'
    if not private.exists():
        fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(os.urandom(32)); stream.flush(); os.fsync(stream.fileno())
    initialize_verifier(config['science_source'], private)
    from high_fidelity_schema_study.four_category.mercury import code_identity
    from .science import plan_groups
    condition = read(root / 'condition.json')
    if code_identity() != condition['source_file_bytes_sha256']:
        raise ValueError('frozen_execution_source_mismatch')
    pins = mirror_immutable(old, root, quiescent=config.get('ancestor_quiescent', False))
    write_once(root / 'imports' / ('members-' + digest(pins) + '.json'), pins)
    plans_path = root / 'group_plans.json'
    if not plans_path.exists():
        write_once(plans_path, plan_groups(condition, config['sources']))
    plans = read(plans_path)
    imported = {}
    # Read-only database backup API gives a consistent snapshot while V1 continues.
    legacy_rows = ancestor_rows(old, root, quiescent=config.get('ancestor_quiescent', False))
    for job in condition['jobs']:
        jid = job['job_id']
        if job['kind'] == 'soft_reference':
            imported[jid] = {'parent_status': 'deferred'}
        elif job['kind'] == 'dataset_parse':
            row = legacy_rows[jid]
            if row['status'] != 'success' or not row['result']:
                raise ValueError('dataset_parse_not_finalized_for_migration')
            from high_fidelity_schema_study.four_category.scheduler_tasks import verify_result
            request = read(root / 'requests' / jid / ('attempt-' + str(row['attempt']) + '.json'))
            result = read(root / row['result'])
            errors, _ = verify_result(condition, request, result, Path(config['sources']), root)
            if errors:
                raise ValueError('dataset_import_invalid:' + repr(errors[:3]))
            imported[jid] = dict(parent_status='success', result_ref=dict(path=row['result'], file_bytes_sha256=file_sha(root / row['result'])))
    jobs = [j for j in condition['jobs'] if j['kind'] in {'classification', 'local_extraction'}]
    # Validate classifier ancestors first, before any extraction references them.
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_verifier,
                             initargs=(config['science_source'], private)) as pool:
        for kind in ('classification', 'local_extraction'):
            futures = {}
            for job in jobs:
                if job['kind'] != kind:
                    continue
                folder = 'classification_groups' if kind == 'classification' else 'extraction_groups'
                if not (root / folder / job['job_id']).exists():
                    imported[job['job_id']] = {'parent_status': 'pending', 'groups': {}}
                    continue
                f = pool.submit(verify_import, root, config['sources'], job['job_id'], config['execution_identity'])
                futures[f] = job['job_id']
            for f in as_completed(futures):
                jid, result = f.result()
                imported[jid] = result
                atomic_json(root / 'prepare-progress.json', dict(time=time.time(), verified_jobs=len(imported),
                            total_jobs=len(condition['jobs']), last_job=jid,
                            verified_groups=sum(len(i.get('groups', {})) for i in imported.values())))
    # Old infrastructure outcomes remain in ancestor results. The new pending
    # continuation never rewrites those outcomes or resets group transport budgets.
    manifest = dict(version='scheduler-v2-import/v1', created_at=time.time(), ancestor_root=str(old),
                    ancestor_condition=condition['condition'], ancestor_ledger_rows=legacy_rows,
                    artifact_manifest_sha256=digest(pins), group_plan_sha256=digest(plans),
                    execution_identity=config['execution_identity'], imported=imported,
                    authority='replayed_at_migration_not_proof_of_historical_time')
    path = root / 'imports' / ('replay-' + str(time.time_ns()) + '.json')
    sha = write_once(path, manifest)
    atomic_json(root / 'import-latest.json', dict(path=path.relative_to(root).as_posix(), file_bytes_sha256=sha))
    print(json.dumps(dict(status='prepared', jobs=len(imported), groups=sum(len(x.get('groups', {})) for x in imported.values()), import_path=str(path))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    prepare(parser.parse_args().config)


if __name__ == '__main__':
    main()
