"""Prepare a fresh frozen condition for scheduler V2 without GPU execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .io import digest, file_sha, local_storage_preflight, read, write_once, Lock


def _private_key(local):
    path = local / 'verification.key'
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(os.urandom(32))
                stream.flush()
                os.fsync(stream.fileno())
    if not path.is_file() or path.stat().st_size != 32:
        raise ValueError('private_verification_key_invalid')
    if os.name == 'posix' and path.stat().st_mode & 0o077:
        raise ValueError('private_verification_key_permissions_invalid')
    return path


def _condition(config, source_path, root):
    destination = root / 'condition.json'
    if source_path is not None:
        source_path = Path(source_path)
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError('frozen_condition_source_invalid')
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Preserve the exact frozen bytes referenced by the deployment pin.
            temporary = destination.with_name('.condition-' + uuid.uuid4().hex + '.tmp')
            try:
                with source_path.open('rb') as src, temporary.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        if file_sha(destination) != file_sha(source_path):
            raise ValueError('frozen_condition_copy_conflict')
    if destination.is_symlink() or file_sha(destination) != config['condition_file_sha256']:
        raise ValueError('condition_identity_changed')
    condition = read(destination)
    from high_fidelity_schema_study.four_category.common import seal_errors
    if seal_errors(condition, 'condition'):
        raise ValueError('frozen_condition_seal_invalid')
    from .cli import _science_code_identity
    if _science_code_identity(config['science_source']) != condition['source_file_bytes_sha256']:
        raise ValueError('frozen_execution_source_mismatch')
    return condition


def _dataset_import(condition, job, sources, root):
    from high_fidelity_schema_study.four_category.common import digest as scientific_digest, seal
    from high_fidelity_schema_study.four_category.scheduler_tasks import execute, verify_result

    jid = job['job_id']
    request = {'condition': condition['condition'], 'job_id': jid, 'job': job,
               'attempt': 1, 'worker_id': 'v2-bootstrap-cpu', 'index_path': None}
    request['request_id'] = scientific_digest(request)
    request_path = root / 'requests' / jid / 'attempt-1.json'
    result_path = root / 'results' / jid / 'attempt-1.json'
    write_once(request_path, request)
    if result_path.exists():
        result = read(result_path)
    else:
        started = time.monotonic()
        body = execute(condition, request, sources, None, lambda *args, **kwargs: None, root)
        result = seal({key: request[key] for key in ('condition', 'job_id', 'worker_id', 'attempt')} |
                      {'body': body, 'duration_s': time.monotonic() - started}, 'result_sha256')
        write_once(result_path, result)
    errors, artifact = verify_result(condition, request, result, sources, root)
    if errors or artifact is not None:
        raise ValueError('dataset_bootstrap_replay_failed:' + repr(errors[:4]))
    if result['body']['status'] != 'success':
        raise ValueError('dataset_bootstrap_not_successful:' + jid)
    return {'parent_status': 'success', 'result_ref':
            {'path': result_path.relative_to(root).as_posix(), 'file_bytes_sha256': file_sha(result_path)}}


def prepare(config_path, *, condition_path=None):
    config = read(config_path)
    root = Path(config['root']).resolve()
    local = local_storage_preflight(config['local_root'])
    sources = Path(config['sources']).resolve()
    if root == sources or root.is_relative_to(sources) or sources.is_relative_to(root):
        raise ValueError('scheduler_state_and_sources_must_be_disjoint')
    root.mkdir(parents=True, exist_ok=True)
    with Lock(local / 'bootstrap.lock'):
        sys.path.insert(0, str(config['science_source']))
        condition = _condition(config, condition_path, root)
        for name, sha in config['operational_files'].items():
            if file_sha(Path(config['runtime_source']) / name) != sha:
                raise ValueError('operational_source_changed:' + name)
        write_once(root / 'deployment.json', config)
        private = _private_key(local)
        os.environ['MERCURY_VERIFICATION_KEY'] = str(private)
        from .science import plan_groups
        plans = plan_groups(condition, sources)
        write_once(root / 'group_plans.json', plans)
        imported = {}
        for job in condition['jobs']:
            jid = job['job_id']
            if job['kind'] == 'dataset_parse':
                imported[jid] = _dataset_import(condition, job, sources, root)
            elif job['kind'] == 'soft_reference':
                imported[jid] = {'parent_status': 'deferred'}
            elif job['kind'] in {'classification', 'local_extraction'}:
                imported[jid] = {'parent_status': 'pending', 'groups': {}}
            else:
                raise ValueError('unsupported_fresh_job_kind:' + str(job['kind']))
        members = {}
        members_sha = digest(members)
        write_once(root / 'imports' / ('members-' + members_sha + '.json'), members)
        manifest = {'version': 'scheduler-v2-import/v1', 'source': 'fresh-frozen-condition',
                    'artifact_manifest_sha256': members_sha, 'group_plan_sha256': digest(plans),
                    'execution_identity': config['execution_identity'], 'ancestor_ledger_rows': {},
                    'imported': imported, 'authority': 'dataset_replayed_at_bootstrap'}
        path = root / 'imports' / ('fresh-' + digest(manifest) + '.json')
        sha = write_once(path, manifest)
        pointer = {'path': path.relative_to(root).as_posix(), 'file_bytes_sha256': sha}
        write_once(root / 'import-latest.json', pointer)
        return {'status': 'prepared', 'jobs': len(imported),
                'dataset_jobs': sum(j['kind'] == 'dataset_parse' for j in condition['jobs']),
                'import_path': str(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--condition', type=Path,
                        help='frozen condition file; omit when already staged at root/condition.json')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.config, condition_path=args.condition), sort_keys=True))


if __name__ == '__main__':
    main()
