"""Immutable grouped extraction attempts and replay verification."""
from __future__ import annotations

import copy
import os
from pathlib import Path

from .common import contained, digest, file_digest, read_json, write_new
from .extraction_protocol import module_for_task
from .workflow import _attempt, replay_run
from .structured_output import applied_control


def mock_group_transport(request: dict) -> dict:
    """Return an explicitly synthetic empty-observation response for a group."""
    import json

    user = request['messages'][-1]['content']
    schema = request['structured_output']['schema']
    version = schema['properties']['schema_version']['const']
    if version == 'paper-extraction-group-response/v8':
        body = json.loads(user)
        target = body['target']
        payload = {
            'schema_version': version,
            'coverage': [{'window_id': row['properties']['window_id']['const'], 'state': 'reviewed'}
                         for row in schema['properties']['coverage']['prefixItems']],
            'objects': [],
            'table_reviews': [{'table_id': table['table_id'], 'purpose': 'other',
                               'field_axis': 'neither', 'rationale': 'Synthetic fixture only.',
                               'source_windows': table['eligible_evidence_ids'][:1]}
                              for table in target['table_candidates']],
            'feature_coverage': [{'candidate_id': cell['candidate_id'], 'decision': 'not_field',
                                  'emitted_labels': [], 'rationale': 'Synthetic fixture only.'}
                                 for cell in target['feature_cell_candidates']],
        }
        return {'text': json.dumps(payload, ensure_ascii=False), 'model': request['model'],
                'finish_reason': 'stop', 'usage': None,
                'structured_output_applied': applied_control(request, synthetic=True)}
    marker = {'paper-extraction-group-response/v4': 'TARGET GROUP FOCUS: assigned windows and their exact text\n',
              'paper-extraction-group-response/v5': 'TARGET GROUP FOCUS: assigned windows and exact primary_quote options\n'}.get(version, 'TARGET GROUP\n')
    target, _ = json.JSONDecoder().raw_decode(user.split(marker, 1)[1].lstrip())
    payload = {
        'schema_version': schema['properties']['schema_version']['const'],
        'coverage': [{'window_id': wid, 'state': 'reviewed'} for wid in target['window_ids']],
        'mentions': [],
        'facts': [],
    }
    return {'text': json.dumps(payload, ensure_ascii=False), 'model': request['model'],
            'finish_reason': 'stop', 'usage': None,
            'structured_output_applied': applied_control(request, synthetic=True)}


def group_job(job: dict, group: dict, index: dict) -> dict:
    """Derive a child job bound to its exact parent, group, and source index."""
    value = copy.deepcopy(job)
    value.pop('job_id', None)
    value['parent_job_id'] = job['job_id']
    value['index_sha256'] = index['index_sha256']
    value['extraction_group'] = copy.deepcopy(group)
    value['job_id'] = digest(value)
    return value


def group_path(root: Path, outer_job_id: str, group_id: str, attempt: int) -> Path:
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError('invalid_group_attempt')
    path = Path('extraction_groups') / outer_job_id / group_id / f'attempt-{attempt}.json'
    return _extended(contained(root, path.as_posix()))


def _extended(path: Path) -> Path:
    """Keep long SHA path components usable on Windows."""
    path = Path(path).resolve()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def record_ref(path: Path, root: Path) -> dict:
    path = _extended(path)
    return {'path': path.relative_to(_extended(root)).as_posix(),
            'file_bytes_sha256': file_digest(path), 'bytes': path.stat().st_size}


def _load_checked(path: Path, root: Path, job: dict, profile: dict, task: dict,
                  paper_input: dict, index: dict, group: dict, max_attempt: int) -> dict:
    if path.parent != group_path(root, job['parent_job_id'], group['group_id'], 1).parent:
        raise ValueError('foreign_group_record_path')
    name = path.name
    if not name.startswith('attempt-') or not name.endswith('.json'):
        raise ValueError('foreign_group_record_name')
    stem = name[len('attempt-'):-len('.json')]
    if not stem.isdecimal() or str(int(stem)) != stem or not 1 <= int(stem) <= max_attempt:
        raise ValueError('foreign_group_record_attempt')
    record = read_json(path)
    if record.get('job') != job or record.get('profile') != profile or record.get('task') != task:
        raise ValueError('foreign_group_record_identity')
    if record.get('attempt') != int(stem) or record.get('extraction_group') != group:
        raise ValueError('foreign_group_record_group')
    errors = replay_run(record, task, paper_input, index=index, extraction_group=group)
    if errors:
        raise ValueError('group_record_replay_failed:' + ','.join(errors[:4]))
    return record


def _existing(root: Path, job: dict, profile: dict, task: dict, paper_input: dict,
              index: dict, group: dict, attempt: int, *, historical: bool = False) -> list[tuple[int, Path, dict]]:
    folder = group_path(root, job['parent_job_id'], group['group_id'], 1).parent
    if not folder.exists():
        return []
    found = []
    for path in folder.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError('foreign_group_cache_entry')
        stem = path.name[len('attempt-'):-len('.json')] if path.name.startswith('attempt-') and path.name.endswith('.json') else ''
        if historical and stem.isdecimal() and str(int(stem)) == stem and int(stem) > attempt:
            # Later retries cannot change the validity of this earlier replay.
            continue
        record = _load_checked(path, root, job, profile, task, paper_input, index, group, attempt)
        found.append((record['attempt'], path, record))
    return sorted(found)


def _check_cache_tree(root: Path, outer_job_id: str, groups: list[dict]) -> None:
    base = _extended(contained(root, (Path('extraction_groups') / outer_job_id).as_posix()))
    if not base.exists():
        return
    expected = {group['group_id'] for group in groups}
    for child in base.iterdir():
        if child.is_symlink() or not child.is_dir() or child.name not in expected:
            raise ValueError('foreign_group_cache_directory')


def _chosen(cached: list[tuple[int, Path, dict]], attempt: int):
    successful = [row for row in cached if row[2]['status'] == 'success']
    if successful:
        return successful[-1]
    if cached and cached[-1][0] == attempt:
        return cached[-1]
    if cached and cached[-1][2]['status'] not in {'transport_error'}:
        # Contract, truncation, incomplete, and blocked outcomes are terminal.
        return cached[-1]
    return None


def execute_grouped(job: dict, profile: dict, task: dict, paper_input: dict, index: dict,
                    policy: dict, attempt: int, root: Path, *, allow_live: bool,
                    transport=None, counter=None, progress=None) -> dict:
    protocol = module_for_task(task)
    groups = protocol.plan_groups(paper_input, policy)
    _check_cache_tree(root, job['job_id'], groups)
    plan_sha = digest(groups)
    refs = []
    records = []
    for group in groups:
        if progress:
            progress('verifying', group_index=group['index'], group_total=len(groups))
        child = group_job(job, group, index)
        cached = _existing(root, child, profile, task, paper_input, index, group, attempt)
        chosen = _chosen(cached, attempt)
        if chosen:
            _, path, record = chosen
        else:
            if progress:
                progress('tokenizing', group_index=group['index'], group_total=len(groups))
            record = _attempt(child, profile, task, paper_input, index, None, None, attempt,
                              allow_live=allow_live, transport=transport, counter=counter,
                              classification_group=None, extraction_group=group)
            path = group_path(root, job['job_id'], group['group_id'], attempt)
            write_new(path, record)
        if progress:
            progress('verifying', group_index=group['index'], group_total=len(groups))
        refs.append(record_ref(path, root))
        records.append(record)
        if record['status'] != 'success':
            return {'status': record['status'], 'group_plan_sha256': plan_sha,
                    'group_record_refs': refs}
    return {'status': 'success', 'group_plan_sha256': plan_sha,
            'group_record_refs': refs}


def verify_grouped(job: dict, profile: dict, task: dict, paper_input: dict, index: dict,
                   policy: dict, attempt: int, body: dict, root: Path) -> tuple[list[str], dict | None]:
    errors = []
    try:
        protocol = module_for_task(task)
        groups = protocol.plan_groups(paper_input, policy)
        _check_cache_tree(root, job['job_id'], groups)
        if body.get('group_plan_sha256') != digest(groups):
            errors.append('group_plan_mismatch')
        if set(body) != {'status', 'group_plan_sha256', 'group_record_refs'}:
            errors.append('group_result_shape_mismatch')
        refs = body['group_record_refs']
        if not isinstance(refs, list) or not 1 <= len(refs) <= len(groups):
            raise ValueError('group_ref_count_invalid')
        records = []
        for group, ref in zip(groups, refs):
            if not isinstance(ref, dict) or set(ref) != {'path', 'file_bytes_sha256', 'bytes'}:
                raise ValueError('group_ref_shape_invalid')
            path = _extended(contained(root, ref['path']))
            child = group_job(job, group, index)
            if path.parent != group_path(root, job['job_id'], group['group_id'], 1).parent:
                raise ValueError('group_ref_order_or_identity_mismatch')
            if file_digest(path) != ref['file_bytes_sha256'] or path.stat().st_size != ref['bytes']:
                raise ValueError('group_ref_file_identity_mismatch')
            cached = _existing(root, child, profile, task, paper_input, index, group,
                               attempt, historical=True)
            chosen = _chosen(cached, attempt)
            if chosen is None or chosen[1] != path:
                raise ValueError('group_ref_not_selected_cache_record')
            records.append(chosen[2])
        failed = [i for i, record in enumerate(records) if record['status'] != 'success']
        if failed:
            if failed != [len(records) - 1] or body['status'] != records[-1]['status']:
                errors.append('group_failure_prefix_mismatch')
        elif len(records) != len(groups) or body['status'] != 'success':
            errors.append('group_success_coverage_mismatch')
        if errors or failed:
            return errors, None
        bundle = protocol.build_bundle(paper_input, index, task, groups, records)
        return errors, bundle
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append('group_replay:' + str(exc))
        return errors, None
