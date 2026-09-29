"""Immutable group attempts and replay for grouped classifier protocols."""
from __future__ import annotations

import copy
import os
from pathlib import Path

from .common import contained, digest, file_digest, read_json, write_new
from .workflow import _attempt, replay_run
from .classification_v2 import plan_groups, build_index_v2


def _build_index(paper_input: dict, task: dict, groups: list[dict], records: list[dict]) -> dict:
    admission = task.get('admission_rules_version')
    if admission == 'classification-quotes/v2':
        return build_index_v2(paper_input, task, groups, records)
    if admission == 'classification-anchors/v3':
        from .classification_v3 import build_index_v3
        return build_index_v3(paper_input, task, groups, records)
    if admission == 'classification-anchors/v13r2':
        from .classification_navigation_v4 import build_index_v4
        return build_index_v4(paper_input, task, groups, records)
    raise ValueError('unsupported_grouped_classification_task')


def mock_group_transport(request: dict) -> dict:
    """Synthetic transport fixture; it makes no semantic classification claim."""
    import json
    user = request['messages'][-1]['content']
    decoder = json.JSONDecoder()
    def section(name: str):
        return decoder.raw_decode(user.split(name + '\n', 1)[1].lstrip())[0]
    group = section('TARGET GROUP')
    entries = []
    contract = section('OUTPUT CONTRACT')
    if contract.get('$id') == 'paper-category-response/v3':
        options = {row['unit_id']: row['options'] for row in section('EXACT TARGET QUOTE OPTIONS')}
        for uid in group['unit_ids']:
            quote_id = options[uid][0]['quote_id'] if options[uid] else None
            entries.append({'unit_id': uid, 'state': 'classified' if quote_id else 'none',
                            'categories': ['structure'] if quote_id else [],
                            'evidence_quote_ids': [quote_id] if quote_id else [],
                            'rationale': 'Synthetic fixture only.'})
        response_version = 'paper-category-response/v3'
    elif contract.get('$id') == 'paper-category-response/v2':
        catalog = {unit['unit_id']: unit for unit in section('FULL CANONICAL PAPER UNIT CATALOG')}
        for uid in group['unit_ids']:
            text = catalog[uid]['text']
            quote = next((text[:n] for n in range(min(240, len(text)), 0, -1)
                          if text.count(text[:n]) == 1), None)
            entries.append({'unit_id': uid, 'state': 'classified' if quote else 'none',
                            'categories': ['structure', 'encoding', 'value', 'syntax'] if quote else [],
                            'evidence_quotes': [quote] if quote else [],
                            'rationale': 'Synthetic fixture only.'})
        response_version = 'paper-category-response/v2'
    else:
        raise ValueError('unsupported_grouped_mock_output_contract')
    payload = {'schema_version': response_version,
               'source_identity': section('SOURCE IDENTITY'), 'entries': entries}
    return {'text': json.dumps(payload, ensure_ascii=False), 'model': request['model'],
            'finish_reason': 'stop', 'usage': None}


def group_job(job: dict, group: dict) -> dict:
    """A normal classification job with an unambiguous parent and target group."""
    value = copy.deepcopy(job)
    value.pop('job_id', None)
    value['parent_job_id'] = job['job_id']
    value['classification_group'] = copy.deepcopy(group)
    value['job_id'] = digest(value)
    return value


def group_path(root: Path, outer_job_id: str, group_id: str, attempt: int) -> Path:
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError('invalid_group_attempt')
    path = Path('classification_groups') / outer_job_id / group_id / f'attempt-{attempt}.json'
    return _extended(contained(root, path.as_posix()))


def _extended(path: Path) -> Path:
    """Keep the mandated two SHA path components usable on Windows."""
    path = Path(path).resolve()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def record_ref(path: Path, root: Path) -> dict:
    path = _extended(path)
    return {'path': path.relative_to(_extended(root)).as_posix(),
            'file_bytes_sha256': file_digest(path), 'bytes': path.stat().st_size}


def _load_checked(path: Path, root: Path, job: dict, profile: dict, task: dict,
                  paper_input: dict, group: dict, max_attempt: int) -> dict:
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
    if record.get('attempt') != int(stem) or record.get('classification_group') != group:
        raise ValueError('foreign_group_record_group')
    errors = replay_run(record, task, paper_input, classification_group=group)
    if errors:
        raise ValueError('group_record_replay_failed:' + ','.join(errors[:4]))
    return record


def _existing(root: Path, job: dict, profile: dict, task: dict, paper_input: dict,
              group: dict, attempt: int, *, historical: bool = False) -> list[tuple[int, Path, dict]]:
    folder = group_path(root, job['parent_job_id'], group['group_id'], 1).parent
    if not folder.exists():
        return []
    found = []
    for path in folder.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError('foreign_group_cache_entry')
        stem = path.name[len('attempt-'):-len('.json')] if path.name.startswith('attempt-') and path.name.endswith('.json') else ''
        if historical and stem.isdecimal() and str(int(stem)) == stem and int(stem) > attempt:
            # Later retries cannot change whether an earlier result was valid.
            continue
        record = _load_checked(path, root, job, profile, task, paper_input, group, attempt)
        found.append((record['attempt'], path, record))
    return sorted(found)


def _check_cache_tree(root: Path, outer_job_id: str, groups: list[dict]) -> None:
    base = _extended(contained(root, (Path('classification_groups') / outer_job_id).as_posix()))
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
    if cached and cached[-1][2]['status'] not in {'transport_error', 'infrastructure_failed'}:
        # A contract failure remains terminal even if called outside the scheduler.
        return cached[-1]
    return None


def execute_grouped(job: dict, profile: dict, task: dict, paper_input: dict, policy: dict,
                    attempt: int, root: Path, *, allow_live: bool, transport=None, counter=None,
                    progress=None) -> dict:
    groups = plan_groups(paper_input, policy)
    _check_cache_tree(root, job['job_id'], groups)
    plan_sha = digest(groups)
    refs = []
    records = []
    for group in groups:
        if progress:
            progress('verifying', group_index=group['index'], group_total=len(groups))
        child = group_job(job, group)
        cached = _existing(root, child, profile, task, paper_input, group, attempt)
        # A successful earlier attempt is reusable even after another group failed.
        chosen = _chosen(cached, attempt)
        if chosen:
            _, path, record = chosen
        else:
            if progress:
                progress('tokenizing', group_index=group['index'], group_total=len(groups))
            record = _attempt(child, profile, task, paper_input, None, None, None, attempt,
                              allow_live=allow_live, transport=transport, counter=counter,
                              classification_group=group)
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


def verify_grouped(job: dict, profile: dict, task: dict, paper_input: dict,
                   policy: dict, attempt: int, body: dict, root: Path) -> tuple[list[str], dict | None]:
    errors = []
    try:
        groups = plan_groups(paper_input, policy)
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
            child = group_job(job, group)
            if path.parent != group_path(root, job['job_id'], group['group_id'], 1).parent:
                raise ValueError('group_ref_order_or_identity_mismatch')
            if file_digest(path) != ref['file_bytes_sha256'] or path.stat().st_size != ref['bytes']:
                raise ValueError('group_ref_file_identity_mismatch')
            cached = _existing(root, child, profile, task, paper_input, group, attempt, historical=True)
            chosen = _chosen(cached, attempt)
            if chosen is None or chosen[1] != path:
                raise ValueError('group_ref_not_selected_cache_record')
            records.append(chosen[2])
        failed = [i for i, record in enumerate(records) if record['status'] != 'success']
        if failed:
            if failed != [len(records)-1] or body['status'] != records[-1]['status']:
                errors.append('group_failure_prefix_mismatch')
        elif len(records) != len(groups) or body['status'] != 'success':
            errors.append('group_success_coverage_mismatch')
        if errors or failed:
            return errors, None
        index = _build_index(paper_input, task, groups, records)
        return errors, index
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append('group_replay:' + str(exc))
        return errors, None
