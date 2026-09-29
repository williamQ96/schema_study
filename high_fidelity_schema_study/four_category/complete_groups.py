"""Complete-group execution with bounded transport retries and replay."""
from __future__ import annotations

from pathlib import Path

from .common import digest
from .replay_cache import replay_validation_scope
from .workflow import _attempt


BODY_KEYS = {
    'schema_version', 'status', 'execution_status', 'generation_status',
    'admission_status', 'group_plan_sha256', 'group_outcomes', 'counts',
}
REF_KEYS = {'path', 'file_bytes_sha256', 'bytes'}
RETRYABLE = {'transport_error'}
RETURNED_BACKEND_STATUSES = {'success', 'truncated', 'refused'}


def _configuration(job: dict) -> int:
    value = job.get('group_execution')
    if (not isinstance(value, dict) or set(value) != {'version', 'max_attempts', 'retry_statuses'}
            or value.get('version') != 'all-groups/v1'
            or type(value.get('max_attempts')) is not int or not 1 <= value['max_attempts'] <= 10
            or value.get('retry_statuses') != ['transport_error']):
        raise ValueError('group_execution_policy_invalid')
    return value['max_attempts']


def _context(task, paper_input, index, policy):
    if task.get('kind') == 'classification':
        from . import classification_v2, grouped_classification
        groups = classification_v2.plan_groups(paper_input, policy)
        return grouped_classification, groups
    if task.get('kind') == 'extraction':
        from .extraction_protocol import module_for_task
        from . import grouped_extraction
        protocol = module_for_task(task)
        return grouped_extraction, protocol.plan_groups(paper_input, policy)
    raise ValueError('group_execution_task_kind_invalid')


def _child_job(module, job, group, index):
    if job['kind'] == 'classification':
        return module.group_job(job, group)
    return module.group_job(job, group, index)


def _path(module, root, job, group, attempt):
    return module.group_path(root, job['job_id'], group['group_id'], attempt)


def _load_group_history(module, root, child, profile, task, paper_input, index, group, cap):
    if task['kind'] == 'classification':
        cached = module._existing(root, child, profile, task, paper_input, group, cap)
    else:
        cached = module._existing(root, child, profile, task, paper_input, index, group, cap)
    history = sorted(cached)
    statuses = [row[2]['status'] for row in history]
    if [row[0] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError('group_attempt_sequence_invalid')
    if any(status not in RETRYABLE for status in statuses[:-1]):
        raise ValueError('group_attempt_after_terminal_outcome')
    if len(history) > cap:
        raise ValueError('group_attempt_limit_exceeded')
    return history


def _generation_returned(record: dict) -> bool:
    backend = record.get('backend_result', {})
    return (backend.get('status') in RETURNED_BACKEND_STATUSES
            and backend.get('dispatch_started') is True
            and backend.get('raw_response') is not None)


def _body(groups, plan_sha, outcomes):
    statuses = [row['status'] for row in outcomes]
    returned = sum(row['generation_returned'] for row in outcomes)
    admitted = sum(status == 'success' for status in statuses)
    complete = returned == len(groups)
    if admitted == len(groups):
        status, admission = 'success', 'full'
    elif admitted:
        status, admission = 'completed_with_rejections', 'partial'
    else:
        status, admission = 'completed_with_rejections', 'none'
    if not complete:
        status = 'generation_incomplete'
    return {
        'schema_version': 'group-execution-result/v2',
        'status': status,
        'execution_status': 'all_groups_accounted',
        'generation_status': 'complete' if complete else 'incomplete',
        'admission_status': admission,
        'group_plan_sha256': plan_sha,
        'group_outcomes': outcomes,
        'counts': {'planned_groups': len(groups), 'accounted_groups': len(outcomes),
                   'returned_groups': returned, 'admitted_groups': admitted,
                   'missing_generation_groups': len(groups) - returned},
    }


def execute_complete(job: dict, profile: dict, task: dict, paper_input: dict, index: dict | None,
                     policy: dict, outer_attempt: int, root: Path, *, allow_live: bool,
                     transport=None, counter=None, progress=None) -> dict:
    with replay_validation_scope():
        return _execute_complete(job, profile, task, paper_input, index, policy, outer_attempt,
                                 root, allow_live=allow_live, transport=transport,
                                 counter=counter, progress=progress)


def _execute_complete(job: dict, profile: dict, task: dict, paper_input: dict, index: dict | None,
                       policy: dict, outer_attempt: int, root: Path, *, allow_live: bool,
                       transport=None, counter=None, progress=None) -> dict:
    cap = _configuration(job)
    module, groups = _context(task, paper_input, index, policy)
    module._check_cache_tree(root, job['job_id'], groups)
    plan_sha = digest(groups)
    outcomes = []
    for group in groups:
        if progress:
            progress('verifying', group_index=group['index'], group_total=len(groups))
        child = _child_job(module, job, group, index)
        history = _load_group_history(module, root, child, profile, task, paper_input,
                                      index, group, cap)
        while not history or history[-1][2]['status'] in RETRYABLE and len(history) < cap:
            if progress:
                progress('tokenizing', group_index=group['index'], group_total=len(groups),
                         group_attempt=len(history) + 1, outer_attempt=outer_attempt)
            number = len(history) + 1
            if task['kind'] == 'classification':
                record = _attempt(child, profile, task, paper_input, None, None, None, number,
                                  allow_live=allow_live, transport=transport, counter=counter,
                                  classification_group=group)
            else:
                record = _attempt(child, profile, task, paper_input, index, None, None, number,
                                  allow_live=allow_live, transport=transport, counter=counter,
                                  extraction_group=group)
            path = _path(module, root, job, group, number)
            module.write_new(path, record)
            history.append((number, path, record))
            if progress:
                progress('verifying', group_index=group['index'], group_total=len(groups),
                         group_attempt=number, outer_attempt=outer_attempt)
        if not history:
            raise ValueError('group_attempt_missing')
        attempts = [{'path': ref['path'], 'file_bytes_sha256': ref['file_bytes_sha256'], 'bytes': ref['bytes']}
                    for _, path, _ in history for ref in [module.record_ref(path, root)]]
        number, path, selected = history[-1]
        outcomes.append({'group_id': group['group_id'], 'attempt_refs': attempts,
                         'selected_attempt': number, 'selected_record_sha256': selected['record_sha256'],
                         'status': selected['status'], 'generation_returned': _generation_returned(selected)})
    return _body(groups, plan_sha, outcomes)


def verify_complete(job: dict, profile: dict, task: dict, paper_input: dict, index: dict | None,
                    policy: dict, outer_attempt: int, body: dict, root: Path) -> tuple[list[str], list[dict]]:
    with replay_validation_scope():
        return _verify_complete(job, profile, task, paper_input, index, policy,
                                outer_attempt, body, root)


def _verify_complete(job: dict, profile: dict, task: dict, paper_input: dict, index: dict | None,
                     policy: dict, outer_attempt: int, body: dict, root: Path) -> tuple[list[str], list[dict]]:
    errors = []
    records: list[dict] = []
    try:
        cap = _configuration(job)
        module, groups = _context(task, paper_input, index, policy)
        module._check_cache_tree(root, job['job_id'], groups)
        if not isinstance(body, dict) or set(body) != BODY_KEYS:
            raise ValueError('group_execution_body_shape_invalid')
        if body.get('schema_version') != 'group-execution-result/v2':
            raise ValueError('group_execution_body_version_invalid')
        expected_plan = digest(groups)
        if body.get('group_plan_sha256') != expected_plan:
            raise ValueError('group_execution_plan_mismatch')
        outcomes = body.get('group_outcomes')
        if not isinstance(outcomes, list) or len(outcomes) != len(groups):
            raise ValueError('group_execution_outcome_count_invalid')
        recomputed = []
        for group, outcome in zip(groups, outcomes):
            if not isinstance(outcome, dict) or set(outcome) != {
                'group_id', 'attempt_refs', 'selected_attempt', 'selected_record_sha256', 'status', 'generation_returned'
            }:
                raise ValueError('group_execution_outcome_shape_invalid')
            if outcome['group_id'] != group['group_id']:
                raise ValueError('group_execution_outcome_order_mismatch')
            child = _child_job(module, job, group, index)
            history = _load_group_history(module, root, child, profile, task, paper_input,
                                          index, group, cap)
            if history[-1][2]['status'] in RETRYABLE and len(history) < cap:
                raise ValueError('group_transport_retries_not_exhausted')
            refs = outcome['attempt_refs']
            if not isinstance(refs, list) or len(refs) != len(history) or not refs:
                raise ValueError('group_execution_attempt_refs_invalid')
            for ref, (number, path, record) in zip(refs, history):
                if not isinstance(ref, dict) or set(ref) != REF_KEYS:
                    raise ValueError('group_execution_attempt_ref_shape_invalid')
                if ref != module.record_ref(path, root):
                    raise ValueError('group_execution_attempt_ref_mismatch')
                # `_existing` calls the version-specific checked loader, which
                # replays every raw response and all job/profile/group identities.
                if record['attempt'] != number:
                    raise ValueError('group_execution_record_attempt_mismatch')
            number, _, selected = history[-1]
            records.append(selected)
            if (outcome['selected_attempt'] != number
                    or outcome['selected_record_sha256'] != selected['record_sha256']
                    or outcome['status'] != selected['status']
                    or outcome['generation_returned'] is not _generation_returned(selected)):
                raise ValueError('group_execution_selected_record_mismatch')
            recomputed.append({'group_id': group['group_id'], 'attempt_refs': refs,
                               'selected_attempt': number, 'selected_record_sha256': selected['record_sha256'],
                               'status': selected['status'], 'generation_returned': _generation_returned(selected)})
        expected = _body(groups, expected_plan, recomputed)
        if body != expected:
            raise ValueError('group_execution_body_derivation_mismatch')
        return errors, records
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append('group_execution_replay:' + str(exc))
        return errors, []
