import copy
import json

import pytest

from high_fidelity_schema_study.four_category.common import digest, read_json, seal
from high_fidelity_schema_study.four_category.complete_groups import _body, execute_complete, verify_complete
from high_fidelity_schema_study.four_category.extraction_v5 import plan_groups as plan_extraction_v5
from high_fidelity_schema_study.four_category.grouped_classification import mock_group_transport as mock_classification
from high_fidelity_schema_study.four_category.grouped_extraction import (
    _extended, group_path as extraction_group_path, mock_group_transport as mock_extraction,
)
from high_fidelity_schema_study.four_category.workflow import _attempt
from .test_grouped_classification import setup as setup_classification
from .test_grouped_extraction import setup as setup_extraction


def _bind(job, cap=2):
    value = copy.deepcopy(job)
    value.pop('job_id')
    value['group_execution'] = {'version': 'all-groups/v1', 'max_attempts': cap,
                                'retry_statuses': ['transport_error']}
    value['job_id'] = digest(value)
    return value


def _case(tmp_path, kind):
    if kind == 'classification':
        paper, task, profile, job, policy = setup_classification(tmp_path)
        index = None
    else:
        paper, index, task, profile, job, policy = setup_extraction(tmp_path)
    job = _bind(job)
    return paper, task, profile, job, policy, index


def _success_transport(kind):
    return mock_classification if kind == 'classification' else mock_extraction


@pytest.mark.parametrize('kind', ['classification', 'extraction'])
def test_invalid_first_group_does_not_stop_later_groups_and_replays_counts(tmp_path, kind):
    paper, task, profile, job, policy, index = _case(tmp_path, kind)
    root = tmp_path / 'run'
    delegate = _success_transport(kind)
    calls = [0]

    def transport(request):
        calls[0] += 1
        if calls[0] == 1:
            return {'text': '{"invalid":true}', 'model': request['model'],
                    'finish_reason': 'stop', 'usage': None,
                    **({'structured_output_applied': delegate(request)['structured_output_applied']}
                       if kind == 'extraction' else {})}
        return delegate(request)

    body = execute_complete(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=transport)
    errors, selected = verify_complete(job, profile, task, paper, index, policy, 1, body, root)
    assert errors == []
    assert body['status'] == 'completed_with_rejections'
    assert body['execution_status'] == 'all_groups_accounted'
    assert body['generation_status'] == 'complete' and body['admission_status'] == 'partial'
    assert body['counts'] == {'planned_groups': len(body['group_outcomes']),
                              'accounted_groups': len(body['group_outcomes']),
                              'returned_groups': len(body['group_outcomes']),
                              'admitted_groups': len(body['group_outcomes']) - 1,
                              'missing_generation_groups': 0}
    assert len(selected) == len(body['group_outcomes']) == calls[0] > 1
    assert selected[0]['status'] == 'contract_invalid'
    assert all(record['status'] == 'success' for record in selected[1:])


def test_transient_transport_is_retried_per_group_then_later_groups_continue(tmp_path):
    paper, task, profile, job, policy, index = _case(tmp_path, 'extraction')
    root = tmp_path / 'run'
    delegate = mock_extraction
    calls = [0]

    def transport(request):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError('temporary connection reset')
        return delegate(request)

    body = execute_complete(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=transport)
    errors, selected = verify_complete(job, profile, task, paper, index, policy, 1, body, root)
    assert errors == [] and body['status'] == 'success'
    assert body['counts']['planned_groups'] > 1
    assert len(body['group_outcomes'][0]['attempt_refs']) == 2
    assert body['group_outcomes'][0]['selected_attempt'] == 2
    assert all(len(row['attempt_refs']) == 1 for row in body['group_outcomes'][1:])
    assert len(selected) == body['counts']['planned_groups']


def test_exhausted_transport_records_every_group_but_marks_generation_incomplete(tmp_path):
    paper, task, profile, job, policy, index = _case(tmp_path, 'extraction')
    root = tmp_path / 'run'

    def fail(_):
        raise RuntimeError('persistent transport interruption')

    body = execute_complete(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=fail)
    errors, records = verify_complete(job, profile, task, paper, index, policy, 1, body, root)
    assert errors == [] and len(records) == body['counts']['planned_groups']
    assert body['status'] == 'generation_incomplete'
    assert body['execution_status'] == 'all_groups_accounted'
    assert body['generation_status'] == 'incomplete' and body['admission_status'] == 'none'
    assert body['counts']['accounted_groups'] == body['counts']['planned_groups']
    assert body['counts']['returned_groups'] == 0
    assert all(len(row['attempt_refs']) == 2 and not row['generation_returned'] for row in body['group_outcomes'])


def test_interruption_resumes_durable_prefix_without_repeating_success(tmp_path, monkeypatch):
    paper, task, profile, job, policy, index = _case(tmp_path, 'extraction')
    root = tmp_path / 'run'
    real_attempt = _attempt
    calls = [0]

    class SimulatedWorkerExit(BaseException):
        pass

    def interrupt_second(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            raise SimulatedWorkerExit()
        return real_attempt(*args, **kwargs)

    monkeypatch.setattr('high_fidelity_schema_study.four_category.complete_groups._attempt', interrupt_second)
    with pytest.raises(SimulatedWorkerExit):
        execute_complete(job, profile, task, paper, index, policy, 1, root,
                         allow_live=False, transport=mock_extraction)
    monkeypatch.setattr('high_fidelity_schema_study.four_category.complete_groups._attempt', real_attempt)
    resumed = execute_complete(job, profile, task, paper, index, policy, 2, root,
                               allow_live=False, transport=mock_extraction)
    errors, selected = verify_complete(job, profile, task, paper, index, policy, 2, resumed, root)
    assert errors == [] and resumed['status'] == 'success'
    assert len(selected) == resumed['counts']['planned_groups']
    assert len(resumed['group_outcomes'][0]['attempt_refs']) == 1


def test_terminal_failure_cannot_be_followed_by_another_group_attempt(tmp_path):
    paper, task, profile, job, policy, index = _case(tmp_path, 'extraction')
    root = tmp_path / 'run'

    def invalid(request):
        return {'text': '{"invalid":true}', 'model': request['model'], 'finish_reason': 'stop',
                'usage': None, 'structured_output_applied': mock_extraction(request)['structured_output_applied']}

    body = execute_complete(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=invalid)
    assert body['status'] == 'completed_with_rejections'
    first = body['group_outcomes'][0]
    ref = first['attempt_refs'][0]
    path = _extended(root / ref['path'])
    record = read_json(path)
    record['attempt'] = 2
    record['run_id'] = digest({'job_id': record['job']['job_id'], 'index_sha256': record['index_sha256'], 'attempt': 2})
    from high_fidelity_schema_study.four_category.grouped_extraction import group_path
    next_path = group_path(root, job['job_id'], first['group_id'], 2)
    next_path.parent.mkdir(parents=True, exist_ok=True)
    next_path.write_text(json.dumps(seal(record, 'record_sha256')), encoding='utf-8')
    with pytest.raises(ValueError, match='group_attempt_after_terminal_outcome'):
        execute_complete(job, profile, task, paper, index, policy, 2, root,
                         allow_live=False, transport=mock_extraction)


def test_resealed_raw_response_tamper_is_rejected(tmp_path):
    paper, task, profile, job, policy, index = _case(tmp_path, 'extraction')
    root = tmp_path / 'run'
    body = execute_complete(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=mock_extraction)
    path = _extended(root / body['group_outcomes'][0]['attempt_refs'][0]['path'])
    record = read_json(path)
    record['backend_result']['raw_text'] = '{"tampered":true}'
    path.write_text(json.dumps(seal(record, 'record_sha256')), encoding='utf-8')
    errors, records = verify_complete(job, profile, task, paper, index, policy, 1, body, root)
    assert errors and records == []


def test_resigned_counts_and_unreferenced_attempts_are_rejected(tmp_path):
    paper, task, profile, job, policy, index = _case(tmp_path, 'extraction')
    root = tmp_path / 'run'
    body = execute_complete(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=mock_extraction)
    wrong_counts = copy.deepcopy(body)
    wrong_counts['counts']['admitted_groups'] = 0
    assert verify_complete(job, profile, task, paper, index, policy, 1, wrong_counts, root)[0]

    extra = copy.deepcopy(body)
    extra['group_outcomes'][0]['attempt_refs'].append(copy.deepcopy(extra['group_outcomes'][0]['attempt_refs'][0]))
    assert verify_complete(job, profile, task, paper, index, policy, 1, extra, root)[0]


def test_unexhausted_transport_body_is_rejected_but_executor_resumes_retry(tmp_path, monkeypatch):
    paper, task, profile, job, policy, index = _case(tmp_path, 'extraction')
    policy = {**policy, 'max_windows': 24}
    root = tmp_path / 'run'
    group = plan_extraction_v5(paper, policy)[0]
    real_attempt = _attempt
    invocations = [0]

    class SimulatedWorkerExit(BaseException):
        pass

    def stop_after_transport(*args, **kwargs):
        invocations[0] += 1
        if invocations[0] == 1:
            return real_attempt(*args, **kwargs)
        raise SimulatedWorkerExit()

    monkeypatch.setattr('high_fidelity_schema_study.four_category.complete_groups._attempt', stop_after_transport)
    with pytest.raises(SimulatedWorkerExit):
        execute_complete(job, profile, task, paper, index, policy, 1, root,
                         allow_live=False, transport=lambda _: (_ for _ in ()).throw(RuntimeError('retry me')))

    ref_path = extraction_group_path(root, job['job_id'], group['group_id'], 1)
    from high_fidelity_schema_study.four_category.grouped_extraction import record_ref
    ref = record_ref(ref_path, root)
    record = read_json(_extended(ref_path))
    forged = _body([group], digest([group]), [{
        'group_id': group['group_id'], 'attempt_refs': [ref], 'selected_attempt': 1,
        'selected_record_sha256': record['record_sha256'], 'status': 'transport_error',
        'generation_returned': False,
    }])
    errors, records = verify_complete(job, profile, task, paper, index, policy, 1, forged, root)
    assert errors and 'group_transport_retries_not_exhausted' in errors[0]
    assert records == []

    monkeypatch.setattr('high_fidelity_schema_study.four_category.complete_groups._attempt', real_attempt)
    resumed = execute_complete(job, profile, task, paper, index, policy, 2, root,
                               allow_live=False, transport=mock_extraction)
    errors, selected = verify_complete(job, profile, task, paper, index, policy, 2, resumed, root)
    assert errors == [] and resumed['status'] == 'success'
    assert len(resumed['group_outcomes'][0]['attempt_refs']) == 2
    assert [record['status'] for record in selected] == ['success']
