import copy
import json

from high_fidelity_schema_study.four_category.common import digest, read_json, seal
from high_fidelity_schema_study.four_category.extraction_protocol import module_for_task
from high_fidelity_schema_study.four_category.grouped_extraction import (
    _extended, execute_grouped, mock_group_transport, verify_grouped,
)
from high_fidelity_schema_study.four_category.structured_output import applied_control
from high_fidelity_schema_study.four_category import extraction_v6
from .test_grouped_extraction import setup as setup_v5


def setup(tmp_path):
    paper, index, _, profile, job, policy = setup_v5(tmp_path)
    task = extraction_v6.make_task()
    job = copy.deepcopy(job)
    job['task_sha256'] = task['task_sha256']
    job.pop('job_id')
    job['job_id'] = digest(job)
    return paper, index, task, profile, job, policy


def test_v6_grouped_execution_replay_and_transport_resume(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    assert module_for_task(task) is extraction_v6
    groups = extraction_v6.plan_groups(paper, policy)
    assert groups and all(type(window_id) is int for group in groups for window_id in group['window_ids'])
    root = tmp_path / 'run'
    calls = [0]

    def fail_second(request):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError('synthetic v6 transport interruption')
        return mock_group_transport(request)

    first = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=fail_second)
    assert first['status'] == 'transport_error' and len(first['group_record_refs']) == 2
    assert verify_grouped(job, profile, task, paper, index, policy, 1, first, root) == ([], None)
    resumed = execute_grouped(job, profile, task, paper, index, policy, 2, root,
                              allow_live=False, transport=mock_group_transport)
    errors, bundle = verify_grouped(job, profile, task, paper, index, policy, 2, resumed, root)
    assert errors == [] and resumed['status'] == 'success'
    assert bundle['schema_version'] == 'paper-derived-observations/v6'
    assert resumed['group_record_refs'][0] == first['group_record_refs'][0]


def test_v6_caps_and_overflow_are_terminal(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'

    def overflow(request):
        target = json.JSONDecoder().raw_decode(
            request['messages'][-1]['content'].split('TARGET GROUP\n', 1)[1].lstrip())[0]
        payload = {'schema_version': request['structured_output']['schema']['properties']['schema_version']['const'],
                   'coverage': [{'window_id': wid, 'state': 'overflow'} for wid in target['window_ids']],
                   'mentions': [], 'facts': []}
        return {'text': json.dumps(payload), 'model': request['model'], 'finish_reason': 'stop',
                'usage': None, 'structured_output_applied': applied_control(request, synthetic=True)}

    first = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=overflow)
    assert first['status'] == 'incomplete'
    assert verify_grouped(job, profile, task, paper, index, policy, 1, first, root) == ([], None)
    replay = execute_grouped(job, profile, task, paper, index, policy, 2, root,
                             allow_live=False, transport=lambda _: (_ for _ in ()).throw(AssertionError('retried incomplete')))
    assert replay['status'] == 'incomplete' and replay['group_record_refs'] == first['group_record_refs']

    capped_root = tmp_path / 'capped-run'

    def over_cap(request):
        target = json.JSONDecoder().raw_decode(
            request['messages'][-1]['content'].split('TARGET GROUP\n', 1)[1].lstrip())[0]
        rows = [{'window_id': wid, 'state': 'reviewed'} for wid in target['window_ids']]
        rows.append(copy.deepcopy(rows[-1]))
        payload = {'schema_version': request['structured_output']['schema']['properties']['schema_version']['const'],
                   'coverage': rows, 'mentions': [], 'facts': []}
        return {'text': json.dumps(payload), 'model': request['model'], 'finish_reason': 'stop',
                'usage': None, 'structured_output_applied': applied_control(request, synthetic=True)}

    capped = execute_grouped(job, profile, task, paper, index, policy, 1, capped_root,
                             allow_live=False, transport=over_cap)
    assert capped['status'] == 'contract_invalid'
    again = execute_grouped(job, profile, task, paper, index, policy, 2, capped_root,
                            allow_live=False, transport=lambda _: (_ for _ in ()).throw(AssertionError('retried cap failure')))
    assert again['status'] == 'contract_invalid' and again['group_record_refs'] == capped['group_record_refs']


def test_v6_foreign_refs_and_tampered_source_group_or_profile_fail(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    body = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                           allow_live=False, transport=mock_group_transport)
    errors, bundle = verify_grouped(job, profile, task, paper, index, policy, 1, body, root)
    assert errors == [] and bundle is not None

    foreign = copy.deepcopy(body)
    foreign['group_record_refs'][0]['path'] = 'extraction_groups/foreign/attempt-1.json'
    assert verify_grouped(job, profile, task, paper, index, policy, 1, foreign, root)[0]

    changed_paper = copy.deepcopy(paper)
    changed_paper['pages'][0]['text_regions'][0]['text'] += ' changed'
    assert verify_grouped(job, profile, task, changed_paper, index, policy, 1, body, root)[0]

    ref = body['group_record_refs'][0]
    record_path = _extended(root / ref['path'])
    record = read_json(record_path)
    record['extraction_group']['window_ids'][0] += 1
    record['job']['extraction_group'] = copy.deepcopy(record['extraction_group'])
    record_path.write_text(json.dumps(seal(record, 'record_sha256')), encoding='utf-8')
    assert verify_grouped(job, profile, task, paper, index, policy, 1, body, root)[0]

    changed_profile = copy.deepcopy(profile)
    changed_profile['model_id'] += '-foreign'
    assert verify_grouped(job, changed_profile, task, paper, index, policy, 1, body, root)[0]
