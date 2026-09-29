import copy
import json

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import digest, read_json, seal
from high_fidelity_schema_study.four_category.extraction_v5 import make_task, plan_groups
from high_fidelity_schema_study.four_category.grouped_extraction import (
    _extended, execute_grouped, group_path, mock_group_transport, verify_grouped,
)
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category.paper import build_index, source_identity, unit_catalog
from high_fidelity_schema_study.four_category.common import taxonomy


def setup(tmp_path):
    sources = tmp_path / 'sources'
    config, _ = make_fixture(sources)
    paper = read_json(sources / 'paper_input.json')
    region = paper['pages'][0]['text_regions'][0]
    for suffix in ('B', 'C'):
        clone = copy.deepcopy(region)
        clone['unit_id'] = region['unit_id'] + suffix
        paper['pages'][0]['text_regions'].append(clone)
    task = make_task()
    profile = config['profiles'][-1]
    profile['runtime']['structured_output'] = {'engine': 'xgrammar', 'version': '0.2.8', 'channel': 'json'}
    producer = {'run_id': 'synthetic-classification-run', 'profile_sha256': profile_hash(profile),
                'task_sha256': task['task_sha256'], 'request_sha256': 'synthetic-request'}
    response = {'schema_version': 'paper-category-response/v1', 'source_identity': source_identity(paper),
                'entries': [{'unit_id': uid, 'page': unit['page'], 'state': 'none', 'categories': [],
                             'evidence_spans': [], 'rationale': 'Synthetic empty index fixture.'}
                            for uid, unit in unit_catalog(paper).items()]}
    index = build_index(json.dumps(response), paper, producer, taxonomy_value=taxonomy())
    job = {'kind': 'extraction', 'paper_id': paper['paper_id'], 'profile_id': profile['profile_id'],
           'profile_sha256': profile_hash(profile), 'task_sha256': task['task_sha256'],
           'parameters': config['classification']['parameters']}
    job['job_id'] = digest(job)
    policy = {'max_windows': 1, 'max_target_chars': 4000, 'max_mentions': 16, 'max_facts': 24}
    return paper, index, task, profile, job, policy


def test_grouped_extraction_success_replays_full_bundle(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    requests = []

    def transport(request):
        requests.append(request)
        return mock_group_transport(request)

    body = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                           allow_live=False, transport=transport)
    errors, bundle = verify_grouped(job, profile, task, paper, index, policy, 1, body, root)
    assert errors == [] and body['status'] == 'success'
    assert len(body['group_record_refs']) == len(plan_groups(paper, policy)) > 1
    assert bundle['observation_count'] == 0 and bundle['mention_record_count'] == 0
    assert len(bundle['coverage']) == sum(len(group['window_ids']) for group in plan_groups(paper, policy))
    assert all('DATASET_SECRET_CANARY' not in request['messages'][-1]['content'] for request in requests)


def test_transport_failure_resumes_prefix_and_preserves_immutable_attempts(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    calls = [0]

    def fail_second(request):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError('synthetic transport interruption')
        return mock_group_transport(request)

    first = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=fail_second)
    assert first['status'] == 'transport_error' and len(first['group_record_refs']) == 2
    assert verify_grouped(job, profile, task, paper, index, policy, 1, first, root) == ([], None)
    first_bytes = _extended(root / first['group_record_refs'][0]['path']).read_bytes()
    resumed = execute_grouped(job, profile, task, paper, index, policy, 2, root,
                              allow_live=False, transport=mock_group_transport)
    errors, bundle = verify_grouped(job, profile, task, paper, index, policy, 2, resumed, root)
    assert errors == [] and bundle is not None
    assert resumed['group_record_refs'][0] == first['group_record_refs'][0]
    assert _extended(root / first['group_record_refs'][0]['path']).read_bytes() == first_bytes
    first_group = plan_groups(paper, policy)[0]
    assert not group_path(root, job['job_id'], first_group['group_id'], 2).exists()
    assert verify_grouped(job, profile, task, paper, index, policy, 1, first, root) == ([], None)


def test_refs_are_bound_to_order_identity_and_raw_response(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    body = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                           allow_live=False, transport=mock_group_transport)
    missing = copy.deepcopy(body)
    missing['group_record_refs'].pop()
    assert verify_grouped(job, profile, task, paper, index, policy, 1, missing, root)[0]
    reordered = copy.deepcopy(body)
    reordered['group_record_refs'][:2] = reversed(reordered['group_record_refs'][:2])
    assert verify_grouped(job, profile, task, paper, index, policy, 1, reordered, root)[0]
    foreign_index = copy.deepcopy(index)
    foreign_index['index_sha256'] = digest({'foreign': True})
    assert verify_grouped(job, profile, task, paper, foreign_index, policy, 1, body, root)[0]
    path = _extended(root / body['group_record_refs'][0]['path'])
    record = read_json(path)
    record['backend_result']['raw_text'] = '{}'
    path.write_text(json.dumps(seal(record, 'record_sha256')), encoding='utf-8')
    assert verify_grouped(job, profile, task, paper, index, policy, 1, body, root)[0]


def test_terminal_incomplete_and_contract_failures_do_not_retry(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'

    def invalid(request):
        target = json.loads(request['messages'][-1]['content'].split('TARGET GROUP\n', 1)[1].split('\n', 1)[0])
        payload = {'schema_version': 'paper-extraction-group-response/v1',
                   'coverage': [{'window_id': 'wrong', 'state': 'reviewed'}], 'mentions': [], 'facts': []}
        return {'text': json.dumps(payload), 'model': request['model'], 'finish_reason': 'stop', 'usage': None}

    first = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=invalid)
    assert first['status'] == 'contract_invalid'
    retried = execute_grouped(job, profile, task, paper, index, policy, 2, root,
                              allow_live=False, transport=lambda _: (_ for _ in ()).throw(AssertionError('retried terminal record')))
    assert retried['group_record_refs'] == first['group_record_refs']


def test_overflow_incomplete_stops_job_and_is_not_retried(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    calls = [0]

    def overflow(request):
        calls[0] += 1
        if calls[0] > 1:
            raise AssertionError('incomplete group must stop the outer job')
        target = json.loads(request['messages'][-1]['content'].split('TARGET GROUP\n', 1)[1].split('\n', 1)[0])
        payload = {'schema_version': 'paper-extraction-group-response/v1',
                   'coverage': [{'window_id': wid, 'state': 'overflow'} for wid in target['window_ids']],
                   'mentions': [], 'facts': []}
        from high_fidelity_schema_study.four_category.structured_output import applied_control
        return {'text': json.dumps(payload), 'model': request['model'], 'finish_reason': 'stop',
                'usage': None, 'structured_output_applied': applied_control(request, synthetic=True)}

    first = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=overflow)
    assert first['status'] == 'incomplete' and len(first['group_record_refs']) == 1
    assert verify_grouped(job, profile, task, paper, index, policy, 1, first, root) == ([], None)
    replay = execute_grouped(job, profile, task, paper, index, policy, 2, root,
                             allow_live=False, transport=overflow)
    assert replay['status'] == 'incomplete' and replay['group_record_refs'] == first['group_record_refs']
    assert calls == [1]


def test_later_group_attempt_does_not_invalidate_earlier_replay(tmp_path):
    paper, index, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    calls = [0]

    def interrupt(request):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError('temporary failure')
        return mock_group_transport(request)

    first = execute_grouped(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=interrupt)
    assert first['status'] == 'transport_error'
    execute_grouped(job, profile, task, paper, index, policy, 2, root,
                    allow_live=False, transport=mock_group_transport)
    assert verify_grouped(job, profile, task, paper, index, policy, 1, first, root) == ([], None)
