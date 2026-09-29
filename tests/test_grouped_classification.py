import copy
import json

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.classification_v2 import make_task, plan_groups, verify_index_v2
from high_fidelity_schema_study.four_category.classification_v3 import make_task as make_anchor_task, verify_index_v3
from high_fidelity_schema_study.four_category.common import digest, read_json, seal
from high_fidelity_schema_study.four_category.grouped_classification import (
    _extended, execute_grouped, group_path, mock_group_transport, verify_grouped,
)
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category.scheduler import default_policy, run_scheduler
from high_fidelity_schema_study.four_category.workflow import tasks_for_config


def setup(tmp_path, task_factory=make_task):
    sources = tmp_path / 'sources'
    config, _ = make_fixture(sources)
    paper_input = read_json(sources / 'paper_input.json')
    # The tiny synthetic PDF normally has one region; exercise actual grouping.
    region = paper_input['pages'][0]['text_regions'][0]
    for suffix in ('B', 'C'):
        clone = copy.deepcopy(region)
        clone['unit_id'] = region['unit_id'] + suffix
        paper_input['pages'][0]['text_regions'].append(clone)
    task = task_factory()
    profile = config['profiles'][-1]
    job = {'kind': 'classification', 'paper_id': paper_input['paper_id'],
           'profile_id': profile['profile_id'], 'profile_sha256': profile_hash(profile),
           'task_sha256': task['task_sha256'], 'parameters': config['classification']['parameters']}
    job['job_id'] = digest(job)
    policy = {'max_units': 1, 'max_target_chars': 16000}
    return paper_input, task, profile, job, policy


def test_grouped_success_replay_and_full_context_without_dataset(tmp_path):
    paper, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    seen = []
    def transport(request):
        seen.append(request['messages'][-1]['content'])
        return mock_group_transport(request)
    body = execute_grouped(job, profile, task, paper, policy, 1, root,
                           allow_live=False, transport=transport)
    errors, index = verify_grouped(job, profile, task, paper, policy, 1, body, root)
    assert errors == [] and body['status'] == 'success'
    assert len(seen) == len(plan_groups(paper, policy)) > 1
    assert len({text.split('FULL CANONICAL PAPER UNIT CATALOG\n', 1)[1].split('\nTARGET GROUP\n', 1)[0]
                for text in seen}) == 1
    assert all('DATASET_SECRET_CANARY' not in text for text in seen)
    assert verify_index_v2(index, paper) == []
    assert len(index['entries']) == sum(len(g['unit_ids']) for g in plan_groups(paper, policy))


def test_failure_prefix_resumes_successful_groups_without_repeat(tmp_path):
    paper, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    calls = [0]
    def fail_second(request):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError('synthetic transport interruption')
        return mock_group_transport(request)
    first = execute_grouped(job, profile, task, paper, policy, 1, root,
                            allow_live=False, transport=fail_second)
    assert first['status'] == 'transport_error' and len(first['group_record_refs']) == 2
    assert verify_grouped(job, profile, task, paper, policy, 1, first, root) == ([], None)
    first_bytes = _extended(root / first['group_record_refs'][0]['path']).read_bytes()
    resumed = execute_grouped(job, profile, task, paper, policy, 2, root,
                              allow_live=False, transport=mock_group_transport)
    errors, index = verify_grouped(job, profile, task, paper, policy, 2, resumed, root)
    assert errors == [] and index is not None
    assert resumed['group_record_refs'][0] == first['group_record_refs'][0]
    assert _extended(root / first['group_record_refs'][0]['path']).read_bytes() == first_bytes
    assert not group_path(root, job['job_id'], plan_groups(paper, policy)[0]['group_id'], 2).exists()
    assert verify_grouped(job, profile, task, paper, policy, 1, first, root) == ([], None)


def test_reseal_tamper_foreign_cache_and_missing_or_reordered_refs(tmp_path):
    paper, task, profile, job, policy = setup(tmp_path)
    root = tmp_path / 'run'
    body = execute_grouped(job, profile, task, paper, policy, 1, root,
                           allow_live=False, transport=mock_group_transport)
    missing = copy.deepcopy(body)
    missing['group_record_refs'].pop()
    assert verify_grouped(job, profile, task, paper, policy, 1, missing, root)[0]
    reordered = copy.deepcopy(body)
    reordered['group_record_refs'][:2] = list(reversed(reordered['group_record_refs'][:2]))
    assert verify_grouped(job, profile, task, paper, policy, 1, reordered, root)[0]
    path = _extended(root / body['group_record_refs'][0]['path'])
    record = read_json(path)
    record['messages'][0]['content'] = 'tampered'
    path.write_text(json.dumps(seal(record, 'record_sha256')), encoding='utf-8')
    assert verify_grouped(job, profile, task, paper, policy, 1, body, root)[0]
    try:
        execute_grouped(job, profile, task, paper, policy, 2, root,
                        allow_live=False, transport=mock_group_transport)
    except ValueError as exc:
        assert 'group_record_replay_failed' in str(exc)
    else:
        raise AssertionError('foreign cache was accepted')


def test_scheduler_v2_grouped_mock_end_to_end(tmp_path):
    sources = tmp_path / 'sources'
    config, corpus = make_fixture(sources)
    config['classification'].update(protocol='classification-quotes/v2',
                                    grouping={'max_units': 24, 'max_target_chars': 16000})
    config['classification']['profile_id'] = config['roles']['locals'][0]
    config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
    config['task_hashes']['classification'] = tasks_for_config(config)['classification']['task_sha256']
    policy = default_policy()
    policy.update(poll_s=.03, heartbeat_s=.03)
    root = tmp_path / 'run'
    summary = run_scheduler(config, corpus, policy, root=root, sources=sources,
                            leases=tmp_path / 'leases', start_watchdog=False)
    assert summary['terminal']
    assert summary['slot_counts'] == {'success': 11, 'deferred': 1}
    group_records = list((root / 'classification_groups').rglob('attempt-*.json'))
    assert len(group_records) == 1
    assert 'DATASET_SECRET_CANARY' not in _extended(group_records[0]).read_text(encoding='utf-8')
    assert run_scheduler(config, corpus, policy, root=root, sources=sources,
                         leases=tmp_path / 'leases', start_watchdog=False) == summary


def test_grouped_v3_anchor_mock_replay_and_resume(tmp_path):
    paper, task, profile, job, policy = setup(tmp_path, make_anchor_task)
    root = tmp_path / 'run'
    body = execute_grouped(job, profile, task, paper, policy, 1, root,
                           allow_live=False, transport=mock_group_transport)
    errors, index = verify_grouped(job, profile, task, paper, policy, 1, body, root)
    assert errors == [] and body['status'] == 'success'
    assert index['schema_version'] == 'paper-category-index/v3'
    assert verify_index_v3(index, paper) == []
    assert all(entry['evidence_spans'] for entry in index['entries'])
    replayed = execute_grouped(job, profile, task, paper, policy, 2, root,
                               allow_live=False, transport=lambda _: (_ for _ in ()).throw(AssertionError('re-dispatched')))
    assert replayed['group_record_refs'] == body['group_record_refs']


def test_scheduler_v3_grouped_mock_end_to_end(tmp_path):
    sources = tmp_path / 'sources'
    config, corpus = make_fixture(sources)
    config['classification'].update(protocol='classification-anchors/v3',
                                    grouping={'max_units': 24, 'max_target_chars': 16000})
    config['classification']['profile_id'] = config['roles']['locals'][0]
    config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
    config['task_hashes']['classification'] = tasks_for_config(config)['classification']['task_sha256']
    policy = default_policy()
    policy.update(poll_s=.03, heartbeat_s=.03)
    summary = run_scheduler(config, corpus, policy, root=tmp_path / 'run', sources=sources,
                            leases=tmp_path / 'leases', start_watchdog=False)
    assert summary['terminal'] and summary['slot_counts'] == {'success': 11, 'deferred': 1}
