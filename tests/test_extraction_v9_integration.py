"""Exercise the new request contract through execution, replay and packet export."""
import copy
import json
import tempfile
from pathlib import Path

from high_fidelity_schema_study.four_category.common import digest, read_json, seal
from high_fidelity_schema_study.four_category.complete_groups import execute_complete, verify_complete
from high_fidelity_schema_study.four_category.execution_artifacts import build_artifact
from high_fidelity_schema_study.four_category.extraction_v9 import make_task
from high_fidelity_schema_study.four_category.grouped_extraction import _extended, mock_group_transport
from high_fidelity_schema_study.four_category.scheduler import run_scheduler
from high_fidelity_schema_study.four_category.scheduled_packet import build_scheduled_packet, verify_scheduled_packet
from high_fidelity_schema_study.four_category.workflow import experiment_errors, plan_jobs, tasks_for_config
from .test_complete_groups import _bind
from .test_complete_scheduler import configure
from .test_grouped_extraction import setup as extraction_setup
from .test_matrix_scheduler import setup


def test_common_fact_view_preserves_quote_and_uncertain_semantics():
    from high_fidelity_schema_study.four_category.paper import fact_view, category_counts
    fact = {'fact_id': 'f', 'subject_mention_id': 'm', 'categories': ['structure', 'value'],
            'primary_evidence': {'exact_quote': 'energy in Wh', 'unit_id': 'u'},
            'claim': {'kind': 'attribute', 'predicate': 'unit',
                      'assertion': {'status': 'reported', 'value': 'Wh', 'basis': None}}}
    for version in ('v8', 'v9'):
        viewed = fact_view({'schema_version': 'paper-derived-observations/' + version, 'facts': [fact]})
        assert viewed[0]['primary_evidence'] == fact['primary_evidence']
        assert viewed[0]['basis_authority'] == 'model_asserted_not_semantically_adjudicated'
        assert viewed[0]['origin'] == 'paper'
        assert category_counts(viewed)['unique_facts'] == 1


def test_v9_configuration_routes_new_task_without_changing_classifier(tmp_path):
    _, config, corpus, _ = setup(tmp_path)
    configure(config)
    old_tasks, old_plan = tasks_for_config(config), plan_jobs(config, corpus)
    config['extraction_input_protocol'] = 'extraction-local-quotes/v7'
    tasks = tasks_for_config(config)
    config['task_hashes'] = {k: v['task_sha256'] for k, v in tasks.items()}
    assert experiment_errors(config) == []
    assert tasks['classification'] == old_tasks['classification']
    assert tasks['extraction'] == make_task()
    assert tasks['extraction']['task_sha256'] != old_tasks['extraction']['task_sha256']
    assert tasks['extraction']['output_schema'] != old_tasks['extraction']['output_schema']
    old_ids = {j['job_id'] for j in old_plan['jobs'] if j['kind'] == 'local_extraction'}
    new_ids = {j['job_id'] for j in plan_jobs(config, corpus)['jobs'] if j['kind'] == 'local_extraction'}
    assert len(new_ids) == 9 and old_ids.isdisjoint(new_ids)


def test_v9_partial_observations_replay_real_rejections_without_salvage(tmp_path):
    paper, index, _, profile, job, policy = extraction_setup(tmp_path)
    task = make_task()
    job = copy.deepcopy(job)
    job['task_sha256'] = task['task_sha256']
    job = _bind(job)
    calls = []
    def transport(request):
        calls.append(request)
        response = mock_group_transport(request)
        if len(calls) == 1:
            response['text'] = '{"invalid":true}'
        return response
    root = tmp_path / 'run'
    body = execute_complete(job, profile, task, paper, index, policy, 1, root,
                            allow_live=False, transport=transport)
    errors, records = verify_complete(job, profile, task, paper, index, policy, 1, body, root)
    assert not errors
    assert body['status'] == 'completed_with_rejections'
    artifact = build_artifact(job, profile, task, paper, index, policy, records)
    assert artifact['schema_version'] == 'paper-partial-observations/v1'
    assert artifact['task_sha256'] == task['task_sha256']
    assert len(artifact['excluded_groups']) == 1
    assert artifact['excluded_groups'][0]['status'] == 'contract_invalid'
    assert len(artifact['admitted_groups']) == len(records) - 1 > 0


def test_v9_resident_matrix_packet_and_resume():
    short_base = Path.home() / '.codex/tmp'
    with tempfile.TemporaryDirectory(prefix='v9-', dir=short_base if short_base.is_dir() else None) as folder:
        root = Path(folder)
        sources, config, corpus, policy = setup(root)
        configure(config)
        config['extraction_input_protocol'] = 'extraction-local-quotes/v7'
        config['task_hashes'] = {k: v['task_sha256'] for k, v in tasks_for_config(config).items()}
        run = root / 'run'
        summary = run_scheduler(config, corpus, policy, root=run, sources=sources,
                                leases=root / 'leases', start_watchdog=False)
        assert summary['terminal']
        assert summary['group_execution']['returned_local_cells'] == 9
        assert summary['group_execution']['fully_admitted_local_cells'] == 9
        records = [_extended(p) for p in (run / 'extraction_groups').rglob('attempt-*.json')]
        before = {str(p): p.read_bytes() for p in records}
        assert all(read_json(p)['task']['schema_version'] == 'four-category-task/v9' for p in records)
        assert all('DATASET_SECRET_CANARY' not in json.dumps(read_json(p)['messages']) for p in records)
        again = run_scheduler(config, corpus, policy, root=run, sources=sources,
                              leases=root / 'leases', start_watchdog=False)
        assert again == summary and before == {str(p): p.read_bytes() for p in records}
        packet = root / 'packet'
        built = build_scheduled_packet(run, sources, packet)
        expected = read_json(run / 'condition.json')['condition']
        verified = verify_scheduled_packet(packet, sources, expected_condition=expected)
        assert built['status'] == verified['status'] == 'pass'
        assert verified['coverage']['complete_generation_cells'] == 9
        assert verified['semantic_accuracy'] is None
