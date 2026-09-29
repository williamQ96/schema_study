import copy
import json

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import digest, read_json, seal, write_new
from high_fidelity_schema_study.four_category.scheduler import Store, collect, compile_condition, envelope_for, run_scheduler
from high_fidelity_schema_study.four_category.scheduler_io import Telemetry
from high_fidelity_schema_study.four_category import scheduler_tasks
from high_fidelity_schema_study.four_category.workflow import experiment_errors, plan_jobs, tasks_for_config
from .test_matrix_scheduler import setup


def configure(config):
    config['classification'].update(protocol='classification-anchors/v3',
                                    grouping={'max_units': 1, 'max_target_chars': 4000})
    config['extraction_input_protocol'] = 'extraction-pointers/v5'
    config['extraction_grouping'] = {'max_windows': 1, 'max_target_chars': 240,
                                     'max_mentions': 32, 'max_facts': 48}
    config['execution']['grouped_mode'] = 'all-groups/v1'
    for profile in config['profiles']:
        profile['runtime']['structured_output'] = {'engine': 'xgrammar', 'version': '0.2.8', 'channel': 'json'}
        profile['context_window'] = 131072
    config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
    config['task_hashes'] = {key: task['task_sha256'] for key, task in tasks_for_config(config).items()}


def test_complete_policy_is_identity_bound_and_legacy_mode_cannot_be_reinterpreted(tmp_path):
    _, config, corpus, _ = setup(tmp_path)
    configure(config)
    plan = plan_jobs(config, corpus)
    old = copy.deepcopy(config); del old['execution']['grouped_mode']
    assert all(a['job_id'] != b['job_id'] for a, b in zip(plan['jobs'], plan_jobs(old, corpus)['jobs']))
    config['extraction_input_protocol'] = 'extraction-surfaces/v4'
    assert 'complete_execution_requires_navigation_v4_and_extraction_v7' in experiment_errors(config)


def test_rejected_classifier_still_produces_one_shared_verified_navigation(tmp_path, monkeypatch):
    sources, config, corpus, policy = setup(tmp_path)
    configure(config)
    root = tmp_path / 'run'; root.mkdir()
    condition = compile_condition(config, corpus, policy)
    store = Store(root, condition)
    original = scheduler_tasks.mock_group_transport
    calls = []
    def partly_invalid(request):
        calls.append(request)
        if len(calls) == 1:
            return {'text': '{broken}', 'model': request['model'], 'finish_reason': 'stop'}
        return original(request)
    monkeypatch.setattr(scheduler_tasks, 'mock_group_transport', partly_invalid)
    try:
        store.refresh(100)
        row = next(r for r in store.rows() if r['job']['kind'] == 'classification')
        assert store.claim(row, policy['workers'][0]['worker_id'], 100)
        row = next(r for r in store.rows() if r['id'] == row['id'])
        envelope = envelope_for(row, condition)
        body = scheduler_tasks.execute(condition, envelope, sources, None, lambda *a, **k: None, root)
        assert body['status'] == 'completed_with_rejections' and len(calls) == 1
        result = seal({k: envelope[k] for k in ('condition', 'job_id', 'worker_id', 'attempt')} |
                      {'body': body, 'duration_s': 1.0}, 'result_sha256')
        write_new(root / 'results' / row['id'] / 'attempt-1.json', result)
        collect(store, sources, Telemetry(root / 'events.jsonl', condition['condition']))
        index = read_json(root / 'indexes' / (digest(row['job']['paper_id']) + '.json'))
        assert index['schema_version'] == 'paper-category-index/v4'
        assert index['entries'][0]['availability'] == 'unavailable'
        assert index['entries'][0]['prediction'] is None
        store.refresh(101)
        local = [r for r in store.rows() if r['job']['kind'] == 'local_extraction']
        assert len(local) == 9 and all(r['ready_at'] is not None and r['status'] == 'pending' for r in local)
        index_paths = {envelope_for(r, condition)['index_path'] for r in local}
        assert len(index_paths) == 1
        # An existing unusable file does not bypass the worker's derivation check.
        index['entries'][0]['prediction'] = {'state': 'none'}
        path = root / next(iter(index_paths))
        path.write_text(json.dumps(seal(index, 'index_sha256')), encoding='utf-8')
        import pytest
        with pytest.raises(ValueError, match='classification_index_invalid'):
            scheduler_tasks.execute(condition, envelope_for(local[0], condition), sources, None,
                                    lambda *a, **k: None, root)
    finally:
        store.db.close()


def test_complete_resident_matrix_resume_preserves_records_and_no_dataset_leak(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    configure(config)
    root = tmp_path / 'run'
    summary = run_scheduler(config, corpus, policy, root=root, sources=sources,
                            leases=tmp_path / 'leases', start_watchdog=False)
    assert summary['terminal'] and summary['slot_counts'] == {'success': 11, 'deferred': 1}
    coverage = summary['group_execution']
    assert coverage['returned_local_cells'] == 9
    assert coverage['fully_admitted_local_cells'] == 9
    assert not coverage['full_local_inference_coverage']  # Mock, not a real generation claim.
    paths = list((root / 'classification_groups').rglob('*.json')) + list((root / 'extraction_groups').rglob('*.json'))
    from high_fidelity_schema_study.four_category.grouped_extraction import _extended
    paths = [_extended(path) for path in paths]
    before = {str(p): p.read_bytes() for p in paths}
    assert all('DATASET_SECRET_CANARY' not in json.dumps(read_json(p)['messages']) for p in paths)
    again = run_scheduler(config, corpus, policy, root=root, sources=sources,
                          leases=tmp_path / 'leases', start_watchdog=False)
    assert summary == again and before == {str(p): p.read_bytes() for p in paths}
