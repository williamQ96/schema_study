import copy
import json

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import read_json
from high_fidelity_schema_study.four_category.scheduler import run_scheduler, TERMINAL
from high_fidelity_schema_study.four_category.grouped_extraction import _extended
from high_fidelity_schema_study.four_category.workflow import plan_jobs, tasks_for_config, experiment_errors, run_batch
from .test_matrix_scheduler import setup


def configure(config):
    config['extraction_input_protocol'] = 'extraction-anchors/v3'
    config['extraction_grouping'] = {'max_windows': 24, 'max_target_chars': 4000, 'max_mentions': 16, 'max_facts': 24}
    for profile in config['profiles']:
        profile['runtime']['structured_output'] = {'engine': 'xgrammar', 'version': '0.2.8', 'channel': 'json'}
        profile['context_window'] = 131072
    config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
    config['task_hashes'] = {kind: task['task_sha256'] for kind, task in tasks_for_config(config).items()}


def test_grouping_changes_extraction_jobs_only_and_invalid_policy_rejected(tmp_path):
    sources, config, corpus, _ = setup(tmp_path)
    configure(config)
    plan = plan_jobs(config, corpus)
    changed = copy.deepcopy(config)
    changed['extraction_grouping']['max_windows'] = 1
    other = plan_jobs(changed, corpus)
    assert plan['task_hashes'] == other['task_hashes']
    assert plan['jobs'][0] == other['jobs'][0]
    assert all(a['job_id'] != b['job_id'] for a, b in zip(plan['jobs'][1:], other['jobs'][1:]))
    changed['extraction_grouping']['max_facts'] = True
    assert 'extraction_grouping_positive_integers_required' in experiment_errors(changed)
    assert run_batch(config, corpus, source_root=sources, output=tmp_path/'unused')['status'] == 'blocked'
    assert 'incomplete' in TERMINAL


def test_resident_grouped_extraction_saves_separate_bundles_and_resumes(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    configure(config)
    root = tmp_path/'run'
    result = run_scheduler(config, corpus, policy, root=root, sources=sources,
                           leases=tmp_path/'leases', start_watchdog=False)
    assert result['terminal'] and result['slot_counts'] == {'success': 11, 'deferred': 1}
    indexes = list((root/'indexes').glob('*.json'))
    bundles = list((root/'extractions').glob('*.json'))
    assert len(indexes) == 1 and read_json(indexes[0])['schema_version'].startswith('paper-category-index/')
    assert len(bundles) == 9
    for path in bundles:
        bundle = read_json(path)
        assert bundle['schema_version'] == 'paper-derived-observations/v5'
        assert bundle['resolved_object_count'] is None
        assert bundle['semantic_correctness'] == 'not_established'
    records = [_extended(p) for p in (root/'extraction_groups').rglob('*.json')]
    assert len(records) >= 9
    for path in records:
        r = read_json(path)
        assert 'DATASET_SECRET_CANARY' not in json.dumps(r['messages'])
        assert r['backend_result']['structured_output_applied']['synthetic'] is True
    before = {p: p.read_bytes() for p in bundles+records+indexes}
    again = run_scheduler(config, corpus, policy, root=root, sources=sources,
                          leases=tmp_path/'leases', start_watchdog=False)
    assert again == result and before == {p: p.read_bytes() for p in before}
