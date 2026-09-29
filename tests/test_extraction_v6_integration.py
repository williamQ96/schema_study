import copy
import json

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import read_json
from high_fidelity_schema_study.four_category.scheduler import run_scheduler
from high_fidelity_schema_study.four_category.grouped_extraction import _extended
from high_fidelity_schema_study.four_category.workflow import plan_jobs, tasks_for_config, experiment_errors, run_batch
from high_fidelity_schema_study.four_category.extraction_protocol import module_for_task
from .test_matrix_scheduler import setup


def configure(config):
    config['extraction_input_protocol'] = 'extraction-surfaces/v4'
    config['extraction_grouping'] = {'max_windows': 8, 'max_target_chars': 1200, 'max_mentions': 32, 'max_facts': 48}
    for profile in config['profiles']:
        profile['runtime']['structured_output'] = {'engine': 'xgrammar', 'version': '0.2.8', 'channel': 'json'}
        profile['context_window'] = 131072
    config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
    config['task_hashes'] = {kind: task['task_sha256'] for kind, task in tasks_for_config(config).items()}


def test_new_condition_changes_extraction_identity_and_preserves_legacy(tmp_path):
    sources, config, corpus, _ = setup(tmp_path)
    configure(config)
    plan = plan_jobs(config, corpus)
    old = copy.deepcopy(config)
    old['extraction_input_protocol'] = 'extraction-anchors/v3'
    old['task_hashes'] = {k: t['task_sha256'] for k, t in tasks_for_config(old).items()}
    legacy = plan_jobs(old, corpus)
    assert plan['jobs'][0] == legacy['jobs'][0]
    assert all(a['job_id'] != b['job_id'] for a, b in zip(plan['jobs'][1:], legacy['jobs'][1:]))
    assert module_for_task(tasks_for_config(config)['extraction']).__name__.endswith('extraction_v6')
    assert module_for_task(tasks_for_config(old)['extraction']).__name__.endswith('extraction_v5')
    changed = copy.deepcopy(config)
    changed['extraction_grouping']['max_windows'] = 1
    other = plan_jobs(changed, corpus)
    assert plan['jobs'][0] == other['jobs'][0]
    assert plan['jobs'][1]['job_id'] != other['jobs'][1]['job_id']
    changed['extraction_grouping']['max_mentions'] = True
    assert 'extraction_grouping_positive_integers_required' in experiment_errors(changed)
    assert run_batch(config, corpus, source_root=sources, output=tmp_path/'unused')['status'] == 'blocked'


def test_v6_resident_packet_bundles_and_resume_do_not_leak_dataset(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    configure(config)
    root = tmp_path/'run'
    result = run_scheduler(config, corpus, policy, root=root, sources=sources,
                           leases=tmp_path/'leases', start_watchdog=False)
    assert result['terminal'] and result['slot_counts'] == {'success': 11, 'deferred': 1}
    indexes = list((root/'indexes').glob('*.json'))
    bundles = list((root/'extractions').glob('*.json'))
    assert len(indexes) == 1 and len(bundles) == 9
    for path in bundles:
        bundle = read_json(path)
        assert bundle['schema_version'] == 'paper-derived-observations/v6'
        assert bundle['resolved_object_count'] is None
        assert bundle['semantic_correctness'] == 'not_established'
    records = [_extended(p) for p in (root/'extraction_groups').rglob('*.json')]
    for path in records:
        r = read_json(path)
        assert 'DATASET_SECRET_CANARY' not in json.dumps(r['messages'])
        assert r['backend_result']['structured_output_applied']['synthetic'] is True
        assert r['task']['schema_version'] == 'four-category-task/v6'
    before = {p: p.read_bytes() for p in bundles+records+indexes}
    again = run_scheduler(config, corpus, policy, root=root, sources=sources,
                          leases=tmp_path/'leases', start_watchdog=False)
    assert again == result and before == {p: p.read_bytes() for p in before}


def test_fact_projection_preserves_assertion_and_unverified_authority():
    from high_fidelity_schema_study.four_category.paper import fact_view, category_counts
    bundle = {'schema_version': 'paper-derived-observations/v6', 'facts': [
        {'fact_id': 'one', 'subject_mention_id': 'subject', 'categories': ['structure', 'value'],
         'claim': {'kind': 'attribute', 'predicate': 'unit',
                   'assertion': {'status': 'inferred', 'value': 'kg', 'basis': 'explicit model rationale'}}}
    ]}
    before = copy.deepcopy(bundle)
    facts = fact_view(bundle)
    assert facts[0]['basis'] == 'inferred' and facts[0]['inference_basis'] == 'explicit model rationale'
    assert facts[0]['basis_authority'] == 'model_asserted_not_semantically_adjudicated'
    assert category_counts(facts)['unique_facts'] == 1
    assert bundle == before
