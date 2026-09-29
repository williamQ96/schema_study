"""Fresh bootstrap uses the frozen CPU parser and leaves GPU work pending."""
import copy
from pathlib import Path

import pytest

from high_fidelity_schema_study.four_category.common import seal
from high_fidelity_schema_study.four_category.scheduler import compile_condition
from high_fidelity_schema_study.four_category.scheduler_tasks import verify_result
from scheduler_v2 import bootstrap, cli, runtime
from scheduler_v2.io import file_sha, read, write_once
from scheduler_v2.state import Ledger

from .test_complete_scheduler import configure
from .test_matrix_scheduler import setup


def _fresh(tmp_path):
    sources, experiment, corpus, policy = setup(tmp_path)
    configure(experiment)
    condition = compile_condition(experiment, corpus, policy)
    frozen = tmp_path / 'frozen-condition.json'
    write_once(frozen, condition)
    package = Path(__file__).resolve().parents[1]
    deployment = {'root': str(tmp_path / 'fresh'), 'local_root': str(tmp_path / 'local'),
                  'sources': str(sources), 'science_source': str(package),
                  'runtime_source': str(package), 'deployment_id': 'fresh-mock',
                  'execution_identity': {'mode': 'mock'},
                  'condition_file_sha256': file_sha(frozen),
                  'operational_files': {'scheduler_v2/bootstrap.py': file_sha(package / 'scheduler_v2/bootstrap.py')}}
    config_path = tmp_path / 'deployment.json'
    write_once(config_path, deployment)
    return deployment, config_path, frozen, condition


def test_fresh_bootstrap_replays_cpu_results_and_preserves_pending_gpu_jobs(tmp_path, monkeypatch):
    deployment, config_path, frozen, condition = _fresh(tmp_path)
    from high_fidelity_schema_study.four_category import backends
    monkeypatch.setattr(backends, '_transformers_transport',
                        lambda *args, **kwargs: pytest.fail('GPU/model transport called'))
    first = bootstrap.prepare(config_path, condition_path=frozen)
    root = Path(deployment['root'])
    manifest = runtime.checked_ref(root, read(root / 'import-latest.json'))
    dataset_jobs = [j for j in condition['jobs'] if j['kind'] == 'dataset_parse']
    assert first['dataset_jobs'] == len(dataset_jobs) == 1
    assert first == bootstrap.prepare(config_path, condition_path=frozen)
    assert cli.verify(deployment)['status'] == 'pass'
    assert len(list((root / 'imports').glob('fresh-*.json'))) == 1
    for job in dataset_jobs:
        jid = job['job_id']
        request = read(root / 'requests' / jid / 'attempt-1.json')
        result = read(root / 'results' / jid / 'attempt-1.json')
        assert request['job'] == job and request['attempt'] == 1
        assert verify_result(condition, request, result, Path(deployment['sources']), root) == ([], None)
        assert manifest['imported'][jid]['parent_status'] == 'success'
    assert all(manifest['imported'][j['job_id']]['parent_status'] == 'pending'
               for j in condition['jobs'] if j['kind'] in {'classification', 'local_extraction'})
    assert all(manifest['imported'][j['job_id']]['parent_status'] == 'deferred'
               for j in condition['jobs'] if j['kind'] == 'soft_reference')
    with Ledger(Path(deployment['local_root']), root, deployment['deployment_id']) as ledger:
        snapshot = ledger.initialize(condition['jobs'], read(root / 'group_plans.json'), manifest['imported'])
        assert sum(j['status'] == 'success' for j in snapshot['jobs']) == len(dataset_jobs)
        assert sum(j['status'] == 'pending' for j in snapshot['jobs']) == sum(
            j['kind'] in {'classification', 'local_extraction'} for j in condition['jobs'])
    assert not list((root / 'classification_groups').rglob('*.json'))
    assert not list((root / 'extraction_groups').rglob('*.json'))


def test_fresh_bootstrap_rejects_changed_condition_and_deployment(tmp_path):
    deployment, config_path, frozen, condition = _fresh(tmp_path)
    bootstrap.prepare(config_path, condition_path=frozen)
    changed = copy.deepcopy(condition)
    changed['live'] = not changed['live']
    changed = seal(changed, 'condition')
    conflicting = tmp_path / 'other-condition.json'
    write_once(conflicting, changed)
    with pytest.raises(ValueError, match='frozen_condition_copy_conflict'):
        bootstrap.prepare(config_path, condition_path=conflicting)
    changed_deployment = dict(deployment, deployment_id='different')
    changed_path = tmp_path / 'other-deployment.json'
    write_once(changed_path, changed_deployment)
    with pytest.raises(ValueError, match='immutable_record_conflict'):
        bootstrap.prepare(changed_path, condition_path=frozen)
