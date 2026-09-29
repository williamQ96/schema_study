"""Offline lineage import keeps raw group provenance across a new condition."""
import copy
from pathlib import Path
import tempfile
import shutil

import pytest

from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import ROOT, digest, file_digest, read_json, seal
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category import scheduler
from high_fidelity_schema_study.four_category.cache_lineage import (
    compatibility_errors, stage_group_cache, verify_cache_lineage,
)
from high_fidelity_schema_study.four_category.scheduled_packet import build_scheduled_packet, verify_scheduled_packet
from .test_complete_scheduler import configure


@pytest.fixture
def completed_source(monkeypatch):
    short_base = Path.home() / '.codex' / 'tmp'
    with tempfile.TemporaryDirectory(prefix='cl-', dir=short_base if short_base.is_dir() else None) as folder:
        root = Path(folder)
        sources = root / 'sources'
        config, corpus = make_fixture(sources)
        config['classification']['profile_id'] = config['roles']['locals'][0]
        config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
        configure(config)
        policy = scheduler.default_policy()
        policy.update(poll_s=.03, heartbeat_s=.03)
        original = root / 'old'
        scheduler.run_scheduler(config, corpus, policy, root=original, sources=sources,
                                leases=root / 'leases', start_watchdog=False)
        ancestor = read_json(original / 'condition.json')
        original_identity = scheduler.code_identity
        old_map = ancestor['source_file_bytes_sha256']
        policy['execution_source'] = {'path': str(ROOT), 'source_file_bytes_sha256': old_map,
                                      'source_tree_sha256': digest(old_map)}
        monkeypatch.setattr(scheduler, 'code_identity', lambda execution_root=None: (
            original_identity(execution_root) if execution_root is not None else
            {**old_map, 'validation_only_test.py': '1' * 64}))
        target = scheduler.compile_condition(config, corpus, policy)
        yield root, sources, config, corpus, policy, original, ancestor, target


def test_full_import_replays_and_relocates_packet(completed_source):
    root, sources, config, corpus, policy, original, ancestor, target = completed_source
    assert not compatibility_errors(ancestor, target)
    new_root = root / 'new'
    binding = stage_group_cache(original, new_root, sources, root / 'leases',
                                expected_ancestor_condition=ancestor['condition'],
                                target_condition_without_lineage=target)
    new_policy = copy.deepcopy(policy)
    new_policy['group_cache_import'] = binding
    new_condition = scheduler.compile_condition(config, corpus, new_policy)
    assert new_condition['condition'] != ancestor['condition']
    errors, members = verify_cache_lineage(new_root, new_condition, source_root=sources)
    assert not errors, errors
    assert any(p.startswith('lineage/') for p in members)
    try:
        summary = scheduler.run_scheduler(config, corpus, new_policy, root=new_root, sources=sources,
                                          leases=root / 'leases', start_watchdog=False)
    except Exception as exc:
        logs = [p.read_text(encoding='utf-8') for p in (new_root / 'workers').glob('*/process.log')]
        raise AssertionError((exc, logs)) from exc
    assert summary['terminal']
    packet = root / 'packet'
    report = build_scheduled_packet(new_root, sources, packet, paper_id=corpus['papers'][0]['paper_id'])
    assert report['status'] == 'pass', report
    relocated = root / 'relocated'
    shutil.copytree(packet, relocated)
    assert verify_scheduled_packet(relocated, sources, expected_condition=new_condition['condition'])['status'] == 'pass'


def test_rejects_changed_logical_job_and_rehashed_member(completed_source):
    root, sources, config, corpus, policy, original, ancestor, target = completed_source
    wrong = seal({**target, 'jobs': []}, 'condition')
    assert compatibility_errors(ancestor, wrong)
    wrong_binding = copy.deepcopy(target)
    wrong_binding['policy']['execution_source']['source_tree_sha256'] = '0' * 64
    assert 'execution_source_not_ancestor' in compatibility_errors(ancestor, seal(wrong_binding, 'condition'))
    wrong_profile = copy.deepcopy(target)
    wrong_profile['config']['profiles'][0]['context_window'] += 1
    assert 'semantic_condition_mismatch' in compatibility_errors(ancestor, seal(wrong_profile, 'condition'))
    wrong_deployment = copy.deepcopy(target)
    wrong_deployment['config']['deployment'] = {'source_file_bytes_sha256': {'fake.py': '0' * 64}}
    assert 'semantic_condition_mismatch' in compatibility_errors(ancestor, seal(wrong_deployment, 'condition'))
    wrong_task = copy.deepcopy(target)
    wrong_task['config']['task_hashes']['classification'] = '0' * 64
    assert compatibility_errors(ancestor, seal(wrong_task, 'condition'))
    new_root = root / 'new'
    binding = stage_group_cache(original, new_root, sources, root / 'leases',
                                expected_ancestor_condition=ancestor['condition'],
                                target_condition_without_lineage=target)
    new_policy = copy.deepcopy(policy)
    new_policy['group_cache_import'] = binding
    new_condition = scheduler.compile_condition(config, corpus, new_policy)
    group = next(new_root.glob('classification_groups/*/*/attempt-*.json'))
    group.write_bytes(group.read_bytes() + b' ')
    errors, _ = verify_cache_lineage(new_root, new_condition, source_root=sources)
    assert errors and 'lineage_member_identity_mismatch' in errors[0]


def test_partial_prefix_import_generates_only_missing_group(completed_source):
    root, sources, config, corpus, policy, original, ancestor, target = completed_source
    all_groups = sorted(original.glob('extraction_groups/*/*/attempt-*.json'))
    assert len(all_groups) > 1
    missing = all_groups[-1]
    relative = missing.relative_to(original)
    job_id = relative.parts[1]
    missing.unlink()
    for kind in ('requests', 'results'):
        shutil.rmtree(original / kind / job_id)
    new_root = root / 'new'
    binding = stage_group_cache(original, new_root, sources, root / 'leases',
                                expected_ancestor_condition=ancestor['condition'],
                                target_condition_without_lineage=target)
    imported = {p.relative_to(new_root).as_posix(): file_digest(p)
                for p in new_root.glob('extraction_groups/*/*/attempt-*.json')}
    assert relative.as_posix() not in imported
    new_policy = copy.deepcopy(policy)
    new_policy['group_cache_import'] = binding
    condition = scheduler.compile_condition(config, corpus, new_policy)
    summary = scheduler.run_scheduler(config, corpus, new_policy, root=new_root, sources=sources,
                                      leases=root / 'leases', start_watchdog=False)
    assert summary['terminal']
    assert (new_root / relative).is_file()
    assert all(file_digest(new_root / path) == sha for path, sha in imported.items())
    assert not verify_cache_lineage(new_root, condition, source_root=sources)[0]
    packet = root / 'packet'
    report = build_scheduled_packet(new_root, sources, packet, paper_id=next(
        j['paper_id'] for j in ancestor['jobs'] if j['job_id'] == job_id))
    assert report['status'] == 'pass', report
