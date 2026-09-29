import copy
import json
from pathlib import Path
import sqlite3

import pytest

from high_fidelity_schema_study.four_category.common import read_json
from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category.resident_worker import ResidentCache
from high_fidelity_schema_study.four_category.scheduler import (
    compile_condition, default_policy, run_scheduler, validate_policy, Store, live_gate,
)
from high_fidelity_schema_study.four_category.scheduler_io import Lock


def setup(tmp_path):
    sources = tmp_path/'sources'
    config, corpus = make_fixture(sources)
    config['classification']['profile_id'] = config['roles']['locals'][0]
    config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
    policy = default_policy()
    policy.update(poll_s=.03, heartbeat_s=.03)
    return sources, config, corpus, policy


def test_resident_cache_loads_once_and_switches_only_on_profile_change():
    calls, closed = [], []
    pool = ResidentCache(lambda p: calls.append(p['model_id']) or object(), lambda obj: closed.append(obj))
    a = {'profile_id': 'a', 'model_id': 'a'}
    b = {'profile_id': 'b', 'model_id': 'b'}
    first = pool.acquire(a)
    assert pool.acquire(a) is first
    pool.acquire(b)
    assert calls == ['a', 'b'] and len(closed) == 1 and pool.reuses == 1
    pool.close()
    assert len(closed) == 2


def test_process_workers_real_contract_resume_and_no_dataset_leak(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    root = tmp_path/'run'
    summary = run_scheduler(config, corpus, policy, root=root, sources=sources,
                            leases=tmp_path/'leases', start_watchdog=False)
    assert summary['terminal']
    assert summary['slot_counts'] == {'success': 11, 'deferred': 1}
    assert len(summary['matrices'][0]['cells']) == 9
    assert summary['matrices'][0]['successful_cells'] == 9
    result_paths = sorted((root/'results').rglob('*.json'))
    assert len(result_paths) == 11
    before = {p: p.read_bytes() for p in result_paths}
    for p in result_paths:
        body = read_json(p)['body']
        if 'record' in body:
            assert 'DATASET_SECRET_CANARY' not in json.dumps(body['record']['messages'])
    result = run_scheduler(config, corpus, policy, root=root, sources=sources,
                           leases=tmp_path/'leases', start_watchdog=False)
    assert result == summary and before == {p: p.read_bytes() for p in result_paths}
    events = [json.loads(line) for p in (root/'workers').glob('*/events.jsonl') for line in p.read_text(encoding='utf-8').splitlines()]
    assert any(e['event'] == 'model_reused' for e in events)
    assert all('messages' not in e and 'raw_response' not in e for e in events)
    changed = copy.deepcopy(policy); changed['max_active_matrices'] = 3
    with pytest.raises(ValueError, match='resume_condition_changed'):
        run_scheduler(config, corpus, changed, root=root, sources=sources, leases=tmp_path/'leases', start_watchdog=False)
    from high_fidelity_schema_study.four_category.common import seal
    target = next(p for p in result_paths if 'record' in read_json(p)['body'])
    tampered = read_json(target)
    tampered['body']['record']['profile']['model_id'] = 'changed-model'
    tampered['body']['record'] = seal(tampered['body']['record'], 'record_sha256')
    target.write_text(json.dumps(seal(tampered, 'result_sha256')), encoding='utf-8')
    with pytest.raises(ValueError, match='result_replay_failed'):
        run_scheduler(config, corpus, policy, root=root, sources=sources, leases=tmp_path/'leases', start_watchdog=False)


def test_atomic_claim_and_gpu_lease_reject_duplicate_ownership(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    condition = compile_condition(config, corpus, policy)
    root = tmp_path/'run'; root.mkdir()
    store = Store(root, condition)
    store.refresh(100)
    row = store.choose(policy['workers'][0], None, 100)
    assert store.claim(row, 'gpu-0', 100)
    assert not store.claim(row, 'gpu-1', 100)
    store.db.close()
    with Lock(tmp_path/'lease'):
        with pytest.raises(RuntimeError, match='lease_already_owned'):
            with Lock(tmp_path/'lease'):
                pass


def test_dependency_failure_blocks_only_its_matrix_and_cpu_work_continues(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    condition = compile_condition(config, corpus, policy)
    root = tmp_path/'run'; root.mkdir()
    store = Store(root, condition)
    classifier = next(j for j in condition['jobs'] if j['kind'] == 'classification')
    store.db.execute("UPDATE jobs SET status='contract_invalid' WHERE id=?", (classifier['job_id'],))
    store.db.commit(); store.refresh(100)
    assert sum(r['status'] == 'blocked_dependency' for r in store.rows()) == 9
    assert store.choose(policy['workers'][2], None, 100)['job']['kind'] == 'dataset_parse'
    store.db.close()


def test_overlap_budget_and_live_without_qualification_rejected(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    duplicate = copy.deepcopy(policy); duplicate['workers'][1]['gpu_ids'] = ['1', '2']
    with pytest.raises(ValueError, match='overlapping'):
        validate_policy(duplicate)
    small = copy.deepcopy(policy); small['host_cpu_threads'] = 2
    with pytest.raises(ValueError, match='budget'):
        validate_policy(small)
    with pytest.raises(ValueError, match='actual_sif_required'):
        live_gate({'live': True}, None, sources)


def test_existing_headerless_dataset_condition_retains_first_row(tmp_path):
    from high_fidelity_schema_study.four_category.dataset import parse_dataset, verify_dataset
    path = tmp_path/'data.csv'
    path.write_text('001,2,A\n002,3,B\n003,4,C\n', encoding='utf-8')
    bundle = parse_dataset(path, root=tmp_path, format_hint='csv_no_header', sample_limit=2)
    values = [f for f in bundle['facts'] if f['predicate'] == 'observed_lexical_value']
    assert bundle['status'] == 'pass' and values[0]['value'] == '001'
    assert len(values) == 6 and not [f for f in bundle['facts'] if f['predicate'] == 'name']
    assert verify_dataset(bundle, tmp_path) == []


def test_affinity_yields_to_long_waiting_other_model(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    condition = compile_condition(config, corpus, policy)
    root = tmp_path/'run'; root.mkdir()
    store = Store(root, condition)
    classifier = next(j for j in condition['jobs'] if j['kind'] == 'classification')
    store.db.execute("UPDATE jobs SET status='success' WHERE id=?", (classifier['job_id'],))
    store.db.commit(); store.refresh(100)
    pool = store.eligible(policy['workers'][0])
    a, b = config['roles']['locals'][:2]
    old = next(r for r in pool if r['job']['profile_id'] == b)
    store.db.execute('UPDATE jobs SET ready_at=1 WHERE id=?', (old['id'],)); store.db.commit()
    assert store.choose(policy['workers'][0], a, 400)['id'] == old['id']
    store.db.close()
def test_chunked_prefill_is_explicit_validated_and_runtime_supported():
    import pytest
    from high_fidelity_schema_study.four_category.resident_worker import execution_kwargs
    profile = {'context_window': 8192, 'runtime': {'generation_execution': {'prefill_chunk_size': 4096}}}
    assert execution_kwargs(profile, {'prefill_chunk_size': None}) == {'prefill_chunk_size': 4096}
    with pytest.raises(ValueError, match='does_not_support'):
        execution_kwargs(profile, {})
    for value in (True, 0, -1, 8193, 0.5):
        profile['runtime']['generation_execution']['prefill_chunk_size'] = value
        with pytest.raises(ValueError, match='invalid_prefill'):
            execution_kwargs(profile, {'prefill_chunk_size': None})


def test_atomic_snapshot_retries_transient_replacement_denial(tmp_path, monkeypatch):
    from high_fidelity_schema_study.four_category import scheduler_io
    real_replace = scheduler_io.os.replace
    attempts = []
    def replace(src, dest):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError('reader holds destination briefly')
        return real_replace(src, dest)
    monkeypatch.setattr(scheduler_io.os, 'replace', replace)
    scheduler_io.atomic_json(tmp_path/'snapshot.json', {'complete': True})
    assert read_json(tmp_path/'snapshot.json') == {'complete': True} and len(attempts) == 2


def test_heterogeneous_workers_route_only_qualified_profiles(tmp_path):
    sources, config, corpus, policy = setup(tmp_path)
    a, b, c = config['roles']['locals']
    policy['workers'][0]['allowed_profile_ids'] = [a, c]
    policy['workers'][1]['allowed_profile_ids'] = [b]
    condition = compile_condition(config, corpus, policy)
    root = tmp_path/'run'; root.mkdir()
    store = Store(root, condition)
    classifier = next(j for j in condition['jobs'] if j['kind'] == 'classification')
    store.db.execute("UPDATE jobs SET status='success' WHERE id=?", (classifier['job_id'],)); store.db.commit()
    store.refresh(100)
    assert {r['job']['profile_id'] for r in store.eligible(policy['workers'][0])} == {a, c}
    assert {r['job']['profile_id'] for r in store.eligible(policy['workers'][1])} == {b}
    store.db.close()
    policy['workers'][0]['allowed_profile_ids'] = [a]
    with pytest.raises(ValueError, match='no_compatible_worker'):
        compile_condition(config, corpus, policy)


def test_flex_kernel_tuning_is_explicit_and_validated():
    from high_fidelity_schema_study.four_category.resident_worker import execution_kwargs
    p = {'runtime': {'settings': {'attn_implementation': 'flex_attention'},
                    'generation_execution': {'kernel_options': {'BLOCK_M': 32, 'BLOCK_N': 32, 'num_stages': 1}}}}
    assert execution_kwargs(p, {})['kernel_options']['BLOCK_M'] == 32
    p['runtime']['generation_execution']['kernel_options']['num_stages'] = 0
    with pytest.raises(ValueError, match='invalid_flex'):
        execution_kwargs(p, {})


def test_flex_options_reach_forward_without_generate_parser_override():
    from types import SimpleNamespace
    from high_fidelity_schema_study.four_category.resident_worker import install_forward_controls, generation_call_kwargs
    captured = []
    model = SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda: {}),
        register_forward_pre_hook=lambda hook, **kw: captured.append(hook))
    options = {'BLOCK_M': 32, 'BLOCK_N': 32, 'num_stages': 1}
    profile = {'runtime': {'settings': {'attn_implementation': 'flex_attention'},
                          'generation_execution': {'kernel_options': options}}}
    install_forward_controls(model, profile)
    assert generation_call_kwargs({'kernel_options': options}) == {}
    assert captured[0](model, (), {'input_ids': 'tokens'})[1] == {'input_ids': 'tokens', 'kernel_options': options}
    with pytest.raises(ValueError, match='conflict'):
        captured[0](model, (), {'kernel_options': {'BLOCK_M': 64}})
