"""Configuration and dispatch gates are exercised without GPU or model calls."""
from __future__ import annotations

import copy
import json

import pytest

from high_fidelity_schema_study.four_category import mercury
from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import ROOT, read_json
from high_fidelity_schema_study.four_category.tasks import render_task
from high_fidelity_schema_study.four_category.offline import make_fixture


def inputs():
    return tuple(read_json(ROOT / 'config' / name) for name in (
        'four_category_experiment_v1.json', 'mercury_models.example.json',
        'mercury_selection.example.json', 'mercury_host_v1.json'))


def selected_inputs():
    base, catalog, selection, host = inputs()
    for p in catalog['profiles']:
        p['status'] = 'frozen'
        p['model_id'] = '/models/' + p['profile_id'] if p['backend'] == 'transformers' else 'chosen-provider-model'
        p['context_window'] = 131072
        if p['backend'] == 'transformers':
            p['revision'] = 'a' * 40
            p['runtime']['checkpoint_manifest'] = {'path': '/inputs/test-manifest.json', 'file_bytes_sha256': 'e' * 64}
    selection['classifier']['profile_sha256'] = profile_hash(catalog['profiles'][0])
    return base, catalog, selection, host


def test_example_composes_to_nonoperational_draft_without_mutating_sources():
    values = inputs()
    before = copy.deepcopy(values)
    config, report = mercury.compose(*values)
    assert values == before
    assert config['status'] == 'draft' and config['inference_enabled'] is False
    assert config['classification']['profile_sha256'] is None
    assert report['execution_ready'] is False
    assert report['live_requests'] == 0
    assert len(config['profiles']) == 4
    assert config['task_hashes'] == values[0]['task_hashes']


@pytest.mark.parametrize('count', [1, 2, 4])
def test_presets_use_relative_ordinals_and_freeze_bindings(count):
    values = selected_inputs()
    values[2]['resource_preset'] = f'h200_{count}gpu'
    config, report = mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    assert report['configuration_ready'] is True
    assert report['execution_ready'] is False  # byte/GPU checks are run-time work
    for p in config['profiles'][:3]:
        assert set(p['runtime']['settings']['max_memory']) == {str(i) for i in range(count)} | {'cpu'}
    assert config['deployment']['required_gpu_count'] == count
    assert mercury.deployment_errors(config, image_sha256='b' * 64) == []
    assert 'actual_sif_identity_mismatch_or_missing' in mercury.deployment_errors(config, image_sha256='c' * 64)
    config['deployment']['source_file_bytes_sha256']['fake.py'] = 'f' * 64
    assert 'active_source_identity_mismatch' in mercury.deployment_errors(config, image_sha256='b' * 64)


def test_swap_changes_profile_not_shared_task_or_classifier(tmp_path):
    values = selected_inputs()
    original, _ = mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    values[1]['profiles'][1]['model_id'] = '/models/newly-released-family'
    values[1]['profiles'][1]['revision'] = 'c' * 40
    swapped, _ = mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    assert original['classification'] == swapped['classification']
    assert original['task_hashes'] == swapped['task_hashes']
    assert profile_hash(original['profiles'][1]) != profile_hash(swapped['profiles'][1])
    _, corpus = make_fixture(tmp_path)
    # The exact task contents depend on paper/index, never model selection.
    from high_fidelity_schema_study.four_category.workflow import tasks_for_config
    assert tasks_for_config(original) == tasks_for_config(swapped)
    paper_input = read_json(tmp_path / 'paper_input.json')
    assert render_task(tasks_for_config(original)['classification'], paper_input) == render_task(
        tasks_for_config(swapped)['classification'], paper_input)


def test_classifier_does_not_follow_changed_model_or_parameters():
    values = selected_inputs()
    values[1]['profiles'][0]['model_id'] = '/models/new-classifier'
    with pytest.raises(ValueError, match='classifier_catalog_pin_mismatch'):
        mercury.compose(*values)
    values[2]['classifier']['profile_sha256'] = None
    with pytest.raises(ValueError, match='explicit_classifier_catalog_pin_required'):
        mercury.compose(*values, freeze=True, image_sha256='b' * 64)


def test_independent_local_classifier_and_http_parameter_incompatibility():
    values = selected_inputs()
    classifier = copy.deepcopy(values[1]['profiles'][0])
    classifier['profile_id'] = 'independent-classifier'
    values[1]['profiles'].append(classifier)
    values[2]['classifier'] = {'profile_id': classifier['profile_id'], 'profile_sha256': profile_hash(classifier)}
    config, _ = mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    assert len(config['profiles']) == 5
    values[2]['locals'][1] = 'local_http_example_v1'
    with pytest.raises(ValueError, match='do_sample'):
        mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    values[2]['parameters']['local'] = {'max_output_tokens': 16384, 'temperature': .7}
    config, report = mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    assert report['configuration_ready']
    assert config['local_parameters'] == values[2]['parameters']['local']


def test_duplicate_candidates_unsupported_runtime_and_changed_tasks_rejected():
    values = selected_inputs()
    values[1]['profiles'].append(copy.deepcopy(values[1]['profiles'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        mercury.compose(*values)
    values = selected_inputs()
    values[1]['profiles'][1]['runtime']['settings']['silently_ignored_setting'] = True
    with pytest.raises(ValueError, match='unsupported Transformers setting'):
        mercury.compose(*values)
    values = selected_inputs()
    values[0]['task_hashes']['extraction'] = '0' * 64
    with pytest.raises(ValueError, match='base_task_hashes'):
        mercury.compose(*values)


def test_resource_conflict_is_not_silently_overwritten():
    values = selected_inputs()
    values[1]['profiles'][1]['runtime']['settings']['max_memory'] = {'0': '90GiB'}
    with pytest.raises(ValueError, match='profile_resource_preset_conflict'):
        mercury.compose(*values)


def test_changed_sif_invalidates_all_effective_profiles_and_cache_ids(tmp_path):
    from high_fidelity_schema_study.four_category.workflow import plan_jobs
    first, _ = mercury.compose(*selected_inputs(), freeze=True, image_sha256='b' * 64)
    second, _ = mercury.compose(*selected_inputs(), freeze=True, image_sha256='c' * 64)
    _, corpus = make_fixture(tmp_path)
    left = {j['job_id'] for j in plan_jobs(first, corpus)['jobs']}
    right = {j['job_id'] for j in plan_jobs(second, corpus)['jobs']}
    assert left.isdisjoint(right)


def test_freeze_rejects_placeholder_context_and_missing_checkpoint():
    values = selected_inputs()
    values[1]['profiles'][1]['model_id'] = '/models/REPLACE_LOCAL_B'
    with pytest.raises(ValueError, match='placeholder_model_identity'):
        mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    values = selected_inputs()
    values[1]['profiles'][1]['context_window'] = 1
    with pytest.raises(ValueError, match='context_window_cannot_fit'):
        mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    values = selected_inputs()
    del values[1]['profiles'][1]['runtime']['checkpoint_manifest']
    with pytest.raises(ValueError, match='checkpoint_manifest_binding_required'):
        mercury.compose(*values, freeze=True, image_sha256='b' * 64)


def test_checkpoint_bytes_checked_even_with_unchanged_profile(tmp_path):
    from high_fidelity_schema_study.four_category.checkpoints import build_manifest
    from high_fidelity_schema_study.four_category.common import file_digest, write_new
    values = selected_inputs()
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'weights.bin').write_bytes(b'weights')
    manifest = tmp_path / 'manifest.json'
    write_new(manifest, build_manifest(model, 'a' * 40))
    for profile in values[1]['profiles'][:3]:
        profile['model_id'] = str(model)
        profile['runtime']['checkpoint_manifest'] = {'path': str(manifest), 'file_bytes_sha256': file_digest(manifest)}
    values[2]['classifier']['profile_sha256'] = profile_hash(values[1]['profiles'][0])
    config, _ = mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    assert mercury.checkpoint_binding_errors(config, verify_bytes=True) == []
    (model / 'weights.bin').write_bytes(b'changed')
    assert any('sha256_mismatch:weights.bin' in e for e in mercury.checkpoint_binding_errors(config, verify_bytes=True))


def test_budget_cannot_exceed_actual_device_capacity():
    values = selected_inputs()
    values[2]['resource_preset'] = 'h200_1gpu'
    config, _ = mercury.compose(*values, freeze=True, image_sha256='b' * 64)
    observed = {'torch': {'cuda': {'devices': [{'index': 0, 'memory_total_bytes': 141_000_000_000}]}},
                'memory': {'total_bytes': 1024**4}}
    assert mercury.resource_errors(config, observed) == []
    observed['torch']['cuda']['devices'][0]['memory_total_bytes'] = 80 * 1024**3
    assert any(e.endswith(':0') for e in mercury.resource_errors(config, observed))


def test_run_gate_blocks_before_collect_or_dispatch(tmp_path, monkeypatch, capsys):
    from high_fidelity_schema_study.four_category import hardware
    config, _ = mercury.compose(*selected_inputs(), freeze=True, image_sha256='b' * 64)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.delenv('MERCURY_IMAGE_SHA256', raising=False)
    monkeypatch.setattr(hardware, 'collect_hardware', lambda: pytest.fail('hardware collection must follow identity gate'))
    monkeypatch.setattr(mercury, 'run_batch', lambda *a, **k: pytest.fail('must not dispatch'))
    assert mercury.main(['run', '--config', str(path), '--corpus', 'missing.json', '--source-root', str(tmp_path),
                         '--output', str(tmp_path / 'out'), '--allow-live']) == 1
    assert 'sif_identity_mismatch' in capsys.readouterr().out


def test_frontier_token_api_record_survives_packet_and_resume(tmp_path):
    from high_fidelity_schema_study.four_category.offline import mock_transport
    from high_fidelity_schema_study.four_category.workflow import run_batch
    from high_fidelity_schema_study.four_category.packet import build_packet
    sources = tmp_path / 'sources'
    config, corpus = make_fixture(sources)
    config.update(status='frozen', inference_enabled=True)
    classifier = next(p for p in config['profiles'] if p['profile_id'] == config['roles']['locals'][0])
    config['classification'].update(profile_id=classifier['profile_id'], profile_sha256=profile_hash(classifier))
    frontier = next(p for p in config['profiles'] if p['profile_id'] == config['roles']['soft_reference'])
    frontier.update(backend='responses', deployment='remote', endpoint='https://example.invalid/v1/responses')
    frontier['capabilities']['supported_parameters'] = ['max_output_tokens']
    frontier['runtime'] = {'token_counter': {'kind': 'responses_input_tokens',
                                           'endpoint': 'https://example.invalid/v1/responses/input_tokens'}}
    token_calls = []
    def token_transport(request):
        token_calls.append(request)
        return {'object': 'response.input_tokens', 'input_tokens': 10}
    def frontier_transport(request):
        response = mock_transport({'model': request['body']['model'], 'messages': request['body']['input']})
        return {'model': frontier['model_id'], 'status': 'completed',
                'output': [{'content': [{'type': 'output_text', 'text': response['text']}]}]}
    transports = {p['profile_id']: mock_transport for p in config['profiles']}
    transports[frontier['profile_id']] = frontier_transport
    kwargs = {'source_root': sources, 'output': tmp_path / 'runs', 'allow_live': True,
              'transports': transports, 'token_transports': {frontier['profile_id']: token_transport}}
    batch = run_batch(config, corpus, **kwargs)
    verification = build_packet(config, corpus, batch, run_root=kwargs['output'], source_root=sources,
                                output=tmp_path / 'packet')
    assert verification['status'] == 'pass'
    assert len(token_calls) == 1
    resumed = run_batch(config, corpus, **kwargs)
    assert len(token_calls) == 1  # replayed successful run does not re-call counter
    assert resumed['backend_attempts_this_invocation'] == 0
