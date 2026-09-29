import json
from pathlib import Path

import pytest

from scheduler_v2.cli import create_command, reconcile, status, verify, main, _science_code_identity
from scheduler_v2.io import digest, file_sha, write_once
from high_fidelity_schema_study.four_category.mercury import code_identity


@pytest.fixture
def deployment(tmp_path):
    root, local, source, scientific = tmp_path / 'science', tmp_path / 'local', tmp_path / 'runtime', tmp_path / 'scientific'
    root.mkdir(); local.mkdir(); source.mkdir(); scientific.mkdir()
    (source / 'runtime.py').write_text('# pinned', encoding='utf-8')
    (scientific / 'code.py').write_text('# scientific source', encoding='utf-8')
    assert _science_code_identity(scientific) == code_identity(scientific)
    condition = {'policy': {'workers': [{'worker_id': 'gpu-a', 'gpu_ids': ['0']}]},
                 'source_file_bytes_sha256': _science_code_identity(scientific)}
    (root / 'condition.json').write_text(json.dumps(condition), encoding='utf-8')
    execution_identity = {'key_id': 'fixture-key'}
    members = {'source.json': 'abc'}
    member_hash = digest(members)
    write_once(root / 'imports' / ('members-' + member_hash + '.json'), members)
    plans = {'job-1': ['group-1']}
    write_once(root / 'group_plans.json', plans)
    manifest = {'artifact_manifest_sha256': member_hash, 'group_plan_sha256': digest(plans),
                'execution_identity': execution_identity}
    write_once(root / 'imports' / 'manifest.json', manifest)
    (root / 'import-latest.json').write_text(json.dumps({'path': 'imports/manifest.json', 'file_bytes_sha256': file_sha(root / 'imports' / 'manifest.json')}), encoding='utf-8')
    config = {'root': str(root), 'local_root': str(local), 'deployment_id': 'd1',
              'condition_file_sha256': file_sha(root / 'condition.json'), 'runtime_source': str(source),
              'science_source': str(scientific), 'execution_identity': execution_identity,
              'operational_files': {'runtime.py': file_sha(source / 'runtime.py')}}
    path = tmp_path / 'deployment.json'; path.write_text(json.dumps(config), encoding='utf-8')
    return config, path


def test_control_commands_are_immutable_and_unique(deployment):
    config, _ = deployment
    result = create_command(config, 'pause', {'worker_id': 'gpu-a'}, 'command-1')
    assert Path(result['path']).is_file()
    assert json.loads(Path(result['path']).read_text()) == result['command']
    assert create_command(config, 'pause', {'worker_id': 'gpu-a'}, 'command-1')['command'] == result['command']
    with pytest.raises(ValueError, match='immutable_record_conflict'):
        create_command(config, 'resume', {}, 'command-1')


def test_status_and_reconcile_are_read_only(deployment):
    config, _ = deployment
    local = Path(config['local_root'])
    assert status(config)['status'] == 'waiting_for_snapshot'
    (local / 'snapshot.json').write_text(json.dumps({'terminal': False, 'stage_counts': {'classification': {'pending': 2}},
         'workers': [{'waiting_reason': 'resource_busy'}], 'group_coverage': {}, 'controls': {}}), encoding='utf-8')
    assert status(config)['waiting_reasons'] == {'resource_busy': 1}
    result = reconcile(config)
    assert result['action_taken'] is False and result['unreceipted_commands'] == []
    events = Path(config['root']) / 'scheduler_v2' / config['deployment_id'] / 'events'
    events.mkdir(parents=True)
    (events / '00000001.json').write_text(json.dumps({'kind': 'claim', 'assignment': {'assignment_id': 'a1', 'worker_id': 'gpu-a'}}), encoding='utf-8')
    assert reconcile(config)['unfinished_assignments'] == [{'assignment_id': 'a1', 'worker_id': 'gpu-a', 'status': 'assigned'}]


def test_verify_checks_pins_without_model_calls(deployment):
    config, _ = deployment
    result = verify(config)
    assert result['status'] == 'pass' and result['model_calls'] == 0
    (Path(config['runtime_source']) / 'runtime.py').write_text('changed', encoding='utf-8')
    assert verify(config)['status'] == 'failed'


def test_fixed_cli_reserve_and_rejects_unknown_gpu(deployment, capsys):
    config, path = deployment
    main(['--config', str(path), 'reserve', '--gpu', '0', '--reservation-key', 'maint-1', '--command-id', 'maint-1'])
    cmd = json.loads(capsys.readouterr().out)['command']
    assert cmd['action'] == 'reserve' and cmd['payload']['gpu_ids'] == ['0']
    assert cmd['command_id'] == cmd['payload']['reservation_key'] == 'maint-1'
    assert cmd['payload']['expires_at'] - cmd['payload']['grace_until'] == pytest.approx(9900, abs=1)
    first = cmd
    main(['--config', str(path), 'reserve', '--gpu', '0', '--reservation-key', 'maint-1'])
    assert json.loads(capsys.readouterr().out)['command'] == first
    main(['--config', str(path), 'cancel-reservation', '--reservation-key', 'maint-1', '--command-id', 'cancel-1'])
    cancelled = json.loads(capsys.readouterr().out)['command']
    assert cancelled['payload']['reservation_key'] == first['command_id']
    with pytest.raises(ValueError, match='gpu_not_in_deployment'):
        main(['--config', str(path), 'reserve', '--gpu', '8', '--reservation-key', 'x'])
    with pytest.raises(ValueError, match='reservation_key_is_command_id'):
        main(['--config', str(path), 'reserve', '--gpu', '0', '--reservation-key', 'x', '--command-id', 'different'])
    with pytest.raises(SystemExit):
        main(['--config', str(path), 'exec', 'rm', '-rf', '/'])


def test_start_returns_after_fenced_supervisor_launch(deployment, monkeypatch):
    _, path = deployment
    from scheduler_v2 import supervisor
    observed = {}
    monkeypatch.setattr(supervisor, '_fenced_launch', lambda local, role, command, log, env=None:
                        observed.update(local=local, role=role, command=command, log=log, env=env) or {'pid': 42})
    result = supervisor.start_background(path)
    assert result['pid'] == 42
    assert observed['role'] == 'supervisor'
    assert observed['command'][1:3] == ['-m', 'scheduler_v2.supervisor']
    assert 'scheduler_v2.runtime' not in observed['command']


def test_verify_binds_historical_manifest_to_execution_and_science_source(deployment):
    config, _ = deployment
    assert verify(config)['status'] == 'pass'
    changed = dict(config, execution_identity={'key_id': 'different'})
    assert 'historical_execution_identity_mismatch' in verify(changed)['errors']
    (Path(config['science_source']) / 'code.py').write_text('changed source', encoding='utf-8')
    assert 'scientific_source_identity_mismatch' in verify(config)['errors']


def test_supervisor_recovers_crash_after_launch_before_pid_receipt(deployment, monkeypatch):
    from scheduler_v2 import supervisor
    config, _ = deployment
    local = Path(config['local_root'])
    local.mkdir(exist_ok=True)
    (local / 'watchdog-launch-intent.json').write_text(json.dumps({'role': 'watchdog', 'command': ['python', 'watchdog']}), encoding='utf-8')
    candidate = {'pid': 55, 'argv': ['python', 'watchdog']}
    monkeypatch.setattr(supervisor, '_alive', lambda path: None)
    monkeypatch.setattr(supervisor, '_intended_process', lambda path: candidate)
    launched = []
    monkeypatch.setattr(supervisor, '_launch', lambda *a, **k: launched.append(True))
    result = supervisor._fenced_launch(local, 'watchdog', ['python', 'watchdog'], local / 'watchdog.log')
    assert result == candidate and not launched


def test_watchdog_restart_budget_is_independent_and_bounded(deployment, monkeypatch):
    from scheduler_v2 import supervisor
    config, _ = deployment
    local = Path(config['local_root'])
    local.mkdir(exist_ok=True)
    (local / 'watchdog-process.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(supervisor, '_alive', lambda path: None)
    monkeypatch.setattr(supervisor, '_intended_process', lambda path: None)
    launches = []
    monkeypatch.setattr(supervisor, '_fenced_launch', lambda *a, **k: launches.append(True) or {'pid': 77})
    process, times, state = supervisor._ensure_watchdog(local, ['watchdog'], local / 'watchdog.log', {}, [1, 2, 3], 4, 3)
    assert process is None and times == [1, 2, 3] and state == 'restart_limit_reached' and launches == []
