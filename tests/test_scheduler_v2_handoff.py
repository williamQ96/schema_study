"""Safety regression tests for the V1 to V2 authority handoff."""
import json
import os
from pathlib import Path
import signal
import sqlite3

import pytest

from scheduler_v2 import handoff, migrate, processes


def process_identity(pid=101):
    return {'pid': pid, 'start_ticks': 77, 'uid': 1234, 'argv': ['/usr/bin/python', 'worker'], 'state': 'R'}


def setup_handoff(tmp_path):
    old = tmp_path / 'old'
    old.mkdir()
    state = tmp_path / 'legacy-state'
    state.mkdir()
    audit = tmp_path / 'handoff'
    audit.mkdir()
    controller, supervisor = process_identity(101), process_identity(102)
    (state / 'controller-process.json').write_text(json.dumps({'identity': controller}))
    (state / 'supervisor-ready.json').write_text(json.dumps({'identity': supervisor}))
    config = {'legacy_handoff_state': str(state), 'ancestor_root': str(old),
              'local_leases': str(tmp_path / 'local-leases'),
              'legacy_leases': str(tmp_path / 'legacy-leases')}
    return config, audit, old, controller, supervisor


def test_existing_release_receipt_cannot_hide_a_live_same_worker(tmp_path, monkeypatch):
    config, audit, old, _, _ = setup_handoff(tmp_path)
    (audit / 'released-gpu0.json').write_text('{}')
    expected = process_identity(201)
    monkeypatch.setattr(handoff, 'same', lambda identity: True)
    with pytest.raises(ValueError, match='released_worker_still_live'):
        handoff.release_worker(config, {'worker_id': 'gpu0', 'gpu_ids': ['GPU0']},
                               expected, audit, object())


def test_control_freeze_backup_failure_resumes_old_controller_before_any_worker_stop(tmp_path, monkeypatch):
    config, audit, _, controller, _ = setup_handoff(tmp_path)
    signals = []
    class Legacy:
        def send_signal(self, identity, sig):
            signals.append((identity['pid'], sig))
    monkeypatch.setattr(handoff, 'same', lambda ident: ident['pid'] == controller['pid'])
    monkeypatch.setattr(handoff.signal, 'SIGSTOP', 19, raising=False)
    monkeypatch.setattr(handoff.signal, 'SIGCONT', 18, raising=False)
    monkeypatch.setattr(handoff.signal, 'SIGKILL', 9, raising=False)
    monkeypatch.setattr(handoff, 'identity', lambda pid: {'state': 'T'})
    monkeypatch.setattr(handoff.sqlite3, 'connect',
                        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError('busy')))
    with pytest.raises(sqlite3.OperationalError):
        handoff.stop_controllers(config, audit, Legacy())
    assert signals == [(controller['pid'], signal.SIGSTOP), (controller['pid'], signal.SIGCONT)]
    assert all(pid > 0 for pid, _ in signals)  # No negative PID/process-group signal.
    assert not (audit / 'control-transferred.json').exists()


def test_release_waits_for_writer_safe_boundary_then_signals_only_target_pid(tmp_path, monkeypatch):
    config, audit, old, _, _ = setup_handoff(tmp_path)
    wid = 'gpu0'
    worker_dir = old / 'workers' / wid
    worker_dir.mkdir(parents=True)
    (worker_dir / 'heartbeat.json').write_text(json.dumps({'phase': 'idle', 'group_index': 4}))
    expected = process_identity(201)
    monkeypatch.setattr(handoff.signal, 'SIGCONT', 18, raising=False)
    monkeypatch.setattr(handoff.signal, 'SIGKILL', 9, raising=False)
    alive = {'value': True}
    identity_checks = []
    monkeypatch.setattr(handoff, 'same', lambda ident: identity_checks.append(ident) or alive['value'])
    boundary_calls = []
    class Legacy:
        def stop_between_writes(self, ident):
            boundary_calls.append(ident)
            return len(boundary_calls) > 1  # First probe finds an open writer; resume and retry.
    kills = []
    def kill(pid, sig):
        kills.append((pid, sig))
        if sig == signal.SIGCONT:
            alive['value'] = False
    monkeypatch.setattr(handoff.os, 'kill', kill)
    monkeypatch.setattr(handoff, 'resources_free', lambda *a, **kw: True)
    worker = {'worker_id': wid, 'gpu_ids': ['GPU0']}
    handoff.release_worker(config, worker, expected, audit, Legacy())
    assert len(boundary_calls) == 2
    assert kills == [(expected['pid'], signal.SIGTERM), (expected['pid'], signal.SIGCONT)]
    assert all(pid > 0 for pid, _ in kills)
    receipt = json.loads((audit / 'released-gpu0.json').read_text())
    assert receipt['resource_release_verified'] is True


def test_unproven_resource_release_prevents_handoff_receipt_and_next_phase(tmp_path, monkeypatch):
    config, audit, old, _, _ = setup_handoff(tmp_path)
    expected = process_identity(202)
    monkeypatch.setattr(handoff, 'same', lambda ident: False)
    monkeypatch.setattr(handoff, 'resources_free', lambda *a, **kw: False)
    class FastClock:
        now = 1000
        @classmethod
        def time(cls):
            cls.now += 1
            return cls.now
        @staticmethod
        def sleep(_):
            pass
    monkeypatch.setattr(handoff, 'time', FastClock)
    with pytest.raises(RuntimeError, match='legacy_resource_release_unproven:gpu0'):
        handoff.release_worker(config, {'worker_id': 'gpu0', 'gpu_ids': ['GPU0']},
                               expected, audit, object())
    assert not (audit / 'released-gpu0.json').exists()
    assert not (audit / 'completed.json').exists()


def test_mirror_uses_strict_two_second_cutoff_and_quiescent_invalid_json_fails(tmp_path, monkeypatch):
    old, target = tmp_path / 'old', tmp_path / 'target'
    artifact = old / 'results' / 'edge.json'
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"ok":true}')
    monkeypatch.setattr(migrate, 'time', type('Clock', (), {'time': staticmethod(lambda: 100.0)}))
    os.utime(artifact, (98.0, 98.0))
    copied = migrate.mirror_immutable(old, target, quiescent=False)
    assert 'results/edge.json' in copied  # Exactly at cutoff is old enough.
    artifact2 = old / 'results' / 'new.json'
    artifact2.write_text('{"ok":true}')
    os.utime(artifact2, (98.001, 98.001))
    assert 'results/new.json' not in migrate.mirror_immutable(old, target, quiescent=False)
    broken = old / 'requests' / 'partial.json'
    broken.parent.mkdir(parents=True)
    broken.write_text('{partial')
    with pytest.raises(json.JSONDecodeError):
        migrate.mirror_immutable(old, target, quiescent=True)


def test_mirror_never_replaces_an_existing_hardlink(tmp_path):
    old, target = tmp_path / 'old', tmp_path / 'target'
    source = old / 'results' / 'record.json'
    source.parent.mkdir(parents=True)
    source.write_text('{"record":1}')
    target_file = target / 'results' / 'record.json'
    target_file.parent.mkdir(parents=True)
    os.link(source, target_file)
    before = (source.stat().st_ino, target_file.stat().st_ino, source.read_bytes())
    copied = migrate.mirror_immutable(old, target, quiescent=True)
    after = (source.stat().st_ino, target_file.stat().st_ino, source.read_bytes())
    assert 'results/record.json' in copied
    assert before == after


def test_reused_pid_with_different_start_identity_is_never_terminated(monkeypatch):
    expected = process_identity(303)
    replacement = {**expected, 'start_ticks': expected['start_ticks'] + 1}
    monkeypatch.setattr(processes, 'identity', lambda pid: replacement)
    monkeypatch.setattr(processes.os, 'getuid', lambda: expected['uid'], raising=False)
    kills = []
    monkeypatch.setattr(processes.os, 'kill', lambda *args: kills.append(args))
    assert processes.same(expected) is False
    assert processes.terminate(expected, grace=0) is False
    assert kills == []
def test_foreign_account_does_not_block_own_release_or_authorize_launch(monkeypatch):
    worker = {'gpu_ids': ['GPU0']}
    config = {'local_leases': 'local', 'legacy_leases': 'legacy'}
    monkeypatch.setattr(handoff, 'resources_free', lambda *args, observe=True: not observe)
    monkeypatch.setattr(handoff, 'gpu_processes', lambda: [('GPU0', 234)])
    monkeypatch.setattr(handoff, 'identity', lambda pid: {'uid': os.getuid()+1 if hasattr(os,'getuid') else 12345, 'start_ticks': 99})
    monkeypatch.setattr(handoff.os, 'getuid', lambda: 1234, raising=False)
    released, foreign = handoff.released_by_this_deployment(worker, config)
    assert released and foreign[0]['pid'] == 234
    monkeypatch.setattr(handoff, 'identity', lambda pid: {'uid': 1234, 'start_ticks': 99})
    assert handoff.released_by_this_deployment(worker, config) == (False, [])
