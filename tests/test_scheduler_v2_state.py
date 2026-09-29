from __future__ import annotations

import json
import shutil

import pytest

from scheduler_v2.state import Ledger
from scheduler_v2.io import digest, write_once


def plan():
    jobs = [{'job_id': 'c1', 'kind': 'classification', 'paper_id': 'p1', 'dependencies': []},
            {'job_id': 'e1', 'kind': 'extraction', 'paper_id': 'p1', 'dependencies': ['c1']}]
    return jobs, {'c1': ['a', 'b'], 'e1': ['x']}


def test_returned_incomplete_is_terminal_without_regeneration(tmp_path):
    with Ledger(tmp_path/'local', tmp_path/'shared', 'd') as ledger:
        ledger.initialize([{'job_id': 'j', 'kind': 'local_extraction'}], {'j': ['g']})
        epoch=ledger.new_epoch()
        ledger.register_worker('w', 'i', ['GPU0'])
        ledger.mark_ready('j', 'g', 1)
        a=ledger.claim('j', 'g', 'w', 'i', epoch, 2)
        ledger.acknowledge(a['assignment_id'], {'sha': 'raw'}, 'w', 'i', epoch, 3)
        assert ledger.commit(a['assignment_id'], {'status': 'incomplete', 'generation_returned': True}, {'sha': 'raw'}, 4)
        assert ledger.claim('j', 'g', 'w', 'i', epoch, 5) is None
        assert ledger.set_parent('j', 'completed_with_rejections', {'sha': 'parent'}, now=6)


def test_ack_commit_parent_and_replay(tmp_path):
    local, shared = tmp_path / 'local', tmp_path / 'shared'
    with Ledger(local, shared, 'deployment') as ledger:
        jobs, groups = plan()
        ledger.initialize(jobs, groups)
        assert ledger.initialize(jobs, groups)['jobs'][0]['groups'][0]['group_id'] == 'a'
        epoch = ledger.new_epoch()
        ledger.register_worker('gpu0', 'boot1')
        ledger.mark_ready('c1', 'a', 10)
        assignment = ledger.claim('c1', 'a', 'gpu0', 'boot1', epoch, 11)
        assert assignment
        assert not ledger.set_parent('c1', 'success', {'sha': 'parent'}, now=12)
        assert ledger.acknowledge(assignment['assignment_id'], {'sha': 'raw'}, 'gpu0', 'boot1', epoch, 12)
        assert ledger.acknowledge(assignment['assignment_id'], {'sha': 'raw'}, 'gpu0', 'boot1', epoch, 12)
        assert not ledger.acknowledge(assignment['assignment_id'], {'sha': 'other'}, 'gpu0', 'boot1', epoch, 12)
        verdict = {'status': 'success', 'generation_returned': True, 'attempt_refs': [{'sha': 'raw'}]}
        assert ledger.commit(assignment['assignment_id'], verdict, {'sha': 'verified'}, 13)
        assert not ledger.commit(assignment['assignment_id'], verdict, {'sha': 'verified'}, 13)
        assert ledger.snapshot()['jobs'][0]['status'] == 'pending'
        before = ledger.snapshot()
    shutil.rmtree(local)
    with Ledger(local, shared, 'deployment') as ledger:
        assert ledger.snapshot() == before
        with pytest.raises(ValueError, match='ledger_plan_identity_mismatch'):
            ledger.initialize([{'job_id': 'different'}], {})


def test_epoch_fence_recovery_budget_and_verified_raw(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        jobs = [{'job_id': f'j{i}', 'kind': 'classification'} for i in range(4)]
        ledger.initialize(jobs, {j['job_id']: ['g'] for j in jobs})
        first = ledger.new_epoch()
        ledger.register_worker('gpu0', 'one')
        ledger.register_worker('gpu1', 'one')
        ledger.register_worker('gpu2', 'one')
        ids = []
        for i in range(3):
            ledger.mark_ready(f'j{i}', 'g', 1)
            ids.append(ledger.claim(f'j{i}', 'g', f'gpu{i}', 'one', first, i + 2)['assignment_id'])
        second = ledger.new_epoch()
        assert not ledger.acknowledge(ids[0], {'raw': 1}, 'gpu0', 'one', first)
        assert ledger.adopt(ids[0], 'gpu0', 'one', second, {'process_incarnation_confirmed': True})
        assert ledger.acknowledge(ids[0], {'raw': 1}, 'gpu0', 'one', second)
        proof = {aid: {'owner_released': True, 'incarnation': 'one'} for aid in ids}
        assert ledger.recover(100, proof) == ids[1:]
        assert ledger.recover(101, proof) == []
        assert ledger.snapshot()['jobs'][0]['groups'][0]['status'] == 'pending_verification'
        assert not ledger.acknowledge(ids[1], {'raw': 2}, 'gpu1', 'one', first)


def test_controls_and_imports(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        jobs, plans = plan()
        ledger.initialize(jobs, plans, {'c1': {'groups': {'a': {'status': 'success',
                                           'generation_returned': True, 'attempt_refs': []}}}})
        assert ledger.snapshot()['jobs'][0]['groups'][0]['status'] == 'success'
        assert ledger.control('pause', 'pause-1', now=100)
        assert not ledger.control('pause', 'pause-1', now=100)
        assert ledger.control('resume', 'resume-1', now=101)
        assert ledger.control('reserve', 'reservation-1', {'gpu_ids': ['0'],
                             'grace_until': 100, 'expires_at': 200}, now=100)
        assert ledger.control('cancel_reservation', 'cancel-1',
                              {'reservation_key': 'reservation-1', 'resource_release_ack': True}, now=102)
        assert not ledger.snapshot()['controls']['reservations']


def test_imported_rejected_group_remains_terminal_and_unclaimable(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        ledger.initialize([{'job_id': 'j', 'kind': 'classification'}], {'j': ['rejected', 'next']},
                          {'j': {'groups': {'rejected': {'status': 'contract_invalid',
                                  'generation_returned': True,
                                  'attempt_refs': [{'path': 'old.json', 'file_bytes_sha256': 'abc'}]}}}})
        epoch = ledger.new_epoch()
        ledger.register_worker('gpu', 'inc', ['0'])
        assert not ledger.mark_ready('j', 'rejected', 1)
        assert ledger.claim('j', 'rejected', 'gpu', 'inc', epoch, 2) is None
        group = ledger.snapshot()['jobs'][0]['groups'][0]
        assert group['status'] == 'contract_invalid'
        assert group['attempt_refs'][0]['path'] == 'old.json'


def test_pending_raw_adoption_requires_matching_digest(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        old = ledger.new_epoch()
        ledger.register_worker('gpu', 'pid-start-1')
        ledger.mark_ready('j', 'g', 1)
        aid = ledger.claim('j', 'g', 'gpu', 'pid-start-1', old, 2)['assignment_id']
        raw = {'path': 'raw.json', 'file_bytes_sha256': 'abc'}
        assert ledger.acknowledge(aid, raw, 'gpu', 'pid-start-1', old)
        new = ledger.new_epoch()
        assert not ledger.adopt(aid, 'gpu', 'pid-start-1', new,
                                {'raw_ref_verified': True, 'raw_ref_sha256': digest({'path': 'other'})})
        assert ledger.adopt(aid, 'gpu', 'pid-start-1', new,
                            {'raw_ref_verified': True, 'raw_ref_sha256': digest(raw)})
        assert ledger.commit(aid, {'status': 'success', 'generation_returned': True}, raw)


def test_journal_tamper_detected_and_unapplied_event_replayed(tmp_path):
    local, shared = tmp_path / 'local', tmp_path / 'shared'
    with Ledger(local, shared, 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        ledger.new_epoch()
    # Local index behind the shared journal simulates a crash after publication.
    with Ledger(local, shared, 'd') as ledger:
        with ledger.db:
            ledger.db.execute('DELETE FROM events WHERE seq=2')
            ledger.db.execute("UPDATE meta SET value='0' WHERE key='epoch'")
            ledger.db.execute("DELETE FROM meta WHERE key='last_sha'")
            ledger._set_meta('last_sha', digest(json.loads(
                (shared / 'scheduler_v2' / 'd' / 'events' / '00000001.json').read_text())))
    with Ledger(local, shared, 'd') as ledger:
        assert ledger.snapshot()['epoch'] == 1
    path = shared / 'scheduler_v2' / 'd' / 'events' / '00000001.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='journal_existing_event_mismatch'):
        Ledger(local, shared, 'd')


def test_claim_fences_worker_incarnation_and_paused_control(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        epoch = ledger.new_epoch()
        ledger.register_worker('gpu', 'pid-start-1')
        ledger.mark_ready('j', 'g', 1)
        ledger.control('pause', 'p', now=1)
        assert ledger.claim('j', 'g', 'gpu', 'pid-start-1', epoch, 2) is None
        ledger.control('resume', 'r', now=2)
        assert ledger.claim('j', 'g', 'gpu', 'wrong-pid-start', epoch, 3) is None
        assignment = ledger.claim('j', 'g', 'gpu', 'pid-start-1', epoch, 3)
        ledger.register_worker('gpu', 'pid-start-2')
        assert not ledger.acknowledge(assignment['assignment_id'], {}, 'gpu', 'pid-start-1', epoch)


def test_targeted_pause_drain_and_journaled_policy_state(tmp_path):
    local, shared = tmp_path / 'local', tmp_path / 'shared'
    with Ledger(local, shared, 'd') as ledger:
        ledger.initialize([{'job_id': 'a', 'kind': 'extraction', 'paper_id': 'p1'},
                           {'job_id': 'b', 'kind': 'extraction', 'paper_id': 'p2'}],
                          {'a': ['g'], 'b': ['g']})
        epoch = ledger.new_epoch()
        ledger.register_worker('gpu0', 'one')
        ledger.register_worker('gpu1', 'one')
        ledger.mark_ready('a', 'g', 1)
        ledger.mark_ready('b', 'g', 1)
        ledger.control('pause', 'pause-gpu0', {'worker_id': 'gpu0'}, 2)
        assert ledger.claim('a', 'g', 'gpu0', 'one', epoch, 3) is None
        first = ledger.claim('a', 'g', 'gpu1', 'one', epoch, 3)
        assert first
        assert ledger.snapshot()['policy_state']['last_extraction_paper'] == 'p1'
        ledger.control('resume', 'resume-gpu0', {'worker_id': 'gpu0'}, 4)
        second = ledger.claim('b', 'g', 'gpu0', 'one', epoch, 5)
        assert second
        assert ledger.snapshot()['policy_state']['lane_streak'] == 2
        assert ledger.snapshot()['policy_state_by_worker']['gpu0']['lane_streak'] == 1
        assert ledger.snapshot()['policy_state_by_worker']['gpu1']['lane_streak'] == 1
        ledger.control('drain', 'drain-gpu0', {'worker_id': 'gpu0'}, 6)
        assert ledger.snapshot()['controls']['draining_workers'] == ['gpu0']
        before = ledger.snapshot()['policy_state']
    shutil.rmtree(local)
    with Ledger(local, shared, 'd') as ledger:
        assert ledger.snapshot()['policy_state'] == before


def test_expired_reservation_remains_active_until_cancel(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        payload = {'gpu_ids': ['0'], 'grace_until': 101, 'expires_at': 102}
        assert ledger.control('reserve', 'r1', payload, 100)
        assert not ledger.control('reserve', 'r1', payload, 100)
        with pytest.raises(ValueError, match='active_reservation_exists'):
            ledger.control('reserve', 'r2', {'gpu_ids': ['1'],
                           'grace_until': 200, 'expires_at': 201}, 200)
        with pytest.raises(ValueError, match='reservation_release_ack_required'):
            ledger.control('cancel_reservation', 'release-r1', {'reservation_key': 'r1'}, 201)
        ledger.control('cancel_reservation', 'release-r1',
                       {'reservation_key': 'r1', 'resource_release_ack': True}, 201)
        assert ledger.control('reserve', 'r2', {'gpu_ids': ['1'],
                              'grace_until': 201, 'expires_at': 202}, 201)


def test_per_worker_hour_recovery_budget_and_pending_verification_cap(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        ledger.initialize([{'job_id': 'j', 'kind': 'extraction'}],
                          {'j': ['a', 'b', 'c', 'd']})
        epoch = ledger.new_epoch()
        ledger.register_worker('gpu', 'one', ['0'])
        for gid in ('a', 'b', 'c', 'd'):
            ledger.mark_ready('j', gid, 1)
        for gid in ('a', 'b'):
            aid = ledger.claim('j', gid, 'gpu', 'one', epoch, 2)['assignment_id']
            assert ledger.recover(3, {aid: {'owner_released': True, 'incarnation': 'one'}}) == [aid]
        aid = ledger.claim('j', 'c', 'gpu', 'one', epoch, 4)['assignment_id']
        assert ledger.recover(5, {aid: {'owner_released': True, 'incarnation': 'one'}}) == []
        assert ledger.recover(3604, {aid: {'owner_released': True, 'incarnation': 'one'}}) == [aid]
        first = ledger.claim('j', 'a', 'gpu', 'one', epoch, 3605)['assignment_id']
        assert ledger.acknowledge(first, {'sha': 'a'}, 'gpu', 'one', epoch)
        second = ledger.claim('j', 'b', 'gpu', 'one', epoch, 3606)['assignment_id']
        assert ledger.acknowledge(second, {'sha': 'b'}, 'gpu', 'one', epoch)
        assert ledger.claim('j', 'd', 'gpu', 'one', epoch, 3607) is None


def test_claim_reservation_guard_and_resume_drain(tmp_path):
    with Ledger(tmp_path / 'local', tmp_path / 'shared', 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        epoch = ledger.new_epoch()
        ledger.register_worker('gpu', 'one', ['0'])
        ledger.mark_ready('j', 'g', 1)
        ledger.control('reserve', 'r', {'gpu_ids': ['0'], 'expires_at': 100}, 1)
        assert ledger.claim('j', 'g', 'gpu', 'one', epoch, 2) is None
        ledger.control('cancel_reservation', 'release',
                       {'reservation_key': 'r', 'resource_release_ack': True}, 101)
        ledger.control('drain', 'drain', {'worker_id': 'gpu'}, 102)
        assert ledger.claim('j', 'g', 'gpu', 'one', epoch, 103) is None
        ledger.control('resume', 'resume', {'worker_id': 'gpu'}, 104)
        assert ledger.claim('j', 'g', 'gpu', 'one', epoch, 105)


def test_closed_raw_adoption_after_worker_exit(tmp_path):
    root = tmp_path / 'shared'
    with Ledger(tmp_path / 'local', root, 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        old = ledger.new_epoch()
        ledger.register_worker('gpu', 'one', ['0'])
        ledger.mark_ready('j', 'g', 1)
        assignment = ledger.claim('j', 'g', 'gpu', 'one', old, 2)
        write_once(root / 'dispatches' / (assignment['assignment_id'] + '.json'),
                   {**assignment, 'condition': 'frozen'})
        receipt = {**assignment, 'condition': 'frozen', 'status': 'returned',
                   'outcome': {'status': 'success', 'generation_returned': True}}
        write_once(root / 'worker_receipts' / 'gpu' / 'one' /
                   ('result-' + assignment['assignment_id'] + '.json'), receipt)
        new = ledger.new_epoch()
        ref = ledger.adopt_closed_raw(assignment['assignment_id'], new, 3)
        assert ref['path'].endswith(assignment['assignment_id'] + '.json')
        assert ledger.snapshot()['jobs'][0]['groups'][0]['status'] == 'pending_verification'
        assert ledger.commit(assignment['assignment_id'], receipt['outcome'], ref, 4)


def test_closed_raw_rejects_dispatch_identity_conflict(tmp_path):
    root = tmp_path / 'shared'
    with Ledger(tmp_path / 'local', root, 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        old = ledger.new_epoch()
        ledger.register_worker('gpu', 'one', ['0'])
        ledger.mark_ready('j', 'g', 1)
        assignment = ledger.claim('j', 'g', 'gpu', 'one', old, 2)
        write_once(root / 'dispatches' / (assignment['assignment_id'] + '.json'),
                   {**assignment, 'group_id': 'wrong'})
        write_once(root / 'worker_receipts' / 'gpu' / 'one' /
                   ('result-' + assignment['assignment_id'] + '.json'),
                   {**assignment, 'status': 'returned', 'outcome': {}})
        new = ledger.new_epoch()
        with pytest.raises(ValueError, match='closed_raw_dispatch_identity_mismatch'):
            ledger.adopt_closed_raw(assignment['assignment_id'], new)


def test_maintenance_control_deadlines_are_durable_and_idempotent(tmp_path):
    local, shared = tmp_path / 'local', tmp_path / 'shared'
    with Ledger(local, shared, 'd') as ledger:
        ledger.initialize([{'job_id': 'j'}], {'j': ['g']})
        assert ledger.control('pause', 'global-pause', now=100)
        assert not ledger.control('pause', 'global-pause', now=900)
        assert ledger.control('pause', 'target-pause',
                              {'worker_id': 'gpu0', 'deadline_s': 120}, now=110)
        assert ledger.control('drain', 'target-drain',
                              {'worker_id': 'gpu1', 'deadline_at': 500}, now=130)
        before = ledger.snapshot()['controls']
        assert {(m['key'], m['deadline']) for m in before['maintenance']} == {
            ('global-pause', 1000), ('target-pause', 230), ('target-drain', 500)}
        with pytest.raises(ValueError, match='maintenance_deadline_invalid'):
            ledger.control('pause', 'bad', {'deadline_s': True}, now=200)
    with Ledger(local, shared, 'd') as ledger:
        assert ledger.snapshot()['controls'] == before
        assert not ledger.control('pause', 'global-pause', now=2000)
        assert ledger.snapshot()['controls']['maintenance'] == before['maintenance']
        assert ledger.control('pause', 'target-pause-2',
                              {'worker_id': 'gpu0'}, now=220)
        assert next(m for m in ledger.snapshot()['controls']['maintenance']
                    if m['worker_id'] == 'gpu0')['deadline'] == 1120
        ledger.control('resume', 'resume-global', now=221)
        controls = ledger.snapshot()['controls']
        assert not controls['paused']
        assert controls['paused_workers'] == ['gpu0']
        assert controls['draining_workers'] == ['gpu1']
        assert {m['worker_id'] for m in controls['maintenance']} == {'gpu0', 'gpu1'}
        ledger.control('resume', 'resume-target', {'worker_id': 'gpu0'}, now=222)
        assert [m['key'] for m in ledger.snapshot()['controls']['maintenance']] == ['target-drain']
        ledger.control('resume', 'resume-drain', {'worker_id': 'gpu1'}, now=223)
        assert ledger.snapshot()['controls']['maintenance'] == []
