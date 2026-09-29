from __future__ import annotations

from scheduler_v2.policy import choose


def job(jid, kind, paper, ready, order=0):
    return {'job_id': jid, 'status': 'pending',
            'job': {'kind': kind, 'paper_id': paper, 'profile_id': 'model'},
            'groups': [{'group_id': jid + '-g', 'ordinal': order,
                        'status': 'pending', 'ready_at': ready}]}


def test_urgent_and_ordinary_lane_order():
    snap = {'jobs': [job('c', 'classification', 'p', 0),
                     job('e', 'extraction', 'q', 0)], 'controls': {}}
    worker = {'gpu_ids': ['0'], 'allowed_profile_ids': ['model']}
    assert choose(snap, worker, 100, {})['lane'] == 'extraction'
    assert choose(snap, worker, 1900, {})['lane'] == 'classification'
    assert choose(snap, worker, 1900, {'last_lane': 'classification', 'lane_streak': 2})['lane'] == 'extraction'
    assert choose(snap, worker, 100, {'last_lane': 'extraction', 'lane_streak': 3})['lane'] == 'classification'


def test_extraction_round_robin_and_reservation():
    snap = {'jobs': [job('a1', 'extraction', 'a', 0), job('a2', 'extraction', 'a', 0),
                     job('b1', 'extraction', 'b', 0)], 'controls': {}}
    worker = {'gpu_ids': ['0']}
    assert choose(snap, worker, 1, {})['job_id'] == 'a1'
    assert choose(snap, worker, 1, {'last_extraction_paper': 'a'})['job_id'] == 'b1'
    snap['controls']['reservations'] = [{'gpu_ids': ['0'], 'grace_until': 2, 'expires_at': 100}]
    assert choose(snap, worker, 3, {}) is None


def test_thousand_matchsets_without_matrix_admission_cap():
    snap = {'jobs': [job(f'j{i}', 'extraction', f'p{i}', 0) for i in range(1000)],
            'controls': {}}
    assert choose(snap, {'gpu_ids': ['0']}, 1, {}) == {
        'job_id': 'j0', 'group_id': 'j0-g', 'lane': 'extraction'}


def test_compatible_downstream_idle_makes_classifier_urgent():
    blocked = job('blocked', 'extraction', 'p', None)
    snap = {'jobs': [job('c', 'classification', 'p', 100),
                     blocked, job('other', 'extraction', 'q', 0)],
            'workers': [{'gpu_ids': ['1'], 'allowed_profile_ids': ['model'], 'idle_since': 0}],
            'controls': {}}
    assert choose(snap, {'gpu_ids': ['0']}, 301, {})['lane'] == 'classification'
    snap['workers'][0]['allowed_profile_ids'] = ['unrelated']
    assert choose(snap, {'gpu_ids': ['0']}, 301, {})['lane'] == 'extraction'


def test_classification_lookahead_eight_and_started_paper_stickiness():
    candidates = [job(f'c{i}', 'classification', f'p{i}', i) for i in range(9)]
    candidates[2]['groups'].append({'group_id': 'already-done', 'ordinal': 1,
                                    'status': 'success', 'ready_at': 0})
    snap = {'jobs': candidates, 'controls': {}}
    result = choose(snap, {'gpu_ids': ['0']}, 20, {})
    assert result['job_id'] == 'c2'
    assert choose(snap, {'gpu_ids': ['0']}, 20,
                  {'current_classifier_paper': 'p5'})['job_id'] == 'c5'


def test_targeted_control_and_expired_reservation_block():
    snap = {'jobs': [job('a', 'extraction', 'p', 0)],
            'controls': {'paused_workers': ['gpu0'], 'draining_workers': [],
                         'reservations': []}}
    assert choose(snap, {'worker_id': 'gpu0', 'gpu_ids': ['0']}, 100, {}) is None
    assert choose(snap, {'worker_id': 'gpu1', 'gpu_ids': ['1']}, 100, {})
    snap['controls']['reservations'] = [{'gpu_ids': ['1'], 'expires_at': 50}]
    assert choose(snap, {'worker_id': 'gpu1', 'gpu_ids': ['1']}, 100, {}) is None


def test_one_group_per_job_and_eight_distinct_papers():
    jobs = [job(f'c{i}', 'classification', f'p{i}', 0) for i in range(10)]
    jobs[0]['groups'].append({'group_id': 'second', 'ordinal': 1,
                             'status': 'pending', 'ready_at': 0})
    jobs[1]['groups'][0]['status'] = 'assigned'
    snap = {'jobs': jobs, 'controls': {}}
    assert choose(snap, {'gpu_ids': ['0']}, 1, {})['job_id'] == 'c0'
    assert choose(snap, {'gpu_ids': ['0']}, 1,
                  {'current_classifier_paper': 'p9'})['job_id'] == 'c0'


def test_lane_quota_uses_this_worker_claims_not_other_workers():
    snap = {'jobs': [job('c', 'classification', 'p', 0),
                     job('e', 'extraction', 'q', 0)],
            'controls': {},
            'policy_state': {'last_lane': 'extraction', 'lane_streak': 100},
            'policy_state_by_worker': {
                'qwen': {'last_lane': 'classification', 'lane_streak': 1},
                'muse': {'last_lane': 'extraction', 'lane_streak': 100}}}
    assert choose(snap, {'worker_id': 'qwen', 'gpu_ids': ['0']}, 1900)['lane'] == 'classification'
    assert choose(snap, {'worker_id': 'muse', 'gpu_ids': ['1']}, 1900)['lane'] == 'classification'
