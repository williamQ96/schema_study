"""Pure, bounded group dispatch policy for scheduler V2."""
from __future__ import annotations


def _compatible(job, worker):
    body = job['job']
    if not worker.get('gpu_ids'):
        return body.get('kind') == 'dataset_parse'
    if body.get('kind') == 'dataset_parse':
        return False
    allowed = worker.get('allowed_profile_ids')
    return not allowed or body.get('profile_id') in allowed


def _available_worker(job, worker, snapshot):
    controls = snapshot.get('controls', {})
    worker_id = worker.get('worker_id')
    if (controls.get('paused') or controls.get('draining') or
            worker_id in controls.get('paused_workers', ()) or
            worker_id in controls.get('draining_workers', ())):
        return False
    gpu_ids = set(worker.get('gpu_ids', []))
    if any(gpu_ids.intersection(r.get('gpu_ids', []))
           for r in controls.get('reservations', [])):
        return False
    return _compatible(job, worker)


def choose(snapshot, worker, now, state=None):
    """Return {job_id, group_id, lane} or None; never mutate input.

    Caller records each dispatch in ``state`` (lane streak, last paper, current
    classifier paper). Jobs and groups are already in original plan order.
    """
    if state is None:
        by_worker = snapshot.get('policy_state_by_worker')
        state = (by_worker.get(worker['worker_id'], {}) if by_worker is not None and worker.get('worker_id')
                 else snapshot.get('policy_state', {}))
    controls = snapshot.get('controls', {})
    worker_id = worker.get('worker_id')
    if (controls.get('paused') or controls.get('draining') or
            worker_id in controls.get('paused_workers', ()) or
            worker_id in controls.get('draining_workers', ())):
        return None
    reservations = controls.get('reservations', [])
    gpu_ids = set(worker.get('gpu_ids', []))
    if any(gpu_ids.intersection(r.get('gpu_ids', [])) for r in reservations):
        return None
    candidates = []
    for job_order, job in enumerate(snapshot.get('jobs', [])):
        if job.get('status') != 'pending' or not _compatible(job, worker):
            continue
        body = job['job']
        kind = body.get('kind')
        if kind == 'classification':
            lane = 'classification'
        elif kind in ('extraction', 'local_extraction'):
            lane = 'extraction'
        elif kind == 'dataset_parse':
            lane = 'dataset'
        else:
            continue
        statuses = [g['status'] for g in job.get('groups', [])]
        if 'assigned' in statuses or statuses.count('pending_verification') >= 2:
            continue
        for group in job.get('groups', []):
            if group['status'] == 'pending' and group.get('ready_at') is not None:
                candidates.append({'job_id': job['job_id'], 'group_id': group['group_id'],
                                   'lane': lane, 'paper_id': body.get('paper_id'),
                                   'replicate_id': body.get('replicate_id'),
                                   'ready_at': group['ready_at'], 'job_order': job_order,
                                   'group_order': group['ordinal']})
                break
    if not candidates:
        return None
    classification = []
    seen_papers = set()
    for candidate in candidates:
        if candidate['lane'] == 'classification' and candidate['paper_id'] not in seen_papers:
            classification.append(candidate)
            seen_papers.add(candidate['paper_id'])
            if len(classification) == 8:
                break
    extraction = [x for x in candidates if x['lane'] == 'extraction']
    dataset = [x for x in candidates if x['lane'] == 'dataset']
    if dataset and not gpu_ids:
        selected = min(dataset, key=lambda x: (x['job_order'], x['group_order']))
    elif classification and not extraction:
        selected = _classification(classification, snapshot, state, now)
    elif extraction and not classification:
        selected = _extraction(extraction, state)
    elif classification and extraction:
        urgent = any(now - x['ready_at'] >= 1800 or _downstream_idle(x, snapshot, now)
                     for x in classification)
        streak_lane = state.get('last_lane')
        streak = state.get('lane_streak', 0)
        # Urgent cycle: C,C,E. Ordinary cycle: E,E,E,C.
        class_limit, extract_limit = (2, 1) if urgent else (1, 3)
        if streak_lane == 'classification':
            lane = 'extraction' if streak >= class_limit else 'classification'
        elif streak_lane == 'extraction':
            lane = 'classification' if streak >= extract_limit else 'extraction'
        else:
            lane = 'classification' if urgent else 'extraction'
        selected = (_classification(classification, snapshot, state, now) if lane == 'classification'
                    else _extraction(extraction, state))
    else:
        return None
    return {key: selected[key] for key in ('job_id', 'group_id', 'lane')}


def _downstream_idle(candidate, snapshot, now):
    """A classification blocks compatible extraction for its paper for 300 s."""
    paper = candidate['paper_id']
    if paper is None:
        return False
    for job in snapshot.get('jobs', []):
        body = job['job']
        if body.get('paper_id') != paper or body.get('kind') not in ('extraction', 'local_extraction'):
            continue
        if not any(g['status'] == 'pending' and g.get('ready_at') is None for g in job.get('groups', [])):
            continue
        workers = snapshot.get('workers')
        if workers is None:
            return now - candidate['ready_at'] >= 300
        for worker in workers:
            idle = worker.get('idle_since')
            if idle is not None and now - idle >= 300 and _available_worker(job, worker, snapshot):
                return True
    return False


def _classification(candidates, snapshot, state, now):
    started = set()
    for job in snapshot.get('jobs', []):
        if job['job'].get('kind') == 'classification' and any(
                g['status'] != 'pending' for g in job.get('groups', [])):
            started.add(job['job'].get('paper_id'))
    current = state.get('current_classifier_paper')
    if current is not None and any(c['paper_id'] == current for c in candidates):
        candidates = [c for c in candidates if c['paper_id'] == current]
    def key(c):
        unblock = _idle_unblock_count(c['paper_id'], snapshot, now)
        return (c['paper_id'] not in started, -unblock, c['ready_at'],
                c['job_order'], c['group_order'])
    return min(candidates, key=key)


def _idle_unblock_count(paper, snapshot, now):
    workers = snapshot.get('workers', [])
    blocked = [job for job in snapshot.get('jobs', [])
               if job['job'].get('paper_id') == paper
               and job['job'].get('kind') in ('extraction', 'local_extraction')
               and job['status'] == 'pending'
               and any(g['status'] == 'pending' and g.get('ready_at') is None
                       for g in job.get('groups', []))]
    return sum(1 for worker in workers if worker.get('idle_since') is not None
               and worker['idle_since'] <= now
               and any(_available_worker(job, worker, snapshot) for job in blocked))


def _extraction(candidates, state):
    paper_order = list(dict.fromkeys(c['paper_id'] for c in candidates))
    last = state.get('last_extraction_paper')
    if last in paper_order and len(paper_order) > 1:
        start = (paper_order.index(last) + 1) % len(paper_order)
    else:
        start = 0
    paper = paper_order[start]
    subset = [c for c in candidates if c['paper_id'] == paper]
    return min(subset, key=lambda c: (c['job_order'], c['group_order']))
