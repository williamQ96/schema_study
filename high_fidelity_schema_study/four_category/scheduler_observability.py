"""Passive stage and worker queue summaries for scheduler snapshots."""
from __future__ import annotations

from collections import defaultdict

STAGES = ('classification', 'local_extraction', 'dataset_parse', 'soft_reference')


def summarize_stages(rows):
    """Count scheduler row statuses independently for each execution stage."""
    counts = {stage: {} for stage in STAGES}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        kind = (row.get('job') or {}).get('kind')
        status = row.get('status')
        if kind in counts and isinstance(status, str):
            counts[kind][status] = counts[kind].get(status, 0) + 1
    return counts


def worker_queue_context(rows, workers, at):
    """Count profile and device compatible ready and dependency-waiting jobs."""
    result = {}
    for worker in workers or []:
        wid = worker['worker_id']
        profiles = worker.get('allowed_profile_ids')
        gpu = bool(worker.get('gpu_ids'))
        ready = waiting = 0
        for row in rows or []:
            if row.get('status') != 'pending':
                continue
            job = row.get('job') or {}
            if (job.get('kind') != 'dataset_parse') != gpu:
                continue
            if profiles and job.get('profile_id') not in profiles:
                continue
            if row.get('ready_at') is not None and row['ready_at'] <= at:
                ready += 1
            elif row.get('ready_at') is None:
                waiting += 1
        result[wid] = {'compatible_ready_jobs': ready, 'dependency_wait_jobs': waiting}
    return result
