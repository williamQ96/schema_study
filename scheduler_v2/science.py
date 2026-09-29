"""One-group adapter around the unmodified, explicitly pinned research runtime.

No prompt, seed, model parameter, group plan or admission rule is defined here.
Import this module only after selecting the frozen scientific source on sys.path.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import hmac
import os
from pathlib import Path
import time

from .io import digest, file_sha, read, write_once

_PAPERS = OrderedDict()


def context(condition, job, sources, root):
    from high_fidelity_schema_study.four_category.scheduler_tasks import paper_context
    from high_fidelity_schema_study.four_category.workflow import tasks_for_config
    from high_fidelity_schema_study.four_category.paper import verify_index
    from high_fidelity_schema_study.four_category.complete_groups import _context
    from high_fidelity_schema_study.four_category.common import digest as scientific_digest
    paper = next(p for p in condition['corpus']['papers'] if p['paper_id'] == job['paper_id'])
    pins = {v['path']: file_sha(Path(sources) / v['path']) for v in paper['source']['artifacts'].values()}
    key = digest([str(sources), paper, pins])
    if key not in _PAPERS:
        _PAPERS[key] = paper_context(condition, job, Path(sources))['input']
        if len(_PAPERS) > 8:
            _PAPERS.popitem(last=False)
    _PAPERS.move_to_end(key)
    paper_input = _PAPERS[key]
    classifier = job['kind'] == 'classification'
    index = None
    if not classifier:
        index = read(Path(root) / 'indexes' / (scientific_digest(job['paper_id']) + '.json'))
        if verify_index(index, paper_input):
            raise ValueError('classification_index_invalid')
    task = tasks_for_config(condition['config'])['classification' if classifier else 'extraction']
    profile = next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id'])
    policy = condition['config']['classification']['grouping'] if classifier else condition['config']['extraction_grouping']
    module, groups = _context(task, paper_input, index, policy)
    return dict(module=module, groups=groups, paper=paper_input, index=index,
                task=task, profile=profile, policy=policy, source_pins=pins)


def plan_groups(condition, sources):
    """Group planning never needs dataset contents or an extraction index."""
    from high_fidelity_schema_study.four_category.scheduler_tasks import paper_context
    from high_fidelity_schema_study.four_category.workflow import tasks_for_config
    from high_fidelity_schema_study.four_category.complete_groups import _context
    tasks = tasks_for_config(condition['config'])
    plans, by_paper = {}, {}
    for job in condition['jobs']:
        if job['kind'] not in {'classification', 'local_extraction'}:
            plans[job['job_id']] = []
            continue
        if job.get('group_execution', {}).get('version') != 'all-groups/v1':
            raise ValueError('v2_requires_frozen_all_groups_jobs')
        key = (job['paper_id'], job['kind'])
        if key not in by_paper:
            paper = paper_context(condition, job, Path(sources))['input']
            classifier = job['kind'] == 'classification'
            task = tasks['classification' if classifier else 'extraction']
            policy = condition['config']['classification']['grouping'] if classifier else condition['config']['extraction_grouping']
            by_paper[key] = _context(task, paper, None, policy)[1]
        plans[job['job_id']] = [g['group_id'] for g in by_paper[key]]
    return plans


def _history(ctx, job, group, root):
    from high_fidelity_schema_study.four_category.complete_groups import _child_job, _load_group_history, _configuration
    child = _child_job(ctx['module'], job, group, ctx['index'])
    cap = _configuration(job)
    history = _load_group_history(ctx['module'], root, child, ctx['profile'], ctx['task'],
                                  ctx['paper'], ctx['index'], group, cap)
    return child, cap, history


def outcome(ctx, group, history, root):
    from high_fidelity_schema_study.four_category.complete_groups import _generation_returned
    number, path, record = history[-1]
    return dict(group_id=group['group_id'], status=record['status'],
                generation_returned=_generation_returned(record), selected_attempt=number,
                selected_record_sha256=record['record_sha256'],
                attempt_refs=[ctx['module'].record_ref(p, root) for _, p, _ in history])


def execute_group(condition, job, group_id, sources, root, session, progress):
    from high_fidelity_schema_study.four_category import backends
    from high_fidelity_schema_study.four_category.workflow import _attempt
    from high_fidelity_schema_study.four_category.grouped_classification import mock_group_transport as mc
    from high_fidelity_schema_study.four_category.grouped_extraction import mock_group_transport as me
    root = Path(root)
    ctx = context(condition, job, sources, root)
    group = next(g for g in ctx['groups'] if g['group_id'] == group_id)
    child, cap, history = _history(ctx, job, group, root)
    original = backends._transformers_transport
    classifier = job['kind'] == 'classification'
    try:
        if ctx['profile']['backend'] == 'transformers':
            backends._transformers_transport = session
        while not history or history[-1][2]['status'] == 'transport_error' and len(history) < cap:
            number = len(history) + 1
            progress('tokenizing', group_index=group['index'], group_total=len(ctx['groups']), group_attempt=number)
            record = _attempt(child, ctx['profile'], ctx['task'], ctx['paper'], ctx['index'], None, None,
                              number, allow_live=condition['live'],
                              transport=(mc if classifier else me) if ctx['profile']['backend'] == 'mock' else None,
                              counter=session.count if ctx['profile']['backend'] == 'transformers' else None,
                              classification_group=group if classifier else None,
                              extraction_group=None if classifier else group)
            path = ctx['module'].group_path(root, job['job_id'], group_id, number)
            write_once(path, record)
            history.append((number, path, record))
        return outcome(ctx, group, history, root)
    finally:
        backends._transformers_transport = original


def verify_group(condition, job, group_id, sources, root, execution_identity):
    root = Path(root)
    ctx = context(condition, job, sources, root)
    group = next(g for g in ctx['groups'] if g['group_id'] == group_id)
    # A receipt is only reusable when EVERY source and attempt still has its pinned bytes.
    folder = ctx['module'].group_path(root, job['job_id'], group_id, 1).parent
    files = sorted(folder.glob('attempt-*.json'))
    if set(folder.iterdir()) != set(files) or any(p.is_symlink() for p in files):
        raise ValueError('foreign_group_cache_entry')
    pins = {ctx['module'].record_ref(p, root)['path']: file_sha(p) for p in files}
    private_path = os.environ.get('MERCURY_VERIFICATION_KEY')
    private_key = Path(private_path).read_bytes() if private_path else None
    key_id = hashlib.sha256(private_key).hexdigest() if private_key else None
    key = digest(dict(condition=condition, job=job, group=group, index=ctx['index'],
                      sources=ctx['source_pins'], records=pins, verifier=execution_identity, cache_key_id=key_id))
    receipt_path = root / 'verification' / (key + '.json')
    if private_key and receipt_path.exists():
        receipt = read(receipt_path)
        signature = receipt.pop('cache_hmac', '')
        expected_signature = hmac.new(private_key, digest(receipt).encode(), hashlib.sha256).hexdigest()
        if (not hmac.compare_digest(signature, expected_signature) or receipt.get('key') != key
                or digest(receipt['outcome']) != receipt.get('outcome_sha256')):
            raise ValueError('verification_receipt_corrupt')
        return receipt['outcome']
    _, cap, history = _history(ctx, job, group, root)
    if not history or history[-1][2]['status'] == 'transport_error' and len(history) < cap:
        raise ValueError('group_not_terminal')
    value = outcome(ctx, group, history, root)
    receipt = dict(key=key, source_pins=ctx['source_pins'], record_pins=pins,
                   verifier=execution_identity, outcome=value, outcome_sha256=digest(value))
    if private_key:
        receipt['cache_hmac'] = hmac.new(private_key, digest(receipt).encode(), hashlib.sha256).hexdigest()
    write_once(receipt_path, receipt)
    return value


def finalize(condition, job, sources, root, execution_identity):
    from high_fidelity_schema_study.four_category.complete_groups import _body
    from high_fidelity_schema_study.four_category.execution_artifacts import build_artifact
    from high_fidelity_schema_study.four_category.common import digest as scientific_digest
    root = Path(root)
    ctx = context(condition, job, sources, root)
    outcomes, records = [], []
    for group in ctx['groups']:
        value = verify_group(condition, job, group['group_id'], sources, root, execution_identity)
        outcomes.append(value)
        records.append(read(root / value['attempt_refs'][-1]['path']))
    body = _body(ctx['groups'], scientific_digest(ctx['groups']), outcomes)
    artifact = build_artifact(job, ctx['profile'], ctx['task'], ctx['paper'], ctx['index'], ctx['policy'], records)
    relative = ('indexes/' + scientific_digest(job['paper_id']) + '.json' if job['kind'] == 'classification'
                else 'extractions/' + job['job_id'] + '.json')
    artifact_sha = write_once(root / relative, artifact)
    result = dict(version='scheduler-v2-parent-result/v1', condition=condition['condition'],
                  job_id=job['job_id'], body=body, artifact={'path': relative, 'file_bytes_sha256': artifact_sha},
                  execution_identity=execution_identity, semantic_accuracy=None)
    path = root / 'v2_results' / (job['job_id'] + '.json')
    result_sha = write_once(path, result)
    return dict(status=body['status'], result_ref={'path': path.relative_to(root).as_posix(),
                                                'file_bytes_sha256': result_sha},
                index_ref=result['artifact'] if job['kind'] == 'classification' else None)
