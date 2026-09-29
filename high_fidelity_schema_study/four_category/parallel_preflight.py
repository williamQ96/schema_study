"""Local-only 3-model x 3-replicate planning. No SSH, CUDA or inference calls.

Scheduling waves are dependency simulations, never wall-time or throughput claims.
The resulting plan is deliberately not accepted by the serial production runner.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from .common import digest, file_digest, read_json, seal, write_new

GIB = 1024 ** 3


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate(spec, corpus):
    if spec.get('schema_version') != 'mercury-parallel-preflight/v1':
        raise ValueError('unsupported_preflight_spec')
    models = spec.get('models', [])
    ids = [m.get('profile_id') for m in models]
    if len(ids) != 3 or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != 3:
        raise ValueError('three_distinct_models_required')
    reps = spec.get('replicates', [])
    if (len(reps) != 3 or any(not isinstance(r.get('replicate_id'), str) for r in reps)
            or len({r['replicate_id'] for r in reps}) != 3
            or any(not isinstance(r.get('seed'), int) or isinstance(r['seed'], bool) for r in reps)
            or len({r['seed'] for r in reps}) != 3):
        raise ValueError('three_distinct_replicates_and_seeds_required')
    if spec.get('classifier_profile_id') not in ids:
        raise ValueError('explicit_classifier_binding_required')
    for m in models:
        if not m.get('model_id') or not m.get('revision') or not positive(m.get('weight_file_bytes')) or not positive(m.get('context_window')):
            raise ValueError('model_identity_and_capacity_required')
        for field in ('kv_upper_bound_bytes_per_token', 'runtime_overhead_bytes'):
            if m.get(field) is not None and not positive(m[field]):
                raise ValueError('invalid_memory_assumption:' + field)
        if m.get('kv_upper_bound_bytes_per_token') is not None and not m.get('memory_assumption_basis'):
            raise ValueError('memory_assumption_basis_required')
    for role in ('classification', 'extraction'):
        if not positive(spec.get('parameters', {}).get(role, {}).get('max_output_tokens')):
            raise ValueError('output_allowance_required:' + role)
    for key, ident in [('papers', 'paper_id'), ('datasets', 'dataset_id'), ('matches', 'match_id')]:
        rows = corpus.get(key, [])
        values = [r.get(ident) for r in rows]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError('duplicate_or_invalid_' + key)
    if not corpus['papers']:
        raise ValueError('papers_required')
    papers = {p['paper_id'] for p in corpus['papers']}
    datasets = {d['dataset_id'] for d in corpus['datasets']}
    for match in corpus['matches']:
        if match['paper_id'] not in papers or match['dataset_id'] not in datasets:
            raise ValueError('unresolved_match')
    host = spec['host_budget']
    for field in ('gpu_count', 'gpu_memory_bytes_each', 'cpu_threads', 'ram_bytes',
                  'gpu_reserve_bytes_each', 'cpu_threads_per_gpu_worker',
                  'ram_bytes_per_gpu_worker', 'dataset_workers',
                  'cpu_threads_per_dataset_worker', 'ram_bytes_per_dataset_worker'):
        if not positive(host.get(field)):
            raise ValueError('invalid_host_budget:' + field)
    for field in ('gpu_count', 'cpu_threads', 'cpu_threads_per_gpu_worker', 'dataset_workers', 'cpu_threads_per_dataset_worker'):
        if not isinstance(host[field], int):
            raise ValueError('integer_budget_required:' + field)


def make_jobs(spec, corpus):
    """Keep dataset content out of paper jobs; matches affect packet membership only."""
    validate(spec, corpus)
    profiles = {m['profile_id']: m for m in spec['models']}
    jobs, matrices = [], []
    for paper in corpus['papers']:
        pid = paper['paper_id']
        def add(kind, profile, rep, dependencies):
            params = copy.deepcopy(spec['parameters'][kind])
            if rep:
                params['seed'] = rep['seed']
            body = {'kind': kind, 'paper_id': pid, 'paper_source_sha256': paper['source']['source_sha256'],
                    'profile_id': profile, 'profile_spec_sha256': digest(profiles[profile]),
                    'task_binding': spec['task_binding'], 'parameters': params,
                    'replicate_id': rep['replicate_id'] if rep else None,
                    'dependencies': dependencies, 'index_policy': 'one_verified_frozen_index_per_paper',
                    'request_identity': 'pending_exact_render_and_tokenization'}
            body['job_id'] = digest(body)
            body['output_key'] = 'jobs/' + body['job_id']
            jobs.append(body)
            return body['job_id']
        cid = add('classification', spec['classifier_profile_id'], None, [])
        cells = []
        for m in spec['models']:
            for rep in spec['replicates']:
                jid = add('extraction', m['profile_id'], rep, [cid])
                cells.append({'model': m['profile_id'], 'replicate': rep['replicate_id'], 'seed': rep['seed'], 'job_id': jid})
        matrices.append({'paper_id': pid, 'rows': [m['profile_id'] for m in spec['models']],
                         'columns': [r['replicate_id'] for r in spec['replicates']], 'cells': cells,
                         'classification_job_id': cid,
                         'dataset_ids': sorted({m['dataset_id'] for m in corpus['matches'] if m['paper_id'] == pid}),
                         'soft_reference': 'deferred'})
    datasets = []
    for row in corpus['datasets']:
        binding = {'dataset': row, 'parser_binding': spec['dataset_parser_binding']}
        datasets.append({'job_id': digest(binding), 'kind': 'dataset_parse', 'dataset_id': row['dataset_id'],
                         'gpu_count': 0, 'output_key': 'datasets/' + digest(binding),
                         'dependencies': [], 'binding': binding})
    return jobs, matrices, datasets


def allocations(host, width):
    count = host['gpu_count']
    if not isinstance(width, int) or width < 1 or width > count or count % width:
        raise ValueError('gpu_width_must_partition_all_devices')
    result = [{'worker_id': 'worker-' + str(i // width), 'gpu_ids': list(range(i, i + width))}
              for i in range(0, count, width)]
    n = len(result)
    if n * host['cpu_threads_per_gpu_worker'] + host['dataset_workers'] * host['cpu_threads_per_dataset_worker'] > host['cpu_threads']:
        raise ValueError('cpu_oversubscription')
    if n * host['ram_bytes_per_gpu_worker'] + host['dataset_workers'] * host['ram_bytes_per_dataset_worker'] > host['ram_bytes']:
        raise ValueError('ram_oversubscription')
    return result


def simulate(jobs, workers, *, failed_ids=(), completed_ids=()):
    """One request per process; replicas never share a process-global RNG."""
    ids = {j['job_id'] for j in jobs}
    worker_ids = [w['worker_id'] for w in workers]
    devices = [d for w in workers for d in w['gpu_ids']]
    if (not workers or len(set(worker_ids)) != len(worker_ids) or any(not w['gpu_ids'] for w in workers)
            or len(set(devices)) != len(devices)):
        raise ValueError('empty_or_overlapping_worker_allocation')
    if len(ids) != len(jobs) or any(set(j['dependencies']) - ids for j in jobs):
        raise ValueError('invalid_dependency_graph')
    done, failed = set(completed_ids), set(failed_ids)
    if (done | failed) - ids or done & failed:
        raise ValueError('invalid_resume_or_failure_set')
    for j in jobs:
        if j['job_id'] in done and not set(j['dependencies']) <= done:
            raise ValueError('completed_job_missing_completed_dependency')
    pending = {j['job_id']: j for j in jobs if j['job_id'] not in done | failed}
    blocked, waves, affinity = set(), [], {}
    while pending:
        newly_blocked = {jid for jid, j in pending.items() if set(j['dependencies']) & (failed | blocked)}
        if newly_blocked:
            blocked.update(newly_blocked)
            for jid in newly_blocked:
                pending.pop(jid)
            continue
        ready = [j for j in pending.values() if set(j['dependencies']) <= done]
        if not ready:
            raise ValueError('dependency_cycle')
        wave = []
        for worker in workers:
            if not ready:
                break
            wid = worker['worker_id']
            ready.sort(key=lambda j: (j['profile_id'] != affinity.get(wid), j['paper_id'], j['kind'], j['replicate_id'] or ''))
            job = ready.pop(0)
            wave.append({'worker_id': wid, 'gpu_ids': worker['gpu_ids'], 'job_id': job['job_id'],
                         'paper_id': job['paper_id'], 'kind': job['kind'], 'profile_id': job['profile_id']})
            affinity[wid] = job['profile_id']
            pending.pop(job['job_id'])
        waves.append(wave)
        done.update(j['job_id'] for j in wave)
    return {'waves': waves, 'scheduled': sum(map(len, waves)), 'blocked_dependency': sorted(blocked),
            'failed': sorted(failed), 'resumed': sorted(completed_ids),
            'timing_basis': 'unit-duration dependency simulation; no throughput or speedup prediction'}


def memory_screen(model, width, host):
    # Disk bytes and ideal balancing are planning proxies, not measured residency.
    usable = host['gpu_memory_bytes_each'] - host['gpu_reserve_bytes_each']
    kv = model.get('kv_upper_bound_bytes_per_token')
    overhead = model.get('runtime_overhead_bytes')
    estimate = None if kv is None or overhead is None else (
        model['weight_file_bytes'] + kv * model['context_window'] + overhead) / width
    return {'profile_id': model['profile_id'], 'gpu_count': width,
            'checkpoint_disk_gib': round(model['weight_file_bytes'] / GIB, 3),
            'ideal_balanced_weight_proxy_gib_per_gpu': round(model['weight_file_bytes'] / width / GIB, 3),
            'usable_budget_gib_per_gpu': round(usable / GIB, 3),
            'ideal_balanced_estimate_gib_per_gpu': None if estimate is None else round(estimate / GIB, 3),
            'screen': 'unknown_kv_or_runtime_peak' if estimate is None else
                      'exceeds_even_ideal_balance' if estimate > usable else 'needs_measured_per_device_peak',
            'execution_ready': False,
            'limitations': ['Checkpoint disk bytes are not actual GPU residency.',
                           'device_map does not guarantee balanced weight or KV placement.',
                           'Prefill activations, cache architecture, workspace, fragmentation and concurrent I/O require measurement.']}


def build_report(spec, corpus):
    jobs, matrices, datasets = make_jobs(spec, corpus)
    candidates = []
    failed = {m['classification_job_id'] for m in matrices if m['paper_id'] in spec.get('historical_context_blocked_papers', [])}
    for width in spec['candidate_gpu_widths']:
        workers = allocations(spec['host_budget'], width)
        candidates.append({'id': '{}x{}gpu'.format(len(workers), width), 'workers': workers,
                           'memory': [memory_screen(m, width, spec['host_budget']) for m in spec['models']],
                           'hypothetical_all_requests_qualified': simulate(jobs, workers),
                           'historical_context_failure_scenario': simulate(jobs, workers, failed_ids=failed)})
    return seal({'schema_version': 'mercury-parallel-preflight-report/v1', 'mode': 'local_offline_only',
                 'spec_sha256': digest(spec), 'corpus_sha256': digest(corpus),
                 'planner_file_bytes_sha256': file_digest(Path(__file__)),
                 'static_plan_status': 'pass', 'execution_ready': False, 'model_calls': 0,
                 'counts': {'papers': len(matrices), 'matches': len(corpus['matches']),
                            'dataset_parse_jobs': len(datasets), 'classification_jobs': len(matrices),
                            'extraction_cells': len(matrices) * 9, 'deferred_references': len(matrices)},
                 'matrices': matrices, 'paper_jobs': jobs, 'dataset_jobs': datasets, 'candidates': candidates,
                 'required_runtime_gates': ['new frozen lane-specific profiles and source/SIF identity',
                     'scheduler/allocation ownership and fresh GPU UUID/free-memory observations',
                     'exact rendered request plus output context check per model; no truncation',
                     'single frozen verified classification index reused by all nine cells',
                     'measured per-device loading, longest prefill and decode peaks with margin',
                     'concurrent canary throughput, OOM, CPU RAM, NFS/local-NVMe I/O qualification',
                     'job-isolated writes, atomic verified resume and final matrix/packet aggregation'],
                 'production_dispatch': 'not_implemented_in_this_preflight; do not start multiple serial run_batch instances'},
                'report_sha256')


def verify_report(report, spec, corpus):
    return [] if report == build_report(spec, corpus) else ['preflight_derivation_mismatch']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = build_report(read_json(args.spec), read_json(args.corpus))
    write_new(args.output, report)
    print(json.dumps({'static_plan_status': report['static_plan_status'], 'execution_ready': False, 'counts': report['counts']}))


if __name__ == '__main__':
    main()
