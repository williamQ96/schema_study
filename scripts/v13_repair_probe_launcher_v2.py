"""Resume only the output probes after an audited, pre-generation launcher fix."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import time

JOB = Path(os.environ['V13_JOB']).resolve()
sys.path[:0] = [str(JOB), str(JOB / 'runtime-v1')]
import v13_repair_launcher as base
from scheduler_v2.io import read, write_once, file_sha, digest, Lock
from scheduler_v2.processes import gpu_processes


def run_worker(worker, deployment, profile):
    root = JOB / 'probes-v2' / worker['worker_id']
    with ExitStack() as stack:
        for gpu in sorted(worker['gpu_ids']):
            for directory in (deployment['legacy_leases'], deployment['local_leases']):
                stack.enter_context(Lock(Path(directory) / (digest(gpu) + '.lock')))
        if any(gpu in worker['gpu_ids'] for gpu, _ in gpu_processes()):
            raise RuntimeError('allocated_gpu_not_free')
        command, env = base.container(profile, worker, deployment, suffix='probe-v2')
        command += ['/job/qualification-tools-v2/v13_repair_probe.py', '--job', '/job', '--profile-id', profile['profile_id']]
        code = base.run_owned(command, env, root)
        if any(gpu in worker['gpu_ids'] for gpu, _ in gpu_processes()):
            raise RuntimeError('gpu_release_unconfirmed')
        write_once(root / 'release.json', {'exit_code': code, 'time': time.time(), 'gpu_release_verified': True})
        return {'profile_id': profile['profile_id'], 'worker_id': worker['worker_id'], 'exit_code': code}


def main():
    with Lock(JOB / 'qualification-launcher.lock'):
        started = time.time()
        manifests = [('science-manifest.json', 'source/high_fidelity_schema_study'),
                     ('runtime-manifest.json', 'runtime-v1'), ('qualification-tools.json', ''),
                     ('qualification-tools-v2/manifest.json', 'qualification-tools-v2')]
        for name, prefix in manifests:
            for path, sha in read(JOB / name)['files'].items():
                if file_sha(JOB / prefix / path) != sha:
                    raise ValueError('source_changed:' + path)
        config = read(JOB / 'config.json')
        previous = read(JOB / 'audit/qualification-summary.json')
        assert previous['status'] == 'probe_failed' and previous['config_sha256'] == digest(config)
        assert all(row['exit_code'] == 0 for row in previous['cpu'])
        assert read(JOB / 'audit/capacity-summary.json')['status'] == 'pass'
        assert not (JOB / 'audit/v13-repair-probe').exists(), 'prior_probe_artifacts_exist_requires_separate_condition'
        assert all(row['exit_code'] == 1 for row in previous['gpu'])
        for row in previous['gpu']:
            log = (JOB / 'probes' / row['worker_id'] / 'process.log').read_text()
            assert 'job_plan_not_execution_ready' in log and 'frontier_reference-unselected-v1' in log
        pins = {p.relative_to(JOB).as_posix(): file_sha(p) for p in (JOB / 'audit').glob('cpu-preflight-*.json')}
        pins['audit/capacity-summary.json'] = file_sha(JOB / 'audit/capacity-summary.json')
        pins['audit/qualification-summary.json'] = file_sha(JOB / 'audit/qualification-summary.json')
        write_once(JOB / 'audit/probe-v2-resume-intent.json',
                   {'time': started, 'reason': 'deferred_frontier_readiness_check_prevented_all_prior_output_generations',
                    'config_sha256': digest(config), 'reused_preflight_files': pins,
                    'tool_manifest_file_sha256': file_sha(JOB / 'qualification-tools-v2/manifest.json')})
        deployment = read(JOB / 'previous-deployment.json')
        profiles = {p['profile_id']: p for p in config['profiles']}
        workers = [w for w in read(JOB / 'policy-pending.json')['workers'] if w['gpu_ids']]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_worker, w, deployment, profiles[w['allowed_profile_ids'][0]]) for w in workers]
            rows = [f.result() for f in futures]
        write_once(JOB / 'audit/qualification-summary-v2.json',
                   {'status': 'pass' if all(row['exit_code'] == 0 for row in rows) else 'probe_failed',
                    'config_sha256': digest(config), 'gpu': rows, 'cpu': previous['cpu'],
                    'started_at': started, 'finished_at': time.time(), 'semantic_accuracy': 'not_established',
                    'tool_manifest_file_sha256': file_sha(JOB / 'qualification-tools-v2/manifest.json'),
                    'reused_preflight_files': pins})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_once(JOB / 'audit/qualification-launcher-v2-failure.json',
                   {'status': 'failed', 'error': type(exc).__name__ + ':' + str(exc), 'time': time.time()})
        raise
