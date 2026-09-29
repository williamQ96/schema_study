"""Bounded, lease-owning V13 repair qualification; no production dispatch."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time

JOB = Path(os.environ['V13_JOB']).resolve()
sys.path.insert(0, str(JOB / 'runtime-v1'))
from scheduler_v2.io import Lock, digest, read, write_once, atomic_json, file_sha
from scheduler_v2.processes import identity, same, gpu_processes, resources_free


def run_owned(command, env, base, timeout=10800):
    base.mkdir(parents=True, exist_ok=True)
    write_once(base / 'launch-intent.json', {'command': command, 'time': time.time()})
    with (base / 'process.log').open('xb') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
        expected = identity(process.pid)
        write_once(base / 'process.json', expected)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            write_once(base / 'cancel-intent.json', {'reason': 'qualification_deadline', 'identity': expected})
            if same(expected):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if same(expected):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            code = -1
    return code


def container(profile, worker, deployment, *, cpu=False, suffix='probe'):
    scratch = Path('/var/tmp/schema-study-williamq/v13-repair-qualification') / JOB.name / suffix / profile['profile_id']
    scratch.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('APPTAINERENV_', 'SINGULARITYENV_'))}
    env.update(APPTAINERENV_PYTHONPATH='/scientific',
               APPTAINERENV_CUDA_VISIBLE_DEVICES='' if cpu else ','.join(worker['gpu_ids']),
               APPTAINERENV_OMP_NUM_THREADS='2' if cpu else str(worker['cpu_threads']),
               APPTAINERENV_RAYON_NUM_THREADS='2', APPTAINERENV_HF_HUB_OFFLINE='1',
               APPTAINERENV_TRANSFORMERS_OFFLINE='1', APPTAINERENV_HF_HOME='/scratch/hf',
               APPTAINERENV_XDG_CACHE_HOME='/scratch/cache', APPTAINERENV_TMPDIR='/scratch')
    cmd = [deployment['apptainer'], 'exec'] + ([] if cpu else ['--nv'])
    cmd += ['--cleanenv', '--containall', '--bind', str(JOB / 'source') + ':/scientific:ro',
            '--bind', str(JOB) + ':/job:rw', '--bind', str(scratch) + ':/scratch:rw',
            '--bind', profile['model_id'] + ':' + profile['model_id'] + ':ro',
            deployment['image'], '/opt/phase1-venv/bin/python', '-u']
    return cmd, env


def gpu_qualify(worker, deployment, profile):
    base = JOB / 'probes' / worker['worker_id']
    with ExitStack() as stack:
        for gpu in sorted(worker['gpu_ids']):
            for root in (deployment['legacy_leases'], deployment['local_leases']):
                stack.enter_context(Lock(Path(root) / (digest(gpu) + '.lock')))
        if any(gpu in worker['gpu_ids'] for gpu, _ in gpu_processes()):
            raise RuntimeError('unowned_gpu_process:' + worker['worker_id'])
        cmd, env = container(profile, worker, deployment)
        cmd += ['/job/v13_repair_probe.py', '--job', '/job', '--profile-id', profile['profile_id']]
        code = run_owned(cmd, env, base)
        if any(gpu in worker['gpu_ids'] for gpu, _ in gpu_processes()):
            raise RuntimeError('gpu_release_unconfirmed:' + worker['worker_id'])
        write_once(base / 'release.json', {'exit_code': code, 'time': time.time(), 'gpu_release_verified': True})
        return {'worker_id': worker['worker_id'], 'profile_id': profile['profile_id'], 'exit_code': code}


def cpu_qualify(worker, deployment, profile, shard, count):
    suffix = '' if count == 1 else '-shard-' + str(shard)
    cmd, env = container(profile, worker, deployment, cpu=True, suffix='cpu' + suffix)
    cmd += ['/job/v13_cpu_preflight.py', '--job', '/job', '--profile-id', profile['profile_id'],
            '--shard-index', str(shard), '--shard-count', str(count)]
    code = run_owned(cmd, env, JOB / 'cpu-processes' / (profile['profile_id'] + suffix))
    return {'profile_id': profile['profile_id'], 'shard_index': shard, 'exit_code': code}


def main():
    deployment = read(JOB / 'previous-deployment.json')
    config = read(JOB / 'config.json')
    workers = [w for w in read(JOB / 'policy-pending.json')['workers'] if w['gpu_ids']]
    profiles = {p['profile_id']: p for p in config['profiles']}
    for manifest, prefix in [('science-manifest.json', 'source/high_fidelity_schema_study'),
                             ('runtime-manifest.json', 'runtime-v1'), ('qualification-tools.json', '')]:
        for path, expected in read(JOB / manifest)['files'].items():
            if file_sha(JOB / prefix / path) != expected:
                raise ValueError('qualification_source_changed:' + path)
    started = time.time()
    with Lock(JOB / 'qualification-launcher.lock'), ThreadPoolExecutor(max_workers=9) as pool:
        cpu = []
        for worker in workers:
            profile = profiles[worker['allowed_profile_ids'][0]]
            count = 4 if profile['profile_id'] == config['classification']['profile_id'] else 1
            cpu.extend(pool.submit(cpu_qualify, worker, deployment, profile, s, count) for s in range(count))
        # A previous fixed qualification may still own these GPUs. Wait without
        # preemption; acquisition inside the capacity launcher remains decisive.
        wait_started = time.time()
        while not all(resources_free(w, deployment['local_leases'], deployment['legacy_leases']) for w in workers):
            atomic_json(JOB / 'audit/resource-wait.json',
                        {'state': 'waiting_for_lease_return', 'started_at': wait_started,
                         'time': time.time(), 'deadline': wait_started + 10800})
            if time.time() - wait_started > 10800:
                raise TimeoutError('qualification_resource_wait_deadline_no_preemption')
            time.sleep(20)
        atomic_json(JOB / 'audit/resource-wait.json',
                    {'state': 'resources_available', 'started_at': wait_started, 'time': time.time()})
        # Synthetic capacity is independently recorded using the established launch path.
        capacity = subprocess.run([sys.executable, str(JOB / 'v13_capacity_launcher.py')],
                                  env=dict(os.environ, V13_JOB=str(JOB)), capture_output=True, text=True, timeout=2100)
        write_once(JOB / 'audit/capacity-launcher-result.json',
                   {'exit_code': capacity.returncode, 'stdout': capacity.stdout[-4000:], 'stderr': capacity.stderr[-4000:]})
        cpu_rows = [f.result() for f in cpu]
        if capacity.returncode or any(r['exit_code'] for r in cpu_rows):
            result = {'status': 'blocked_before_output_probes', 'cpu': cpu_rows, 'capacity_exit_code': capacity.returncode}
        elif read(JOB / 'audit/capacity-summary.json')['status'] != 'pass':
            result = {'status': 'capacity_failed', 'cpu': cpu_rows}
        else:
            futures = [pool.submit(gpu_qualify, w, deployment, profiles[w['allowed_profile_ids'][0]]) for w in workers]
            gpu_rows = [f.result() for f in futures]
            result = {'status': 'pass' if all(r['exit_code'] == 0 for r in gpu_rows) else 'probe_failed',
                      'cpu': cpu_rows, 'gpu': gpu_rows}
    write_once(JOB / 'audit/qualification-summary.json',
               {**result, 'started_at': started, 'finished_at': time.time(),
                'config_sha256': digest(config), 'semantic_accuracy': 'not_established'})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_once(JOB / 'audit/qualification-launcher-failure.json',
                   {'status': 'failed', 'error': type(exc).__name__ + ':' + str(exc), 'time': time.time()})
        raise
