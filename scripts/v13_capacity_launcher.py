"""Own both GPU lease namespaces while testing frozen V13 capacity."""
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

JOB = Path(os.environ.get('V13_JOB', '/storage/users/williamq/schema-study-deployment-20260926/jobs/v13-full-20260928-v1'))
sys.path.insert(0, str(JOB / 'runtime-v1'))
from scheduler_v2.io import Lock, digest, read, write_once, atomic_json
from scheduler_v2.processes import identity

def probe(worker, deployment):
    wid = worker['worker_id']
    base = JOB / 'capacity' / wid
    base.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(worker['gpu_ids']):
            for root in (deployment['legacy_leases'], deployment['local_leases']):
                stack.enter_context(Lock(Path(root) / (digest(gpu) + '.lock')))
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if any(gpu in active for gpu in worker['gpu_ids']):
            raise RuntimeError('allocated_gpu_has_unowned_process:' + wid)
        profile_id = worker['allowed_profile_ids'][0]
        scratch = Path('/var/tmp/schema-study-williamq/v13-capacity') / wid
        scratch.mkdir(parents=True, exist_ok=True)
        env = {k:v for k,v in os.environ.items() if not k.startswith(('APPTAINERENV_', 'SINGULARITYENV_'))}
        env.update(APPTAINERENV_PYTHONPATH='/scientific', APPTAINERENV_CUDA_VISIBLE_DEVICES=','.join(worker['gpu_ids']),
                   APPTAINERENV_OMP_NUM_THREADS=str(worker['cpu_threads']), APPTAINERENV_HF_HUB_OFFLINE='1',
                   APPTAINERENV_TRANSFORMERS_OFFLINE='1', APPTAINERENV_HF_HOME='/scratch/hf',
                   APPTAINERENV_XDG_CACHE_HOME='/scratch/cache', APPTAINERENV_TMPDIR='/scratch')
        command=[deployment['apptainer'],'exec','--nv','--cleanenv','--containall',
                 '--bind',str(JOB / 'source')+':/scientific:ro', '--bind',str(JOB)+':/job:rw',
                 '--bind',str(scratch)+':/scratch:rw']
        for profile in read(JOB/'config.json')['profiles']:
            if profile['profile_id']==profile_id:
                command+=['--bind',profile['model_id']+':'+profile['model_id']+':ro']
        command += [deployment['image'],'/opt/phase1-venv/bin/python','-u','-m',
                    'high_fidelity_schema_study.four_category.resident_qualification',
                    '--profile','/job/profiles/'+profile_id+'.json','--output','/job/capacity/'+wid+'/measurement']
        write_once(base/'intent.json',dict(command=command,worker=worker,time=time.time()))
        with (base/'process.log').open('ab') as log:
            p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            write_once(base/'process.json',identity(p.pid))
            try:
                code=p.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                # Only the process group this launcher created, while retaining its leases.
                import signal
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=30)
                code=-1
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if any(gpu in active for gpu in worker['gpu_ids']):
            raise RuntimeError('probe_gpu_release_unconfirmed:' + wid)
        row=read(base/'measurement/result.json') if (base/'measurement/result.json').exists() else {'status':'failed','error':'no_result'}
        write_once(base/'release.json',dict(exit_code=code,gpu_release_verified=True,time=time.time()))
        return wid,row

def main():
    deployment=read(JOB/'previous-deployment.json')
    workers=[w for w in read(JOB/'policy-pending.json')['workers'] if w['gpu_ids']]
    with Lock(JOB/'capacity-launcher.lock'), ThreadPoolExecutor(max_workers=3) as pool:
        rows=dict(pool.map(lambda w:probe(w,deployment),workers))
    atomic_json(JOB/'audit/capacity-summary.json',dict(time=time.time(),results=rows,
                 status='pass' if all(r['status']=='pass' for r in rows.values()) else 'failed'))

if __name__=='__main__':main()
