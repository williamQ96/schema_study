"""Local deployment helper. Credentials are imported only from the private SSH helper."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import sys
import time

ARTIFACTS = Path(__file__).resolve().parents[1] / 'data/experiments/mercury_scheduler_v2_2026_09_28_v1'
BASE = '/storage/users/williamq/schema-study-deployment-20260926/jobs'


def connect():
    private = Path.home() / '.oaciss-secrets'
    sys.path[:0] = [str(private / 'python-deps'), str(private)]
    from oaciss_access import mercury
    return mercury()


def execute(code, timeout=60):
    gateway, client = connect()
    try:
        _, out, err = client.exec_command('python3 -c ' + shlex.quote(code), timeout=timeout)
        value, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error[-6000:])
        return json.loads(value)
    finally:
        client.close()
        gateway.close()


def baseline():
    code = r'''
import json,hashlib,os,time,importlib.util
from pathlib import Path
base=Path('/storage/users/williamq/schema-study-deployment-20260926/jobs')
old=base/'spatial-inference-20260927-v3/run'
state=base/'gpu-priority-handoff-20260928-v1/state'
spec=importlib.util.spec_from_file_location('handoff',base/'gpu-priority-handoff-20260928-v1/runtime-v2/gpu_priority_handoff.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plan=m.load_plan(state)
m.verify_pins(plan)
pins=m.preserve(old)
workers={}
for p in (old/'workers').iterdir():
 if (p/'heartbeat.json').exists():
  hb=m.read(p/'heartbeat.json'); workers[p.name]={'heartbeat':hb,'process':m.proc(hb['pid'])}
snapshot=m.read(old/'snapshot.json')
value=dict(created_at=time.time(),original_root=str(old),condition=plan['condition'],
           scientific_pins=plan['pins'],artifact_pins=pins,workers=workers,
           controller=m.read(state/'controller-process.json'),supervisor=m.read(state/'supervisor-ready.json'),
           snapshot=snapshot,authority='repair_time_baseline_not_historical_proof')
target=base/'scheduler-v2-20260928-v1/audit';target.mkdir(parents=True,exist_ok=True)
p=target/('baseline-'+str(time.time_ns())+'.json');m.atomic(p,value)
value['remote_baseline']=str(p)
print(json.dumps(value))
'''
    value = execute(code, 180)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / ('baseline-' + str(time.time_ns()) + '.json')
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(dict(local=str(path), remote=value['remote_baseline'],
                         scientific_files=len(value['scientific_pins']),
                         artifacts=len(value['artifact_pins']),
                         stage_counts=value['snapshot']['stage_counts'])))


def stage_preparer(revision=1):
    repo = Path(__file__).resolve().parents[1]
    members = ['scheduler_v2/' + n + '.py' for n in ('__init__', 'io', 'science', 'migrate')]
    runtime = BASE + '/scheduler-v2-20260928-v1/preparer-v' + str(revision)
    gateway, client = connect()
    try:
        sftp = client.open_sftp()
        def mkdir(path):
            try:
                sftp.stat(path)
            except OSError:
                parent = path.rsplit('/', 1)[0]
                if parent: mkdir(parent)
                sftp.mkdir(path)
        for member in members:
            target = runtime + '/' + member
            mkdir(target.rsplit('/', 1)[0])
            payload = (repo / member).read_bytes()
            try:
                with sftp.open(target, 'rb') as stream:
                    if stream.read() != payload:
                        raise ValueError('staged_source_conflict:' + member)
            except FileNotFoundError:
                with sftp.open(target, 'wb') as stream:
                    stream.write(payload)
        sftp.close()
        code = r'''
import json,sys,os,subprocess,hashlib,time
from pathlib import Path
base=Path('/storage/users/williamq/schema-study-deployment-20260926/jobs')
job=base/'scheduler-v2-20260928-v1'; runtime=job/'preparer-v1'
sys.path.insert(0,str(runtime))
from scheduler_v2.io import read,write_once,file_sha,digest
plan=read(base/'gpu-priority-handoff-20260928-v1/state/plan-v2.json')
condition=read(Path(plan['root'])/'condition.json')
qualification=condition['policy']['qualification']['path']
config=dict(deployment_id='mercury-scheduler-v2-20260928-v1',preparation_only=True,
 root=str(job/'run'),ancestor_root=plan['root'],local_root='/var/tmp/schema-study-williamq/mercury-scheduler-v2-20260928-v1',
 runtime_source=str(runtime),science_source=plan['source'],sources=plan['sources'],image=plan['image'],
 qualification=qualification,legacy_leases=plan['leases'],local_leases='/var/tmp/schema-study-williamq/gpu-leases',
 condition_file_sha256=file_sha(Path(plan['root'])/'condition.json'),
 authority_lock=str(Path(plan['root'])/'coordinator.lock'),
 execution_identity={'scientific_source':condition['source_file_bytes_sha256'],
                     'adapter_files':{n:file_sha(runtime/n) for n in ['scheduler_v2/science.py','scheduler_v2/io.py']},
                     'image_file_sha256':read(qualification)['image_file_bytes_sha256']},
 operational_files={p.relative_to(runtime).as_posix():file_sha(p) for p in runtime.rglob('*.py')})
configpath=job/'prepare-config.json';write_once(configpath,config)
os.environ['PYTHONPATH']=str(runtime)
control_python=read(base/'gpu-priority-handoff-20260928-v1/state/supervisor-ready.json')['identity']['argv'][0]
with (job/'audit/prepare.log').open('a') as log:
 p=subprocess.Popen([control_python,'-u','-m','scheduler_v2.migrate','--config',str(configpath)],
   stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt=dict(pid=p.pid,time=time.time(),config=str(configpath),python=control_python)
write_once(job/'audit'/('prepare-start-'+str(time.time_ns())+'.json'),receipt)
print(json.dumps(receipt))
'''
        code = code.replace("preparer-v1", "preparer-v" + str(revision))
        if revision != 1:
            code = code.replace("prepare-config.json", "prepare-config-v" + str(revision) + ".json")
        _, out, err = client.exec_command('python3 -c ' + shlex.quote(code), timeout=45)
        value, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error[-5000:])
        result = json.loads(value)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS/'prepare-start.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    finally:
        client.close(); gateway.close()


def stage_runtime(revision=1):
    """Stage an immutable operational release; no production signals or model calls."""
    repo = Path(__file__).resolve().parents[1]
    runtime = BASE + '/scheduler-v2-20260928-v1/runtime-v' + str(revision)
    gateway, client = connect()
    try:
        with client.open_sftp() as sftp:
            def mkdir(path):
                try: sftp.stat(path)
                except OSError:
                    mkdir(path.rsplit('/', 1)[0]); sftp.mkdir(path)
            for source in sorted((repo/'scheduler_v2').glob('*.py')):
                member = 'scheduler_v2/' + source.name
                target = runtime + '/' + member
                mkdir(target.rsplit('/', 1)[0])
                payload = source.read_bytes()
                try:
                    with sftp.open(target, 'rb') as f:
                        if f.read() != payload: raise ValueError('release_conflict:' + member)
                except FileNotFoundError:
                    with sftp.open(target, 'wb') as f: f.write(payload)
        code = r'''
import json,sys,time
from pathlib import Path
job=Path('/storage/users/williamq/schema-study-deployment-20260926/jobs/scheduler-v2-20260928-v1')
runtime=Path(RUNTIME)
sys.path.insert(0,str(runtime))
from scheduler_v2.io import read,write_once,file_sha,local_storage_preflight
c=read(job/'prepare-config-v2.json')
c.update(preparation_only=False,ancestor_quiescent=True,runtime_source=str(runtime),
 apptainer='/home/users/williamq/schema-study-deployment-20260926/apptainer-1.5.4/bin/apptainer',
 legacy_handoff_module=str(job.parent/'gpu-priority-handoff-20260928-v1/runtime-v2/gpu_priority_handoff.py'),
 legacy_handoff_state=str(job.parent/'gpu-priority-handoff-20260928-v1/state'),
 baseline_path=str(job/'audit/baseline-1790621855815221564.json'))
c['operational_files']={p.relative_to(runtime).as_posix():file_sha(p) for p in runtime.rglob('*.py')}
c['execution_identity']['adapter_files']={n:file_sha(runtime/n) for n in ['scheduler_v2/science.py','scheduler_v2/io.py']}
path=job/('deployment-config-v'+str(REVISION)+'.json')
write_once(path,c)
local_storage_preflight(c['local_root'])
for p in (c['local_leases'],c['legacy_leases']):Path(p).mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(config=str(path),runtime=str(runtime),files=len(c['operational_files']),time=time.time())))
'''.replace('RUNTIME', repr(runtime)).replace('REVISION', str(revision))
        _, out, err = client.exec_command('python3 -c ' + shlex.quote(code), timeout=45)
        stdout, stderr = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status(): raise RuntimeError(stderr[-5000:])
        result=json.loads(stdout)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS/('runtime-v'+str(revision)+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
    finally:
        client.close(); gateway.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['baseline', 'exec', 'stage-preparer', 'stage-runtime'])
    parser.add_argument('--file', type=Path)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--revision', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'baseline':
        baseline()
    elif args.mode == 'stage-preparer':
        stage_preparer(args.revision)
    elif args.mode == 'stage-runtime':
        stage_runtime(args.revision)
    else:
        print(json.dumps(execute(args.file.read_text(encoding='utf-8'), args.timeout), indent=2))


if __name__ == '__main__':
    main()
