"""Launch bounded CPU preflight shards and independent GPU capacity tests."""
import argparse
import json
from pathlib import Path
import sys
from mercury_scheduler_v2_access import connect, execute
from deploy_v13_full import REMOTE, PYTHON, LOCAL

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--revision',type=int,required=True)
    args=parser.parse_args()
    remote=REMOTE+'/candidate-'+format(args.revision,'02d')
    gateway,client=connect()
    try:
        with client.open_sftp() as sftp:
            for name in ['v13_cpu_preflight.py','v13_capacity_launcher.py','freeze_v13_remote.py']:
                sftp.put(str(Path(__file__).parent/name),remote+'/'+name)
            sftp.put(str(LOCAL/'previous-deployment.json'),remote+'/previous-deployment.json')
    finally:client.close();gateway.close()
    code=r'''
import subprocess,json,os
from pathlib import Path
j=Path(@@REMOTE@@);d=json.loads((j/'previous-deployment.json').read_text());c=json.loads((j/'config.json').read_text());rows=[]
(j/'audit').mkdir(exist_ok=True)
env=dict(os.environ,V13_JOB=str(j))
with (j/'capacity-launcher.log').open('xb') as log:
 p=subprocess.Popen([@@PYTHON@@,str(j/'v13_capacity_launcher.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
rows.append(dict(kind='capacity_launcher',pid=p.pid))
for p in c['profiles']:
 if p['profile_id'] not in c['roles']['locals']:continue
 shards=4 if p['profile_id']==c['classification']['profile_id'] else 1
 for shard in range(shards):
  suffix='' if shards==1 else '-shard-'+str(shard)
  scratch=Path('/var/tmp/schema-study-williamq/v13-cpu-preflight')/j.name/(p['profile_id']+suffix);scratch.mkdir(parents=True,exist_ok=True)
  env={k:v for k,v in os.environ.items() if not k.startswith(('APPTAINERENV_','SINGULARITYENV_'))}
  env.update(APPTAINERENV_PYTHONPATH='/scientific',APPTAINERENV_CUDA_VISIBLE_DEVICES='',APPTAINERENV_OMP_NUM_THREADS='2',APPTAINERENV_RAYON_NUM_THREADS='2',APPTAINERENV_HF_HUB_OFFLINE='1',APPTAINERENV_TRANSFORMERS_OFFLINE='1',APPTAINERENV_XDG_CACHE_HOME='/scratch/cache',APPTAINERENV_HF_HOME='/scratch/hf',APPTAINERENV_TMPDIR='/scratch')
  cmd=[d['apptainer'],'exec','--cleanenv','--containall','--bind',str(j/'source')+':/scientific:ro','--bind',str(j)+':/job:rw','--bind',str(scratch)+':/scratch:rw','--bind',p['model_id']+':'+p['model_id']+':ro',d['image'],'/opt/phase1-venv/bin/python','-u','/job/v13_cpu_preflight.py','--job','/job','--profile-id',p['profile_id'],'--shard-count',str(shards),'--shard-index',str(shard)]
  with (j/'audit'/('cpu-preflight-'+p['profile_id']+suffix+'.log')).open('xb') as log:
   proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
  rows.append(dict(kind='cpu_preflight',profile=p['profile_id'],shard=shard,pid=proc.pid))
(j/'audit/preflight-launches.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
'''.replace('@@REMOTE@@',repr(remote)).replace('@@PYTHON@@',repr(PYTHON))
    print(json.dumps(execute(code)))

if __name__=='__main__':main()
