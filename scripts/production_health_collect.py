"""Bounded, read-only Mercury collection for a verified production incident."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import production_health_events as protocol
from telegram_codex_bridge import read, sha, atomic


def collect(config,event_id):
    if not protocol.hex_id(event_id):
        raise ValueError('invalid_event_id')
    state=Path(config['state_dir']);name=event_id+'.json'
    envelope=read(state/'inbox'/name);event=protocol.verify(envelope,config)
    if sha(state/'reports'/name)!=event['report_sha256']:
        raise ValueError('captured_report_changed')
    output=Path(config['analysis_root'])/('diagnostic-'+str(time.time_ns()))
    output.mkdir(parents=True,exist_ok=False)
    (output/'event.json').write_bytes((state/'inbox'/name).read_bytes())
    (output/'event-report.json').write_bytes((state/'reports'/name).read_bytes())
    sys.path[:0]=config['python_paths']
    from oaciss_access import mercury
    root=config['production_root'].rstrip('/')
    provenance=[]
    g,c=mercury()
    try:
        with c.open_sftp() as sftp:
            def get(remote,local,limit=16*1024*1024):
                with sftp.open(remote,'rb') as f:
                    data=f.read(limit+1)
                if len(data)>limit:
                    raise ValueError('source_exceeds_collection_bound')
                value=json.loads(data)
                path=output/local;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                provenance.append({'source':remote,'local':local,'file_bytes_sha256':hashlib.sha256(data).hexdigest(),
                                   'collected_at':time.time()})
                return value
            condition=get(root+'/condition.json','condition.json')
            if sha(output/'condition.json')!=config['condition_file_bytes_sha256']:
                raise ValueError('production_condition_changed')
            snapshot=get(root+'/snapshot.json','snapshot.json')
            v2 = config.get('scheduler_version') == 'v2'
            summary=get(root+('/snapshot.json' if v2 else '/summary.json'),'summary.json')
            health=get(root+'/health/state.json','health.json')
            if snapshot['condition']!=config['condition'] or health['condition']!=config['condition']:
                raise ValueError('live_condition_mismatch')
            for worker in snapshot.get('workers',[]):
                wid=worker['worker_id']
                if not re.fullmatch('[A-Za-z0-9_-]{1,80}',wid):
                    raise ValueError('unsafe_worker_id')
                worker_root=config.get('worker_state_root',root) if v2 else root
                try:
                    get(worker_root+'/workers/'+wid+'/heartbeat.json','workers/'+wid+'.json')
                except FileNotFoundError:
                    if not v2:
                        raise
                    # An unallocated or failed-to-start V2 worker has no real
                    # heartbeat. Preserve that absence instead of inventing one.
                    atomic(output/'workers'/(wid+'.json'), {'heartbeat_available': False,
                           'worker_id': wid, 'coordinator_observation': worker})
            get(config['remote_bridge_root']+'/status.json','publisher-status.json',65536)
            if config.get('v12_job'):
                get(config['v12_job']+'/audit/job-state.json','v12-job-state.json',65536)
        _,out,err=c.exec_command('nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader',timeout=30)
        text,error=out.read(65537).decode(),err.read(4096).decode()
        if out.channel.recv_exit_status()!=0 or len(text)>65536:
            raise RuntimeError('gpu_observation_failed')
        (output/'gpu.csv').write_text(text,encoding='utf-8')
    finally:
        c.close();g.close()
    atomic(output/'collection.json',{'event_id':event_id,'condition':config['condition'],
        'captured_report_hash_verified':True,'collected_at':time.time(),'sources':provenance,
        'production_writes':0,'semantic_accuracy':None})
    atomic(output/'input-manifest.json',{'files':{p.relative_to(output).as_posix():sha(p)
                                                  for p in output.rglob('*') if p.is_file()}})
    return {'output':str(output),'event':event['kind'],'ready_jobs':snapshot.get('ready_jobs'),
            'eligible_idle_workers':snapshot.get('eligible_idle_workers'),'slot_counts':summary.get('slot_counts')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--event-id',required=True)
    a=p.parse_args();print(json.dumps(collect(read(a.config),a.event_id),ensure_ascii=False))
