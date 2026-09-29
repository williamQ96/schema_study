"""Run on Mercury with a staged operational release; never generate model output."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

def main(config_path):
    config=json.loads(Path(config_path).read_text())
    sys.path[:0]=[config['runtime_source'],config['science_source']]
    from scheduler_v2.io import read,write_once,file_sha
    from scheduler_v2.cli import verify
    from scheduler_v2.runtime import worker_command
    from scheduler_v2.processes import resources_free
    from scheduler_v2.policy import choose
    from scheduler_v2.state import Ledger
    from scheduler_v2.runtime import checked_ref,make_ready
    root=Path(config['root'])
    condition=read(root/'condition.json')
    checks=verify(config)
    if checks['status']!='pass':raise ValueError(checks['errors'])
    write_once(root/'deployment.json',config)
    imported=checked_ref(root,read(root/'import-latest.json'))
    shadow=root.parent/('shadow-'+str(time.time_ns()))
    decisions=[]
    with Ledger(Path(config['local_root'])/'shadow'/shadow.name,shadow,'shadow') as ledger:
        ledger.initialize(condition['jobs'],read(root/'group_plans.json'),imported['imported'])
        ledger.new_epoch()
        make_ready(ledger,ledger.snapshot(),time.time(),{k:v.get('ready_at') for k,v in imported['ancestor_ledger_rows'].items()})
        snap=ledger.snapshot()
        for worker in condition['policy']['workers']:
            if not worker['gpu_ids']:continue
            decision=choose(snap,worker,time.time())
            job=next((j for j in condition['jobs'] if decision and j['job_id']==decision['job_id']),None)
            decisions.append(dict(worker=worker['worker_id'],decision=decision,paper_id=job.get('paper_id') if job else None,
                                  kind=job.get('kind') if job else None))
    probes=[]
    for worker in condition['policy']['workers']:
        if not worker['gpu_ids']:continue
        if resources_free(worker,config['local_leases'],config['legacy_leases']):
            raise ValueError('expected_legacy_owner_missing_before_preflight')
        incarnation='preflight-'+str(time.time_ns())
        cmd,env=worker_command(config,condition,worker,incarnation)
        # The release must fail at the held legacy lease before session/model creation.
        result=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=60)
        safe=result.returncode!=0 and 'lease_already_owned:/legacy-leases/' in result.stderr
        probes.append(dict(worker=worker['worker_id'],returncode=result.returncode,dual_lease_block_verified=safe,
                           log_tail=result.stderr[-2500:]))
        if not safe:raise RuntimeError(json.dumps(probes[-1]))
    report=dict(time=time.time(),status='pass',pin_verification=checks,shadow_decisions=decisions,
                container_entry_probes=probes,model_generation_calls=0)
    write_once(root.parent/'audit'/'runtime-preflight.json',report)
    print(json.dumps(report))

if __name__=='__main__':main(sys.argv[1])
