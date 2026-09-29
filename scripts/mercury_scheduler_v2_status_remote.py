"""Bounded read-only observation of the September 28 V2 deployment."""
from pathlib import Path
import json
import subprocess
import time

job=Path('/storage/users/williamq/schema-study-deployment-20260926/jobs/scheduler-v2-20260928-v1')
local=Path('/var/tmp/schema-study-williamq/mercury-scheduler-v2-20260928-v1')
def read(p):
    return json.loads(p.read_text()) if p.exists() else None
status={'time':time.time(),'handoff':read(job/'handoff/handoff-status.json'),
        'activation_success':read(job/'audit/activation-success.json'),
        'activation_failed':read(job/'audit/activation-failed.json'),
        'activation_v2_failed':read(job/'audit/activation-v2-failed.json'),
        'prepare':read(job/'run/prepare-progress.json'),'snapshot':read(local/'snapshot.json'),
        'supervisor':read(local/'supervisor-process.json'),
        'coordinator':read(local/'coordinator-process.json'),
        'watchdog':read(local/'watchdogheartbeat.json'),
        'released':[p.name for p in (job/'handoff').glob('released-*.json')]}
status['logs']={str(p.relative_to(local)):p.read_text(errors='replace')[-2500:] for p in local.glob('*.log')}
status['worker_logs']={p.parent.name:p.read_text(errors='replace')[-2000:] for p in (local/'workers').glob('*/process.log')}
status['activation_logs']={p.name:p.read_text(errors='replace')[-2500:] for p in (job/'audit').glob('activation*.log')}
status['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
print(json.dumps(status))
