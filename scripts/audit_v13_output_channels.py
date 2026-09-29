"""Capture immutable V13 returned records for offline diagnosis; no model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from mercury_scheduler_v2_access import connect, execute


def capture(target: Path):
    manifest = json.loads((target/'capture.json').read_text()) if (target/'capture.json').exists() else execute(r'''
import json,sqlite3,time,hashlib
from pathlib import Path
root=Path('/storage/users/williamq/schema-study-deployment-20260926/jobs/v13-full-20260928-v1')
run=root/'run'; local=Path('/var/tmp/schema-study-williamq/v13-full-20260928-v1')
db=sqlite3.connect('file:'+str(local/'scheduler_v2.sqlite')+'?mode=ro',uri=True)
db.row_factory=sqlite3.Row; db.execute('BEGIN')
jobs={r['job_id']:json.loads(r['body']) for r in db.execute('select job_id,body from jobs')}
assignments=[dict(r) for r in db.execute('select * from assignments where verdict is not null order by started_at')]
rows=[]; members={}; papers=set()
def member(path, dest):
 data=path.read_bytes(); members[dest]={'remote':str(path),'bytes':len(data),'file_bytes_sha256':hashlib.sha256(data).hexdigest()}
for a in assignments:
 v=json.loads(a['verdict']); j=jobs[a['job_id']]; papers.add(j['paper_id'])
 ref=v['attempt_refs'][-1]
 member(run/ref['path'],'run/'+ref['path'])
 if members['run/'+ref['path']]['file_bytes_sha256']!=ref['file_bytes_sha256']:raise ValueError('record_hash_mismatch')
 rows.append({'assignment_id':a['assignment_id'],'worker':a['worker_id'],'job':j,'verdict':v,'record':'run/'+ref['path']})
condition=json.loads((run/'condition.json').read_text())
member(run/'condition.json','condition.json')
for path in (run/'indexes').glob('*.json'):member(path,'indexes/'+path.name)
for paper in condition['corpus']['papers']:
 if paper['paper_id'] in papers:
  ref=paper['source']['artifacts']['input']; path=root/'candidate-02/source_bundle'/ref['path']
  member(path,'sources/'+ref['path'])
  if members['sources/'+ref['path']]['file_bytes_sha256']!=ref['sha256']:raise ValueError('source_hash_mismatch')
source=root/'candidate-02/source/high_fidelity_schema_study'
source_pins={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in [
 'four_category/structured_output.py','four_category/response_channels.py','four_category/resident_worker.py',
 'four_category/workflow.py','four_category/classification_v3.py','four_category/extraction_v12.py',
 'four_category/extraction_v13.py','four_category/compact_prompt_v13.py','four_category/backends.py']}
print(json.dumps({'captured_at':time.time(),'root':str(root),'condition':condition['condition_sha256'] if 'condition_sha256' in condition else None,
 'snapshot':json.loads((local/'snapshot.json').read_text()),'rows':rows,'members':members,'source_pins':source_pins}))
''', 90)
    target.mkdir(parents=True, exist_ok=True)
    for member in manifest['members'].values():
        member['local'] = 'members/' + member['file_bytes_sha256'] + '.json'
    (target / 'capture.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    gateway, client = connect()
    try:
        with client.open_sftp() as sftp:
            for relative, member in manifest['members'].items():
                local = target / member['local']
                if local.exists():
                    if hashlib.sha256(local.read_bytes()).hexdigest() != member['file_bytes_sha256']:
                        raise ValueError('existing_download_hash_mismatch:' + relative)
                    continue
                local.parent.mkdir(parents=True, exist_ok=True)
                with sftp.open(member['remote'], 'rb') as stream:
                    stream.prefetch(member['bytes'])
                    data = stream.read()
                if hashlib.sha256(data).hexdigest() != member['file_bytes_sha256']:
                    raise ValueError('download_hash_mismatch:' + relative)
                local.write_bytes(data)
    finally:
        client.close()
        gateway.close()
    repo = Path(__file__).resolve().parents[1]
    checks = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() == sha
              for name, sha in manifest['source_pins'].items()}
    (target/'source-checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps({'target':str(target),'records':len(manifest['rows']),
                      'bytes':sum(m['bytes'] for m in manifest['members'].values()),'source_checks':checks}))


if __name__ == '__main__':
    capture(Path(sys.argv[1]))
