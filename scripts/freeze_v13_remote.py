"""Finalize the V13 condition after measured qualification, then bootstrap CPU work."""
from pathlib import Path
import os
import sys
import time

BASE = Path('/storage/users/williamq/schema-study-deployment-20260926/jobs/v13-full-20260928-v1')
JOB = Path(os.environ.get('V13_JOB', str(BASE)))
sys.path[:0] = [str(JOB / 'runtime-v1'), str(JOB / 'source')]
from scheduler_v2.io import read, write_once, file_sha, digest
from scheduler_v2.bootstrap import prepare
from scheduler_v2.cli import verify
from high_fidelity_schema_study.four_category.scheduler import compile_condition, live_gate
from high_fidelity_schema_study.four_category.common import seal
from high_fidelity_schema_study.four_category.backends import profile_hash

def main():
    old=read(JOB/'previous-deployment.json')
    config=read(JOB/'config.json')
    policy=read(JOB/'policy-pending.json')
    corpus=read(JOB/'corpus.json')
    source=read(JOB/'science-manifest.json')
    measured=read(JOB/'audit/capacity-summary.json')
    assert measured['status']=='pass', 'capacity_failed'
    profiles={p['profile_id']:p for p in config['profiles']}
    classifier_id=config['classification']['profile_id']
    shards=[read(JOB/'audit'/('cpu-preflight-'+classifier_id+'-shard-'+str(i)+'.json')) for i in range(4)]
    assert all(r['status']=='pass' and r['shard_index']==i and r['shard_count']==4 and r['config_sha256']==digest(config) for i,r in enumerate(shards))
    papers=[pid for r in shards for pid in r['checked_papers']]
    assert len(papers)==len(set(papers))==10 and set(papers)=={p['paper_id'] for p in corpus['papers']}
    rows=[row for r in shards for row in r['rows']]
    assert len(rows)==len({(r['kind'],r['paper_id'],r['group_id']) for r in rows})
    combined={**shards[0],'shard_count':1,'shard_index':0,'checked_papers':papers,'rows':rows,
        'extraction_group_count':sum(r['kind']=='extraction' for r in rows),
        'classification_group_count':sum(r['kind']=='classification' for r in rows),
        'shard_file_hashes':{str(i):file_sha(JOB/'audit'/('cpu-preflight-'+classifier_id+'-shard-'+str(i)+'.json')) for i in range(4)}}
    assert combined['extraction_group_count']==147 and combined['classification_group_count']==385
    write_once(JOB/'audit'/('cpu-preflight-'+classifier_id+'.json'),combined)
    qualified={}
    for worker in policy['workers']:
        if not worker['gpu_ids']:continue
        pid=worker['allowed_profile_ids'][0]
        report=read(JOB/'audit'/('cpu-preflight-'+pid+'.json'))
        assert report['status']=='pass' and report['config_sha256']==digest(config)
        row=measured['results'][worker['worker_id']]
        assert row['profile_sha256']==profile_hash(profiles[pid])
        assert row['source_tree_sha256']==source['source_tree_sha256']
        assert row['gpu_ids']==worker['gpu_ids'] and row['status']=='pass'
        qualified[profile_hash(profiles[pid])]=row
    qualification={'schema_version':'resident-scheduler-qualification/v1','status':'pass',
        'source_tree_sha256':source['source_tree_sha256'],'image_file_bytes_sha256':file_sha(old['image']),
        'profiles':qualified,'created_at':measured['time'],'semantic_evaluation':False,
        'purpose':'fresh_V13_synthetic_capacity_and_all_corpus_cpu_contract_checks'}
    qpath=JOB/'qualification.json'
    write_once(qpath,qualification)
    policy['qualification']={'path':str(qpath),'file_bytes_sha256':file_sha(qpath)}
    write_once(JOB/'policy.json',policy)
    condition=compile_condition(config,corpus,policy,live=True)
    condition=seal({**condition,'qualification_record':qualification},'condition')
    assert condition['source_file_bytes_sha256']==source['files']
    write_once(JOB/'condition.json',condition)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(g for w in policy['workers'] for g in w['gpu_ids'])
    live_gate(condition,Path(old['image']),JOB/'source_bundle')
    write_once(JOB/'audit/live-gate.json',{'status':'pass','condition':condition['condition'],
               'actual_checkpoint_bytes_verified':True,'actual_image_sha256':qualification['image_file_bytes_sha256']})
    operations=read(JOB/'runtime-manifest.json')['files']
    deployment=dict(deployment_id='v13-full-20260928-v1',preparation_only=False,ancestor_quiescent=True,
        root=str(BASE/'run'),local_root='/var/tmp/schema-study-williamq/v13-full-20260928-v1',
        runtime_source=str(JOB/'runtime-v1'),science_source=str(JOB/'source'),sources=str(JOB/'source_bundle'),
        image=old['image'],apptainer=old['apptainer'],qualification=str(qpath),
        legacy_leases=old['legacy_leases'],local_leases=old['local_leases'],authority_lock=old['authority_lock'],
        condition_file_sha256=file_sha(JOB/'condition.json'),operational_files=operations,
        execution_identity={'scientific_source':source['files'],
            'image_file_sha256':qualification['image_file_bytes_sha256'],
            'adapter_files':{n:operations[n] for n in ['scheduler_v2/io.py','scheduler_v2/science.py']}})
    write_once(JOB/'deployment-config.json',deployment)
    result=prepare(JOB/'deployment-config.json',condition_path=JOB/'condition.json')
    checked=verify(deployment)
    assert checked['status']=='pass', checked
    write_once(JOB/'audit/bootstrap.json',{'bootstrap':result,'verification':checked})
    print(__import__('json').dumps({'status':'prepared','condition':condition['condition'],'bootstrap':result,'verify':checked}),flush=True)

if __name__=='__main__':main()
