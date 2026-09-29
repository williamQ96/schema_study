import copy
import json

import pytest

from high_fidelity_schema_study.four_category.common import read_json, seal
from high_fidelity_schema_study.four_category.offline import mock_transport
from high_fidelity_schema_study.four_category.scheduler import run_scheduler
from high_fidelity_schema_study.four_category.workflow import experiment_errors, tasks_for_config, _attempt, replay_run, plan_jobs
from high_fidelity_schema_study.four_category.paper import build_index
from high_fidelity_schema_study.four_category.tasks import load_task, render_task
from high_fidelity_schema_study.four_category.grouped_classification import _extended
from .test_matrix_scheduler import setup


def new_config(config, protocol='classification-quotes/v2'):
    config['classification'].update(protocol=protocol, grouping={'max_units':24,'max_target_chars':16000})
    config['extraction_input_protocol']='extraction-compact/v2'
    config['task_hashes']={k:t['task_sha256'] for k,t in tasks_for_config(config).items()}
    return config


@pytest.mark.parametrize('protocol',['classification-quotes/v2','classification-anchors/v3'])
def test_new_condition_complete_matrix_resume_and_stage_counts(tmp_path, protocol):
    sources,config,corpus,policy=setup(tmp_path)
    old=plan_jobs(config,corpus)
    new_config(config,protocol)
    new=plan_jobs(config,corpus)
    assert all(a['job_id']!=b['job_id'] for a,b in zip(old['jobs'],new['jobs']))
    root=tmp_path/'run'
    summary=run_scheduler(config,corpus,policy,root=root,sources=sources,leases=tmp_path/'leases',start_watchdog=False)
    assert summary['stage_counts']=={'classification':{'success':1},'local_extraction':{'success':9},'dataset_parse':{'success':1},'soft_reference':{'deferred':1}}
    before={_extended(p):_extended(p).read_bytes() for p in root.rglob('attempt-*.json')}
    assert run_scheduler(config,corpus,policy,root=root,sources=sources,leases=tmp_path/'leases',start_watchdog=False)==summary
    assert all(p.read_bytes()==value for p,value in before.items())


def test_fence_normalization_replay_is_new_task_only(tmp_path):
    sources,config,corpus,_=setup(tmp_path)
    paper=read_json(sources/'paper_input.json')
    legacy=load_task('classification')
    raw=mock_transport({'model':'mock','messages':render_task(legacy,paper)})['text']
    index=build_index(raw,paper,{k:'fixture' for k in ('run_id','profile_sha256','task_sha256','request_sha256')})
    def fenced(request):
        response=mock_transport(request)
        response['text']='```json\n'+response['text']+'\n```'
        return response
    for modern in (False,True):
        candidate=copy.deepcopy(config)
        if modern:new_config(candidate)
        task=tasks_for_config(candidate)['extraction']
        job=next(j for j in plan_jobs(candidate,corpus)['jobs'] if j['kind']=='local_extraction')
        profile=next(p for p in candidate['profiles'] if p['profile_id']==job['profile_id'])
        record=_attempt(job,profile,task,paper,index,None,None,1,allow_live=False,transport=fenced,counter=None)
        assert record['status']==('success' if modern else 'contract_invalid')
        assert replay_run(record,task,paper,index=index)==[]
        if modern:
            assert record['response_normalization']=={'kind':'json_code_fence'}
            altered=copy.deepcopy(record);altered['response_normalization']={'kind':'bare_json'}
            assert 'response_normalization_replay_mismatch' in replay_run(seal(altered,'record_sha256'),task,paper,index=index)


@pytest.mark.parametrize('policy',[None,{}, {'max_units':True,'max_target_chars':1}, {'max_units':0,'max_target_chars':1}])
def test_invalid_group_config_rejected_before_job_planning(tmp_path,policy):
    _,config,corpus,_=setup(tmp_path);new_config(config);config['classification']['grouping']=policy
    assert 'classification_grouping_positive_integers_required' in experiment_errors(config)
    with pytest.raises(ValueError):plan_jobs(config,corpus)
