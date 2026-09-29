import copy
import json
import pytest

from scheduler_v2 import science
from scheduler_v2.io import write_once, read
from .test_complete_groups import _case, _success_transport
from high_fidelity_schema_study.four_category.complete_groups import _context, execute_complete


@pytest.mark.parametrize('kind', ['classification', 'extraction'])
def test_one_group_quantum_preserves_requests_and_rejected_answers(tmp_path, monkeypatch, kind):
    paper, task, profile, job, policy, index = _case(tmp_path, kind)
    module, groups = _context(task, paper, index, policy)
    ctx = dict(module=module, groups=groups, task=task, paper=paper, profile=profile,
               index=index, policy=policy, source_pins={})
    monkeypatch.setattr(science, 'context', lambda *a: ctx)
    delegate = _success_transport(kind)
    old_requests, new_requests = [], []
    def capture(target):
        def transport(req):
            target.append(copy.deepcopy(req))
            if len(target) == 1:
                return {'text': '{"invalid":true}', 'model': req['model'], 'finish_reason': 'stop',
                        'usage': None, **({'structured_output_applied': delegate(req)['structured_output_applied']}
                                        if kind == 'extraction' else {})}
            return delegate(req)
        return transport
    expected = execute_complete(job, profile, task, paper, index, policy, 1, tmp_path/'old',
                                allow_live=False, transport=capture(old_requests))
    monkeypatch.setattr(module, 'mock_group_transport', capture(new_requests))
    values = [science.execute_group({'live': False}, job, g['group_id'], tmp_path,
                                    tmp_path/'new', None, lambda *a, **k: None) for g in groups]
    assert new_requests == old_requests
    assert values[0]['status'] == 'contract_invalid'
    assert [v['status'] for v in values] == [v['status'] for v in expected['group_outcomes']]
    count = len(new_requests)
    for group in groups:
        science.execute_group({'live': False}, job, group['group_id'], tmp_path,
                              tmp_path/'new', None, lambda *a, **k: None)
    assert len(new_requests) == count


def test_atomic_immutable_write_cannot_replace_historical_hardlink(tmp_path):
    import os
    old, new = tmp_path/'old.json', tmp_path/'new.json'
    write_once(old, {'answer': 1})
    os.link(old, new)
    with pytest.raises(ValueError, match='immutable_record_conflict'):
        write_once(new, {'answer': 2})
    assert read(old) == {'answer': 1}


def test_verification_receipt_cannot_be_forged_with_self_hash(tmp_path, monkeypatch):
    from scheduler_v2.io import digest
    paper, task, profile, job, policy, index = _case(tmp_path, 'classification')
    module, groups = _context(task, paper, index, policy)
    ctx = dict(module=module, groups=groups, task=task, paper=paper, profile=profile,
               index=index, policy=policy, source_pins={})
    monkeypatch.setattr(science, 'context', lambda *a: ctx)
    key = tmp_path/'private.key'; key.write_bytes(b'private-cache-key')
    monkeypatch.setenv('MERCURY_VERIFICATION_KEY', str(key))
    root = tmp_path/'run'; condition = {'live': False}
    gid = groups[0]['group_id']
    science.execute_group(condition, job, gid, tmp_path, root, None, lambda *a, **k: None)
    expected = science.verify_group(condition, job, gid, tmp_path, root, 'verifier-1')
    assert science.verify_group(condition, job, gid, tmp_path, root, 'verifier-1') == expected
    receipt = next((root/'verification').glob('*.json'))
    value = read(receipt); value['outcome']['status'] = 'invented'
    value['outcome_sha256'] = digest(value['outcome'])
    receipt.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='verification_receipt_corrupt'):
        science.verify_group(condition, job, gid, tmp_path, root, 'verifier-1')
