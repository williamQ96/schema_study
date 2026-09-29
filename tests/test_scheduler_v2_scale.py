"""Bounded scheduler state with one thousand independent matchsets."""
from __future__ import annotations

import gc
import tracemalloc

from scheduler_v2.runtime import make_ready
from scheduler_v2.state import Ledger


MATCHSETS = 1000


def graph():
    jobs, groups, imported = [], {}, {}
    for number in range(MATCHSETS):
        dataset, classifier = f'd{number}', f'c{number}'
        paper = f'p{number}'
        jobs.append({'job_id': dataset, 'kind': 'dataset_parse'})
        groups[dataset] = []
        imported[dataset] = {'parent_status': 'success',
                             'result_ref': {'path': f'cpu/{dataset}.json', 'file_bytes_sha256': 'frozen'}}
        jobs.append({'job_id': classifier, 'kind': 'classification',
                     'paper_id': paper, 'profile_id': 'classifier', 'dependencies': [dataset]})
        groups[classifier] = [f'{classifier}-g0', f'{classifier}-g1']
        for model in range(3):
            for replicate in range(3):
                jid = f'e{number}-{model}-{replicate}'
                jobs.append({'job_id': jid, 'kind': 'local_extraction',
                             'paper_id': paper, 'profile_id': f'model{model}',
                             'replicate_id': f'r{replicate}', 'dependencies': [classifier]})
                groups[jid] = [f'{jid}-g0', f'{jid}-g1']
    imported['c1'] = {'parent_status': 'contract_invalid',
                      'groups': {gid: {'status': 'contract_invalid',
                                      'generation_returned': True, 'attempt_refs': []}
                                 for gid in groups['c1']}}
    return jobs, groups, imported


def _commit_group(ledger, jid, gid, epoch, at):
    assignment = ledger.claim(jid, gid, 'gpu', 'incarnation-1', epoch, at)
    assert assignment
    raw = {'path': f'raw/{assignment["assignment_id"]}.json', 'file_bytes_sha256': 'pinned'}
    assert ledger.acknowledge(assignment['assignment_id'], raw, 'gpu', 'incarnation-1', epoch, at + 1)
    assert ledger.commit(assignment['assignment_id'],
                         {'status': 'success', 'generation_returned': True, 'attempt_refs': [raw]}, raw, at + 2)


def test_thousand_matchset_graph_admission_dependencies_and_rebuild(tmp_path):
    gc.collect()
    tracemalloc.start()
    try:
        jobs, groups, imported = graph()
        root = tmp_path / 'shared'
        with Ledger(tmp_path / 'local-a', root, 'scale') as ledger:
            snapshot = ledger.initialize(jobs, groups, imported)
            assert len(snapshot['jobs']) == 11000
            assert sum(len(job['groups']) for job in snapshot['jobs']) == 20000
            epoch = ledger.new_epoch()
            ledger.register_worker('gpu', 'incarnation-1', ['0'])
            make_ready(ledger, snapshot, 100)
            snapshot = ledger.snapshot()
            ready_classifiers = [job['job_id'] for job in snapshot['jobs']
                                 if job['job']['kind'] == 'classification'
                                 and any(group['ready_at'] is not None for group in job['groups'])]
            assert ready_classifiers == ['c0', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
            assert not any(group['ready_at'] is not None for job in snapshot['jobs']
                           if job['job']['kind'] == 'local_extraction'
                           for group in job['groups'])
            _commit_group(ledger, 'c0', 'c0-g0', epoch, 101)
            make_ready(ledger, ledger.snapshot(), 104)
            _commit_group(ledger, 'c0', 'c0-g1', epoch, 105)
            assert ledger.set_parent('c0', 'success', {'path': 'parent/c0.json'},
                                     {'path': 'index/c0.json'}, 108)
            make_ready(ledger, ledger.snapshot(), 109)
            snapshot = ledger.snapshot()
            by_id = {job['job_id']: job for job in snapshot['jobs']}
            assert by_id['e0-0-0']['groups'][0]['ready_at'] == 109
            assert all(by_id[f'e1-{m}-{r}']['status'] == 'blocked_dependency'
                       for m in range(3) for r in range(3))
            assert by_id['e2-0-0']['groups'][0]['ready_at'] is None
            assert by_id['c9']['groups'][0]['ready_at'] == 109
            assignment = ledger.claim('c2', 'c2-g0', 'gpu', 'incarnation-1', epoch, 110)
            assert assignment
            before = ledger.snapshot()
        # A new local SQLite index reconstructs from immutable shared events.
        with Ledger(tmp_path / 'local-b', root, 'scale') as rebuilt:
            after = rebuilt.snapshot()
            assert after == before
            assert rebuilt.initialize(jobs, groups, imported)['jobs'][2]['status'] == 'pending'
            assert rebuilt.recover(111) == []
            new_epoch = rebuilt.new_epoch()
            assert new_epoch == epoch + 1
            assert rebuilt.recover(112, {assignment['assignment_id']:
                    {'owner_released': True, 'incarnation': 'incarnation-1'}}) == [assignment['assignment_id']]
            assert next(job for job in rebuilt.snapshot()['jobs'] if job['job_id'] == 'c2')['groups'][0]['status'] == 'pending'
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    print(f'scheduler_v2_1000_matchsets_peak_mib={peak / 1024**2:.1f}')
    assert peak < 256 * 1024**2
