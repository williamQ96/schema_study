"""Derive navigation or explicitly partial observations from complete group records."""
from __future__ import annotations

import copy

from .common import digest, seal
from .extraction_protocol import module_for_task
from .paper import source_identity


def build_artifact(job, profile, task, paper_input, index, policy, records):
    if task['kind'] == 'classification':
        from .classification_v2 import plan_groups
        from .classification_navigation_v4 import build_index_v4
        return build_index_v4(paper_input, task, plan_groups(paper_input, policy), records)
    module = module_for_task(task)
    groups = module.plan_groups(paper_input, policy)
    if len(records) != len(groups):
        raise ValueError('complete_execution_record_count_mismatch')
    if all(record['status'] == 'success' for record in records):
        return module.build_bundle(paper_input, index, task, groups, records)
    if task.get('admission_rules_version') not in {'extraction-pointers/v5', 'extraction-pointers/v6', 'extraction-local-quotes/v7', 'extraction-quote-options/v8', 'extraction-page-regions/v13', 'extraction-page-regions/v13r2'}:
        raise ValueError('partial_observations_require_pointer_protocol')
    admitted, excluded = [], []
    for group, record in zip(groups, records):
        if record['job']['parent_job_id'] != job['job_id'] or record['profile'] != profile:
            raise ValueError('partial_observation_binding_mismatch')
        if record['status'] == 'success':
            admitted.append(module.materialize_group(paper_input, index, task, group, record))
        else:
            from .workflow import replay_run
            errors = replay_run(record, task, paper_input, index=index, extraction_group=group)
            if errors:
                raise ValueError('partial_observation_replay_failed:' + ','.join(errors[:4]))
            excluded.append({'group_id': group['group_id'], 'window_ids': copy.deepcopy(group['window_ids']),
                             'status': record['status'], 'run_id': record['run_id'],
                             'record_sha256': record['record_sha256'],
                             'validation_errors': copy.deepcopy(record['validation_errors'])})
    return seal({'schema_version': 'paper-partial-observations/v1',
                 'source_identity': source_identity(paper_input),
                 'paper_input_canonical_sha256': digest(paper_input),
                 'index_sha256': index['index_sha256'], 'task_sha256': task['task_sha256'],
                 'parent_job_id': job['job_id'], 'profile_sha256': records[0]['profile_sha256'],
                 'group_plan_sha256': digest(groups), 'groups': groups,
                 'admission_status': 'partial' if admitted else 'none',
                 'admitted_groups': admitted, 'excluded_groups': excluded,
                 'planned_group_count': len(groups), 'admitted_group_count': len(admitted),
                 'planned_window_count': sum(len(g['window_ids']) for g in groups),
                 'admitted_window_count': sum(len(g['coverage']) for g in admitted),
                 'resolved_object_count': None, 'semantic_correctness': 'not_established',
                 'authority': 'admitted_whole_groups_only_not_complete_paper_schema'}, 'artifact_sha256')
