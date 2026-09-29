"""Scientific task execution/replay shared by worker and coordinator."""
from pathlib import Path
from . import backends
from .common import contained, digest, read_json, seal, seal_errors
from .dataset import parse_dataset, verify_dataset
from .paper import verify_paper, build_index, verify_index
from .workflow import _attempt, replay_run, tasks_for_config
from .offline import mock_transport
from .grouped_classification import execute_grouped, verify_grouped, mock_group_transport
from .extraction_protocol import GROUPED_PROTOCOLS


def paper_context(condition, job, sources):
    paper = next(p for p in condition['corpus']['papers'] if p['paper_id'] == job['paper_id'])
    issues = verify_paper(paper['source'], sources)
    if issues:
        raise ValueError('paper_source_verification_failed')
    paths = {k: contained(sources, v['path']) for k, v in paper['source']['artifacts'].items()}
    return {'input': read_json(paths['input']), 'layout': read_json(paths['layout']),
            'reading': paths['reading_text'].read_text(encoding='utf-8')}


def execute(condition, envelope, sources, session, progress, root):
    job = envelope['job']
    if job['kind'] == 'dataset_parse':
        old = read_json(contained(sources, job['dataset']['bundle_path']))
        if old['bundle_sha256'] != job['dataset']['bundle_sha256'] or verify_dataset(old, sources):
            raise ValueError('dataset_source_verification_failed')
        raw = next(s for s in old['sources'] if s['role'] == 'dataset')
        bundle = parse_dataset(contained(sources, raw['path']), root=sources,
                               format_hint=old['parser']['format'], sample_limit=condition['config']['dataset_parser']['sample_limit'])
        return {'status': 'success' if bundle['status'] == 'pass' else 'dataset_failed', 'dataset_bundle': bundle}
    progress('verifying')
    context = paper_context(condition, job, sources)
    index = read_json(contained(root, envelope['index_path'])) if envelope.get('index_path') else None
    if index and verify_index(index, context['input']):
        raise ValueError('classification_index_invalid')
    tasks = tasks_for_config(condition['config'])
    task = tasks['classification' if job['kind'] == 'classification' else 'extraction']
    profile = next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id'])
    old = backends._transformers_transport
    try:
        if profile['backend'] == 'transformers':
            backends._transformers_transport = session
        progress('tokenizing')
        if job.get('group_execution', {}).get('version') == 'all-groups/v1':
            from .complete_groups import execute_complete
            from .grouped_extraction import mock_group_transport as mock_extraction
            classifier = job['kind'] == 'classification'
            policy = condition['config']['classification']['grouping'] if classifier else condition['config']['extraction_grouping']
            mock = mock_group_transport if classifier else mock_extraction
            return execute_complete(job, profile, task, context['input'], index, policy,
                                    envelope['attempt'], root, allow_live=condition['live'],
                                    transport=mock if profile['backend'] == 'mock' else None,
                                    counter=session.count if profile['backend'] == 'transformers' else None,
                                    progress=progress)
        if job['kind'] == 'classification' and condition['config']['classification'].get('protocol') in {'classification-quotes/v2', 'classification-anchors/v3', 'classification-anchors/v13r2'}:
            return execute_grouped(job, profile, task, context['input'],
                                   condition['config']['classification']['grouping'],
                                   envelope['attempt'], root, allow_live=condition['live'],
                                   transport=mock_group_transport if profile['backend'] == 'mock' else None,
                                   counter=session.count if profile['backend'] == 'transformers' else None,
                                   progress=progress)
        if job['kind'] != 'classification' and condition['config'].get('extraction_input_protocol') in GROUPED_PROTOCOLS:
            from .grouped_extraction import execute_grouped as execute_extraction, mock_group_transport as mock_extraction
            return execute_extraction(job, profile, task, context['input'], index,
                                      condition['config']['extraction_grouping'], envelope['attempt'], root,
                                      allow_live=condition['live'],
                                      transport=mock_extraction if profile['backend'] == 'mock' else None,
                                      counter=session.count if profile['backend'] == 'transformers' else None,
                                      progress=progress)
        record = _attempt(job, profile, task, context['input'], index, context['layout'], context['reading'],
                          envelope['attempt'], allow_live=condition['live'],
                          transport=mock_transport if profile['backend'] == 'mock' else None,
                          counter=session.count if profile['backend'] == 'transformers' else None)
    finally:
        backends._transformers_transport = old
    return {'status': record['status'], 'record': record}


def verify_result(condition, envelope, result, sources, root):
    from .replay_cache import replay_validation_scope
    with replay_validation_scope():
        return _verify_result(condition, envelope, result, sources, root)


def _verify_result(condition, envelope, result, sources, root):
    errors = seal_errors(result, 'result_sha256')
    for key in ('condition', 'job_id', 'attempt', 'worker_id'):
        if result.get(key) != envelope[key]:
            errors.append('result_binding:' + key)
    body, job = result.get('body', {}), envelope['job']
    if body.get('status') == 'infrastructure_failed':
        if set(body) != {'status', 'error_type'} or not isinstance(body['error_type'], str):
            errors.append('malformed_infrastructure_failure')
        return errors, None
    if job['kind'] == 'dataset_parse':
        bundle = body.get('dataset_bundle', {})
        errors.extend(verify_dataset(bundle, sources))
        expected = 'success' if bundle.get('status') == 'pass' else 'dataset_failed'
        if body.get('status') != expected or bundle.get('dataset_id') != job['dataset']['dataset_id']:
            errors.append('dataset_result_identity_mismatch')
        return errors, None
    context = paper_context(condition, job, sources)
    task = tasks_for_config(condition['config'])['classification' if job['kind'] == 'classification' else 'extraction']
    index = read_json(contained(root, envelope['index_path'])) if envelope.get('index_path') else None
    if job.get('group_execution', {}).get('version') == 'all-groups/v1':
        from .complete_groups import verify_complete
        from .execution_artifacts import build_artifact
        profile = next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id'])
        policy = condition['config']['classification']['grouping'] if job['kind'] == 'classification' else condition['config']['extraction_grouping']
        group_errors, records = verify_complete(job, profile, task, context['input'], index,
                                               policy, envelope['attempt'], body, root)
        errors.extend(group_errors)
        if errors:
            return errors, None
        try:
            return errors, build_artifact(job, profile, task, context['input'], index, policy, records)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return errors + ['complete_execution_artifact:' + str(exc)], None
    if job['kind'] == 'classification' and condition['config']['classification'].get('protocol') in {'classification-quotes/v2', 'classification-anchors/v3', 'classification-anchors/v13r2'}:
        group_errors, verified_index = verify_grouped(job,
            next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id']),
            task, context['input'], condition['config']['classification']['grouping'],
            envelope['attempt'], body, root)
        return errors + group_errors, verified_index
    if job['kind'] != 'classification' and condition['config'].get('extraction_input_protocol') in GROUPED_PROTOCOLS:
        from .grouped_extraction import verify_grouped as verify_extraction
        group_errors, bundle = verify_extraction(job,
            next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id']),
            task, context['input'], index, condition['config']['extraction_grouping'],
            envelope['attempt'], body, root)
        return errors + group_errors, bundle
    record = body.get('record', {})
    expected_profile = next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id'])
    if record.get('job') != job or record.get('profile') != expected_profile or record.get('attempt') != envelope['attempt']:
        errors.append('record_job_profile_attempt_mismatch')
    errors.extend(replay_run(record, task, context['input'], index=index, layout=context['layout'], reading_text=context['reading']))
    if record.get('status') != body.get('status'):
        errors.append('status_mismatch')
    if not errors and job['kind'] == 'classification' and record['status'] == 'success':
        producer = {k: record[k] for k in ('run_id', 'profile_sha256', 'request_sha256')}
        producer['task_sha256'] = task['task_sha256']
        index = build_index(record['backend_result']['raw_text'], context['input'], producer, taxonomy_value=task['taxonomy'])
    else:
        index = None
    return errors, index
