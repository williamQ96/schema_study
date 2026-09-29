"""One-hop, byte-pinned import of immutable scheduler group attempts."""
from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
import re
import shutil

from .common import contained, digest, file_digest, identity, read_json, seal_errors, write_new
from .complete_groups import _child_job, _context, _load_group_history, _configuration
from .paper import verify_index
from .replay_cache import replay_validation_scope
from .scheduler_io import Lock
from .scheduler_tasks import paper_context
from .workflow import plan_jobs, tasks_for_config

VERSION = 'group-cache-lineage/v1'
HEX = re.compile(r'[0-9a-f]{64}\Z')
ATTEMPT = re.compile(r'attempt-([1-9][0-9]*)\.json\Z')


def _normalized(condition):
    value = copy.deepcopy(condition)
    value.pop('condition', None)
    value.pop('qualification_record', None)
    value.pop('source_file_bytes_sha256', None)
    value.pop('coordinator_source_file_bytes_sha256', None)
    value['policy'].pop('qualification', None)
    value['policy'].pop('group_cache_import', None)
    value['policy'].pop('execution_source', None)
    return value


def compatibility_errors(ancestor_condition: dict, target_condition_without_lineage: dict) -> list[str]:
    """Pure semantic gate, suitable for checking before stopping a source run."""
    errors = []
    for label, condition in (('ancestor', ancestor_condition), ('target', target_condition_without_lineage)):
        errors += [label + ':' + e for e in seal_errors(condition, 'condition')]
        try:
            plan = plan_jobs(condition['config'], condition['corpus'])
            if plan['plan_sha256'] != condition['logical_plan_sha256']:
                errors.append(label + ':logical_plan_mismatch')
            if plan['task_hashes'] != {k: v['task_sha256'] for k, v in tasks_for_config(condition['config']).items()}:
                errors.append(label + ':task_hash_mismatch')
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(label + ':plan_invalid:' + str(exc))
    try:
        if _normalized(ancestor_condition) != _normalized(target_condition_without_lineage):
            errors.append('semantic_condition_mismatch')
        if ancestor_condition['condition'] == target_condition_without_lineage['condition']:
            errors.append('new_condition_required')
        if ancestor_condition['jobs'] != target_condition_without_lineage['jobs']:
            errors.append('job_set_mismatch')
        if ancestor_condition['policy'].get('group_cache_import'):
            errors.append('one_hop_only')
        if target_condition_without_lineage['policy'].get('group_cache_import'):
            errors.append('target_already_bound')
        control_source = target_condition_without_lineage.get('coordinator_source_file_bytes_sha256',
                                                               target_condition_without_lineage['source_file_bytes_sha256'])
        prior_control = ancestor_condition.get('coordinator_source_file_bytes_sha256',
                                               ancestor_condition['source_file_bytes_sha256'])
        if control_source == prior_control:
            errors.append('source_change_required')
        execution_source = target_condition_without_lineage['policy'].get('execution_source')
        if execution_source is not None:
            prior_execution = ancestor_condition['policy'].get('execution_source')
            prior_manifest = (prior_execution['source_file_bytes_sha256'] if prior_execution is not None
                              else ancestor_condition['source_file_bytes_sha256'])
            if (not isinstance(execution_source, dict)
                    or set(execution_source) != {'path', 'source_file_bytes_sha256', 'source_tree_sha256'}
                    or not isinstance(execution_source['path'], str) or not execution_source['path']
                    or execution_source['source_file_bytes_sha256'] != prior_manifest
                    or execution_source['source_tree_sha256'] != digest(prior_manifest)
                    or target_condition_without_lineage['source_file_bytes_sha256'] != prior_manifest):
                errors.append('execution_source_not_ancestor')
        elif ancestor_condition['policy'].get('execution_source') is not None:
            errors.append('execution_source_missing')
    except (KeyError, TypeError, ValueError) as exc:
        errors.append('condition_shape_invalid:' + str(exc))
    return errors


def _member(root: Path, relative: str):
    if not isinstance(relative, str) or not relative or '\\' in relative or relative.startswith('/') or any(
            part in ('', '.', '..') for part in relative.split('/')):
        raise ValueError('lineage_path_invalid:' + str(relative))
    raw = root
    for part in relative.split('/'):
        raw = raw / part
        if raw.is_symlink():
            raise ValueError('lineage_symlink_invalid:' + relative)
    path = contained(root, relative)
    if path.is_symlink() or not path.is_file():
        raise ValueError('lineage_member_invalid:' + relative)
    return path


def _group_files(root, condition, selected_job_ids=None):
    jobs = {j['job_id']: j for j in condition['jobs']}
    for name, kind in (('classification_groups', 'classification'), ('extraction_groups', 'local_extraction')):
        base = root / name
        if base.is_symlink():
            raise ValueError('group_cache_root_symlink:' + name)
        if base.exists():
            if not base.is_dir():
                raise ValueError('group_cache_root_invalid:' + name)
            allowed = {jid for jid, job in jobs.items() if job['kind'] == kind}
            for child in base.iterdir():
                if child.is_symlink() or not child.is_dir() or child.name not in allowed:
                    raise ValueError('out_of_plan_group_cache:' + str(child))
    selected = set(jobs) if selected_job_ids is None else set(selected_job_ids)
    if not selected <= set(jobs):
        raise ValueError('unknown_selected_job')
    result = []
    for job_id in sorted(selected):
        job = jobs[job_id]
        folder_name = ('classification_groups' if job['kind'] == 'classification' else
                       'extraction_groups' if job['kind'] == 'local_extraction' else None)
        for name in ('classification_groups', 'extraction_groups'):
            if (root / name).is_symlink() or (root / name / job_id).is_symlink():
                raise ValueError('foreign_group_cache_symlink:' + job_id)
            folder = contained(root, f'{name}/{job_id}')
            if not folder.exists():
                continue
            if name != folder_name or folder.is_symlink() or not folder.is_dir():
                raise ValueError('foreign_group_cache_directory:' + job_id)
            for path in folder.rglob('*'):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValueError('foreign_group_cache_entry:' + str(path))
                if path.is_file():
                    result.append(path.relative_to(root).as_posix())
    return sorted(result)


def _validate_groups(root, condition, sources, members, *, index_root=None):
    expected = set(members)
    by_job = {j['job_id']: j for j in condition['jobs']}
    task_map = tasks_for_config(condition['config'])
    index_cache = {}
    with replay_validation_scope():
        for job_id in sorted({p.split('/')[1] for p in members}):
            job = by_job[job_id]
            task = task_map['classification' if job['kind'] == 'classification' else 'extraction']
            profile = next(p for p in condition['config']['profiles'] if p['profile_id'] == job['profile_id'])
            paper = paper_context(condition, job, sources)['input']
            index = None
            if job['kind'] == 'local_extraction':
                paper_id = job['paper_id']
                if paper_id not in index_cache:
                    path = _member(index_root or root, ('indexes/' if index_root else 'lineage/indexes/') + digest(paper_id) + '.json')
                    candidate = read_json(path)
                    errors = verify_index(candidate, paper)
                    if errors:
                        raise ValueError('ancestor_index_invalid:' + ','.join(errors[:4]))
                    index_cache[paper_id] = candidate
                index = index_cache[paper_id]
            policy = (condition['config']['classification']['grouping'] if job['kind'] == 'classification'
                      else condition['config']['extraction_grouping'])
            module, groups = _context(task, paper, index, policy)
            module._check_cache_tree(root, job_id, groups)
            cap = _configuration(job)
            for group in groups:
                child = _child_job(module, job, group, index)
                history = _load_group_history(module, root, child, profile, task, paper, index, group, cap)
                expected.difference_update(module.record_ref(path, root)['path'] for _, path, _ in history)
        if expected:
            raise ValueError('unplanned_or_unchecked_groups:' + ','.join(sorted(expected)[:3]))


def _prove_original_indexes(root, condition, sources, members, *, prefix=''):
    """An extraction cache may only inherit the classifier's finalized index."""
    from .scheduler_tasks import verify_result
    papers = {j['paper_id'] for j in condition['jobs'] if j['kind'] == 'local_extraction'
              and any(p.split('/')[1] == j['job_id'] for p in members)}
    for paper_id in sorted(papers):
        job = next(j for j in condition['jobs'] if j['kind'] == 'classification' and j['paper_id'] == paper_id)
        folder = contained(root, prefix + 'results/' + job['job_id'])
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError('ancestor_classifier_result_missing:' + paper_id)
        attempts = sorted(int(m.group(1)) for p in folder.iterdir() if (m := ATTEMPT.fullmatch(p.name)))
        if not attempts or attempts != list(range(1, attempts[-1] + 1)):
            raise ValueError('ancestor_classifier_attempt_sequence_invalid:' + paper_id)
        number = attempts[-1]
        relative = f'{job["job_id"]}/attempt-{number}.json'
        envelope = read_json(_member(root, prefix + 'requests/' + relative))
        result = read_json(_member(root, prefix + 'results/' + relative))
        if envelope.get('condition') != condition['condition'] or envelope.get('job') != job:
            raise ValueError('ancestor_classifier_envelope_invalid:' + paper_id)
        errors, derived = verify_result(condition, envelope, result, sources, root)
        saved = read_json(_member(root, prefix + 'indexes/' + digest(paper_id) + '.json'))
        if errors or derived is None or derived != saved:
            raise ValueError('ancestor_classifier_index_derivation_invalid:' + paper_id + ':' + ','.join(errors[:3]))


def stage_group_cache(previous_root, target_root, sources, leases, *, expected_ancestor_condition,
                      target_condition_without_lineage) -> dict:
    """Stage a fresh run's raw cache; return the binding to place in its policy."""
    previous_root, target_root = Path(previous_root).resolve(), Path(target_root).resolve()
    sources, leases = Path(sources).resolve(), Path(leases).resolve()
    if previous_root == target_root or target_root.is_relative_to(previous_root) or previous_root.is_relative_to(target_root):
        raise ValueError('source_and_target_must_be_disjoint')
    ancestor = read_json(_member(previous_root, 'condition.json'))
    if ancestor.get('condition') != expected_ancestor_condition or not HEX.fullmatch(str(expected_ancestor_condition)):
        raise ValueError('ancestor_condition_identity_mismatch')
    errors = compatibility_errors(ancestor, target_condition_without_lineage)
    if errors:
        raise ValueError('incompatible_conditions:' + ','.join(errors))
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError('target_root_must_be_empty')
    with ExitStack() as stack:
        stack.enter_context(Lock(previous_root / 'coordinator.lock'))
        for worker in ancestor['policy']['workers']:
            stack.enter_context(Lock(previous_root / 'workers' / worker['worker_id'] / 'worker.lock'))
        for gpu in sorted(g for w in ancestor['policy']['workers'] for g in w['gpu_ids']):
            stack.enter_context(Lock(leases / (digest(gpu) + '.lock')))
        members = _group_files(previous_root, ancestor)
        _validate_groups(previous_root, ancestor, sources, members, index_root=previous_root)
        _prove_original_indexes(previous_root, ancestor, sources, members)
        target_root.mkdir(parents=True, exist_ok=True)
        ancestor_relative = 'lineage/ancestor-condition.json'
        index_paths = ['lineage/indexes/' + digest(j['paper_id']) + '.json' for j in ancestor['jobs']
                       if j['kind'] == 'local_extraction' and any(p.split('/')[1] == j['job_id'] for p in members)]
        index_paths = sorted(set(index_paths))
        parent_attempt_paths = []
        for paper_id in {j['paper_id'] for j in ancestor['jobs'] if j['kind'] == 'local_extraction'
                         and any(p.split('/')[1] == j['job_id'] for p in members)}:
            classifier = next(j for j in ancestor['jobs'] if j['kind'] == 'classification' and j['paper_id'] == paper_id)
            folder = contained(previous_root, 'results/' + classifier['job_id'])
            for result in folder.iterdir():
                if not ATTEMPT.fullmatch(result.name):
                    raise ValueError('ancestor_classifier_result_name_invalid')
                for kind in ('requests', 'results'):
                    parent_attempt_paths.append(f'lineage/{kind}/{classifier["job_id"]}/{result.name}')
        for relative in [ancestor_relative] + index_paths + sorted(parent_attempt_paths) + members:
            source_relative = ('condition.json' if relative == ancestor_relative else
                               relative.removeprefix('lineage/') if relative in index_paths or relative in parent_attempt_paths else relative)
            source = _member(previous_root, source_relative)
            destination = contained(target_root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if file_digest(source) != file_digest(destination):
                raise ValueError('source_changed_during_copy:' + source_relative)
        manifest = {'schema_version': VERSION, 'ancestor_condition': ancestor['condition'],
                    'target_coordinator_source_tree_sha256': digest(target_condition_without_lineage.get(
                        'coordinator_source_file_bytes_sha256', target_condition_without_lineage['source_file_bytes_sha256'])),
                    'jobs_sha256': digest(ancestor['jobs']),
                    'files': [identity(contained(target_root, p), target_root) for p in [ancestor_relative] + index_paths + sorted(parent_attempt_paths) + members]}
        manifest_path = target_root / 'lineage' / 'manifest.json'
        write_new(manifest_path, manifest)
        return {'path': 'lineage/manifest.json', 'file_bytes_sha256': file_digest(manifest_path),
                'bytes': manifest_path.stat().st_size, 'ancestor_condition': ancestor['condition']}


def verify_cache_lineage(root, condition, selected_job_ids=None, *, source_root=None) -> tuple[list[str], list[str]]:
    """Verify the sealed binding and all selected raw imports; return packet members."""
    root = Path(root).resolve()
    errors, paths = [], []
    try:
        binding = condition['policy']['group_cache_import']
        if not isinstance(binding, dict) or set(binding) != {'path', 'file_bytes_sha256', 'bytes', 'ancestor_condition'}:
            raise ValueError('lineage_binding_shape_invalid')
        if binding['path'] != 'lineage/manifest.json':
            raise ValueError('lineage_manifest_path_invalid')
        manifest_path = _member(root, binding['path'])
        if file_digest(manifest_path) != binding['file_bytes_sha256'] or manifest_path.stat().st_size != binding['bytes']:
            raise ValueError('lineage_manifest_binding_mismatch')
        manifest = read_json(manifest_path)
        if manifest.get('schema_version') != VERSION or manifest.get('ancestor_condition') != binding['ancestor_condition']:
            raise ValueError('lineage_manifest_identity_mismatch')
        ancestor_path = 'lineage/ancestor-condition.json'
        ancestor = read_json(_member(root, ancestor_path))
        if ancestor.get('condition') != binding['ancestor_condition']:
            raise ValueError('ancestor_identity_mismatch')
        target_unbound = copy.deepcopy(condition)
        target_unbound['policy'].pop('group_cache_import')
        target_unbound.pop('condition', None)
        # A provisional seal is needed only to run the pure semantic check.
        from .common import seal
        target_unbound = seal(target_unbound, 'condition')
        compatibility = compatibility_errors(ancestor, target_unbound)
        if compatibility:
            raise ValueError('lineage_compatibility:' + ','.join(compatibility))
        if manifest.get('target_coordinator_source_tree_sha256') != digest(condition.get(
                'coordinator_source_file_bytes_sha256', condition['source_file_bytes_sha256'])) or manifest.get('jobs_sha256') != digest(condition['jobs']):
            raise ValueError('lineage_target_binding_mismatch')
        entries = manifest.get('files')
        if not isinstance(entries, list) or not entries:
            raise ValueError('lineage_file_list_invalid')
        declared = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes', 'hash_mode'} or entry.get('hash_mode') != 'file_bytes':
                raise ValueError('lineage_file_entry_invalid')
            relative = entry['path']
            if relative in declared:
                raise ValueError('lineage_duplicate_member:' + relative)
            declared[relative] = entry
        index_paths = {p for p in declared if p.startswith('lineage/indexes/')}
        parent_paths = {p for p in declared if p.startswith(('lineage/requests/', 'lineage/results/'))}
        if any(not HEX.fullmatch(p.split('/')[-1].removesuffix('.json')) or not p.endswith('.json') for p in index_paths):
            raise ValueError('lineage_index_path_invalid')
        if any(len(p.split('/')) != 4 or not HEX.fullmatch(p.split('/')[2]) or not ATTEMPT.fullmatch(p.split('/')[3]) for p in parent_paths):
            raise ValueError('lineage_parent_attempt_path_invalid')
        if ancestor_path not in declared or set(declared) != {ancestor_path} | index_paths | parent_paths | set(_group_files_manifest_paths(declared)):
            raise ValueError('lineage_member_names_invalid')
        selected = set(j['job_id'] for j in condition['jobs']) if selected_job_ids is None else set(selected_job_ids)
        if not selected <= {j['job_id'] for j in condition['jobs']}:
            raise ValueError('unknown_selected_job')
        selected_groups = [p for p in declared if p != ancestor_path and p.split('/')[1] in selected]
        selected_groups = [p for p in selected_groups if p.startswith(('classification_groups/', 'extraction_groups/'))]
        selected_papers = {j['paper_id'] for j in condition['jobs'] if j['job_id'] in selected and j['kind'] == 'local_extraction'
                           and any(p.split('/')[1] == j['job_id'] for p in selected_groups)}
        selected_indexes = {'lineage/indexes/' + digest(p) + '.json' for p in selected_papers}
        selected_parent_jobs = {j['job_id'] for j in condition['jobs'] if j['kind'] == 'classification' and j['paper_id'] in selected_papers}
        selected_parent_paths = {p for p in parent_paths if p.split('/')[2] in selected_parent_jobs}
        if not selected_indexes <= index_paths:
            raise ValueError('lineage_index_missing')
        paths = [binding['path'], ancestor_path] + sorted(selected_indexes | selected_parent_paths | set(selected_groups))
        for relative in [ancestor_path] + sorted(selected_indexes | selected_parent_paths | set(selected_groups)):
            path = _member(root, relative)
            entry = declared[relative]
            if file_digest(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
                raise ValueError('lineage_member_identity_mismatch:' + relative)
        if source_root is not None and selected_groups:
            _validate_groups(root, condition, Path(source_root).resolve(), selected_groups)
            _prove_original_indexes(root, ancestor, Path(source_root).resolve(), selected_groups, prefix='lineage/')
        actual = set(_group_files(root, condition, selected))
        if not set(selected_groups) <= actual:
            raise ValueError('lineage_imported_member_missing')
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(str(exc))
    return errors, paths


def _group_files_manifest_paths(declared):
    for relative in declared:
        if relative == 'lineage/ancestor-condition.json' or relative.startswith(('lineage/indexes/', 'lineage/requests/', 'lineage/results/')):
            continue
        parts = relative.split('/')
        if (len(parts) != 4 or parts[0] not in {'classification_groups', 'extraction_groups'}
                or not HEX.fullmatch(parts[1]) or not HEX.fullmatch(parts[2]) or not ATTEMPT.fullmatch(parts[3])):
            raise ValueError('lineage_group_path_invalid:' + relative)
        yield relative
