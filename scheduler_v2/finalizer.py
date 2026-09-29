"""Independent paper packet freeze, with byte and scientific derivation gates."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys
import uuid

from .io import Lock, atomic_json, digest, file_sha, read, write_once
from .migrate import initialize_verifier
from .state import Ledger, TERMINAL


def verify_packet(packet, sources, *, expected_condition):
    packet = Path(packet)
    manifest = read(packet / 'manifest.json')
    if manifest.get('schema_version') != 'scheduled-paper-packet/v3':
        raise ValueError('packet_version_invalid')
    expected = set(manifest['files'])
    actual = {p.relative_to(packet).as_posix() for p in packet.rglob('*') if p.is_file()}
    if actual != expected | {'manifest.json'}:
        raise ValueError('packet_member_set_mismatch')
    for name, sha in manifest['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or (packet / rel).is_symlink() or file_sha(packet / rel) != sha:
            raise ValueError('packet_member_changed:' + name)
    condition = read(packet / 'condition.json')
    if condition['condition'] != expected_condition:
        raise ValueError('packet_condition_mismatch')
    from high_fidelity_schema_study.four_category.common import seal_errors
    from high_fidelity_schema_study.four_category.scheduled_packet import _jobs, _selection
    from high_fidelity_schema_study.four_category.complete_groups import verify_complete
    from high_fidelity_schema_study.four_category.execution_artifacts import build_artifact
    from high_fidelity_schema_study.four_category.scheduler_tasks import verify_result
    from .science import context
    if seal_errors(condition, 'condition'):
        raise ValueError('packet_condition_invalid')
    jobs = {j['job_id']: j for j in _jobs(condition)}
    selection = _selection(condition, manifest['paper_id'])
    if selection['job_ids'] != manifest['job_ids']:
        raise ValueError('packet_selection_invalid')
    terminal_states = manifest.get('terminal_states')
    if not isinstance(terminal_states, dict) or set(terminal_states) != set(selection['job_ids']):
        raise ValueError('packet_terminal_state_set_invalid')
    journal = sorted((packet / 'scheduler_v2' / manifest['deployment_id'] / 'events').glob('*.json'))
    events = [read(path) for path in journal]
    previous = None
    for order, event in enumerate(events, 1):
        if event.get('seq') != order or event.get('prev_sha') != previous:
            raise ValueError('packet_journal_chain_invalid')
        previous = digest(event)
    for jid in selection['job_ids']:
        job = jobs[jid]
        terminal = terminal_states[jid]
        if terminal['status'] not in TERMINAL:
            raise ValueError('packet_job_not_terminal')
        if terminal['status'] == 'blocked_dependency':
            dependencies = set(job.get('dependencies', []))
            if (terminal.get('result_ref') is not None or terminal.get('index_ref') is not None
                    or any(status != 'blocked_dependency' for status in terminal.get('group_statuses', []))
                    or (packet / 'v2_results' / (jid + '.json')).exists()):
                raise ValueError('packet_blocked_job_shape_invalid')
            proofs = [e for e in events if e.get('kind') == 'blocked_dependency'
                      and e.get('job_id') == jid and e.get('dependency_id') in dependencies]
            if len(proofs) != 1:
                raise ValueError('packet_blocked_dependency_proof_missing')
            dependency = terminal_states.get(proofs[0]['dependency_id'])
            if (dependency is None or dependency['status'] not in TERMINAL
                    or dependency['status'] == 'success' or dependency.get('index_ref') is not None):
                raise ValueError('packet_blocked_dependency_not_failed')
            continue
        if job['kind'] == 'soft_reference':
            continue
        if job['kind'] == 'dataset_parse':
            reference = manifest['dataset_results'][jid]
            if terminal['result_ref']['path'] != reference:
                raise ValueError('packet_dataset_terminal_ref_mismatch')
            result = read(packet / reference)
            request = read(packet / 'requests' / jid / ('attempt-' + str(result['attempt']) + '.json'))
            errors, _ = verify_result(condition, request, result, Path(sources), packet)
            if errors:
                raise ValueError('packet_dataset_derivation_invalid:' + repr(errors[:3]))
            continue
        result = read(packet / 'v2_results' / (jid + '.json'))
        if (result['job_id'] != jid or result['condition'] != expected_condition
                or result['body']['status'] != terminal['status']):
            raise ValueError('packet_parent_identity_invalid')
        ctx = context(condition, job, sources, packet)
        errors, records = verify_complete(job, ctx['profile'], ctx['task'], ctx['paper'], ctx['index'],
                                          ctx['policy'], 1, result['body'], packet)
        if errors:
            raise ValueError('packet_group_derivation_invalid:' + repr(errors[:3]))
        artifact = build_artifact(job, ctx['profile'], ctx['task'], ctx['paper'], ctx['index'], ctx['policy'], records)
        artifact_ref = result['artifact']
        if file_sha(packet / artifact_ref['path']) != artifact_ref['file_bytes_sha256'] or read(packet / artifact_ref['path']) != artifact:
            raise ValueError('packet_artifact_derivation_invalid')
    return dict(byte_integrity='pass', derivation_validity='pass', semantic_accuracy=None,
                semantic_evaluation='pending_independent_reference', deferred_soft_reference=True)


def build_paper(config, ledger_snapshot, paper_id):
    root = Path(config['root'])
    condition = read(root / 'condition.json')
    from high_fidelity_schema_study.four_category.scheduled_packet import _selection
    selection = _selection(condition, paper_id)
    rows = {j['job_id']: j for j in ledger_snapshot['jobs']}
    destination = root.parent / 'packets' / paper_id
    if destination.exists():
        return verify_packet(destination, config['sources'], expected_condition=condition['condition'])
    staging = destination.with_name('.p-' + uuid.uuid4().hex[:12])
    staging.mkdir(parents=True)
    members = {'condition.json', 'deployment.json', 'group_plans.json', 'import-latest.json'}
    datasets = {}
    terminal_states = {}
    for jid in selection['job_ids']:
        row = rows[jid]
        if row['status'] not in TERMINAL:
            raise ValueError('paper_not_terminal')
        terminal_states[jid] = dict(status=row['status'], result_ref=row['result_ref'],
                                    index_ref=row['index_ref'],
                                    group_statuses=[group['status'] for group in row['groups']])
        job = row['job']
        for directory in ('requests', 'results', 'classification_groups', 'extraction_groups'):
            members.update(p.relative_to(root).as_posix() for p in (root / directory / jid).rglob('*.json'))
        if row['status'] == 'blocked_dependency':
            continue
        if job['kind'] in {'classification', 'local_extraction'}:
            result_path = 'v2_results/' + jid + '.json'
            members.add(result_path)
            members.add(read(root / result_path)['artifact']['path'])
        elif job['kind'] == 'dataset_parse':
            datasets[jid] = row['result_ref']['path']
    # Include the complete migration graph/journal so origin and operational
    # continuation remain independently auditable, including old failures.
    for directory in ('imports', 'ancestors', 'lineage', 'scheduler_v2'):
        members.update(p.relative_to(root).as_posix() for p in (root / directory).rglob('*.json'))
    for name in sorted(members):
        source, target = root / name, staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    manifest = dict(schema_version='scheduled-paper-packet/v3', paper_id=paper_id, job_ids=selection['job_ids'],
                    scientific_condition=condition['condition'], deployment_id=config['deployment_id'],
                    files={n: file_sha(staging / n) for n in sorted(members)}, hash_mode='file_bytes_sha256',
                    dataset_results=datasets, terminal_states=terminal_states,
                    external_source_root_parameter='sources', semantic_accuracy=None,
                    authority='auditable_group_outputs_including_rejections_not_semantic_gold')
    write_once(staging / 'manifest.json', manifest)
    report = verify_packet(staging, config['sources'], expected_condition=condition['condition'])
    os.rename(staging, destination)
    return report


def run(config_path):
    config = read(config_path)
    root, local = Path(config['root']), Path(config['local_root'])
    initialize_verifier(config['science_source'], local / 'verification.key')
    with Lock(local / 'finalizer.lock'), Ledger(local, root, config['deployment_id']) as ledger:
        snapshot = ledger.snapshot()
        if not all(j['status'] in TERMINAL for j in snapshot['jobs']):
            raise ValueError('finalization_requires_terminal_ledger')
        reports = {}
        try:
            for paper in read(root / 'condition.json')['corpus']['papers']:
                reports[paper['paper_id']] = build_paper(config, snapshot, paper['paper_id'])
                atomic_json(root / 'finalization-status.json', dict(status='running', reports=reports))
            release = dict(version='scheduler-v2-packet-release/v1', deployment_id=config['deployment_id'],
                           packets={pid: file_sha(root.parent/'packets'/pid/'manifest.json') for pid in reports},
                           reports=reports, semantic_accuracy=None)
            write_once(root.parent / 'packet-release.json', release)
            atomic_json(root / 'finalization-status.json', dict(status='pass', reports=reports,
                         release_path=str(root.parent / 'packet-release.json')))
        except Exception as exc:
            atomic_json(root / 'finalization-status.json', dict(status='fail', reports=reports, error=str(exc)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)


if __name__ == '__main__':
    main()
