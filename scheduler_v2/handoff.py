"""One-time audited V1-to-V2 authority transfer; no scientific artifacts are edited."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from .io import Lock, atomic_json, file_sha, optional, read, write_once
from .processes import identity, same, resources_free, gpu_processes


def released_by_this_deployment(worker, config):
    """An unrelated account may occupy a released card; never evict or adopt it."""
    if resources_free(worker, config['local_leases'], config['legacy_leases'], observe=bool(worker['gpu_ids'])):
        return True, []
    if not resources_free(worker, config['local_leases'], config['legacy_leases'], observe=False):
        return False, []
    foreign = []
    for gpu, pid in gpu_processes():
        if gpu not in worker['gpu_ids']:
            continue
        observed = identity(pid)
        if observed is None or observed['uid'] == os.getuid():
            return False, []
        foreign.append(dict(gpu=gpu, pid=pid, uid=observed['uid'], start_ticks=observed['start_ticks']))
    return True, foreign


def frozen_handoff(config):
    spec = importlib.util.spec_from_file_location('frozen_compatibility_handoff', config['legacy_handoff_module'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stop_controllers(config, audit, legacy):
    if (audit / 'control-transferred.json').exists():
        return
    state = Path(config['legacy_handoff_state'])
    controller = read(state / 'controller-process.json')['identity']
    supervisor = read(state / 'supervisor-ready.json')['identity']
    write_once(audit / 'control-transfer-intent.json', dict(controller=controller, supervisor=supervisor))
    if same(controller):
        legacy.send_signal(controller, signal.SIGSTOP)
        deadline = time.time() + 5
        while same(controller) and identity(controller['pid'])['state'] != 'T' and time.time() < deadline:
            time.sleep(.05)
        if not same(controller) or identity(controller['pid'])['state'] != 'T':
            raise RuntimeError('legacy_controller_not_stopped')
        try:
            started = time.monotonic()
            def bounded(status, remaining, total):
                if time.monotonic() - started > 5:
                    raise RuntimeError('legacy_sqlite_backup_busy')
            with sqlite3.connect('file:' + config['ancestor_root'] + '/scheduler.sqlite?mode=ro', uri=True) as src:
                with sqlite3.connect(audit / 'legacy-ledger.sqlite') as dest:
                    src.backup(dest, pages=100, progress=bounded, sleep=.1)
                    if dest.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                        raise ValueError('legacy_ledger_backup_invalid')
        except BaseException:
            if same(controller):
                legacy.send_signal(controller, signal.SIGCONT)
            raise
        if same(supervisor):
            legacy.send_signal(supervisor, signal.SIGTERM)
        # Only the identified CPU controller. GPU workers keep their current request.
        legacy.send_signal(controller, signal.SIGKILL)
    # Also stop the pinned supervisor if its controller had already exited.
    # Any independently restarted controller must still release the authority lock.
    if same(supervisor):
        legacy.send_signal(supervisor, signal.SIGTERM)
    deadline = time.time() + 10
    while same(controller) and time.time() < deadline:
        time.sleep(.1)
    if same(controller):
        raise RuntimeError('legacy_dispatch_authority_still_live')
    write_once(audit / 'control-transferred.json', dict(time=time.time(), controller=controller, supervisor=supervisor))


def stop_old_watchdog(config, audit):
    root = Path(config['ancestor_root']).resolve()
    stopped = []
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():
            continue
        row = identity(int(p.name))
        if not row or row['uid'] != os.getuid() or row['state'] == 'Z':
            continue
        argv = row['argv']
        if ('high_fidelity_schema_study.four_category.queue_health' in argv and '--root' in argv
                and Path(argv[argv.index('--root') + 1]).resolve() == root):
            write_once(audit / ('watchdog-stop-' + str(row['pid']) + '.json'), row)
            if same(row):
                os.kill(row['pid'], signal.SIGTERM)
                stopped.append(row['pid'])
    return stopped


def release_worker(config, worker, expected, audit, legacy):
    wid = worker['worker_id']
    receipt_path = audit / ('released-' + wid + '.json')
    if receipt_path.exists():
        if same(expected):
            raise ValueError('released_worker_still_live')
        released, _ = released_by_this_deployment(worker, config)
        if not released:
            raise ValueError('released_resource_reoccupied')
        return
    old = Path(config['ancestor_root'])
    inbox_path = old / 'workers' / wid / 'inbox.json'
    envelope = read(inbox_path) if inbox_path.exists() else None
    group_root = old / ('classification_groups' if envelope and envelope['job']['kind'] == 'classification' else 'extraction_groups') / (envelope['job_id'] if envelope else 'none')
    intent_path = audit / ('drain-intent-' + wid + '.json')
    intent = optional(intent_path)
    if intent is None:
        started = time.time()
        intent = dict(worker=expected, envelope=envelope,
                      initial_records=sorted(str(p) for p in group_root.rglob('attempt-*.json')),
                      time=started, deadline=started+900)
        write_once(intent_path, intent)
    if intent['worker'] != expected or intent['envelope'] != envelope:
        raise ValueError('handoff_resume_identity_conflict')
    initial = {Path(p) for p in intent['initial_records']}
    started = intent['time']
    while same(expected):
        hb_path = old / 'workers' / wid / 'heartbeat.json'
        heartbeat = read(hb_path) if hb_path.exists() else {}
        new_records = set(group_root.rglob('attempt-*.json')) - initial
        boundary = bool(new_records) or heartbeat.get('phase') == 'idle' or not worker['gpu_ids']
        atomic_json(audit / 'handoff-status.json', dict(time=time.time(), phase='draining', worker_id=wid,
                    group_index=heartbeat.get('group_index'), elapsed_s=time.time()-started))
        if boundary or time.time() - started >= 900:
            if not legacy.stop_between_writes(expected):
                time.sleep(.1)
                continue
            pins = {str(p): file_sha(p) for p in group_root.rglob('attempt-*.json') if read(p) is not None}
            suspension_path = audit / ('suspension-' + wid + '.json')
            existing = optional(suspension_path)
            if existing is not None and (existing['worker'] != expected or existing['envelope'] != envelope):
                raise ValueError('suspension_identity_conflict')
            if existing is None:
                write_once(suspension_path, dict(time=time.time(), worker=expected,
                        saved_group_pins=pins, envelope=envelope, heartbeat=heartbeat,
                        inflight_unsaved_generation_cancelled=True, retry_budget_consumed=False))
            # Pending SIGTERM is delivered after SIGCONT. No process-group signals.
            if same(expected):
                os.kill(expected['pid'], signal.SIGTERM)
            if same(expected):
                os.kill(expected['pid'], signal.SIGCONT)
            deadline = time.time() + 30
            while same(expected) and time.time() < deadline:
                time.sleep(.1)
            if same(expected):
                os.kill(expected['pid'], signal.SIGKILL)
            break
        time.sleep(1)
    deadline = time.time() + 120
    while time.time() < deadline:
        released, foreign = released_by_this_deployment(worker, config) if not same(expected) else (False, [])
        if released:
            write_once(receipt_path, dict(time=time.time(), worker_id=wid, identity=expected,
                                          resource_release_verified=True, available_for_launch=not foreign,
                                          foreign_gpu_processes=foreign))
            return
        time.sleep(1)
    raise RuntimeError('legacy_resource_release_unproven:' + wid)


def run(config_path):
    config = read(config_path)
    root = Path(config['root'])
    audit = root.parent / 'handoff'
    audit.mkdir(parents=True, exist_ok=True)
    legacy = frozen_handoff(config)
    plan = legacy.load_plan(Path(config['legacy_handoff_state']))
    legacy.verify_pins(plan)
    baseline = read(config['baseline_path'])
    legacy.verify_preserved(baseline['artifact_pins'])
    if not (root / 'import-latest.json').exists():
        raise ValueError('replayed_import_required_before_handoff')
    condition = read(root / 'condition.json')
    if condition['condition'] != baseline['condition']:
        raise ValueError('handoff_condition_mismatch')
    inventory_path = audit / 'legacy-workers.json'
    if not inventory_path.exists():
        write_once(inventory_path, legacy.find_workers(Path(config['ancestor_root']), condition))
    inventory = read(inventory_path)
    with Lock(Path(config['local_root']) / 'handoff.lock'):
        stop_controllers(config, audit, legacy)
        with Lock(config['authority_lock']):
            stop_old_watchdog(config, audit)
            # Idle workers first; other workers retain their existing requests until their turn.
            workers = sorted(condition['policy']['workers'], key=lambda w: bool(
                read(Path(config['ancestor_root'])/'workers'/w['worker_id']/'heartbeat.json').get('job_id')))
            for worker in workers:
                release_worker(config, worker, inventory[worker['worker_id']], audit, legacy)
            legacy.verify_preserved(baseline['artifact_pins'])
            legacy.verify_pins(plan)
            atomic_json(audit / 'handoff-status.json', dict(time=time.time(), phase='final_delta_replay'))
            from .migrate import prepare
            if config.get('ancestor_quiescent') is not True:
                raise ValueError('final_sync_requires_quiescent_ancestor')
            prepare(config_path)
            write_once(audit / 'completed.json', dict(time=time.time(), original_condition=condition['condition'],
                        historical_hashes_unchanged=True, old_dispatch_stopped=True,
                        workers_released=list(inventory), original_attempt_records_unchanged=True))
        # Transfer lock has been released before the new coordinator starts.
        # Keep launch invocation explicit in deployment tooling; the handoff itself
        # never silently assumes an environment or Telegram credential path.
        atomic_json(audit / 'handoff-status.json', dict(time=time.time(), phase='ready_for_v2'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    run(parser.parse_args().config)


if __name__ == '__main__':
    main()
