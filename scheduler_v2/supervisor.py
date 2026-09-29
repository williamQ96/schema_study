"""Standalone coordinator and independent watchdog process supervisor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import Lock, atomic_json, optional, read
from .processes import identity, same


def _alive(path):
    process = optional(path)
    return process if process and same(process) else None


def _launch(command, log_path, env=None):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open('ab') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   env=env, start_new_session=True)
    return identity(process.pid)


def _intended_process(intent_path):
    """Find a launched child whose PID receipt was lost in a controller crash."""
    intent = optional(intent_path)
    command = intent.get('command') if isinstance(intent, dict) else None
    proc = Path('/proc')
    if not isinstance(command, list) or not proc.is_dir():
        return None
    current_uid = os.getuid() if hasattr(os, 'getuid') else None
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        row = identity(int(entry.name))
        if (row and row.get('state') != 'Z' and row.get('argv') == command
                and (current_uid is None or row.get('uid') == current_uid)):
            return row
    return None


def _fenced_launch(local, role, command, log, env=None):
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    with Lock(local / 'supervisor.lock'):
        pid_path = local / (role + '-process.json')
        current = _alive(pid_path)
        if current:
            return current
        intent_path = local / (role + '-launch-intent.json')
        recovered = _intended_process(intent_path)
        if recovered:
            atomic_json(pid_path, recovered)
            return recovered
        atomic_json(intent_path, {'role': role, 'command': command, 'time': time.time()})
        launched = _launch(command, log, env)
        if not launched:
            raise RuntimeError(role + '_identity_unavailable')
        atomic_json(pid_path, launched)
        return launched


def _ensure_watchdog(local, command, log, env, times, now, restart_limit):
    local = Path(local)
    receipt = local / 'watchdog-process.json'
    alive = _alive(receipt)
    if alive:
        return alive, times, 'running'
    if _intended_process(local / 'watchdog-launch-intent.json'):
        return _fenced_launch(local, 'watchdog', command, log, env=env), times, 'running'
    had_receipt = receipt.exists()
    if had_receipt and len(times) >= restart_limit:
        atomic_json(local / 'watchdog-restarts.json', {'times': times,
                    'status': 'restart_limit_reached', 'updated_at': now})
        return None, times, 'restart_limit_reached'
    if had_receipt:
        times = [*times, now]
    process = _fenced_launch(local, 'watchdog', command, log, env=env)
    atomic_json(local / 'watchdog-restarts.json', {'times': times,
                'status': 'running', 'updated_at': now})
    return process, times, 'running'


def _supervise(config_path, *, telegram_secret=None, restart_window=3600, restart_limit=3):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    local, root = Path(config['local_root']), Path(config['root'])
    local.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(Path(config['runtime_source']).resolve()),
                                         str(Path(config['science_source']).resolve()),
                                         env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    coordinator_cmd = [sys.executable, '-m', 'scheduler_v2.runtime', '--config', str(config_path)]
    watchdog_cmd = [sys.executable, '-m', 'scheduler_v2.watchdog',
                    '--root', str(root), '--local', str(local), '--science-source', config['science_source']]
    if telegram_secret:
        watchdog_cmd += ['--telegram-secret', str(Path(telegram_secret).resolve())]
    watchdog_state = optional(local / 'watchdog-restarts.json') or {'times': []}
    watchdog_times = [t for t in watchdog_state.get('times', []) if time.time() - t < restart_window]
    watchdog, watchdog_times, _ = _ensure_watchdog(local, watchdog_cmd, local / 'watchdog.log', env,
                                                   watchdog_times, time.time(), restart_limit)
    restarts = optional(local / 'coordinator-restarts.json') or {'times': []}
    times = [t for t in restarts.get('times', []) if time.time() - t < restart_window]
    coordinator = (_alive(local / 'coordinator-process.json') or
                   _fenced_launch(local, 'coordinator', coordinator_cmd, local / 'coordinator.log', env=env))
    finalizer_attempts = 0
    while True:
        time.sleep(1)
        times = [t for t in times if time.time() - t < restart_window]
        watchdog_times = [t for t in watchdog_times if time.time() - t < restart_window]
        watchdog, watchdog_times, _ = _ensure_watchdog(local, watchdog_cmd, local / 'watchdog.log', env,
                                                       watchdog_times, time.time(), restart_limit)
        snapshot = optional(local / 'snapshot.json')
        if snapshot and snapshot.get('terminal'):
            final_status = optional(root / 'finalization-status.json')
            if final_status and final_status.get('status') == 'pass':
                return {'status': 'complete', 'finalization': final_status, 'watchdog': watchdog}
            if finalizer_attempts < 3 and not _alive(local / 'finalizer-process.json'):
                finalizer_attempts += 1
                finalizer_cmd = [sys.executable, '-m', 'scheduler_v2.finalizer', '--config', str(config_path)]
                _fenced_launch(local, 'finalizer', finalizer_cmd, local / 'finalizer.log', env=env)
            if finalizer_attempts >= 3 and final_status and final_status.get('status') == 'fail':
                return {'status': 'finalization_failed', 'finalization': final_status, 'watchdog': watchdog}
            if finalizer_attempts >= 3 and not _alive(local / 'finalizer-process.json'):
                return {'status': 'finalization_failed', 'finalization': final_status, 'watchdog': watchdog}
        if same(coordinator):
            continue
        if snapshot and snapshot.get('terminal'):
            continue
        if len(times) >= restart_limit:
            atomic_json(local / 'coordinator-restarts.json', {'times': times, 'status': 'restart_limit_reached', 'updated_at': time.time()})
            return {'status': 'restart_limit_reached', 'watchdog': watchdog, 'coordinator': coordinator}
        times.append(time.time())
        atomic_json(local / 'coordinator-restarts.json', {'times': times, 'status': 'restarting', 'updated_at': time.time()})
        coordinator = _fenced_launch(local, 'coordinator', coordinator_cmd, local / 'coordinator.log', env=env)


def start(config_path, *, telegram_secret=None, restart_window=3600, restart_limit=3):
    config = read(config_path)
    # Distinct from the short launch lock, which is also used for child roles.
    with Lock(Path(config['local_root']) / 'supervision-owner.lock'):
        return _supervise(config_path, telegram_secret=telegram_secret,
                          restart_window=restart_window, restart_limit=restart_limit)


def start_background(config_path, *, telegram_secret=None):
    """Launch one supervisor process and return immediately."""
    config_path = Path(config_path).resolve()
    config = read(config_path)
    local = Path(config['local_root'])
    command = [sys.executable, '-m', 'scheduler_v2.supervisor', '--config', str(config_path)]
    if telegram_secret:
        command += ['--telegram-secret', str(Path(telegram_secret).resolve())]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(Path(config['runtime_source']).resolve()),
                                         str(Path(config['science_source']).resolve()),
                                         env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    return _fenced_launch(local, 'supervisor', command, local / 'supervisor.log', env=env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--telegram-secret', type=Path, help='Secret file passed directly to the independent watchdog')
    args = parser.parse_args(argv)
    print(json.dumps(start(args.config, telegram_secret=args.telegram_secret), sort_keys=True))


if __name__ == '__main__':
    main()
