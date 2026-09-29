"""Bounded, owned probe execution under an acknowledged single-GPU reservation."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import time

from .io import Lock, atomic_json, digest, file_sha, read, write_once
from .processes import identity, resources_free, same, terminate


def specification(config, key, manifest_path, expected_sha, now):
    if file_sha(manifest_path) != expected_sha:
        raise ValueError('probe_manifest_identity_changed')
    manifest = read(manifest_path)
    argv = manifest.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x or '\0' in x for x in argv):
        raise ValueError('probe_argv_must_be_explicit_list')
    pins = manifest.get('files')
    if not isinstance(pins, dict) or not pins or argv[0] not in pins:
        raise ValueError('probe_executable_pin_required')
    for path, sha in pins.items():
        if not Path(path).is_absolute() or file_sha(path) != sha:
            raise ValueError('probe_file_identity_changed')
    snapshot = read(Path(config['local_root']) / 'snapshot.json')
    matches = [r for r in snapshot['controls']['reservations'] if r['key'] == key]
    if len(matches) != 1:
        raise ValueError('reservation_not_acknowledged')
    reservation = matches[0]
    if len(reservation['gpu_ids']) != 1 or not 0 < reservation['expires_at'] - now <= 10800:
        raise ValueError('probe_reservation_invalid_or_expired')
    return manifest, reservation


def run(config_path, key, manifest_path, expected_sha):
    config = read(config_path)
    manifest, reservation = specification(config, key, manifest_path, expected_sha, time.time())
    local, root = Path(config['local_root']), Path(config['root'])
    audit = root / 'probes' / digest(key)
    gpu = reservation['gpu_ids'][0]
    worker = {'gpu_ids': [gpu]}
    deadline = reservation['expires_at']
    intent = dict(reservation=reservation, manifest=manifest, manifest_sha256=expected_sha)
    write_once(audit / 'intent.json', intent)
    with Lock(local / 'probes' / (digest(key) + '.lock')):
        # A command is launched only once. Ambiguous restarts require reconciliation.
        if (audit / 'launch-intent.json').exists():
            raise ValueError('probe_already_launched_reconcile_required')
        while time.time() < deadline:
            if resources_free(worker, config['local_leases'], config['legacy_leases']):
                break
            time.sleep(1)
        else:
            raise RuntimeError('reservation_expired_before_resource_release')
        with ExitStack() as leases:
            for namespace in ('local_leases', 'legacy_leases'):
                leases.enter_context(Lock(Path(config[namespace]) / (digest(gpu) + '.lock')))
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = gpu
            env['APPTAINERENV_CUDA_VISIBLE_DEVICES'] = gpu
            write_once(audit / 'launch-intent.json', dict(time=time.time(), owner=identity(os.getpid())))
            with (audit / 'output.log').open('xb') as log:
                child = subprocess.Popen(manifest['argv'], env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=log, start_new_session=True)
            expected = identity(child.pid)
            if expected is None:
                raise RuntimeError('probe_process_identity_unavailable')
            write_once(audit / 'process.json', expected)
            while child.poll() is None and time.time() < deadline:
                time.sleep(min(1, max(.01, deadline-time.time())))
            expired = child.poll() is None
            if expired:
                write_once(audit / 'cancel-intent.json', dict(time=time.time(), reason='reservation_expired', process=expected))
                terminate(expected)
            child.wait(timeout=35)
            write_once(audit / 'execution.json', dict(time=time.time(), returncode=child.returncode,
                                                     expired=expired, process_released=not same(expected)))
        # Neither an exit code nor our own lease release proves descendants freed the GPU.
        until = time.time() + 120
        while time.time() < until:
            if resources_free(worker, config['local_leases'], config['legacy_leases']):
                from .cli import create_command
                command = create_command(config, 'cancel_reservation', {'reservation_key': key},
                                         'probe-return-' + digest(key))
                write_once(audit / 'return-request.json', dict(time=time.time(), command=command,
                                                             resources_released=True))
                return
            time.sleep(1)
        atomic_json(audit / 'blocked.json', dict(time=time.time(), reason='probe_resource_release_unproven'))
        raise RuntimeError('probe_resource_release_unproven_production_stays_blocked')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--reservation-key', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    a = p.parse_args()
    run(a.config, a.reservation_key, a.manifest, a.manifest_sha256)


if __name__ == '__main__':
    main()
