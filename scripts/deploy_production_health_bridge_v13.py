"""Prepare the read-only health bridge for the fresh V13 Scheduler V2 run."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import secrets
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path('C:/Users/izayo/.oaciss-secrets')
REMOTE_BASE = '/home/users/williamq/schema-study-deployment-20260926'
PRODUCTION_ROOT = ('/storage/users/williamq/schema-study-deployment-20260926/'
                   'jobs/v13-full-20260928-v1/run')
BRIDGE_ROOT = REMOTE_BASE + '/bridges/v13-full-20260928-v1'
REMOTE_CONFIG = '/home/users/williamq/.config/schema-study/production-health-bridge-v13.json'
REMOTE_TELEGRAM_SECRET = '/home/users/williamq/.config/schema-study/mercury-telegram.json'
REMOTE_CONTROL_PYTHON = (REMOTE_BASE + '/jobs/matrix-scheduler-20260926-v7/'
                         'control-venv/bin/python')
JOB_ID = 'v13-full-20260928-v1'
WORKER_STATE_ROOT = '/var/tmp/schema-study-williamq/v13-full-20260928-v1'


def build_configs(condition_bytes, previous, *, expected_condition_sha256,
                  private=PRIVATE, repo=ROOT, hmac_key=None):
    """Build matching bridge configs without opening credentials or contacting HPC."""
    condition_doc = json.loads(condition_bytes)
    condition_sha = hashlib.sha256(condition_bytes).hexdigest()
    if (not isinstance(expected_condition_sha256, str) or len(expected_condition_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_condition_sha256)
            or condition_sha != expected_condition_sha256):
        raise ValueError('fresh_condition_file_hash_mismatch')
    condition = condition_doc['condition']
    if condition == previous.get('condition'):
        raise ValueError('fresh_condition_reuses_previous_identity')
    preprocessing = condition_doc.get('config', {}).get('preprocessing')
    if preprocessing != {'pipeline': 'V13', 'mineru_enabled': True, 'auxiliary_enabled': False}:
        raise ValueError('v13_mineru_preprocessing_required')
    key = hmac_key if hmac_key is not None else secrets.token_hex(32)
    if (not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key)
            or key == previous.get('hmac_key')):
        raise ValueError('new_private_hmac_key_required')
    shared = {
        'protocol': 'mercury-production-health/v1', 'job_id': JOB_ID,
        'condition': condition, 'condition_file_bytes_sha256': condition_sha,
        'production_root': PRODUCTION_ROOT,
        'hmac_key': key,
        **{key: previous[key] for key in ('bot_id', 'bot_username', 'chat_id')},
    }
    private = Path(private)
    repo = Path(repo)
    local_path = private / 'production-health-bridge-v13-local.json'
    artifact = repo / 'data/experiments/production_health_bridge_v13_2026_09_28_v1'
    local = {
        **shared,
        **{key: previous[key] for key in ('repo', 'python_paths', 'codex_exe', 'rollout_path', 'thread_id')},
        'scheduler_version': 'v2', 'worker_state_root': WORKER_STATE_ROOT,
        'state_dir': str(private / 'production-health-bridge-v13-state'),
        'remote_bridge_root': BRIDGE_ROOT, 'local_config_path': str(local_path),
        'poll_seconds': 30, 'diagnostic_timeout_seconds': 1200,
        'telegram_secret_path': str(private / 'mercury-telegram.json'),
        'remote_telegram_secret_path': REMOTE_TELEGRAM_SECRET,
        'instructions_path': str(repo / 'docs/production_health_codex_bridge_v13.md'),
        'analysis_root': str(artifact / 'analysis'),
    }
    remote = {
        **shared, 'bridge_root': BRIDGE_ROOT,
        'lock_module_root': BRIDGE_ROOT + '/source', 'poll_seconds': 20,
        'telegram_secret_path': REMOTE_TELEGRAM_SECRET,
    }
    return local, remote, condition_sha


def _mkdirs(sftp, remote_path):
    current = ''
    for part in remote_path.strip('/').split('/'):
        current += '/' + part
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def deploy(*, expected_condition_sha256):
    # Network and credential imports stay inside the explicit operator action.
    sys.path[:0] = [str(PRIVATE / 'python-deps'), str(PRIVATE)]
    from oaciss_access import mercury
    from telegram_codex_bridge import read, sha, atomic, telegram_call

    previous = read(PRIVATE / 'production-health-bridge-local.json')
    secret = read(PRIVATE / 'mercury-telegram.json')
    bot = telegram_call(secret, 'getMe', {})
    if (bot['id'] != previous['bot_id'] or bot['username'] != previous['bot_username']
            or secret.get('chat_id') != previous['chat_id'] or secret.get('enabled') is not True):
        raise ValueError('telegram_identity_mismatch')
    local_path = PRIVATE / 'production-health-bridge-v13-local.json'
    if local_path.exists():
        raise FileExistsError('new_v13_configuration_already_exists')
    remote_control_path = PurePosixPath(REMOTE_CONFIG)

    files = {name: ROOT / 'scripts' / name for name in (
        'production_health_events.py', 'production_health_publisher.py', 'telegram_codex_bridge.py')}
    files['bridge_lock.py'] = ROOT / 'four_category/scheduler_io.py'
    source_manifest = {'files': {name: sha(path) for name, path in files.items()}}
    gateway, connection = mercury()
    try:
        with connection.open_sftp() as sftp:
            with sftp.open(PRODUCTION_ROOT + '/condition.json', 'rb') as stream:
                raw = stream.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError('condition_file_too_large')
            local, remote, condition_sha = build_configs(
                raw, previous, expected_condition_sha256=expected_condition_sha256,
                private=PRIVATE, repo=ROOT)
            for existing in (BRIDGE_ROOT, REMOTE_CONFIG):
                try:
                    sftp.stat(existing)
                except OSError:
                    pass
                else:
                    raise FileExistsError('v13_remote_path_already_exists:' + existing)
            bridge_parent = BRIDGE_ROOT.rsplit('/', 1)[0]
            _mkdirs(sftp, bridge_parent)
            sftp.mkdir(BRIDGE_ROOT)
            for name in ('source', 'events', 'reports'):
                sftp.mkdir(BRIDGE_ROOT + '/' + name)
            for name, path in files.items():
                sftp.put(str(path), BRIDGE_ROOT + '/source/' + name)
            with sftp.open(BRIDGE_ROOT + '/code-manifest.json', 'wx') as stream:
                stream.write(json.dumps(source_manifest))
            _mkdirs(sftp, str(remote_control_path.parent))
            with sftp.open(str(remote_control_path), 'wx') as stream:
                stream.write(json.dumps(remote))
            sftp.chmod(str(remote_control_path), 0o600)

        atomic(local_path, local)
        code = '''from pathlib import Path
import json,subprocess,time
b=Path(BRIDGE)
with (b/'publisher.log').open('x') as log:
 p=subprocess.Popen([CONTROL,str(b/'source/production_health_publisher.py'),CONFIG],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'publisher_pid':p.pid,'bridge_root':str(b),'time':time.time(),'production_restarts':0}))'''
        code = (code.replace('BRIDGE', repr(BRIDGE_ROOT))
                .replace('CONTROL', repr(REMOTE_CONTROL_PYTHON))
                .replace('CONFIG', repr(str(remote_control_path))))
        _, stdout, stderr = connection.exec_command(
            shlex.quote(REMOTE_CONTROL_PYTHON) + ' -c ' + shlex.quote(code), timeout=45)
        output, error = stdout.read().decode(), stderr.read().decode()
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError('publisher_launch_failed:' + error[:200])
        receipt = json.loads(output)
        artifact = ROOT / 'data/experiments/production_health_bridge_v13_2026_09_28_v1'
        atomic(artifact / 'deployment.json', {**receipt, 'source_manifest': source_manifest,
                                              'condition': local['condition'],
                                              'condition_file_bytes_sha256': condition_sha})
        print(json.dumps(receipt))
    finally:
        connection.close()
        gateway.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-condition-sha256', required=True)
    args = parser.parse_args(argv)
    deploy(expected_condition_sha256=args.expected_condition_sha256)


if __name__ == '__main__':
    main()
