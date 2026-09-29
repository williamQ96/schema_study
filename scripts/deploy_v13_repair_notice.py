"""Bind the existing signed Telegram/Codex transport to a fixed V13 probe job."""
from pathlib import Path
import argparse
import hashlib
import json
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from mercury_scheduler_v2_access import connect, execute
from deploy_v13_repair import REMOTE, PYTHON, LOCAL
from telegram_codex_bridge import read, atomic, telegram_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=int)
    args = parser.parse_args()
    candidate = 'candidate-' + format(args.candidate, '02d')
    job = REMOTE + '/' + candidate
    private = Path.home() / '.oaciss-secrets'
    prior = read(private / 'production-health-bridge-v13-local.json')
    secret_path = private / 'mercury-telegram.json'
    secret = read(secret_path)
    bot = telegram_call(secret, 'getMe', {})
    if (bot['id'] != prior['bot_id'] or bot['username'] != prior['bot_username']
            or secret.get('enabled') is not True or secret['chat_id'] != prior['chat_id']):
        raise ValueError('telegram_identity_mismatch')
    stem = 'v13-repair-' + candidate
    local_path = private / (stem + '-local.json')
    state = private / (stem + '-state')
    remote_path = '/home/users/williamq/.config/schema-study/' + stem + '.json'
    if local_path.exists() or state.exists():
        raise FileExistsError('notice_scope_already_exists')
    staged = LOCAL / candidate
    raw = (staged / 'config.json').read_bytes()
    source = read(staged / 'science-manifest.json')
    common = {'job_id': 'v13-repair-' + candidate, 'hmac_key': secrets.token_hex(32),
              **{k: prior[k] for k in ('bot_id', 'bot_username', 'chat_id')},
              'remote_job_root': job, 'remote_notice_root': job + '/notice',
              'remote_outbox': job + '/notice/outbox', 'remote_report': job + '/notice/terminal-report.json',
              'config_file_sha256': hashlib.sha256(raw).hexdigest(),
              'source_tree_sha256': source['source_tree_sha256']}
    local = {**common, **{k: prior[k] for k in ('repo', 'python_paths', 'codex_exe', 'rollout_path', 'thread_id')},
             'state_dir': str(state), 'legacy_state_dir': str(state / 'no-legacy-state'),
             'local_config_path': str(local_path), 'telegram_secret_path': str(secret_path), 'poll_seconds': 30,
             'instructions_path': str(ROOT / 'docs/v13_repair_probe_notifications.md'),
             'analysis_root': str(LOCAL / candidate / 'event-analysis')}
    remote = {**common, 'telegram_secret_path': '/home/users/williamq/.config/schema-study/mercury-telegram.json'}
    files = {n: ROOT / 'scripts' / n for n in ['watch_v13_repair_probe.py', 'telegram_codex_bridge.py']}
    gateway, client = connect()
    try:
        with client.open_sftp() as sftp:
            with sftp.open(job + '/config.json', 'rb') as stream:
                if stream.read() != raw:
                    raise ValueError('staged_config_identity_changed')
            sftp.mkdir(job + '/notice')
            sftp.mkdir(job + '/notice/source')
            sftp.mkdir(job + '/notice/outbox')
            for name, path in files.items():
                with sftp.open(job + '/notice/source/' + name, 'wx') as stream:
                    stream.write(path.read_bytes())
            with sftp.open(remote_path, 'wx') as stream:
                stream.write(json.dumps(remote))
            sftp.chmod(remote_path, 0o600)
    finally:
        client.close(); gateway.close()
    state.mkdir()
    atomic(local_path, local)
    code = '''import sys,json,subprocess,time
from pathlib import Path
j=Path(@@JOB@@);sys.path.insert(0,str(j/'runtime-v1'));sys.path.insert(0,str(j/'notice/source'))
from scheduler_v2.io import write_once
from scheduler_v2.processes import identity
import telegram_codex_bridge as p
c=p.read(@@CONFIG@@)
p.LABELS['bridge_test']='Mercury | V13 repair notification route verified; completion will trigger a read-only Codex diagnosis'
test=j/'notice/bridge-test-report.json';write_once(test,{'kind':'transport_test','job_id':c['job_id'],'config_file_sha256':c['config_file_sha256']})
receipt=p.publish(c,p.read(c['telegram_secret_path']),'bridge_test',test,j/'notice/outbox')
with (j/'notice/watcher.log').open('xb') as log:
 child=subprocess.Popen([@@PYTHON@@,str(j/'notice/source/watch_v13_repair_probe.py'),@@CONFIG@@],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
value={'identity':identity(child.pid),'time':time.time(),'test_event_id':receipt.stem,'job_id':c['job_id']};write_once(j/'notice/deployment.json',value);print(json.dumps(value))'''
    value = execute(code.replace('@@JOB@@', repr(job)).replace('@@CONFIG@@', repr(remote_path)).replace('@@PYTHON@@', repr(PYTHON)))
    atomic(staged / 'notice-deployment.json', value)
    print(json.dumps({**value, 'local_config': str(local_path), 'local_receiver_started': False}))


if __name__ == '__main__':
    main()
