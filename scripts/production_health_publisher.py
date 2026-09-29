"""Read-only production watchdog observer; writes only its own bridge outbox."""
import argparse
import sys
import time
from pathlib import Path
import production_health_events as protocol
from telegram_codex_bridge import read, atomic, sha, telegram_call


def run(config_path, *, once=False):
    config=read(config_path)
    sys.path.insert(0,config['lock_module_root'])
    from bridge_lock import Lock
    root=Path(config['production_root']).resolve()
    out=Path(config['bridge_root']).resolve()
    if out==root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('bridge_output_must_be_separate_from_production')
    with Lock(out/'publisher.lock'):
        for name,expected in read(out/'code-manifest.json')['files'].items():
            if sha(out/'source'/name)!=expected:
                raise ValueError('publisher_code_identity')
        bot_verified=False
        while not (out/'STOP').exists():
            try:
                if not bot_verified:
                    secret=read(config['telegram_secret_path'])
                    bot=telegram_call(secret,'getMe',{})
                    if (bot['id']!=config['bot_id'] or bot['username']!=config['bot_username']
                            or secret.get('enabled') is not True or secret['chat_id']!=config['chat_id']):
                        raise ValueError('publisher_bot_identity')
                    bot_verified=True
                if sha(root/'condition.json')!=config['condition_file_bytes_sha256']:
                    raise ValueError('production_condition_bytes_changed')
                state=read(root/'health/state.json')
                emitted=protocol.cycle(state,config,out)
                atomic(out/'status.json',{'state':'observing','pid':__import__('os').getpid(),'time':time.time(),
                    'condition':config['condition'],'published_this_cycle':emitted,
                    'total_events':len(list((out/'events').glob('*.json'))),'production_writes':0})
            except Exception as exc:
                atomic(out/'status.json',{'state':'needs_attention','pid':__import__('os').getpid(),
                    'time':time.time(),'error_type':type(exc).__name__,'error_code':str(exc)[:160] if isinstance(exc,ValueError) else None,
                    'production_writes':0})
            if once:
                return
            time.sleep(config.get('poll_seconds',20))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('--once',action='store_true')
    a=p.parse_args();run(a.config,once=a.once)
