"""Condition-bound production health receipts, separate from frozen probe events."""
from __future__ import annotations
import hashlib
import hmac
import re
import time
from pathlib import Path
from telegram_codex_bridge import canonical, atomic, read, sha, sign

VERSION = 'mercury-production-health/v1'
KINDS = {'ready_queue_overdue', 'allocation_imbalance', 'worker_no_progress',
         'worker_heartbeat_lost', 'scheduler_heartbeat_lost', 'coordinator_failed',
         'gpu_headroom_low', 'sustained_utilization_skew', 'systemic_classification_rejection',
         'run_completed', 'run_completed_with_failures', 'dependency_starvation',
         'classification_queue_overdue', 'extraction_queue_overdue', 'validation_backlog',
         'planned_resume_failure', 'resource_ownership_conflict', 'reservation_expired', 'reservation_wait_delayed'}
TERMINAL_KINDS = {'run_completed', 'run_completed_with_failures'}
RANK = {'info': 0, 'warning': 1, 'critical': 2}


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def hex_id(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def incident_id(condition, kind, scope, opened):
    # Watchdog episode numbers reset after recovery. The opening event ID makes
    # a later recurrence distinct even when its episode counter is again one.
    return digest([VERSION, condition, kind, scope, opened])


def trigger_id(condition, incident, severity):
    return digest([VERSION, condition, incident, severity])


def candidates(state, config):
    """Catch up active incidents once, ignoring inactive historical reminders."""
    if state.get('condition') != config['condition']:
        raise ValueError('watchdog_condition_mismatch')
    by_key = {}
    for key, row in state.get('outbox', {}).items():
        e = row['event']
        if (e.get('condition') != config['condition'] or key != e.get('event_id')
                or digest({k:v for k,v in e.items() if k!='event_id'}) != key):
            raise ValueError('watchdog_event_identity_mismatch')
        if e['kind'] not in KINDS:
            continue
        if (e.get('severity') not in RANK or e.get('transition') not in {'open','reminder','recovered'}
                or not isinstance(e.get('scope'), str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', e['scope'])):
            raise ValueError('watchdog_event_shape')
        by_key.setdefault(e['kind']+':'+e['scope'], []).append(row)
    result = []
    for key, rows in sorted(by_key.items()):
        rows.sort(key=lambda r:r['event']['time'])
        latest = rows[-1]['event']
        active = state.get('incidents', {}).get(key, {})
        terminal = latest['kind'] in TERMINAL_KINDS
        if not terminal and (not active.get('active') or latest['transition']=='recovered'):
            continue
        # Bound the source rows to the currently active episode; no history flood.
        start = max((i for i,r in enumerate(rows) if r['event']['transition']=='open'), default=None)
        if start is None:
            raise ValueError('active_incident_open_event_missing')
        current = rows[start:]
        delivered = [r for r in current if r.get('sent') is True and type(r.get('message_id')) is int
                     and r['message_id'] > 0 and r['event']['transition']!='recovered']
        if not delivered:
            continue
        # One catch-up for the highest severity actually delivered. Later lower
        # severity reminders do not create additional diagnostics.
        row = max(delivered, key=lambda r:(RANK[r['event']['severity']], r['event']['time']))
        e = row['event']
        iid = incident_id(config['condition'],e['kind'],e['scope'],rows[start]['event']['event_id'])
        result.append({'incident_id':iid,'event_id':trigger_id(config['condition'],iid,e['severity']),
                       'opening_event_id':rows[start]['event']['event_id'],'row':row})
    return result


def verify(envelope, config, *, now=None, max_age_s=7*86400):
    if not isinstance(envelope,dict) or set(envelope)!={'payload','hmac_sha256'}:
        raise ValueError('envelope_shape')
    p = envelope['payload']
    expected = sign(p,config['hmac_key'])['hmac_sha256']
    if not isinstance(envelope['hmac_sha256'],str) or not hmac.compare_digest(envelope['hmac_sha256'],expected):
        raise ValueError('event_signature')
    keys = {'protocol','job_id','condition','kind','scope','severity','incident_id','opening_event_id',
            'watchdog_event_id','event_id','created_at','receipt','report_sha256'}
    if not isinstance(p,dict) or set(p)!=keys:
        raise ValueError('event_fields')
    if p['protocol']!=VERSION or p['job_id']!=config['job_id'] or p['condition']!=config['condition'] or p['kind'] not in KINDS:
        raise ValueError('event_scope')
    if p['severity'] not in RANK or not isinstance(p['scope'],str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',p['scope']):
        raise ValueError('event_shape')
    if any(not hex_id(p[k]) for k in ['incident_id','opening_event_id','watchdog_event_id','event_id','report_sha256']):
        raise ValueError('event_hash')
    if (p['incident_id']!=incident_id(p['condition'],p['kind'],p['scope'],p['opening_event_id'])
            or p['event_id']!=trigger_id(p['condition'],p['incident_id'],p['severity'])):
        raise ValueError('event_identity')
    current = time.time() if now is None else now
    if (type(p['created_at']) not in (int,float) or not 0<=current-p['created_at']
            or max_age_s is not None and current-p['created_at']>max_age_s):
        raise ValueError('event_age')
    receipt=p['receipt']
    if not isinstance(receipt,dict) or set(receipt)!={'bot_username','bot_id','chat_id','message_id','basis'}:
        raise ValueError('receipt_shape')
    if (any(receipt.get(k)!=config[k] for k in ['bot_username','bot_id','chat_id'])
            or type(receipt.get('message_id')) is not int or receipt['message_id']<=0
            or receipt['basis']!='condition_bound_watchdog_delivery_record'):
        raise ValueError('telegram_identity')
    return p


def publish_candidate(candidate, state, config, directory, *, at=None):
    """No extra Telegram send: bridge the already acknowledged watchdog alert."""
    directory=Path(directory);eid=candidate['event_id']
    target=directory/'events'/(eid+'.json')
    report_path=directory/'reports'/(eid+'.json')
    if target.exists():
        payload=verify(read(target),config,max_age_s=None)
        if sha(report_path)!=payload['report_sha256']:
            raise ValueError('published_report_changed')
        return payload
    row=candidate['row'];e=row['event']
    if not report_path.exists():
        atomic(report_path,{'protocol':VERSION,'job_id':config['job_id'],'condition':config['condition'],
            'incident_id':candidate['incident_id'],'opening_event_id':candidate['opening_event_id'],
            'watchdog_delivery':row,'captured_at':time.time() if at is None else at,
            'watchdog_state_canonical_sha256':digest(state), 'snapshot':state.get('last_snapshot'),
            'threshold_history':state.get('threshold_history',[]),
            'condition_file_bytes_sha256':config['condition_file_bytes_sha256'],
            'semantic_accuracy':None})
    # Resume an interrupted publish using the preserved report, not a later reminder.
    report=read(report_path);row=report['watchdog_delivery'];e=row['event']
    payload={'protocol':VERSION,'job_id':config['job_id'],'condition':config['condition'],
        'kind':e['kind'],'scope':e['scope'],'severity':e['severity'],'incident_id':candidate['incident_id'],
        'opening_event_id':candidate['opening_event_id'],'watchdog_event_id':e['event_id'],'event_id':eid,
        'created_at':report['captured_at'],'report_sha256':sha(report_path),
        'receipt':{**{k:config[k] for k in ['bot_username','bot_id','chat_id']},'message_id':row['message_id'],
                   'basis':'condition_bound_watchdog_delivery_record'}}
    envelope=sign(payload,config['hmac_key']);verify(envelope,config,now=payload['created_at'])
    atomic(target,envelope)
    return payload


def cycle(state, config, directory):
    directory=Path(directory)
    prior={}
    # Existing envelopes are the durable dedup ledger, including after a crash.
    for path in (directory/'events').glob('*.json'):
        p=verify(read(path),config,max_age_s=None)
        if sha(directory/'reports'/(p['event_id']+'.json'))!=p['report_sha256']:
            raise ValueError('published_report_changed')
        prior[p['incident_id']]=max(prior.get(p['incident_id'],-1),RANK[p['severity']])
    published=[]
    for c in candidates(state,config):
        severity=c['row']['event']['severity']
        if RANK[severity]<=prior.get(c['incident_id'],-1):
            continue
        p=publish_candidate(c,state,config,directory)
        prior[c['incident_id']]=RANK[severity]
        published.append(p['event_id'])
    return published
