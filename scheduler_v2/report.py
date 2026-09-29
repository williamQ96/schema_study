"""Read-only operational rates from immutable V2 dispatch/result/commit records."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from .io import optional, read


def report(root, since=None, now=None):
    root = Path(root)
    now = time.time() if now is None else now
    started, completed, admitted, recovery = [], [], [], []
    kinds = Counter()
    for path in sorted((root/'scheduler_v2').glob('*/events/*.json')):
        event = read(path)
        if event['kind'] == 'claim':
            a = event['assignment']
            if since is None or a['started_at'] >= since:
                started.append(a)
        elif event['kind'] == 'recover':
            if since is None or event['at'] >= since: recovery.append(event)
        elif event['kind'] == 'commit':
            if since is None or event['at'] >= since:
                completed.append(event)
                verdict = event.get('verdict') or event.get('outcome') or {}
                kinds[verdict.get('status', 'unknown')] += 1
                if verdict.get('status') == 'success': admitted.append(event)
    beginning = since if since is not None else min((a['started_at'] for a in started), default=now)
    elapsed = max(0, now-beginning)
    receipts = []
    for p in (root/'worker_receipts').glob('*/*/result-*.json'):
        r = read(p)
        if r.get('status') == 'returned' and r.get('finished_at', 0) >= beginning:
            receipts.append(r)
    return {'window_start': beginning, 'window_end': now, 'elapsed_s': elapsed,
            'dispatched_groups': len(started), 'returned_groups': len(receipts),
            'validated_groups': len(completed), 'contract_admitted_groups': len(admitted),
            'contract_admitted_groups_per_hour': len(admitted)*3600/elapsed if elapsed else None,
            'returned_fraction_of_dispatched': len(receipts)/len(started) if started else None,
            'validated_outcomes': dict(kinds), 'automatic_recoveries': len(recovery),
            'snapshot': optional(root/'snapshot.json'), 'semantic_accuracy': None,
            'interpretation': 'Operational observation includes loading, queueing, and verification; unfinished groups are censored. Not a fidelity estimate.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--since', type=float)
    a = p.parse_args()
    print(json.dumps(report(a.root, a.since), sort_keys=True))
