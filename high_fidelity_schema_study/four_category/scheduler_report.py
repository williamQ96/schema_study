"""Read passive telemetry after a run; never change scheduling or call a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .common import write_new
from .scheduler_io import read_optional


def report(root):
    root = Path(root)
    files = [root/'telemetry/events.jsonl', *sorted((root/'workers').glob('*/events.jsonl'))]
    events, malformed = [], 0
    for path in files:
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                malformed += 1
    summary = read_optional(root/'summary.json') or {}
    health = read_optional(root/'health/state.json') or {}
    dispatches = [e for e in events if e.get('event') == 'task_dispatched']
    completed = [e for e in events if e.get('event') == 'task_committed']
    waits = sorted(e['ready_wait_s'] for e in dispatches)
    samples = [e for e in events if e.get('event') == 'worker_sample']
    timings = {name: [e['duration_s'] for e in events if e.get('event') == name and 'duration_s' in e]
               for name in ('model_loaded', 'prefill_finished', 'generation_finished')}
    sessions, start = [], None
    for e in sorted(events, key=lambda e: e['time']):
        if e['event'] == 'coordinator_started':
            start = e['time']
        elif e['event'] in ('coordinator_completed', 'coordinator_failed') and start is not None:
            sessions.append(max(0, e['time']-start)); start = None
    wall = sum(sessions) if start is None else None
    successes = sum(m['successful_cells'] for m in summary.get('matrices', []))
    return {'schema_version': 'scheduler-performance-report/v1', 'condition': summary.get('condition'),
            'execution_mode': summary.get('execution_mode', 'unknown'),
            'passive_only': True, 'malformed_log_lines': malformed, 'attempts_dispatched': len(dispatches),
            'attempts_committed': len(completed), 'model_loads': len(timings['model_loaded']),
            'model_reuses': sum(e.get('event') == 'model_reused' for e in events),
            'ready_wait_s': {'max': max(waits, default=None), 'p95': waits[min(len(waits)-1, int(.95*len(waits)))] if waits else None},
            'timings_s': {name: {'count': len(v), 'sum': sum(v), 'max': max(v, default=None)} for name, v in timings.items()},
            'peak_sampled_cpu_rss_bytes': max((e['rss_bytes'] for e in samples if e.get('rss_bytes') is not None), default=None),
            'worker_dispatches': dict(Counter(e['worker_id'] for e in dispatches)),
            'observed_coordinator_wall_s': wall, 'successful_extraction_cells': successes,
            'validated_extraction_cells_per_hour': successes*3600/wall if wall and summary.get('execution_mode') == 'live' else None,
            'throughput_scope': 'observed coordinator sessions including resume overhead; mock runs are software tests only',
            'health_events': dict(Counter(row['event']['kind'] for row in health.get('outbox', {}).values())),
            'undelivered_health_events': sum(not row['sent'] for row in health.get('outbox', {}).values()),
            'matrices': summary.get('matrices', []), 'semantic_accuracy': None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = report(args.root)
    write_new(args.output, result)
    print(json.dumps({k: result[k] for k in ('attempts_dispatched', 'model_loads', 'model_reuses', 'health_events')}))


if __name__ == '__main__':
    main()
