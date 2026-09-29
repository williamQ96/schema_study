"""Publish one authenticated completion/failure receipt for the fixed V13 probes."""
import json
import sys
import time
from pathlib import Path
import telegram_codex_bridge as protocol


def terminal_snapshot(job, report, config):
    """Bind downstream probe reports to the signed terminal receipt."""
    qualification = protocol.read(report)
    profiles = protocol.read(job / 'config.json')['roles']['locals']
    reports = {}
    for profile in profiles:
        if not isinstance(profile, str) or not profile or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in profile):
            raise ValueError('unsafe_profile_id')
        relative = 'audit/v13-repair-probe/' + profile + '/report.json'
        path = job / relative
        if path.exists():
            reports[relative] = protocol.sha(path)
    value = {'kind': 'v13-repair-terminal/v1',
             'source_report': {'path': str(report.relative_to(job)).replace('\\', '/'),
                               'sha256': protocol.sha(report)},
             'qualification': qualification, 'probe_reports': reports}
    copied = Path(config['remote_report'])
    raw = protocol.canonical(value)
    if copied.exists():
        if copied.read_bytes() != raw:
            raise ValueError('terminal_report_changed')
    else:
        with copied.open('xb') as stream:
            stream.write(raw)
    return copied, qualification


def main(config_path):
    config = protocol.read(config_path)
    job = Path(config['remote_job_root'])
    state = Path(config['remote_notice_root'])
    state.mkdir(parents=True, exist_ok=True)
    outbox = Path(config['remote_outbox'])
    outbox.mkdir(exist_ok=True)
    protocol.LABELS.update({
        'bridge_test': 'Mercury | V13 repair notification route verified',
        'comparison_completed': 'Mercury | V13 repair tests passed; Codex will verify the outputs. Full inference is still paused.',
        'comparison_failed': 'Mercury | V13 repair tests returned failures; Codex will inspect retained outputs. Full inference is still paused.',
    })
    if protocol.sha(job / 'config.json') != config['config_file_sha256']:
        raise ValueError('qualification_config_changed')
    while not (state / 'STOP').exists():
        report = job / 'audit/qualification-summary.json'
        failure = job / 'audit/qualification-launcher-failure.json'
        protocol.atomic(state / 'status.json', {'state': 'observing', 'time': time.time(), 'job_id': config['job_id']})
        if failure.exists() and not report.exists():
            # A dedicated immutable failure report also has a fixed configured path.
            report = failure
        if report.exists():
            copied, summary = terminal_snapshot(job, report, config)
            kind = 'comparison_completed' if summary.get('status') == 'pass' else 'comparison_failed'
            for attempt in range(3):
                try:
                    event = protocol.publish(config, protocol.read(config['telegram_secret_path']), kind, copied, outbox)
                    protocol.atomic(state / 'status.json', {'state': 'finished', 'time': time.time(),
                                    'event_id': event.stem, 'source_report': str(report), 'status': summary.get('status')})
                    return
                except Exception as exc:
                    protocol.atomic(state / 'delivery-error.json', {'attempt': attempt + 1,
                                    'error_type': type(exc).__name__, 'time': time.time()})
                    if attempt == 2:
                        raise
                    time.sleep(30)
        time.sleep(20)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
