"""Host process identity and conservative dual-namespace GPU lease checks."""
import os
from pathlib import Path
import signal
import subprocess
import time
from contextlib import ExitStack

from .io import Lock, digest


def identity(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=int(pid), start_ticks=int(fields[19]), ppid=int(fields[1]),
                    uid=p.stat().st_uid, state=fields[0],
                    argv=[b.decode() for b in (p / 'cmdline').read_bytes().split(b'\0') if b])
    except (OSError, ValueError):
        return None


def same(expected):
    row = identity(expected['pid']) if expected else None
    return bool(row and row['state'] != 'Z' and row['start_ticks'] == expected['start_ticks']
                and row['uid'] == expected['uid'] == os.getuid() and row['argv'] == expected['argv'])


def worker_identity(incarnation):
    found = []
    for p in Path('/proc').iterdir():
        if p.name.isdecimal():
            row = identity(int(p.name))
            if (row and row['uid'] == os.getuid() and row['state'] != 'Z'
                    and 'scheduler_v2.worker' in row['argv'] and incarnation in row['argv']
                    and 'python' in Path(row['argv'][0]).name):
                found.append(row)
    if len(found) > 1:
        raise RuntimeError('ambiguous_worker_process')
    return found[0] if found else None


def terminate(expected, grace=30):
    if not same(expected):
        return False
    os.kill(expected['pid'], signal.SIGTERM)
    until = time.monotonic() + grace
    while same(expected) and time.monotonic() < until:
        time.sleep(.2)
    if same(expected):
        os.kill(expected['pid'], signal.SIGKILL)
    return True


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'], text=True, timeout=5)
    return [(p[0].strip(), int(p[1])) for line in output.splitlines()
            if len(p := line.split(',')) == 2]


def resources_free(worker, local_leases, legacy_leases, *, observe=True):
    # These locks supplement (not replace) the site's authorized allocation.
    try:
        with ExitStack() as stack:
            for gpu in sorted(worker.get('gpu_ids', [])):
                for root in (local_leases, legacy_leases):
                    stack.enter_context(Lock(Path(root) / (digest(gpu) + '.lock')))
            if observe and any(gpu in worker['gpu_ids'] for gpu, _ in gpu_processes()):
                return False
        return True
    except RuntimeError:
        return False
