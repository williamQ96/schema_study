"""Small local-disk primitives for the single-host scheduler and its watchdog."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers/indexers can briefly deny replacement. Keep atomicity and
    # a bounded retry; persistent filesystem failures still fail closed.
    for attempt in range(12):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.02 * (attempt + 1), .2))


def read_optional(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


class Lock:
    """OS-released advisory lock; never steal leases because a heartbeat is late."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        self.stream.seek(0)
        self.stream.write(b'0')
        self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            self.stream = None
            raise RuntimeError('lease_already_owned:' + self.path.name) from None
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            if os.name == 'nt':
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


class Telemetry:
    """Append-only JSONL, one writer per stream, no prompts or raw responses."""
    def __init__(self, path, condition):
        self.path, self.condition = Path(path), condition
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **fields):
        row = {'schema_version': 'scheduler-telemetry/v1', 'time': time.time(),
               'condition': self.condition, 'event': event, **fields}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
