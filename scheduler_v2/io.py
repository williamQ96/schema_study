"""Durable local control files and immutable shared artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def optional(path):
    try:
        return read(path)
    except FileNotFoundError:
        return None


def sync_directory(path):
    if os.name == 'posix':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _temporary(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                allow_nan=False, indent=2) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    return path, tmp


def atomic_json(path, value):
    path, tmp = _temporary(path, value)
    try:
        for n in range(12):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if n == 11:
                    raise
                time.sleep(.02 * (n + 1))
        sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def write_once(path, value):
    """Atomically publish without replacing an existing record, even a hardlink."""
    path, tmp = _temporary(path, value)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            if read(path) != value:
                raise ValueError('immutable_record_conflict:' + str(path))
        sync_directory(path.parent)
        return file_sha(path)
    finally:
        tmp.unlink(missing_ok=True)


class Lock:
    def __init__(self, path):
        self.path, self.stream = Path(path), None

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
            raise RuntimeError('lease_already_owned:' + str(self.path)) from None
        return self

    def __exit__(self, *args):
        if self.stream:
            if os.name == 'nt':
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def local_storage_preflight(path, *, minimum_free=1024**3):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == 'posix':
        kind = subprocess.check_output(['stat', '-f', '-c', '%T', str(path)], text=True).strip()
        if kind not in {'xfs', 'ext2/ext3', 'btrfs', 'zfs'}:
            raise ValueError('persistent_host_local_filesystem_required:' + kind)
        if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
            raise ValueError('private_owned_state_directory_required')
        stat = os.statvfs(path)
        if stat.f_bavail * stat.f_frsize < minimum_free:
            raise ValueError('local_state_space_insufficient')
    return path
