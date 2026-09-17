#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The queue's shared activity clock.

Every enqueue/acquire/release resets the idle age the fleet records in
idle-recovery.json; the ST supervisor reads the same file's updated_at to
judge how long the queue has been quiet. The consumer of that idle age -- the
five-minute controller that restored the vLLM production fleet -- retired
with the overlay stack (2026-09-18): production recovery is now the ST
supervisor's own boot-start + crash-recovery loop (launchers/st-glm53.service).
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def read(path, default=None):
    try:
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError('idle controller state must be an owned regular file')
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
    temporary.replace(path)


def boot_id():
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except FileNotFoundError:
        if sys.platform == 'darwin':
            # Local CPU fixtures also exercise enqueue/release on macOS.
            return subprocess.check_output(['sysctl', '-n', 'kern.boottime'], text=True).strip()
        raise


def clock():
    return time.monotonic()


@contextmanager
def lock(directory, name='.lock', *, blocking=True):
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def activity(directory, reason):
    """Caller owns fleet .lock. Every enqueue/acquire/release resets idle age."""
    directory = Path(directory)
    state = dict(version=2, boot_id=boot_id(), since=clock(), phase='waiting',
                 reason=reason, generation=uuid.uuid4().hex, updated_at=time.time())
    write(directory / 'idle-recovery.json', state)
    return state


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else (int(fields[1]), fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        value = process(pid)
        if value is None:
            return False
        pid = value[0]
    return False


def holder_live(directory):
    path = directory / 'holder'
    if not path.exists() or not path.read_text().strip():
        return False
    row = path.read_text().strip().split('|')
    if len(row) != 7 or row[2] not in (socket.gethostname(), socket.gethostname().split('.')[0]):
        return True  # Unknown ownership is busy, not permission to take over.
    return bool(row[1].isdigit() and process(int(row[1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['activity', 'status'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('detail', nargs='?')
    args = parser.parse_args()
    try:
        if args.action == 'activity':
            result = activity(args.directory, args.detail or 'fleet activity')
        else:
            result = read(args.directory / 'idle-recovery.json', {})
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print('ABORT: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
