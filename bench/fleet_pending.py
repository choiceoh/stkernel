#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Revise a supervised reservation before admission, without a new ticket."""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time

import fleet_handoff as handoff


@contextmanager
def lock(directory):
    with (directory / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def path(directory, session):
    return directory / 'pending' / (hashlib.sha256(session.encode()).hexdigest() + '.json')


def queued(directory, session):
    if (directory / 'holder').exists() and (directory / 'holder').read_text().split('|')[0] == session:
        raise ValueError('reservation already admitted; edits are closed')
    rows = handoff.rows(directory)
    matches = [(i, row) for i, row in enumerate(rows) if row[1] == session]
    if len(matches) != 1:
        raise ValueError('reservation is no longer queued')
    return rows, *matches[0]


def register(directory, session, command, fleet, kind):
    """Called by the owning supervisor under .lock, before starting its waiter."""
    _, _, row = queued(directory, session)
    pid = os.getpid()
    if row[6] != str(pid):
        raise ValueError('reservation belongs to another process')
    (directory / 'pending').mkdir(mode=0o700, exist_ok=True)
    value = dict(session=session, ticket=row[0], enqueued_at=row[2], pid=pid,
                 start=handoff.identity(pid), host=socket.gethostname(), protocol=handoff.PROTOCOL,
                 state='queued', revision=1, command=list(command), cwd=os.getcwd(),
                 estimate_min=int(row[3]), note=row[4], kind=kind, fleet=fleet,
                 repo=os.environ['REPO'], validation_env={k:os.environ[k] for k in ('LOGD', 'PATH', 'HEAD_URL') if k in os.environ},
                 experiment=os.environ.get('FLEET_EXPERIMENT_ID'),
                 history=[])
    handoff.write(path(directory, session), value)


def inspect(directory, session):
    rows, index, row = queued(directory, session)
    value = handoff.read(path(directory, session))
    if not value or value.get('ticket') != row[0] or str(value.get('pid')) != row[6]:
        raise ValueError('this waiter predates editable reservations or uses request/wait; '
                         'command editing requires a new fleet.sh run reservation')
    if value['state'] != 'queued' or not handoff.live(value):
        raise ValueError('reservation has started or its supervisor is no longer alive')
    return value, rows, index


def validate(value, directory):
    command, cwd = value['command'], value['cwd']
    if not command or not all(isinstance(arg, str) and '\0' not in arg for arg in command) or not command[0]:
        raise ValueError('replacement requires a nonempty argv command')
    if not Path(cwd).is_dir():
        raise ValueError('working directory does not exist')
    env = dict(os.environ, **value['validation_env'])
    env.update(REPO=value['repo'], FLEET_DIR=str(directory))
    executable = command[0]
    if '/' in executable:
        executable = str(Path(cwd) / executable)
    if not shutil.which(executable, path=env.get('PATH')):
        raise ValueError('replacement executable does not exist or is not executable')
    # Use the same pinned controller as this waiter, including its preflight.
    args = ['bash', value['fleet'], 'preflight']
    if value['kind'] == 'probe':
        args.append('--probe')
    result = subprocess.run([*args, value['session'], '--', *command], cwd=cwd,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise ValueError('replacement preflight failed; original reservation retained\n' + result.stdout)


def edit(directory, session, *, command=None, cwd=None, estimate=None, note=None, expected=None):
    # Slow checks never hold the fleet lock. Admission and other editors can
    # proceed, so compare the original revision again before committing.
    with lock(directory):
        original, _, _ = inspect(directory, session)
    if expected is not None and original['revision'] != expected:
        raise ValueError('reservation revision changed; inspect it before editing again')
    updated = copy.deepcopy(original)
    if (command is not None or cwd is not None) and original.get('experiment'):
        raise ValueError('submit experiments have immutable evidence identities; '
                         'use submit --supersedes for a changed manifest')
    if command is not None:
        updated['command'] = command
    if cwd is not None:
        updated['cwd'] = str(Path(cwd).resolve())
    if estimate is not None:
        if estimate < 1:
            raise ValueError('estimate must be a positive number of minutes')
        updated['estimate_min'] = estimate
    if note is not None:
        if any(c in note for c in ('|', '\n', '\r', '\0')):
            raise ValueError('note cannot contain queue separators or newlines')
        updated['note'] = note
    if updated == original:
        return dict(original, changed=False)
    if command is not None or cwd is not None:
        validate(updated, directory)
    with lock(directory):
        current, rows, index = inspect(directory, session)
        if current != original:
            raise ValueError('reservation changed during preflight; inspect it before editing again')
        updated['revision'] += 1
        updated['history'].append(dict(revision=original['revision'], at=time.time(),
                                       **{k:original[k] for k in ('command', 'cwd', 'estimate_min', 'note')}))
        # The command record is authoritative. Admission reads its metadata too,
        # so interruption between these atomic writes cannot execute a stale edit.
        handoff.write(path(directory, session), updated)
        rows[index][3:5] = [str(updated['estimate_min']), updated['note']]
        temporary = directory / 'queue.edit.tmp'
        projection_pending = False
        try:
            temporary.write_text(''.join('|'.join(row) + '\n' for row in rows))
            temporary.replace(directory / 'queue')
        except OSError:
            # The authoritative revision already committed. Never report that
            # the old command was retained merely because its queue view failed.
            projection_pending = True
        return dict(updated, changed=True, position=index + 1, queue_projection_pending=projection_pending)


def metadata(directory, session, pid):
    """Authoritative metadata at GO; caller holds .lock."""
    value = handoff.read(path(directory, session))
    row = next((r for r in handoff.rows(directory) if r[1] == session), None)
    if (row and value and value['ticket'] == row[0] and value['pid'] == pid
            and value['state'] == 'queued' and handoff.live(value)):
        return str(value['estimate_min']), value['note']
    return None


def transition(directory, session, state):
    """Owner-only state change under .lock; return the admitted argv and cwd."""
    value = handoff.read(path(directory, session))
    if not value or value['pid'] != os.getpid() or value['start'] != handoff.identity(os.getpid()):
        raise ValueError('queued command ownership changed')
    value['state'] = state
    handoff.write(path(directory, session), value)
    return value


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = None
    if '--' in argv:
        split = argv.index('--')
        command, argv = argv[split + 1:], argv[:split]
    ap = argparse.ArgumentParser(description=__doc__, epilog='Append -- cmd arg... to replace the queued command.')
    ap.add_argument('session')
    ap.add_argument('--est', type=int)
    ap.add_argument('--note')
    ap.add_argument('--cwd')
    ap.add_argument('--expect-revision', type=int)
    args = ap.parse_args(argv)
    directory = Path(os.environ['FLEET_DIR'])
    try:
        value = edit(directory, args.session, command=command, cwd=args.cwd,
                     estimate=args.est, note=args.note, expected=args.expect_revision)
        print(json.dumps({k:v for k,v in value.items() if k not in ('validation_env', 'repo', 'fleet', 'start', 'host', 'protocol')}, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        print(f'edit refused: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
