#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fleet admission and compatibility receipts (caller holds .lock).

Sessions release their hold without restoring production. Only the central idle
controller may recover after the fleet has been unused for 300 seconds. Legacy
restore debt is retired at the next admission and never blocks a queued probe.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

PROTOCOL = 2
SINGLE = 'single'   # the lane for a check that needs one GPU, not four (fleet_single.py says where)


def lane(kind):
    """'single' for the one-GPU lane; 'fleet' for boot and probe, which take the four Sparks."""
    return SINGLE if kind == SINGLE else 'fleet'


def holder_path(directory, kind='boot'):
    """Each lane owns one holder file: `holder` (the fleet) and `holder-single` (the one GPU).

    Two files, not one file with a lane column: everything that already reads `holder` --
    the ST launcher, the idle controller, restore debt, the AR campaign -- means the FLEET
    by it, and a single-GPU check must never look like the fleet being held to them.
    """
    return Path(directory) / ('holder-single' if lane(kind) == SINGLE else 'holder')


def holders(directory):
    """{lane: row} for every lane whose holder file names a session."""
    found = {}
    for name in ('fleet', SINGLE):
        try:
            row = holder_path(directory, name).read_text().strip().split('|')
        except FileNotFoundError:
            continue
        if row and row[0]:
            found[name] = row
    return found


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(pid):
    try:
        # Linux PID reuse and zombies must not turn a stale receipt into readiness.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def receipt(directory, session):
    return directory / ('ready-' + hashlib.sha256(session.encode()).hexdigest() + '.json')


def ready(directory, session, pid):
    token = identity(pid)
    if not token:
        raise ValueError('boot supervisor is not alive')
    value = dict(session=session, pid=pid, start=token, host=socket.gethostname(), protocol=PROTOCOL)
    write(receipt(directory, session), value)
    return value


def live(value):
    return bool(value and value.get('protocol') == PROTOCOL and value.get('host') == socket.gethostname()
                and value.get('start') and identity(value['pid']) == value['start'])


def rows(directory):
    return [line.split('|') for line in (directory / 'queue').read_text().splitlines() if line]


def successor(directory, session):
    from fleet_priority import rank, downstream
    from experiment_metrics import estimates
    db = Path(os.environ.get('FLEET_EXPERIMENT_ROOT', directory / 'experiments')) / 'experiments.sqlite3'
    from fleet_pause import paused
    lines = ['|'.join(row) for row in rows(directory) if row[1] != session and not paused(directory, row[1], row)
             and (len(row) < 7 or not row[6] or identity(int(row[6])))]
    def marker(name):
        path = directory / name
        return path.read_text().strip() if path.exists() else ''
    ranked = rank(lines, downstream(db), time.time(), marker('priority-front'), marker('priority-yield'),
                  estimates=estimates(db))
    if not ranked:
        return None
    row = ranked[0]['line'].split('|')
    value = read(receipt(directory, row[1]))
    return value if row[5] == 'boot' and live(value) and str(value['pid']) == row[6] else None


def claim_held(directory, session, pid):
    """Reconcile a cancelled admission after holder creation, under fleet lock."""
    holder = (directory / 'holder').read_text().split('|')
    value = read(receipt(directory, session))
    if holder[:2] != [session, str(pid)] or not live(value) or value['pid'] != pid:
        raise ValueError('admission requires this live supervisor hold')
    (directory / 'restore-debt.json').unlink(missing_ok=True)


def admit(directory, session, pid, kind, estimate='30', note=''):
    """Commit ownership and reset the central idle clock without restore debt.

    The single-GPU lane writes its own holder and touches nothing of the fleet's: not the
    idle clock (that GPU is not one of the four, so its work is not fleet activity), not
    restore debt, not the managed-handoff receipt.
    """
    from fleet_pause import paused
    if paused(directory, session):
        return False
    value = read(receipt(directory, session))
    managed = kind == 'boot' and live(value) and value['pid'] == pid
    from fleet_pending import metadata
    current = metadata(directory, session, pid)
    if current:
        estimate, note = current
    holder = holder_path(directory, kind)
    temporary = holder.with_name(holder.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(f'{session}|{pid}|{socket.gethostname().split(".")[0]}|{int(time.time())}|{estimate}|{note}|{kind}\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(holder)
    if lane(kind) == SINGLE:
        return True
    if managed:
        claim_held(directory, session, pid)
    else:
        (directory / 'restore-debt.json').unlink(missing_ok=True)
    from fleet_idle import activity
    activity(directory, 'acquire')
    return True


def offer(directory, session):
    debt = read(directory / 'restore-debt.json')
    if not debt or debt['owner']['session'] != session:
        raise ValueError('restore debt is not owned by this session')
    target = successor(directory, session)
    if not target:
        return None
    debt.update(target=target, offered=time.time())
    write(directory / 'restore-debt.json', debt)
    return target


def clear(directory, session):
    debt = read(directory / 'restore-debt.json')
    if debt and debt['owner']['session'] != session:
        raise ValueError('cannot clear another holder\'s restore responsibility')
    (directory / 'restore-debt.json').unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['admit', 'offer', 'clear', 'ready', 'next'])
    ap.add_argument('directory', type=Path)
    ap.add_argument('session')
    ap.add_argument('pid', nargs='?', type=int)
    ap.add_argument('kind', nargs='?', default='boot')
    ap.add_argument('estimate', nargs='?', default='30')
    ap.add_argument('note', nargs='?', default='')
    args = ap.parse_args()
    if args.action == 'admit':
        return 0 if admit(args.directory, args.session, args.pid, args.kind, args.estimate, args.note) else 1
    if args.action == 'ready':
        ready(args.directory, args.session, args.pid)
    elif args.action == 'next':
        # The waiting boot ticket the fleet lease is handed to when `session` lets go: the
        # queue's own priority order, a live supervisor, and never a probe (it runs beside
        # production, so it is a reason to release, not to hold).
        target = successor(args.directory, args.session)
        if not target:
            return 1
        print(f"{target['session']} {target['pid']}")
    elif args.action == 'clear':
        clear(args.directory, args.session)
    else:
        target = offer(args.directory, args.session)
        if not target:
            return 1
        print(target['session'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
