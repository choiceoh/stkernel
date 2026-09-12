#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read reservation state and bounded logs without probing or acquiring GPUs."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import stat
import sys
import time

import fleet_handoff as handoff
import fleet_pending as pending

MAX_LOG_BYTES = 256 * 1024
FINAL = {'finished', 'cancelled'}


@contextmanager
def snapshot_lock(directory):
    # Observers do not create a fleet or its queue/lock files.
    try:
        stream = (directory / '.lock').open('r')
    except FileNotFoundError:
        yield
        return
    with stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        yield


def text(path):
    try:
        return path.read_text()
    except FileNotFoundError:
        return ''


def read_state(directory):
    rows = [line.split('|') for line in text(directory / 'queue').splitlines() if line]
    rows = [r for r in rows if len(r) >= 7]
    holders = {lane: row for lane, row in handoff.holders(directory).items() if len(row) >= 7}
    return rows, holders


def holder_for(holders, session, rows):
    """The holder that matters to one session: the one naming it, else its lane's.

    A queued single-GPU check waits behind `holder-single`, not behind the fleet's holder,
    and its `waiting_for` must say so.
    """
    for row in holders.values():
        if row[0] == session:
            return row
    queued = next((r for r in rows if r[1] == session), None)
    return holders.get(handoff.lane(queued[5]) if queued else 'fleet')


def process_log(pid):
    """Best-effort log location for older live controllers, never a pipe read."""
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if not any(Path(os.fsdecode(arg)).name in ('fleet.sh', 'fleet_boot.py') for arg in args):
            return None
        descriptor = Path(f'/proc/{pid}/fd/1')
        target = Path(os.readlink(descriptor))
        if not target.is_absolute():
            return None
        opened, named = descriptor.stat(), target.stat()
        if stat.S_ISREG(opened.st_mode) and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino):
            return str(target)
    except (OSError, ValueError):
        pass
    return None


def describe(directory, session, rows, holder, now=None, ticket=None):
    now = time.time() if now is None else now
    row = next((r for r in rows if r[1] == session), None)
    held = holder if holder and holder[0] == session else None
    warnings = []
    try:
        value = pending.read_record(directory, session, ticket)
    except (OSError, ValueError) as exc:
        value = None
        warnings.append('Saved reservation record is unreadable: ' + type(exc).__name__)
    if ticket is not None and value is None:
        raise ValueError('unknown reservation ticket: ' + session + '/' + ticket)
    historical = False
    if ticket is not None:
        try:
            latest = pending.read_record(directory, session)
        except (OSError, ValueError):
            latest = None
        historical = (not latest or latest.get('ticket') != ticket
                      or bool(value.get('state') in FINAL and (row or held))
                      or bool(row and (row[0] != ticket or row[6] != str(value['pid'])))
                      or bool(held and (held[1] != str(value['pid'])
                          or handoff.identity(value['pid']) not in (None, value.get('start')))))
        if historical:
            # A reused session/PID must never lend its current queue position,
            # liveness, edit command or output descriptor to an older ticket.
            row = held = None
    # A reused session must never show a previous run's argv/result as current.
    if value and ((value.get('state') in FINAL and (row or held))
                  or (row and (value.get('ticket') != row[0] or str(value.get('pid')) != row[6]))
                  or (held and str(value.get('pid')) != held[1])):
        value = None
    if value and (row or held):
        current_start = handoff.identity(value['pid'])
        if current_start and current_start != value.get('start'):
            value = None
    if not value and not row and not held:
        raise ValueError('unknown reservation: ' + session)
    result = {k:value[k] for k in ('session', 'ticket', 'enqueued_at', 'revision', 'command', 'cwd',
              'kind', 'estimate_min', 'note', 'experiment', 'phase', 'started_at', 'payload_finished_at',
              'finished_at', 'recovery_policy', 'recovery_deferred', 'pause_reason', 'paused_at', 'resumed_at', 'payload_returncode', 'recovery_returncode', 'returncode', 'outcome', 'log_path', 'error', 'log_error')
              if value and k in value}
    result.update(session=session, source='saved' if value else 'legacy', position=None, editable=False)
    if ticket is not None:
        result['historical'] = historical
    alive = handoff.live(value) if value and not historical else False
    from fleet_pause import paused
    is_parked = bool(value and not historical and not held and paused(directory, session, row))
    if is_parked:
        result.update(state='paused', position=None, ahead=[], editable=True,
                      waiting_for=value.get('pause_reason', 'resume requested by owner'))
    elif row:
        result.update(ticket=row[0], enqueued_at=row[2], kind=row[5], position=rows.index(row) + 1)
        result.setdefault('estimate_min', float(row[3]))
        result.setdefault('note', row[4])
        if not value:
            alive = bool(handoff.identity(int(row[6])))
        result['state'] = ('paused' if value and value['state'] == 'paused' else 'queued') if alive else 'interrupted'
        result['waiting_for'] = (('holder ' + holder[0]) if holder else
                                 'single-GPU admission checks' if handoff.lane(row[5]) == handoff.SINGLE
                                 else 'fleet admission checks')
        result['ahead'] = [r[1] for r in rows[:rows.index(row)] if not paused(directory, r[1], r)]
        result['editable'] = bool(value and alive and value['state'] in ('queued','paused'))
        if result['state'] == 'paused':
            result['waiting_for'] = value.get('pause_reason','resume requested by owner')
    elif held:
        if not value:
            alive = held[2] == socket.gethostname().split('.')[0] and bool(handoff.identity(int(held[1])))
        result.update(state='running' if alive else 'interrupted', estimate_min=float(held[4]),
                      note=held[5], kind=held[6])
        result.setdefault('started_at', int(held[3]))
        if value and alive and value['state'] == 'finishing':
            result['state'] = 'finishing'
    elif value['state'] in FINAL:
        result['state'] = value.get('outcome') or value['state']
    else:
        result['state'] = 'transitioning' if alive else 'interrupted'
    result['supervisor_alive'] = alive
    if value and value['state'] not in FINAL and not alive:
        # A dead process with only a successful payload is not a successful run.
        result.pop('outcome', None)
    if result['editable']:
        result['edit_scope'] = 'metadata' if value.get('experiment') else 'command, cwd, metadata'
        result['edit_reason'] = 'before admission'
    else:
        result['edit_reason'] = ('older controller or request/wait has no editable command record'
                                 if not value else 'editing closes at admission or supervisor exit')
    if not result.get('log_path') and (row or held) and alive:
        result['log_path'] = process_log(int(row[6] if row else held[1]))
        if result['log_path']:
            result['log_source'] = 'existing stdout file'
    elif result.get('log_path'):
        result['log_source'] = 'reservation capture'
    if not result.get('log_path'):
        result['log_reason'] = 'No retained regular log file; older terminal/pipe output cannot be recovered.'
    queued_at = result.get('enqueued_at')
    if queued_at:
        until = result.get('started_at') or result.get('finished_at') or now
        result['wait_seconds'] = round(max(0, float(until) - float(queued_at)), 1)
    if result.get('started_at'):
        until = result.get('payload_finished_at') or result.get('finished_at') or now
        result['payload_seconds'] = round(max(0, float(until) - result['started_at']), 1)
    if warnings:
        result['warnings'] = warnings
    result['actions'] = {}
    if result['editable']:
        result['actions']['edit'] = ['fleet.sh', 'edit', session, '--expect-revision', str(value['revision'])]
    if value and value.get('pause_protocol') == 1 and result['editable']:
        action = 'resume' if result['state'] == 'paused' else 'pause'
        result['actions'][action] = ['fleet.sh', action, session, '--expect-revision', str(value['revision'])]
    if result.get('log_path'):
        result['actions']['logs'] = ['fleet.sh', 'logs', session] + (['--ticket', ticket] if ticket else [])
    if result.get('experiment'):
        result['actions']['result'] = ['fleet.sh', 'result', value['experiment']]
    return result


def show(directory, session=None, ticket=None):
    if ticket is not None and session is None:
        raise ValueError('--ticket requires a reservation session')
    with snapshot_lock(directory):
        rows, holders = read_state(directory)
        if session is not None:
            return describe(directory, session, rows, holder_for(holders, session, rows), ticket=ticket)
        names = list(dict.fromkeys([row[0] for row in holders.values()] + [r[1] for r in rows]))
        from fleet_pause import parked
        names += [v['session'] for v in parked(directory) if v['session'] not in names]
        return [describe(directory, name, rows, holder_for(holders, name, rows)) for name in names]


def history(directory, session, limit=20):
    """Bounded summaries for one session, including its latest active ticket."""
    with snapshot_lock(directory):
        rows, holders = read_state(directory)
        holder = holder_for(holders, session, rows)
        result = []
        for record in pending.history(directory, session, limit):
            value = describe(directory, session, rows, holder, ticket=record['ticket'])
            summary = {k:value[k] for k in ('session', 'ticket', 'state', 'kind', 'revision', 'enqueued_at',
                       'started_at', 'finished_at', 'returncode', 'payload_returncode', 'recovery_returncode',
                       'historical', 'position') if k in value}
            summary['note'] = str(value.get('note', ''))[:240]
            result.append(summary)
        return result


def tail(path, count):
    # Open first, then fstat: a replaced FIFO/device can never hang an observer.
    if not 1 <= count <= 2000:
        raise ValueError('--tail must be between 1 and 2000')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError('log source is not a regular file')
        start = max(0, metadata.st_size - MAX_LOG_BYTES)
        stream.seek(start)
        data = stream.read(MAX_LOG_BYTES)
    if start and b'\n' in data:
        data = data.split(b'\n', 1)[1]
    return b'\n'.join(data.splitlines()[-count:]).decode('utf-8', errors='replace')


def render(value):
    if isinstance(value, list):
        if not value:
            print('No active reservations.')
        for row in value:
            location = f"queue #{row['position']}" if row['position'] else row['state']
            print(f"{row['session']}: {location} [{row['kind']}] {row.get('note', '')}")
        return
    position = f" (queue #{value['position']})" if value['position'] else ''
    print(f"{value['session']}: {value['state']}{position}")
    for key in ('phase', 'ticket', 'revision', 'note', 'cwd', 'wait_seconds', 'payload_seconds',
                'payload_returncode', 'recovery_returncode', 'returncode', 'log_path', 'waiting_for', 'error', 'log_error'):
        if value.get(key) is not None:
            print(f"  {key}: {value[key]}")
    if value.get('command'):
        print('  command: ' + shlex.join(value['command']))
    print('  edit: ' + (value.get('edit_scope', '') if value['editable'] else value['edit_reason']))
    if not value.get('log_path'):
        print('  log: ' + value['log_reason'])
    for command in value['actions'].values():
        print('  ' + shlex.join(command))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    view = sub.add_parser('show', help='inspect one reservation, or list the active queue')
    view.add_argument('session', nargs='?')
    view.add_argument('--json', action='store_true')
    view.add_argument('--ticket')
    logs = sub.add_parser('logs', help='last lines of a reservation log, bounded to 256 KiB')
    logs.add_argument('session')
    logs.add_argument('--tail', type=int, default=80)
    logs.add_argument('--ticket')
    past = sub.add_parser('history', help='recent ticket summaries for one reservation session')
    past.add_argument('session')
    past.add_argument('--limit', type=int, default=20)
    past.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    try:
        directory = Path(os.environ['FLEET_DIR'])
        if args.action == 'history':
            value = history(directory, args.session, args.limit)
            if args.json:
                print(json.dumps(value, ensure_ascii=False))
            elif not value:
                print('No retained reservations for ' + args.session + '.')
            else:
                for row in value:
                    print(f"{row['ticket']}: {row['state']} [{row['kind']}] {row['note']}")
            return 0
        value = show(directory, args.session, args.ticket)
        if args.action == 'logs':
            if not 1 <= args.tail <= 2000:
                raise ValueError('--tail must be between 1 and 2000')
            if not value.get('log_path'):
                raise ValueError(value['log_reason'])
            print(tail(value['log_path'], args.tail))
        elif args.json:
            print(json.dumps(value, ensure_ascii=False))
        else:
            render(value)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
