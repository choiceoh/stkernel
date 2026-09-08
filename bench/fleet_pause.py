#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pause a waiting reservation without losing its ticket or blocking successors."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import time

import fleet_handoff as handoff
import fleet_pending as pending


def paused(directory, session, row=None):
    value = pending.read_record(Path(directory), session)
    if (not value or value.get('state') != 'paused' or value.get('pause_protocol') != 1
            or not handoff.live(value)):
        return False
    saved = pending.parked_row(value)
    if saved:
        return row is None or len(row) == 7 and all(row[i] == saved[i] for i in (0, 1, 2, 5, 6))
    if row is None:
        row = next((r for r in handoff.rows(Path(directory)) if r[1] == session), None)
    return bool(row and value.get('ticket') == row[0] and str(value.get('pid')) == row[6])


def parked(directory):
    """Read a bounded discovery index, verifying every canonical live record."""
    directory = Path(directory)
    result = []
    for session in pending.parked_index(directory):
        try:
            value = pending.read_record(directory, session)
            if value and pending.parked_row(value) and paused(directory, session):
                result.append(value)
        except (OSError, ValueError):
            continue  # One damaged record must not hide other live reservations.
    return result


def write_rows(directory, rows):
    temporary = Path(directory) / 'queue.pause.tmp'
    with temporary.open('w') as stream:
        stream.write(''.join('|'.join(row) + '\n' for row in rows))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(Path(directory) / 'queue')


def reconcile(directory, session, pid=None):
    """Caller holds .lock; repair a committed park/resume without changing IDs.

    Return 0 when this saved reservation owns the projection, 1 for an ordinary
    reservation. A conflicting live session or row is a refusal, never replaced.
    """
    directory = Path(directory)
    value = pending.read_record(directory, session)
    if (not value or value.get('pause_protocol') != 1 or value.get('state') not in ('paused', 'queued')
            or not handoff.live(value)):
        return 1
    saved = pending.parked_row(value)
    if pid is not None and value['pid'] != pid:
        if saved or value['state'] == 'paused':
            raise ValueError('session belongs to a live parked reservation; edit or resume its existing ticket')
        return 1
    rows = handoff.rows(directory)
    physical = [r for r in rows if r[1] == session]
    if not saved:
        if (value['state'] != 'paused' or len(physical) != 1 or value.get('ticket') != physical[0][0]
                or str(value['pid']) != physical[0][6] or str(value['enqueued_at']) != physical[0][2]):
            return 1
        # Upgrade a pause written by an earlier controller from this protocol.
        value = copy.deepcopy(value)
        value['parked_row'] = list(physical[0])
        value['parked_row'][3:5] = [str(value['estimate_min']), value['note']]
        pending.save_record(directory, value)
        saved = value['parked_row']
    # Metadata can change while parked; identity columns still cannot change.
    identity = lambda row: [row[i] for i in (0, 1, 2, 5, 6)]
    if len(physical) > 1 or physical and identity(physical[0]) != identity(saved):
        raise ValueError('saved reservation conflicts with another queued identity')
    if value['state'] == 'paused':
        if physical:
            write_rows(directory, [r for r in rows if r[1] != session])
    elif physical != [saved]:
        write_rows(directory, [r for r in rows if r[1] != session] + [saved])
    return 0


def owned(directory, session):
    value, rows, index = pending.inspect(Path(directory), session)
    if value.get('pause_protocol') != 1:
        raise ValueError('this pinned controller cannot pause; keep its current reservation or submit with a current controller')
    return value, rows, index


def set_paused(directory, value, reason):
    """Caller holds .lock and has matched the queued identity."""
    if value['state'] == 'paused':
        reconcile(directory, value['session'], value['pid'])
        return value
    value = copy.deepcopy(value)
    _, _, row = pending.queued(Path(directory), value['session'])
    row = list(row)
    row[3:5] = [str(value['estimate_min']), value['note']]
    value.update(state='paused', phase='paused', pause_reason=str(reason)[:4000], paused_at=time.time(),
                 revision=value['revision'] + 1, parked_row=list(row))
    pending.save_record(Path(directory), value)
    reconcile(directory, value['session'], value['pid'])
    # A donor may have offered this reservation just before it paused. It will
    # reclaim recovery responsibility instead of waiting for a paused receiver.
    debt = handoff.read(Path(directory) / 'restore-debt.json')
    if debt and debt.get('target', {}).get('session') == value['session']:
        debt.pop('target', None)
        handoff.write(Path(directory) / 'restore-debt.json', debt)
    for name in ('priority-front', 'priority-yield'):
        path = Path(directory) / name
        if path.exists() and path.read_text().strip() == value['session']:
            path.unlink()
    return value


def pause_failed(directory, session, checked, reason, *, locked=False):
    """A failed older check cannot pause a newly edited reservation."""
    def apply():
        try:
            current, _, _ = owned(directory, session)
        except ValueError:
            return False
        keys = ('ticket', 'pid', 'start', 'revision', 'prepare_manifest')
        if any(current.get(k) != checked.get(k) for k in keys) or current['state'] != 'queued':
            return False
        set_paused(directory, current, reason)
        return True
    if locked:
        return apply()
    with pending.lock(Path(directory)):
        return apply()


def pause(directory, session, reason='paused for revision', expected=None):
    with pending.lock(Path(directory)):
        value, _, _ = owned(directory, session)
        if expected is not None and value['revision'] != expected:
            raise ValueError('reservation revision changed; inspect before pausing')
        return set_paused(directory, value, reason)


def resume(directory, session, expected=None):
    with pending.lock(Path(directory)):
        original, _, _ = owned(directory, session)
        if original['state'] == 'queued':
            reconcile(directory, session, original['pid'])
    if expected is not None and original['revision'] != expected:
        raise ValueError('reservation revision changed; inspect before resuming')
    if original['state'] != 'paused':
        return dict(original, changed=False)
    # Validate the bound preparation. Changed source must be explicitly accepted
    # through edit first; resume never invents a new evidence identity.
    with pending.owner_environment(original, directory):
        pending.validate(original, Path(directory))
        if original.get('prepare_manifest'):
            import fleet_prepare
            import fleet_prepared
            value = (fleet_prepared.read(directory,original['prepare_manifest']) if original.get('prepare_receipt_required')
                     else json.loads(Path(original['prepare_manifest']).read_text()))
            fleet_prepare.validate(value, refresh=True, directory=directory)
            if original.get('validation_env', {}).get('FLEET_VALIDATION_REQUIRED') == '1' and original['kind'] == 'boot':
                fleet_prepare.validate_targets(directory, original['prepare_manifest'], verify_only=True,
                                               controller=original)
    with pending.lock(Path(directory)):
        current, _, _ = owned(directory, session)
        if current != original:
            raise ValueError('reservation changed while checking resume; inspect it before retrying')
        value = copy.deepcopy(current)
        value.update(state='queued', phase='waiting', resumed_at=time.time(), revision=value['revision'] + 1)
        value.pop('pause_reason', None)
        pending.save_record(Path(directory), value)
        reconcile(directory, session, value['pid'])
        return dict(value, changed=True)


def admission(directory, session):
    """Called under admission's lock. Never remove the row on preparation failure."""
    if paused(directory, session):
        return 4
    import fleet_prepare
    checked = pending.read_record(Path(directory), session)
    try:
        fleet_prepare.check_pending(Path(directory), session, external=False)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        if checked and pause_failed(directory, session, checked, str(exc), locked=True):
            print('PAUSED before GO: ' + str(exc), file=sys.stderr)
            return 4
        print('PREPARE REFUSED: ' + str(exc), file=sys.stderr)
        return 3
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['pause', 'resume', 'is-paused', 'admission', 'reconcile', 'list', 'pid'])
    ap.add_argument('session', nargs='?')
    ap.add_argument('--pid', type=int)
    ap.add_argument('--format', choices=['json', 'text'], default='json')
    ap.add_argument('--reason', default='paused for revision')
    ap.add_argument('--expect-revision', type=int)
    args = ap.parse_args(argv)
    directory = Path(os.environ['FLEET_DIR'])
    try:
        if args.action == 'list':
            values = parked(directory)
            if args.format == 'text':
                for value in values:
                    print(f"  {value['session']} [paused], ticket {value['ticket']}: {value.get('pause_reason', '')}")
            else:
                print(json.dumps([{k:v[k] for k in ('session','ticket','pid','state','revision','enqueued_at','pause_reason') if k in v}
                                  for v in values], ensure_ascii=False))
            return 0
        if not args.session:
            ap.error('this action requires a session')
        if args.action == 'pid':
            value = pending.read_record(directory, args.session)
            if value and (paused(directory, args.session) or value.get('state') == 'queued'
                          and value.get('pause_protocol') == 1 and pending.parked_row(value) and handoff.live(value)):
                print(value['pid'])
                return 0
            return 1
        if args.action == 'reconcile':
            # This action is used by _enqueue, which already holds .lock.
            return reconcile(directory, args.session, args.pid)
        if args.action == 'is-paused':
            with pending.lock(directory):
                if paused(directory, args.session):
                    reconcile(directory, args.session)
                    return 0
                return 1
        if args.action == 'admission':
            return admission(directory, args.session)
        value = (pause(directory,args.session,args.reason,args.expect_revision) if args.action == 'pause'
                 else resume(directory,args.session,args.expect_revision))
        print(json.dumps({k:value[k] for k in ('session','ticket','state','revision','enqueued_at','pause_reason') if k in value},ensure_ascii=False))
        return 0
    except (ValueError,OSError,subprocess.SubprocessError) as exc:
        print(args.action + ' refused: ' + str(exc),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
