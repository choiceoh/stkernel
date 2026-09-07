# SPDX-License-Identifier: Apache-2.0
"""Withdraw superseded demand; retire only unstarted, unneeded requests."""
import fcntl
import json
from pathlib import Path
import tempfile


def subscribed(store, session, job):
    return store.db.execute('SELECT 1 FROM subscribers WHERE session=? AND job=?',(session,job)).fetchone() is not None


def consumers(store, job):
    from experiments import TERMINAL
    active = store.db.execute('SELECT count(*) FROM subscribers s WHERE s.job=? AND NOT EXISTS '
        '(SELECT 1 FROM withdrawals w WHERE w.job=s.job AND w.session=s.session)',(job,)).fetchone()[0]
    dependents = set()
    for row in store.db.execute('SELECT id,state,payload FROM jobs'):
        if row['state'] not in TERMINAL and row['id'] != job:
            if job in json.loads(row['payload'])['spec'].get('depends_on',[]):
                dependents.add(row['id'])
    for row in store.db.execute('SELECT d.job,j.state FROM dependencies d JOIN jobs j ON j.id=d.job WHERE d.dependency=?',(job,)):
        if row['state'] not in TERMINAL and row['job'] != job:
            dependents.add(row['job'])
    return active, sorted(dependents)


def dequeue(payload, job):
    directory = Path(payload['paths']['FLEET_DIR'])
    directory.mkdir(exist_ok=True,parents=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path = directory/'queue'
        if not path.exists():
            return
        lines = path.read_text().splitlines(True)
        keep = [line for line in lines if len(line.split('|')) < 2 or line.split('|')[1] != 'exp-'+job]
        if lines != keep:
            with tempfile.NamedTemporaryFile(mode='w',dir=directory,delete=False) as out:
                out.writelines(keep)
                temporary = Path(out.name)
            temporary.replace(path)
    # Never signal processes or alter a live holder. Managed wait/execute paths
    # observe the retired state and exit without starting a boot.


def retire(store, session, old, replacement, reason):
    from experiments import TERMINAL
    if old == replacement or not reason.strip():
        raise ValueError('retirement needs a different replacement and a reason')
    retired = False
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        if not subscribed(store,session,old) or not subscribed(store,session,replacement):
            raise ValueError('only a subscriber can replace their request with another subscribed request')
        row, new = store.get(old), store.get(replacement)
        if row['payload']['spec']['kind'] != new['payload']['spec']['kind']:
            raise ValueError('replacement must have the same experiment kind')
        if new['state'] in TERMINAL and new['state'] != 'succeeded':
            raise ValueError('replacement must still be viable')
        store.db.execute('INSERT OR REPLACE INTO withdrawals VALUES(?,?,?,?)',(old,session,replacement,reason))
        active, dependents = consumers(store,old)
        if not active and not dependents and row['started'] is None and row['state'] not in TERMINAL:
            store.state(old,'retired',dict(reason=reason,replacement=replacement))
            retired = True
        store.event(replacement,'supersession',dict(old=old,retired=retired,remaining_subscribers=active,
                                                   dependents=dependents,reason=reason))
    if retired:
        dequeue(row['payload'],old)
    return dict(old=old,replacement=replacement,retired=retired,state=store.get(old)['state'],
                remaining_subscribers=active,dependents=dependents)
