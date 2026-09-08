# SPDX-License-Identifier: Apache-2.0
"""Phase timings and conservative runtime estimates, never performance evidence."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import time

WAIT_PHASES = dict(waiting_dependencies='dependencies', waiting_cpu_evidence='cpu_shared_wait',
                   waiting_cpu='cpu_queue', waiting_baseline='baseline_wait', waiting_group='shared_boot_wait',
                   ready_pair='group_collection')


def transition(store, row, state):
    old = row['state']
    if state == old:
        return
    now = time.time()
    for item in store.db.execute("SELECT phase,started FROM phase_open WHERE job=? AND phase LIKE 'state:%'",(row['id'],)).fetchall():
        store.db.execute('INSERT INTO timings VALUES(?,?,?,?,?)',
                         (row['id'],item['phase'][6:],max(0,now-item['started']),now,1))
        store.db.execute('DELETE FROM phase_open WHERE job=? AND phase=?',(row['id'],item['phase']))
    phase = WAIT_PHASES.get(state)
    kind = row['payload']['spec'].get('kind')
    if state == 'queued_fleet':
        phase = 'cpu_dispatch' if kind == 'cpu' else 'gpu_queue'
    if state == 'running' and kind == 'cpu':
        phase = 'cpu_run'
    if phase:
        store.db.execute('INSERT OR REPLACE INTO phase_open VALUES(?,?,?)',(row['id'],'state:'+phase,now))
    from experiments import TERMINAL
    if state in TERMINAL:
        for item in store.db.execute('SELECT phase,started FROM phase_open WHERE job=?',(row['id'],)).fetchall():
            store.db.execute('INSERT INTO timings VALUES(?,?,?,?,?)',(row['id'],item['phase'],max(0,now-item['started']),now,0))
        store.db.execute('DELETE FROM phase_open WHERE job=?',(row['id'],))


def mark(store, job, phase, action, ok=True):
    if phase not in {'boot','measure','restore','preparation','preflight','gpu_run'}:
        raise ValueError('unknown measured phase')
    now = time.time()
    with store.db:
        if action == 'start':
            store.db.execute('INSERT OR REPLACE INTO phase_open VALUES(?,?,?)',(job,phase,now))
        else:
            row = store.db.execute('SELECT started FROM phase_open WHERE job=? AND phase=?',(job,phase)).fetchone()
            if row:
                store.db.execute('INSERT INTO timings VALUES(?,?,?,?,?)',(job,phase,max(0,now-row['started']),now,int(ok)))
                store.db.execute('DELETE FROM phase_open WHERE job=? AND phase=?',(job,phase))


@contextmanager
def timed(store, job, phase):
    mark(store,job,phase,'start')
    try:
        yield
    except BaseException:
        mark(store,job,phase,'end',False)
        raise
    else:
        mark(store,job,phase,'end')


def signature(payload):
    spec = payload['spec']
    value = {k:spec.get(k) for k in ('kind','context','command','knobs','env','resources','api_port','baseline_policy')}
    value['host'] = payload.get('snapshot',{}).get('host')
    value['inputs'] = payload.get('snapshot',{}).get('inputs',{})
    if spec.get('kind') in {'pair','baseline'}:
        from serving_group import workloads
        value['workloads'] = payload.get('measurement_binding',{}).get('workloads') or workloads(spec)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def predict(conn, payload):
    declared = payload['spec'].get('estimate_min',15)
    answer = dict(minutes=declared,source='declared',samples=0)
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='timings'").fetchone():
        return answer
    phase = 'cpu_run' if payload['spec'].get('kind') == 'cpu' else 'gpu_run'
    wanted = signature(payload)
    values = []
    rows = conn.execute("SELECT j.payload,t.seconds FROM timings t JOIN jobs j ON j.id=t.job "
        "WHERE t.phase=? AND t.ok=1 AND j.state='succeeded' ORDER BY t.at DESC LIMIT 1000",(phase,)).fetchall()
    for raw, seconds in rows:
        if math.isfinite(seconds) and seconds > 0 and signature(json.loads(raw)) == wanted:
            values.append(seconds)
            if len(values) == 20:
                break
    if len(values) >= 3:
        p90 = sorted(values)[math.ceil(.9*len(values))-1]
        answer = dict(minutes=min(720,max(1,math.ceil(p90/60))),source='observed-p90',samples=len(values),
                      p50_s=statistics.median(values),p90_s=p90)
    return answer


def estimates(path):
    if not Path(path).exists():
        return {}
    try:
        with sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True,timeout=.2) as conn:
            rows = conn.execute("SELECT id,payload FROM jobs WHERE state='queued_fleet'").fetchall()
            return {'exp-'+job:predict(conn,json.loads(raw)) for job,raw in rows}
    except (sqlite3.Error,ValueError,KeyError,TypeError):
        return {}


def summary(store):
    values = {}
    for phase,seconds,ok in store.db.execute('SELECT phase,seconds,ok FROM timings'):
        item = values.setdefault(phase,dict(seconds=[],failed_samples=0))
        if ok:
            item['seconds'].append(seconds)
        else:
            item['failed_samples'] += 1
    for item in values.values():
        numbers = sorted(item.pop('seconds'))
        item.update(n=len(numbers),p50_s=statistics.median(numbers) if numbers else None,
                    p95_s=numbers[math.ceil(.95*len(numbers))-1] if numbers else None)
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase')
    parser.add_argument('action',choices=['start','end'])
    parser.add_argument('--failed',action='store_true')
    args = parser.parse_args()
    from experiments import Store
    mark(Store(os.environ['FLEET_EXPERIMENT_ROOT']),os.environ['FLEET_EXPERIMENT_ID'],args.phase,args.action,not args.failed)
