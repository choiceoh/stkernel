# SPDX-License-Identifier: Apache-2.0
"""Coalesce ready pair requests; seal their physical measurement plan at GO."""
import copy
import hashlib
import json
import time
from measurement_contract import evaluations
from serving_group import workloads


def key(payload):
    from experiments import encoded
    spec = payload['spec']
    value = dict(snapshot=payload['snapshot'], paths=payload['paths'],
                 environment={k:v for k,v in payload['environment'].items() if k != 'SSH_AUTH_SOCK'},
                 spec={k:spec.get(k) for k in ('kind','revision','knobs','context','env','inputs','resources','api_port','baseline_policy')},
                 artifacts=[{k:a[k] for k in ('relative','sha256')} for a in payload.get('prepared_artifacts',[])])
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def register(store, job):
    from experiments import encoded, TERMINAL, RetiredJob
    row = store.get(job)
    if row['repeat_reason']:
        return job  # deliberate independent samples never join another boot
    payload = row['payload']
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        if store.get(job)['state'] == 'retired':
            raise RetiredJob(job)
        existing = store.db.execute('SELECT leader FROM group_members WHERE job=?',(job,)).fetchone()
        if existing:
            return existing['leader']
        wanted = workloads(payload['spec'])
        match = None
        for group in store.db.execute('SELECT * FROM execution_groups WHERE signature=? AND sealed=0 ORDER BY created',
                                      (key(payload),)).fetchall():
            if store.get(group['leader'])['state'] in TERMINAL:
                continue
            if store.db.execute('SELECT count(*) FROM group_members WHERE leader=?',(group['leader'],)).fetchone()[0] >= 8:
                continue
            union = json.loads(group['workloads'])
            union += [w for w in wanted if w not in union]
            if len(union) <= 6:
                match = group['leader']
                break
        if match is None:
            match, union = job, wanted
            store.db.execute('INSERT INTO execution_groups VALUES(?,?,0,?,?)',
                             (match,key(payload),encoded(union),time.time()))
        else:
            store.db.execute('UPDATE execution_groups SET workloads=? WHERE leader=?',(encoded(union),match))
        store.db.execute('INSERT INTO group_members VALUES(?,?)',(job,match))
        if job != match:
            store.db.execute('INSERT OR IGNORE INTO dependencies VALUES(?,?,?)',(job,match,'gpu-shared'))
        payload['measurement_binding'] = dict(producer=match, workloads=union)
        store.db.execute('UPDATE jobs SET payload=? WHERE id=?',(encoded(payload),job))
        if job != match:
            # Queue estimates must include workloads added by later consumers.
            owner = store.get(match)['payload']
            owner['measurement_binding'] = dict(producer=match,workloads=union)
            store.db.execute('UPDATE jobs SET payload=? WHERE id=?',(encoded(owner),match))
        store.event(job,'boot_group',dict(leader=match,workloads=len(union)))
    return match


def seal(store, leader, payload):
    from experiments import encoded, RetiredJob
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        if store.get(leader)['state'] == 'retired':
            raise RetiredJob(leader)
        group = store.db.execute('SELECT * FROM execution_groups WHERE leader=?',(leader,)).fetchone()
        if not group:
            return payload
        union = json.loads(group['workloads'])
        store.db.execute('UPDATE execution_groups SET sealed=1 WHERE leader=?',(leader,))
        for member in store.db.execute('SELECT job FROM group_members WHERE leader=?',(leader,)).fetchall():
            row = store.get(member['job'])
            if row['state'] == 'retired':
                continue
            bound = row['payload']
            bound['measurement_binding'] = dict(producer=leader,workloads=union)
            store.db.execute('UPDATE jobs SET payload=? WHERE id=?',(encoded(bound),row['id']))
        store.event(leader,'boot_group_sealed',dict(workloads=union))
    effective = copy.deepcopy(payload)
    effective['spec']['evaluations'] = [dict(objective={'metric':'decode_steps'},workload=w) for w in union]
    return effective


def validate_members(store, leader):
    from experiments import verify, RetiredJob
    from experiment_baselines import ready
    from experiment_resources import readiness
    for member in store.db.execute('SELECT job FROM group_members WHERE leader=?',(leader,)).fetchall():
        row = store.get(member['job'])
        if row['state'] == 'retired':
            continue
        verify(row['payload'])
        readiness(row['payload']['spec']['resources'])
        if not ready(row['payload']):
            raise ValueError('shared boot member no longer has its required baseline evidence')
        if any(store.get(d)['state'] != 'succeeded' for d in row['payload']['spec']['depends_on']):
            raise ValueError('shared boot member prerequisite no longer passes')
        try:
            store.state(row['id'],'running')
        except RetiredJob:
            continue  # a consumer withdrew while its admission checks ran


def publish(store, leader, state, result):
    from experiments import pair_result, verify
    for member in store.db.execute('SELECT job FROM group_members WHERE leader=?',(leader,)).fetchall():
        job = member['job']
        if job == leader or store.get(job)['state'] == 'retired':
            continue
        if state == 'failed':
            status, value = 'failed', dict(evidence='gpu-pair', reason='shared serving execution failed',
                                          execution_job=leader, execution_result=result)
        else:
            try:
                payload = store.get(job)['payload']
                verify(payload)
                status, value = pair_result(payload, job)
            except (OSError, ValueError) as exc:
                status, value = 'failed', dict(evidence='gpu-pair', reason=str(exc))
            value['execution_job'] = leader
        store.state(job,status,value)
