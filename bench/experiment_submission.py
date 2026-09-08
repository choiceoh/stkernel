# SPDX-License-Identifier: Apache-2.0
"""Prepare a bounded DAG once, recheck shared inputs, then publish atomically."""
import copy
import json
import os
from pathlib import Path
import re
import shutil


class Context:
    def __init__(self, repo):
        from experiments import BASE_ENV
        self.repo = repo
        self.environment = {k:os.environ[k] for k in BASE_ENV if k in os.environ}
        logd = Path(os.environ.get('LOGD','/home/choiceoh/glm53-logs'))
        self.paths = dict(LOGD=str(logd),FLEET_DIR=os.environ.get('FLEET_DIR',str(logd/'fleet')),
            ONEPASS_JSONL=os.environ.get('ONEPASS_JSONL',str(logd/'bracket-onepass.jsonl')),
            ONEPASS_VERDICTS=os.environ.get('ONEPASS_VERDICTS',str(logd/'verdicts.jsonl')),
            MK_OVERLAY_STAMP=os.environ.get('MK_OVERLAY_STAMP',str(Path.home()/'glm53-cache/.overlay-sha')))
        self.memo = {}
        self.snapshots = {}

    def payload(self, spec):
        from experiments import snapshot, encoded
        from cpu_evidence import identity
        # Dependencies/objectives do not change source/deployment attestation.
        key = encoded({k:spec.get(k) for k in ('kind','revision','context','inputs','probe_contract')})
        if key not in self.snapshots:
            self.snapshots[key] = snapshot(self.repo,spec,self.paths['MK_OVERLAY_STAMP'])
        payload = dict(spec=copy.deepcopy(spec),repo=str(self.repo),paths=self.paths,
                       bash=shutil.which('bash'),environment=self.environment,snapshot=self.snapshots[key])
        if spec['kind'] == 'cpu':
            payload['cpu_identity'] = identity(self.repo,spec,self.environment,self.memo)
        return payload


def cacheable_dependencies(store, payload):
    # Generated build inputs are not covered by the tracked-tree content key.
    return all(not store.get(d)['payload']['spec'].get('outputs')
               and not (store.get(d)['result'] or {}).get('artifacts')
               for d in payload['spec']['depends_on'])


def reuse(store, job, payload):
    """Only used for freshly attested submissions, before private checkout creation."""
    from experiments import TERMINAL
    row = store.get(job)
    if (row['state'] in TERMINAL or row['started'] is not None or row['repeat_reason']
            or not payload.get('cpu_identity') or not cacheable_dependencies(store,payload)
            or any(store.get(d)['state'] != 'succeeded' for d in payload['spec']['depends_on'])):
        return False
    hit = store.db.execute('SELECT job FROM cpu_cache WHERE key=?',(payload['cpu_identity']['key'],)).fetchone()
    if not hit:
        return False
    source = store.get(hit['job'])
    from prepared_artifacts import intact
    result = source['result'] or {}
    if (source['state'] != 'succeeded' or result.get('checks',{}).get('passed') is not True
            or result.get('checks',{}).get('coverage_complete') is not True or not intact(result.get('artifacts',[]))):
        return False
    store.state(job,'succeeded',dict(result,revision=payload['spec']['revision'],cache_source=source['id'],
        tested_revision=source['payload']['spec']['revision'],cache_identity=payload['cpu_identity'],
        cache_path='before-checkout'))
    return True


def submit_many(store, session, requests, repo, *, repeat=None, launch=True):
    from experiments import normalize, ensure_worker, worker_lock
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',session) or repeat is not None and not repeat.strip():
        raise ValueError('invalid session or empty repeat reason')
    if not isinstance(requests,list) or not 1 <= len(requests) <= 32:
        raise ValueError('batch must contain 1..32 requests')
    context = Context(repo)
    pending = []
    names = set()
    for item in requests:
        if not isinstance(item,dict) or set(item)-{'name','manifest','requires'}:
            raise ValueError('batch request needs name, manifest and optional requires')
        name,requires = item.get('name'),item.get('requires',[])
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',name) or name in names:
            raise ValueError('batch request names must be unique short identifiers')
        if not isinstance(requires,list) or any(not isinstance(n,str) or n not in names for n in requires):
            raise ValueError('batch requires must name earlier requests; cycles are not allowed')
        names.add(name)
        pending.append((name,requires,context.payload(normalize(item.get('manifest'),repo))))
    # This memo is scoped to this request only. Never retain package/source
    # fingerprints across calls or trust a TTL when an agent changes inputs.
    fresh = Context(repo)
    for _,_,payload in pending:
        if fresh.payload(payload['spec']) != payload:
            raise ValueError('batch source, deployment or CPU environment changed during preparation')
    answers,ids = [],{}
    with store.transaction():
        for name,requires,payload in pending:
            payload['spec']['depends_on'] = sorted(set(payload['spec']['depends_on']+[ids[n] for n in requires]))
            answer = store.submit(session,payload,repeat)
            ids[name] = answer['id']
            answers.append(dict(name=name,**answer))
    # Another subscriber can have launched a deduplicated job in the meantime.
    # Retain its worker lock through cache publication; never finish live work.
    for answer,(_,_,payload) in zip(answers,pending):
        lock = worker_lock(store,answer['id'])
        if lock is not None:
            with lock,store.transaction():
                if reuse(store,answer['id'],payload):
                    answer.update(state='succeeded',cache_hit=True)
    if launch:
        for answer in answers:
            ensure_worker(store,answer['id'])
    return dict(requests=answers)
