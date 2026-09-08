# SPDX-License-Identifier: Apache-2.0
"""Compact, actionable evidence summaries; commands are suggestions, never run."""
from pathlib import Path
import json


CACHE_HISTORY_LIMIT = 50


def cpu_family(payload):
    """Compare named gate selections without resolving or executing commands."""
    spec = payload.get('spec', {})
    command = spec.get('command', [])
    if (spec.get('kind') != 'cpu' or not isinstance(command, list) or len(command) < 4
            or not all(isinstance(arg, str) for arg in command) or command[1] != 'bench/cpu_checks.py'):
        return None
    args = command[2:]
    if len(args) % 2 or any(args[i] not in {'--suite', '--test', '--contract'} for i in range(0, len(args), 2)):
        return None
    return tuple(sorted(set(zip(args[::2], args[1::2]))))


def cpu_cache_reuse(store, row):
    """Explain recorded CPU identities using a bounded, read-only history view.

    Matching identities alone do not prove reuse: dependency state, explicit
    repeat requests and artifact integrity have separate admission contracts.
    No current files, runtimes, queues or GPU baseline identities are inspected.
    """
    result = row.get('result') or {}
    answer = dict(status='unknown', source_id=None, changed_paths=[], changed_components=[])
    if result.get('cache_source'):
        return dict(answer, status='cached', source_id=result['cache_source'],
                    reason='This result records reuse of successful CPU evidence')
    payload = row['payload']
    current = payload.get('cpu_identity')
    family = cpu_family(payload)
    if not family or not isinstance(current, dict) or not current.get('key'):
        return dict(answer, reason='No comparable CPU identity was recorded for this command')
    if not isinstance(current.get('dependencies'), dict) or not isinstance(current.get('components'), dict):
        return dict(answer, reason='Recorded CPU identity has no component fingerprints')
    # Bound the rows visited before filtering. A long history dominated by
    # other workloads returns unknown instead of scanning all past jobs.
    rows = store.db.execute('SELECT id,payload,state,result,finished FROM jobs '
                            'WHERE rowid < COALESCE((SELECT rowid FROM jobs WHERE id=?),0) '
                            'ORDER BY rowid DESC LIMIT ?', (row['id'], CACHE_HISTORY_LIMIT)).fetchall()
    previous = None
    for candidate in rows:
        if candidate['state'] != 'succeeded':
            continue
        # Later completion cannot explain the original submission's cache state.
        if candidate['finished'] is None or candidate['finished'] > row.get('created', 0):
            continue
        old_payload = json.loads(candidate['payload'])
        if cpu_family(old_payload) != family:
            continue
        old_result = json.loads(candidate['result']) if candidate['result'] else {}
        checks = old_result.get('checks') or {}
        if checks.get('passed') is not True or checks.get('coverage_complete') is not True:
            continue
        previous = candidate['id'], old_payload.get('cpu_identity')
        break
    if previous is None:
        return dict(answer, reason='No comparable successful CPU evidence in the previous 50 requests',
                    search_limit=CACHE_HISTORY_LIMIT)
    source_id, expected = previous
    if (not isinstance(expected, dict) or not expected.get('key')
            or not isinstance(expected.get('dependencies'), dict) or not isinstance(expected.get('components'), dict)):
        return dict(answer, source_id=source_id, reason='Previous CPU identity has no component fingerprints')
    from cpu_evidence import difference
    change = difference(expected, current)
    if not change['equal']:
        return dict(answer, status='changed', source_id=source_id,
                    changed_paths=change['changed_paths'], changed_components=change['changed_components'],
                    reason='Recorded source or execution requirements differ from the previous CPU evidence')
    reason = ('An independent sample was requested' if row.get('repeat_reason') else
              'CPU identity matches; no reuse was recorded. Dependencies and artifact checks also govern reuse')
    return dict(answer, status='identity_match', source_id=source_id, reason=reason)


def explain(store, row):
    payload = row['payload']
    spec = payload['spec']
    result = row['result'] or {}
    checks = result.get('checks',{}).get('checks',[])
    failed = [c for c in checks if c.get('returncode') or c.get('skipped') or not c.get('tests_run')
              or c.get('counts') and c['counts'].get('passed') is not True]
    blockers = []
    for dep in spec.get('depends_on',[]):
        source = store.get(dep)
        if source['state'] != 'succeeded':
            blockers.append(dict(id=dep,state=source['state'],hypothesis=source['payload']['spec'].get('hypothesis'),
                                 reason=(source['result'] or {}).get('reason'),log=source['log']))
    for dep in result.get('dependencies',[]):
        if not any(b['id']==dep for b in blockers):
            source=store.get(dep)
            blockers.append(dict(id=dep,state=source['state'],log=source['log']))
    fleet = str(Path(payload.get('repo','.'))/'bench/fleet.sh')
    actions = [dict(action='inspect_dependency',argv=['bash',fleet,'result',b['id'],'--details']) for b in blockers]
    if failed and spec.get('kind')=='cpu':
        actions.append(dict(action='reproduce_cpu_failure',cwd=payload['repo'],
            argv=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',*spec['command']],
            note='Reproduces the checked revision; repair and submit the new committed revision before promoting GPU work.'))
    if not actions and row['state'] in {'failed','incomplete','interrupted','blocked'}:
        actions.append(dict(action='inspect_evidence',argv=['bash',fleet,'result',row['id'],'--details'],path=row['log']))
    return dict(state=row['state'],evidence=result.get('evidence'),reason=result.get('reason'),
        **({'cache_reuse':cpu_cache_reuse(store,row)} if spec.get('kind') == 'cpu' else {}),
        **({k:result[k] for k in ('baseline_policy','comparison_complete','promotion_ready') if k in result}),
        failed_checks=[dict(suite=c.get('suite'),log=c.get('log'),skipped=c.get('skipped',[]),
                           failures=(c.get('counts') or {}).get('failure_details',[])) for c in failed],
        blocking_dependencies=blockers,next_actions=actions,automatic_retry=False,
        scope='CPU checks do not establish GPU numerical or serving performance evidence' if spec.get('kind')=='cpu' else result.get('scope'))
