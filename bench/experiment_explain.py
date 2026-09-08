# SPDX-License-Identifier: Apache-2.0
"""Compact, actionable evidence summaries; commands are suggestions, never run."""
from pathlib import Path


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
        **({k:result[k] for k in ('baseline_policy','comparison_complete','promotion_ready') if k in result}),
        failed_checks=[dict(suite=c.get('suite'),log=c.get('log'),skipped=c.get('skipped',[]),
                           failures=(c.get('counts') or {}).get('failure_details',[])) for c in failed],
        blocking_dependencies=blockers,next_actions=actions,automatic_retry=False,
        scope='CPU checks do not establish GPU numerical or serving performance evidence' if spec.get('kind')=='cpu' else result.get('scope'))
