# SPDX-License-Identifier: Apache-2.0
"""Build reviewable CPU plans and submit them through the normal API.

The GPU stage left with the vLLM pair lane (2026-09-18): GPU work is admitted
through fleet.sh run --gpu and the ST bracket lanes, so a plan is CPU checks
and CPU preparation only.
"""
import json
from pathlib import Path
import uuid


def build(raw, repo, base=None):
    from experiments import normalize, git
    gpu = dict(raw)
    cpu_jobs = gpu.pop('cpu_jobs',None)
    if cpu_jobs is None:
        import os
        policy = Path(os.environ.get('FLEET_DIR','/home/choiceoh/glm53-logs/fleet'))/'cpu-policy.json'
        cpu_jobs = min(2,json.loads(policy.read_text()).get('slots',2)) if policy.exists() else 2
    if type(cpu_jobs) is not int or not 1 <= cpu_jobs <= 8:
        raise ValueError('cpu_jobs must be 1..8')
    suites = gpu.pop('cpu_suites', None)
    tests = gpu.pop('cpu_tests', [])
    preparation = gpu.pop('prepare', [])
    revision = git(repo, 'rev-parse', 'HEAD')
    gpu.setdefault('kind', 'cpu')
    gpu.setdefault('revision', revision)
    gpu = normalize(gpu, repo)
    # The overlay math contracts retired with the overlay (2026-09-18); the
    # fleet suite is the one CPU gate this planner still names by default.
    if suites is None and not tests:
        suites = ['fleet']
    if not isinstance(suites, list) or not isinstance(tests, list) or not (suites or tests):
        raise ValueError('plan needs at least one CPU suite/test')
    from cpu_checks import SUITES
    if any(s not in SUITES for s in suites):
        raise ValueError('unknown CPU suite')
    import re
    if any(not isinstance(t, str) or not re.fullmatch(r'tests/test_[A-Za-z0-9_]+\.py', t) or not (repo / t).is_file() for t in tests):
        raise ValueError('CPU tests must name existing tests/test_*.py files')
    common = dict(kind='cpu', revision=gpu['revision'], context={}, inputs={})
    stages = []
    checks = [(suite,['--suite',suite]) for suite in suites]
    if tests:
        checks.append(('individual',[part for test in dict.fromkeys(tests) for part in ('--test',test)]))
    for name,arguments in checks:
        check = normalize(dict(kind='cpu', revision=gpu['revision'], hypothesis='CPU '+name+': '+gpu['hypothesis'],
            command=['python3','bench/cpu_checks.py',*arguments],
            resources={'cpu_slots':cpu_jobs if name=='fleet' else 1}),repo)
        stages.append(dict(name='checks' if len(checks)==1 else 'checks-'+name,manifest=check,requires=[]))
    if not isinstance(preparation, list) or len(preparation) > 8:
        raise ValueError('prepare must list at most eight CPU build stages')
    for index, step in enumerate(preparation):
        if not isinstance(step, dict) or set(step) - {'command', 'outputs', 'resources', 'env', 'timeout_s','requires'}:
            raise ValueError('prepare supports CPU command, outputs, env, resources, timeout_s and requires')
        step = dict(step)
        requires = step.pop('requires',[])
        if not isinstance(requires,list) or any(not isinstance(n,str) or n not in [s['name'] for s in stages] for n in requires):
            raise ValueError('prepare requires must name earlier CPU stages')
        manifest = normalize(dict(kind='cpu', revision=gpu['revision'], hypothesis='CPU preparation: ' + gpu['hypothesis'], **step), repo)
        stages.append(dict(name=f'prepare-{index+1}', manifest=manifest, requires=requires))
    return dict(revision=gpu['revision'], changed=git(repo, 'diff', '--name-only', base, 'HEAD').splitlines() if base else [],
                stages=stages,
                scope='CPU gates and preparation; GPU work is queued directly through fleet.sh run --gpu and the ST lanes')


def run(args, store, repo):
    plan = build(json.loads(args.manifest.read_text()), repo, args.base)
    directory = store.root / 'plans' / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    plan['path'] = str(directory / 'plan.json')
    plan['mode'] = 'submit' if args.submit else 'prepare-only' if args.prepare_only else 'preview'
    manifests = {stage['name']:stage['manifest'] for stage in plan['stages']}
    submissions = {}
    launched = set()
    def save():
        ids = {name:item['id'] for name,item in submissions.items()}
        for stage in plan['stages']:
            original = manifests[stage['name']]
            manifest = dict(original,depends_on=sorted(set(original['depends_on'] +
                [ids.get(d,'pending-stage:'+d) for d in stage['requires']])))
            path = directory / (stage['name'] + '.json')
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(manifest,indent=2) + '\n')
            temporary.replace(path)
            stage.update(manifest=manifest,path=str(path),
                         dependencies_resolved=all(d in ids for d in stage['requires']))
            if stage['name'] in submissions:
                stage['submission'] = submissions[stage['name']]
        temporary = directory / 'plan.tmp'
        temporary.write_text(json.dumps(plan, indent=2) + '\n')
        temporary.replace(plan['path'])
    def launch_registered():
        from experiments import ensure_worker
        for submission in submissions.values():
            if submission['id'] not in launched:
                ensure_worker(store,submission['id'])
                launched.add(submission['id'])
    save()
    if args.submit or args.prepare_only:
        from experiment_submission import submit_many
        try:
            selected = [s for s in plan['stages']]
            answer = submit_many(store,args.session,[dict(name=s['name'],manifest=manifests[s['name']],requires=s['requires'])
                for s in selected],repo,launch=False)
            submissions = {s['name']:s for s in answer['requests']}
            save()
            launch_registered()
        except (OSError,ValueError) as exc:
            plan['error'] = str(exc)
        finally:
            save()
            launch_registered()
    return plan
