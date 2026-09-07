# SPDX-License-Identifier: Apache-2.0
"""Build reviewable CPU/prepare/GPU plans and submit them through the normal API."""
import json
from pathlib import Path
import uuid


def build(raw, repo, base=None):
    from experiments import normalize, git
    from measurement_contract import objective, workload
    gpu = dict(raw)
    cpu_jobs = gpu.pop('cpu_jobs',None)
    if cpu_jobs is None:
        import os
        policy = Path(os.environ.get('FLEET_DIR','/home/choiceoh/glm53-logs/fleet'))/'cpu-policy.json'
        cpu_jobs = min(2,json.loads(policy.read_text()).get('slots',2)) if policy.exists() else 2
    if type(cpu_jobs) is not int or not 1 <= cpu_jobs <= 8:
        raise ValueError('cpu_jobs must be 1..8')
    suites = gpu.pop('cpu_suites', None)
    contracts = gpu.pop('cpu_contracts', None)
    tests = gpu.pop('cpu_tests', [])
    preparation = gpu.pop('prepare', [])
    obj = objective(gpu.pop('objective', None))
    work = gpu.pop('workload', {})
    if obj['metric'] == 'decode_tokens' and not work:
        work = dict(fixed_decode_tokens=2048, fixed_decode_reps=3, require_exclusive=True)
    if 'evaluations' not in gpu:
        gpu['evaluations'] = [dict(objective=obj, workload=workload(work))]
    gpu.setdefault('kind', 'pair')
    gpu.setdefault('revision', git(repo, 'rev-parse', 'HEAD'))
    gpu = normalize(gpu, repo)
    if gpu['kind'] != 'pair':
        raise ValueError('plan expects a serving pair configuration')
    changed = git(repo, 'diff', '--name-only', base, 'HEAD').splitlines() if base else []
    if suites is None and contracts is None and not tests:
        from cpu_contracts import changed_contracts
        contracts = changed_contracts(repo,base)
    contracts = contracts or []
    if suites is None and contracts:
        suites = []
    elif suites is None:
        suites = ['fleet'] if changed and all(p.startswith('bench/') or p.startswith('tests/test_fleet') for p in changed) else ['logic']
        if any(p.startswith('launchers/') or p.startswith('profiles/') for p in changed):
            suites.append('startup')
    if not isinstance(suites, list) or not isinstance(tests, list) or not isinstance(contracts,list) or not (suites or tests or contracts):
        raise ValueError('plan needs at least one CPU suite/test')
    from cpu_contracts import CONTRACTS
    if any(not isinstance(c,str) or c not in CONTRACTS for c in contracts):
        raise ValueError('unknown CPU contract')
    from cpu_checks import SUITES
    if any(s not in SUITES for s in suites):
        raise ValueError('unknown CPU suite')
    import re
    if any(not isinstance(t, str) or not re.fullmatch(r'tests/test_[A-Za-z0-9_]+\.py', t) or not (repo / t).is_file() for t in tests):
        raise ValueError('CPU tests must name existing tests/test_*.py files')
    # Keep the aggregate deployment gate unchanged; plans represent its core
    # and fleet components as independently reusable required jobs.
    suites = list(dict.fromkeys(part for suite in suites for part in (['core','fleet'] if suite=='logic' else [suite])))
    common = dict(kind='cpu', revision=gpu['revision'], context=gpu['context'], inputs=gpu['inputs'])
    stages = []
    checks = [(suite,['--suite',suite]) for suite in suites]
    if tests:
        checks.append(('individual',[part for test in dict.fromkeys(tests) for part in ('--test',test)]))
    for name,arguments in checks:
        check = normalize(dict(common,hypothesis='CPU '+name+': '+gpu['hypothesis'],
            command=['python3','bench/cpu_checks.py',*arguments],
            resources={'cpu_slots':cpu_jobs if name=='fleet' else 1}),repo)
        stages.append(dict(name='checks' if len(checks)==1 else 'checks-'+name,manifest=check,requires=[]))
    for contract in dict.fromkeys(contracts):
        manifest = normalize(dict(common,hypothesis='CPU '+contract+': '+gpu['hypothesis'],
                             command=['python3','bench/cpu_checks.py','--contract',contract]),repo)
        stages.append(dict(name='checks-'+contract,manifest=manifest,requires=[]))
    if contracts and 'sensitivity' not in suites:
        manifest = normalize(dict(common,hypothesis='CPU fault sensitivity: '+gpu['hypothesis'],
                             command=['python3','bench/cpu_checks.py','--suite','sensitivity']),repo)
        stages.append(dict(name='sensitivity',manifest=manifest,requires=[]))
    if not isinstance(preparation, list) or len(preparation) > 8:
        raise ValueError('prepare must list at most eight CPU build stages')
    for index, step in enumerate(preparation):
        if not isinstance(step, dict) or set(step) - {'command', 'outputs', 'resources', 'env', 'timeout_s','requires'}:
            raise ValueError('prepare supports CPU command, outputs, env, resources, timeout_s and requires')
        step = dict(step)
        requires = step.pop('requires',[])
        if not isinstance(requires,list) or any(not isinstance(n,str) or n not in [s['name'] for s in stages] for n in requires):
            raise ValueError('prepare requires must name earlier CPU stages')
        manifest = normalize(dict(common, hypothesis='CPU preparation: ' + gpu['hypothesis'], **step), repo)
        stages.append(dict(name=f'prepare-{index+1}', manifest=manifest, requires=requires))
    stages.append(dict(name='gpu', manifest=gpu, requires=[s['name'] for s in stages]))
    return dict(revision=gpu['revision'], changed=changed, stages=stages,
                scope='CPU gates and preparation precede shared baselines and grouped onepass workloads')


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
    # Persist a preview before registration. The batch validates all included
    # stages first, shares source/runtime reads and registers the DAG atomically.
    selected = [s for s in plan['stages'] if s['name']!='gpu']
    save()
    if args.submit or args.prepare_only:
        from experiment_submission import submit_many
        try:
            answer = submit_many(store,args.session,[dict(name=s['name'],manifest=manifests[s['name']],requires=s['requires'])
                for s in selected],repo,launch=False)
            submissions = {s['name']:s for s in answer['requests']}
            # Make CPU IDs recoverable and start their normal workers before
            # potentially slow or unavailable deployment attestation. GPU jobs
            # still depend on every CPU stage; explicit preparation edges stay.
            save()
            launch_registered()
            if not args.prepare_only:
                gpu = dict(manifests['gpu'])
                gpu['depends_on'] = sorted(set(gpu['depends_on']+[s['id'] for s in submissions.values()]))
                submissions['gpu'] = submit_many(store,args.session,[dict(name='gpu',manifest=gpu)],repo,launch=False)['requests'][0]
            if 'gpu' in submissions and getattr(args,'supersedes',[]):
                from experiment_retirement import retire
                submissions['gpu']['superseded'] = [retire(store,args.session,old,submissions['gpu']['id'],'Replaced by newer plan')
                    for old in args.supersedes if old != submissions['gpu']['id']]
        except (OSError,ValueError) as exc:
            plan['error'] = str(exc)
        finally:
            save()
            launch_registered()
    return plan
