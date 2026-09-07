# SPDX-License-Identifier: Apache-2.0
"""Build reviewable CPU/prepare/GPU plans and submit them through the normal API."""
import json
from pathlib import Path
import subprocess
import sys
import uuid


def build(raw, repo, base=None):
    from experiments import normalize, git
    from measurement_contract import objective, workload
    gpu = dict(raw)
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
    command = ['python3', 'bench/cpu_checks.py']
    for suite in dict.fromkeys(suites):
        command.extend(['--suite', suite])
    for test in dict.fromkeys(tests):
        command.extend(['--test', test])
    common = dict(kind='cpu', revision=gpu['revision'], context=gpu['context'], inputs=gpu['inputs'])
    stages = []
    if suites or tests:
        check = normalize(dict(common, hypothesis='CPU gates: ' + gpu['hypothesis'], command=command), repo)
        stages.append(dict(name='checks',manifest=check,requires=[]))
    for contract in dict.fromkeys(contracts):
        manifest = normalize(dict(common,hypothesis='CPU '+contract+': '+gpu['hypothesis'],
                             command=['python3','bench/cpu_checks.py','--contract',contract]),repo)
        stages.append(dict(name='checks-'+contract,manifest=manifest,requires=[]))
    if contracts:
        manifest = normalize(dict(common,hypothesis='CPU fault sensitivity: '+gpu['hypothesis'],
                             command=['python3','bench/cpu_checks.py','--suite','sensitivity']),repo)
        stages.append(dict(name='sensitivity',manifest=manifest,requires=[]))
    check_names = [s['name'] for s in stages]
    if not isinstance(preparation, list) or len(preparation) > 8:
        raise ValueError('prepare must list at most eight CPU build stages')
    for index, step in enumerate(preparation):
        if not isinstance(step, dict) or set(step) - {'command', 'outputs', 'resources', 'env', 'timeout_s'}:
            raise ValueError('prepare supports CPU command, outputs, env, resources and timeout_s')
        manifest = normalize(dict(common, hypothesis='CPU preparation: ' + gpu['hypothesis'], **step), repo)
        stages.append(dict(name=f'prepare-{index+1}', manifest=manifest, requires=check_names))
    stages.append(dict(name='gpu', manifest=gpu, requires=[s['name'] for s in stages]))
    return dict(revision=gpu['revision'], changed=changed, stages=stages,
                scope='CPU gates and preparation precede shared baselines and grouped onepass workloads')


def run(args, store, repo):
    plan = build(json.loads(args.manifest.read_text()), repo, args.base)
    directory = store.root / 'plans' / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    plan['path'] = str(directory / 'plan.json')
    plan['mode'] = 'submit' if args.submit else 'prepare-only' if args.prepare_only else 'preview'
    def save():
        temporary = directory / 'plan.tmp'
        temporary.write_text(json.dumps(plan, indent=2) + '\n')
        temporary.replace(plan['path'])
    ids = {}
    for stage in plan['stages']:
        manifest = stage['manifest']
        manifest['depends_on'] = sorted(set(manifest['depends_on'] + [ids.get(d, 'pending-stage:' + d) for d in stage['requires']]))
        path = directory / (stage['name'] + '.json')
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        stage['path'] = str(path)
        stage['dependencies_resolved'] = all(d in ids for d in stage['requires'])
        save()
        if (args.submit or args.prepare_only) and (stage['name'] != 'gpu' or not args.prepare_only):
            supersedes = [part for old in getattr(args,'supersedes',[]) for part in ('--supersedes',old)] if stage['name']=='gpu' else []
            process = subprocess.run([sys.executable, str(Path(__file__).with_name('experiments.py')),
                '--root', str(store.root), 'submit', args.session, str(path), *supersedes], text=True, capture_output=True)
            if process.returncode:
                stage['error'] = process.stderr.strip() or process.stdout.strip()
                plan['error'] = 'Submission stopped at ' + stage['name'] + '; earlier submitted jobs remain available'
                save()
                return plan
            output = process.stdout
            stage['submission'] = json.loads(output)
            ids[stage['name']] = stage['submission']['id']
    save()
    return plan
