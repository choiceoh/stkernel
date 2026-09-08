#!/usr/bin/env python3
"""Fleet-held compact/inline transport correctness; fresh four-rank receipts.

Default: one distributed check-only cohort. Memcheck/racecheck are explicit
optional --stage values. No timing microbench, serving boot, reservation,
or legacy evidence reuse is performed by this payload.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import ar_consumer_gpu_identity as identity
import reuse_ar_consumer_gpu_evidence as evidence
import run_ar_consumer_gpu as base
from decode_transport_gpu_probe import FLAGS, SCHEMA, validate_transport_proof

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('probe', 'memcheck', 'racecheck')
HARNESS = ('probes/decode_transport_gpu_probe.py', 'probes/run_decode_transport_gpu.py')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_identity():
    result = evidence.source()
    result['harness_sha256'] = {name: digest((ROOT/name).read_bytes()) for name in HARNESS}
    return result


def validate_cohort(out, stage, completed, runtime, source):
    """Legacy numerical/sanitizer checks plus mandatory new mode proof."""
    if stage not in STAGES:
        raise ValueError('unknown correctness stage')
    expected = {stage+'-rank'+str(rank) for rank in range(4)}
    if (len(completed) != 4 or {row.get('stage') for row in completed} != expected):
        raise ValueError('one complete four-rank cohort required')
    entries, artifacts, _ = evidence.validate_group(out, stage, completed, runtime)
    for entry in entries:
        report = evidence._record(artifacts[entry['stage']+'.json'])
        validate_transport_proof(report)
        if report.get('wrapper_sha256') != source['harness_sha256'][HARNESS[0]]:
            raise ValueError('rank executed another transport wrapper')
    return entries, {name: digest(data) for name, data in artifacts.items()}


def verify_admission(out, *, required_stages=('probe',)):
    """Read-only check for a following onepass; never accepts old AR evidence."""
    out = Path(out)
    report = evidence._record((out/'admission.json').read_bytes())
    if (report.get('schema') != SCHEMA or report.get('status') != 'PASS'
            or report.get('flags') != FLAGS or report.get('image') != base.IMAGE
            or not set(required_stages) <= set(report.get('selected_stages', []))):
        raise ValueError('new compact/inline cohort admission required')
    source = evidence._record((out/'source.json').read_bytes())
    runtime = evidence._record((out/'runtime.json').read_bytes())
    identity.validate_runtime(runtime)
    if source != source_identity():
        raise ValueError('tested source changed before onepass')
    hashes = report.get('artifacts_sha256', {})
    for name in ('source.json', 'runtime.json'):
        if hashes.get(name) != digest((out/name).read_bytes()):
            raise ValueError('identity evidence changed after admission')
    for stage in required_stages:
        cohort = [row for row in report['completed'] if row['stage'].startswith(stage+'-rank')]
        _, artifacts = validate_cohort(out, stage, cohort, runtime, source)
        if any(hashes.get(name) != value for name, value in artifacts.items()):
            raise ValueError('GPU artifact changed after admission')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, action='append',
                        help='default probe only; sanitizers are explicit optional cohorts')
    args = parser.parse_args()
    args.out = args.out.resolve()
    selected = tuple(dict.fromkeys(args.stage or ('probe',)))
    session = os.environ['FLEET_SESSION']
    if not re.fullmatch('[A-Za-z0-9_-]+', session):
        raise ValueError('invalid fleet session')
    holder = Path('/home/choiceoh/glm53-logs/fleet/holder').read_text().strip().split('|')
    if holder[0] != session or holder[-1] != 'boot' or os.environ.get('FLEET_RESTORE_MANAGED') != '1':
        raise ValueError('matching supervised boot hold required')
    source = source_identity()
    runtime = evidence.collect_runtime(base.remote)
    args.out.mkdir(parents=True, exist_ok=False)
    base.atomic_json(args.out/'source.json', source)
    base.atomic_json(args.out/'runtime.json', runtime)
    admission = dict(schema=SCHEMA, status='RUNNING', image=base.IMAGE, flags=FLAGS,
        selected_stages=list(selected), completed=[], artifacts_sha256={
            name: digest((args.out/name).read_bytes()) for name in ('source.json','runtime.json')})
    base.atomic_json(args.out/'admission.json', admission)
    name = 'decode-transport-' + session
    peer_root = Path('/home/choiceoh/decode-transport-probe-' + session)
    peer_out = Path('/home/choiceoh/decode-transport-evidence-' + session)
    archive = subprocess.check_output(['git','-C',str(ROOT),'archive','--format=tar',source['revision']])

    def prepare(node):
        root = ROOT if node == 'local' else peer_root
        out = args.out if node == 'local' else peer_out
        if node != 'local':
            base.remote(node, ['mkdir',str(root),str(out)])
            base.remote(node, ['tar','-xf','-','-C',str(root)], input=archive)
        base.remote(node, ['docker','image','inspect',base.IMAGE], stdout=subprocess.DEVNULL)
        names = base.remote(node, ['docker','ps','--format','{{.Names}}'], capture_output=True, text=True).stdout.splitlines()
        if any(value in ('glm53','glm53-worker') for value in names):
            raise ValueError(f'serving must be stopped by the owning campaign first: {node}')
        base.remote(node, ['mkdir','-p',str(out/'build')])
        return root, out

    def stop_owned():
        for node in base.NODES:
            try:
                base.remote(node, ['docker','stop','-t','2',name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            except subprocess.SubprocessError:
                pass

    paths = None

    def run_rank(rank, stage):
        node = base.NODES[rank]
        root, out = paths[rank]
        memory = base.remote(node, ['python3','-c',
            "from pathlib import Path; print(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))"],
            capture_output=True, text=True)
        limit, required = base.memory_budget(stage)
        if int(memory.stdout.strip()) * 1024 < required:
            raise ValueError(f'{node}: need {required // base.GIB} GiB available before {stage}')
        command = ['docker','run','--name',name,'--gpus','device=0',
            '--network=host','--device=/dev/infiniband','--cap-add=IPC_LOCK',
            '--ulimit','memlock=-1:-1','--cpuset-cpus=14-17',
            f'--memory={limit}g',f'--memory-swap={limit}g','--shm-size=1g',
            '-e','MAX_JOBS=1','-e','OMP_NUM_THREADS=1',
            '-e','AR_CONSUMER_BUILD=/evidence/build',
            '-e','VLLM_GLM53_MK_BUILD_ROOT=/evidence/build/mk',
            '-e','VLLM_DSV4_OSAR_BUILD_ROOT=/evidence/build/osar',
            '-e','AR_CONSUMER_RANK='+str(rank),'-e','AR_CONSUMER_IPS='+base.IPS,
            '-e','AR_CONSUMER_INIT='+identity.INIT,
            '--mount',f'type=bind,src={root},dst=/repo,readonly',
            '--mount',f'type=bind,src={out},dst=/evidence',
            '--mount','type=bind,src=/usr/local/cuda/compute-sanitizer,dst=/san,readonly',
            '--workdir','/repo']
        for flag, value in FLAGS.items():
            command += ['-e',flag+'='+value]
        target = '/repo/probes/decode_transport_gpu_probe.py'
        if stage == 'racecheck':
            command += ['-e','NV_COMPUTE_SANITIZER_MAX_RACECHECK_HAZARDS=100000']
        if stage in ('memcheck','racecheck'):
            command += ['--entrypoint','/san/compute-sanitizer',base.IMAGE,'--tool',stage,
                        '--target-processes','application-only','--error-exitcode','77']
            if stage == 'racecheck':
                command += ['--racecheck-num-workers','4','--print-session-details',
                            '--kernel-name','regex='+base.RACECHECK_KERNELS]
            command += ['python3',target]
        else:
            command += ['--entrypoint','python3',base.IMAGE,target]
        filename = stage+'-rank'+str(rank)
        command += ['--out','/evidence/'+filename+'.json']
        with (args.out/(filename+'.log')).open('w') as log:
            base.run_container(node, command, name, args.out/(filename+'.container.json'), log)
        if node != 'local':
            data = base.remote(node, ['cat',str(out/(filename+'.json'))], capture_output=True, text=True).stdout
            (args.out/(filename+'.json')).write_text(data)
        result = evidence._record((args.out/(filename+'.json')).read_bytes())
        return dict(node=node, stage=filename, source_sha256=result['source_sha256'],
                    kernel_filter=base.RACECHECK_KERNELS if stage == 'racecheck' else None)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(prepare, base.NODES))
        for stage in selected:
            completed = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run_rank, rank, stage) for rank in range(4)]
                try:
                    for future in as_completed(futures):
                        completed.append(future.result())
                except BaseException:
                    stop_owned()
                    raise
            if source_identity() != source or evidence.collect_runtime(base.remote) != runtime:
                raise ValueError('source or GPU runtime changed during correctness cohort')
            completed, hashes = validate_cohort(args.out, stage, completed, runtime, source)
            admission['completed'].extend(completed)
            admission['artifacts_sha256'].update(hashes)
            base.atomic_json(args.out/'admission.json', admission)
        admission['status'] = 'PASS'
        base.atomic_json(args.out/'admission.json', admission)
        verify_admission(args.out, required_stages=selected)
        print('PASS fresh transport correctness: '+','.join(selected)+'; no timing samples', flush=True)
    except BaseException as exc:
        admission.update(status='FAIL', error=repr(exc))
        base.atomic_json(args.out/'admission.json', admission)
        raise
    finally:
        if paths is not None:
            stop_owned()


if __name__ == '__main__':
    main()
