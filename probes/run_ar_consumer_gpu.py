#!/usr/bin/env python3
"""Fleet-held four-node AR/MHC/GEMM probe; stop only its own containers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
NODES = ('local', '10.10.10.1', '10.10.10.3', '10.10.10.4')
IPS = '10.10.10.2,10.10.10.1,10.10.10.3,10.10.10.4'


def remote(node, argv, **kwargs):
    command = argv if node == 'local' else ['ssh', '-o', 'BatchMode=yes',
                'choiceoh@' + node, shlex.join(argv)]
    return subprocess.run(command, check=True, timeout=kwargs.pop('timeout', 900), **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    session = os.environ['FLEET_SESSION']
    assert re.fullmatch('[A-Za-z0-9_-]+', session)
    holder = Path('/home/choiceoh/glm53-logs/fleet/holder').read_text().strip().split('|')
    assert holder[0] == session and holder[-1] == 'boot', holder
    assert os.environ.get('FLEET_RESTORE_MANAGED') == '1', 'supervised boot hold required'
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'source.commit').write_text(revision + '\n')
    name = 'arconsumer-' + session
    peer_root = Path('/home/choiceoh/ar-consumer-probe-' + session)
    peer_out = Path('/home/choiceoh/ar-consumer-evidence-' + session)
    receipts = []
    # Workers need only this committed source, not a pre-existing Git clone
    # or GitHub credentials. Every rank later attests the actual CUDA bytes.
    archive = subprocess.check_output(['git', '-C', str(ROOT), 'archive', '--format=tar', revision])

    def prepare(node):
        root = ROOT if node == 'local' else peer_root
        out = args.out if node == 'local' else peer_out
        if node != 'local':
            remote(node, ['mkdir', str(root), str(out)])
            remote(node, ['tar', '-xf', '-', '-C', str(root)], input=archive)
        remote(node, ['docker', 'image', 'inspect', IMAGE], stdout=subprocess.DEVNULL)
        names = remote(node, ['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True).stdout.splitlines()
        assert not any(n in ('glm53', 'glm53-worker') for n in names), (node, names)
        remote(node, ['mkdir', '-p', str(out / 'build')])
        return root, out

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(prepare, NODES))

    def stop_owned():
        for node in NODES:
            try:
                remote(node, ['docker', 'stop', '-t', '2', name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15)
            except subprocess.SubprocessError:
                pass

    def run_rank(rank, stage, distributed=True):
        node = NODES[rank]
        root, out = paths[rank]
        # This is a maintenance probe: the serving weights have been stopped.
        mem = remote(node, ['python3', '-c',
            "from pathlib import Path; print(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))"],
            capture_output=True, text=True)
        assert int(mem.stdout.strip()) >= 16 * 1024**2, (node, 'less than 16 GiB available')
        cmd = ['docker', 'run', '--rm', '--name', name, '--gpus', 'device=0',
            '--network=host', '--device=/dev/infiniband', '--cap-add=IPC_LOCK',
            '--ulimit', 'memlock=-1:-1', '--cpuset-cpus=14-17', '--memory=8g', '--shm-size=1g',
            '-e', 'MAX_JOBS=1', '-e', 'OMP_NUM_THREADS=1',
            '-e', 'AR_CONSUMER_BUILD=/evidence/build',
            '-e', 'VLLM_GLM53_MK_BUILD_ROOT=/evidence/build/mk',
            '-e', 'VLLM_DSV4_OSAR_BUILD_ROOT=/evidence/build/osar',
            '-e', 'AR_CONSUMER_RANK=' + str(rank), '-e', 'AR_CONSUMER_IPS=' + IPS,
            '-e', 'AR_CONSUMER_INIT=tcp://10.10.10.2:29758',
            '--mount', f'type=bind,src={root},dst=/repo,readonly',
            '--mount', f'type=bind,src={out},dst=/evidence',
            '--mount', 'type=bind,src=/usr/local/cuda/compute-sanitizer,dst=/san,readonly',
            '--workdir', '/repo']
        target = '/repo/probes/ar_consumer_probe.py'
        if stage in ('memcheck', 'racecheck'):
            cmd += ['--entrypoint', '/san/compute-sanitizer', IMAGE, '--tool', stage,
                    '--target-processes', 'application-only', '--error-exitcode', '77',
                    'python3', target, '--check-only']
        else:
            cmd += ['--entrypoint', 'python3', IMAGE, target, '--trace']
        if distributed:
            cmd += ['--distributed']
        filename = ('' if distributed else 'local-') + stage + '-rank' + str(rank)
        cmd += ['--out', '/evidence/' + filename + '.json']
        with (args.out / (filename + '.log')).open('w') as log:
            remote(node, cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        if node != 'local':
            data = remote(node, ['cat', str(out / (filename + '.json'))], capture_output=True, text=True).stdout
            (args.out / (filename + '.json')).write_text(data)
        report = json.loads((args.out / (filename + '.json')).read_text())
        assert report['status'] == 'PASS' and len(report['cases']) == 36, (node, stage)
        return {'node': node, 'stage': filename, 'source_sha256': report['source_sha256']}

    try:
        for stage in ('probe', 'memcheck', 'racecheck'):
            receipts.append(run_rank(0, stage, distributed=False))
            print('PASS delayed producer: ' + stage, flush=True)
        for stage in ('probe', 'memcheck', 'racecheck'):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run_rank, rank, stage) for rank in range(4)]
                results = []
                try:
                    for future in as_completed(futures):
                        results.append(future.result())
                except BaseException:
                    stop_owned()
                    raise
            assert len({json.dumps(r['source_sha256'], sort_keys=True) for r in results}) == 1
            receipts.extend(results)
            print('PASS all ranks: ' + stage, flush=True)
    finally:
        stop_owned()
        (args.out / 'admission.json').write_text(json.dumps(
            {'revision': revision, 'image': IMAGE, 'completed': receipts}, indent=2) + '\n')


if __name__ == '__main__':
    main()
