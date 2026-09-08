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
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
NODES = ('local', '10.10.10.1', '10.10.10.3', '10.10.10.4')
IPS = '10.10.10.2,10.10.10.1,10.10.10.3,10.10.10.4'
GIB = 1024**3
RACECHECK_KERNELS = '(mk_|k_oneshot|ar_consumer_delay)'


def memory_budget(stage):
    # Racecheck's instrumentation exceeded the old 8 GiB cgroup limit.
    # All serving containers are stopped; keep an additional 8 GiB free
    # beyond the bounded probe budget and preserve every kernel check.
    limit = 24 if stage == 'racecheck' else 8
    return limit, (limit + 8) * GIB


def run_container(node, command, name, diagnostics, log):
    """Keep exit/OOM evidence before removing only this probe's container."""
    try:
        remote(node, command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    finally:
        active_error = sys.exc_info()[0]
        cleanup_errors = []
        try:
            state = remote(node, ['docker', 'inspect', '--format',
                '{"state":{{json .State}},"image":{{json .Image}},'
                '"memory_limit":{{json .HostConfig.Memory}},'
                '"memory_swap_limit":{{json .HostConfig.MemorySwap}},'
                '"cpus":{{json .HostConfig.CpusetCpus}}}', name],
                capture_output=True, text=True, timeout=15)
            diagnostics.write_text(state.stdout)
        except subprocess.SubprocessError as exc:
            cleanup_errors.append(str(exc))
            diagnostics.write_text(json.dumps({'inspection_error': str(exc)}) + '\n')
        try:
            remote(node, ['docker', 'rm', '-f', name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        except subprocess.SubprocessError as exc:
            cleanup_errors.append(str(exc))
        # An inspect/cleanup error must not replace the original launch/OOM
        # exception. A successful launch still requires successful cleanup.
        if cleanup_errors and active_error is None:
            raise RuntimeError('container diagnostics/cleanup failed: ' + '; '.join(cleanup_errors))


def remote(node, argv, **kwargs):
    command = argv if node == 'local' else ['ssh', '-o', 'BatchMode=yes',
                'choiceoh@' + node, shlex.join(argv)]
    return subprocess.run(command, check=True, timeout=kwargs.pop('timeout', 900), **kwargs)


def main():
    from ar_consumer_probe import AR_OWNERSHIP_SIZES

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
        limit_gib, required_bytes = memory_budget(stage)
        assert int(mem.stdout.strip()) * 1024 >= required_bytes, (
            node, f'need {required_bytes // GIB} GiB available before {stage}')
        cmd = ['docker', 'run', '--name', name, '--gpus', 'device=0',
            '--network=host', '--device=/dev/infiniband', '--cap-add=IPC_LOCK',
            '--ulimit', 'memlock=-1:-1', '--cpuset-cpus=14-17',
            f'--memory={limit_gib}g', f'--memory-swap={limit_gib}g', '--shm-size=1g',
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
        if stage == 'racecheck':
            # The default reserves capacity for ten million hazards. Bound
            # that storage, not the launches: even one reported hazard fails
            # the zero-warning gate below, and no case is skipped.
            cmd += ['-e', 'NV_COMPUTE_SANITIZER_MAX_RACECHECK_HAZARDS=100000']
        if stage in ('memcheck', 'racecheck'):
            cmd += ['--entrypoint', '/san/compute-sanitizer', IMAGE, '--tool', stage,
                    '--target-processes', 'application-only', '--error-exitcode', '77']
            if stage == 'racecheck':
                # Instrument every production MK/OSAR kernel and the delayed
                # producer, excluding unchanged Torch fixture/oracle kernels.
                # Unfiltered memcheck still covers the complete application.
                cmd += ['--racecheck-num-workers', '4', '--print-session-details',
                        '--kernel-name', 'regex=' + RACECHECK_KERNELS]
            cmd += ['python3', target, '--check-only']
        else:
            cmd += ['--entrypoint', 'python3', IMAGE, target, '--trace']
        if distributed:
            cmd += ['--distributed']
        filename = ('' if distributed else 'local-') + stage + '-rank' + str(rank)
        cmd += ['--out', '/evidence/' + filename + '.json']
        with (args.out / (filename + '.log')).open('w') as log:
            run_container(node, cmd, name, args.out / (filename + '.container.json'), log)
        if stage in ('memcheck', 'racecheck'):
            summary = ('ERROR SUMMARY: 0 errors' if stage == 'memcheck' else
                       'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')
            assert summary in (args.out / (filename + '.log')).read_text(), (node, stage)
        if node != 'local':
            data = remote(node, ['cat', str(out / (filename + '.json'))], capture_output=True, text=True).stdout
            (args.out / (filename + '.json')).write_text(data)
        report = json.loads((args.out / (filename + '.json')).read_text())
        assert report['status'] == 'PASS' and len(report['cases']) == 36, (node, stage)
        assert report['mhc_warmup_capture'] == 'PASS', (node, stage, 'MHC warmup/capture lifecycle')
        pre_cases = report['mhc_pre_view_cases']
        assert len(pre_cases) == 6 and all(c['passed'] for c in pre_cases)
        assert {(c['consumer'], c['input_value']) for c in pre_cases} == {
            (early, value) for early in (False, True) for value in (.03125, 0., -.0625)}
        ownership = report['ar_ownership_cases']
        expected = {(n, seed) for n in AR_OWNERSHIP_SIZES for seed in (17, 0, 29)} if distributed else set()
        assert len(ownership) == len(expected), (node, stage, 'AR ownership coverage')
        assert {(c['elements'], c['seed']) for c in ownership if c['pass']} == expected, (node, stage)
        return {'node': node, 'stage': filename, 'source_sha256': report['source_sha256'],
                'kernel_filter': RACECHECK_KERNELS if stage == 'racecheck' else None}

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
