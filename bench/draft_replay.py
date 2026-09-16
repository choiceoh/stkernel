"""Queue-owned TP4 replay of captured drafter states, without a serving boot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

NODES = ('10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4')


def command(rank, argv):
    return list(argv) if rank == 0 else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                                       'choiceoh@' + NODES[rank], shlex.join(argv)]


def run(rank, argv, **kwargs):
    return subprocess.run(command(rank, argv), check=True, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'checkpoint', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--reader', default='all', help='one reader, or all')
    ap.add_argument('--precision', choices=('fp8-rtn', 'bf16'), default='fp8-rtn')
    ap.add_argument('--rounds', type=int, default=10)
    args = ap.parse_args()
    if not 1 <= args.rounds <= 100:
        raise ValueError('rounds must be 1..100')
    repo = Path(os.environ.get('REPO', Path(__file__).resolve().parents[1])).resolve()
    session, owner = os.environ.get('FLEET_SESSION', ''), os.environ.get('ST_LEASE_OWNER', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', session) or owner != 'queue/' + session:
        raise ValueError('replay needs its queue-owned fleet lease')
    if subprocess.check_output(['hostname', '-s'], text=True).strip() != 'srv2':
        raise ValueError('TP4 replay is launched on the fleet controller srv2')
    run(0, ['bash', '-c', 'FLEET_REPO="$1"; source "$1/launchers/lib/fleet-lease.sh"; fleet_lease verify --owner "$2"',
            'bash', str(repo), owner], stdout=subprocess.DEVNULL)
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('replay source checkout must be committed and clean')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    image = os.environ['ST_IMAGE']  # pin the capture's image explicitly in the admitted command
    tree = Path('/home/choiceoh/st-replay-runs') / session / revision[:12]
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('refusing to overwrite existing replay evidence')
    name = 'st-draft-replay-' + session
    env = {
        'WORLD_SIZE': '4', 'MASTER_ADDR': NODES[0], 'MASTER_PORT': '29666', 'LOCAL_RANK': '0',
        'NCCL_NET': 'IB', 'NCCL_IB_DISABLE': '0', 'NCCL_IB_HCA': 'rocep1s0f0,roceP2p1s0f0',
        'NCCL_SOCKET_IFNAME': 'enp1s0f0np0', 'GLOO_SOCKET_IFNAME': 'enP2p1s0f0np0',
        'NCCL_CROSS_NIC': '1', 'NCCL_PROTO': 'LL,LL128,Simple', 'NCCL_CUMEM_ENABLE': '0',
        'NCCL_IB_ROCE_VERSION_NUM': '2', 'NCCL_IB_ADDR_FAMILY': 'AF_INET', 'NCCL_NVLS_ENABLE': '0',
        'NCCL_IGNORE_CPU_AFFINITY': '1', 'NCCL_DEBUG': 'WARN', 'NCCL_MIN_NCHANNELS': '16',
        'NCCL_MAX_NCHANNELS': '16', 'NCCL_NCHANNELS_PER_NET_PEER': '4', 'NCCL_P2P_LEVEL': 'SYS',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC': '300',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONPATH': '/repo',
        'OMP_NUM_THREADS': '2', 'MAX_JOBS': '2', 'TRITON_CACHE_DIR': '/cache/cu132/triton',
        'DG_JIT_CACHE_DIR': '/cache/cu132/deep_gemm', 'ST_DENSE_BUILD_ROOT': '/cache/cu132/st-dense',
        'ST_ONESHOT_BUILD_ROOT': '/cache/cu132/st-oneshot', 'ST_NATIVE_BUILD_ROOT': '/cache/cu132/st-native',
        'CUDA_CACHE_PATH': '/cache/cu132/driver', 'FLASHINFER_WORKSPACE_BASE': '/cache/cu132',
    }

    def prepare(rank):
        run(rank, ['test', '-s', str(args.capture / f'rank{rank}/manifest.json')])
        run(rank, ['test', '-s', str(args.checkpoint)])
        run(rank, ['mkdir', '-p', str(tree), str(out.parent)])
        if rank == 0:
            run(0, ['rsync', '-a', '--exclude=__pycache__', *[str(repo / p) for p in ('engine', 'probes', 'launchers')], str(tree) + '/'])
        else:
            run(0, ['rsync', '-a', '--exclude=__pycache__', *[str(repo / p) for p in ('engine', 'probes', 'launchers')],
                    'choiceoh@' + NODES[rank] + ':' + str(tree) + '/'])
        actual = run(rank, ['docker', 'image', 'inspect', '--format', '{{.Id}}', image], capture_output=True, text=True).stdout.strip()
        mem = run(rank, ['cat', '/proc/meminfo'], capture_output=True, text=True).stdout
        available = int(re.search(r'^MemAvailable:\s+(\d+)', mem, re.M)[1]) * 1024
        if available < 24 * 2**30:
            raise ValueError(f'rank {rank} lacks 8 GiB replay room plus a 16 GiB floor')
        return dict(rank=rank, host=NODES[rank], image=actual, memory_available=available)

    with ThreadPoolExecutor(max_workers=4) as pool:
        runtime = list(pool.map(prepare, range(4)))
    (out.parent / (out.stem + '-runtime.json')).write_text(json.dumps(dict(revision=revision, runtime=runtime, env=env,
        capture=str(args.capture), checkpoint=str(args.checkpoint), precision=args.precision, reader=args.reader,
        rounds=args.rounds, engine_booted=False, source_tree=str(tree)), indent=2) + '\n')
    probe = ['python3', '-u', '/repo/probes/draft_sensitivity.py', '--capture', str(args.capture),
             '--checkpoint', str(args.checkpoint), '--reader', args.reader, '--precision', args.precision,
             '--rounds', str(args.rounds), '--output', str(out)]
    script = 'source /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; exec ' + shlex.join(probe)
    children, logs = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt('replay interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for rank in range(4):
            argv = ['docker', 'run', '--rm', '--name', name, '--gpus', 'all', '--network', 'host', '--ipc', 'host',
                    '--memory', '16g', '--ulimit', 'memlock=-1:-1', '--cap-add', 'IPC_LOCK',
                    '--device', '/dev/infiniband:/dev/infiniband', '-e', f'RANK={rank}']
            for key, value in env.items():
                argv += ['-e', key + '=' + value]
            for src, dst, readonly in ((tree, '/repo', True), (args.capture, str(args.capture), True),
                                       (args.checkpoint.parent, str(args.checkpoint.parent), True),
                                       (Path('/home/choiceoh/glm53-cache'), '/cache', False),
                                       (out.parent, str(out.parent), False)):
                argv += ['-v', f'{src}:{dst}' + (':ro' if readonly else '')]
            argv += ['--entrypoint', '/bin/bash', image, '-lc', script]
            log = (out.parent / f'{out.stem}-rank{rank}.log').open('w')
            logs.append(log)
            children.append(subprocess.Popen(command(rank, argv), stdout=log, stderr=subprocess.STDOUT))
        started = time.monotonic()
        print(f'TP4 replay started: {revision}, logs {out.parent}/{out.stem}-rankN.log', flush=True)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('a replay rank failed; see retained rank logs')
            if time.monotonic() - started > 1800:
                raise TimeoutError('replay exceeded 30 minutes')
            time.sleep(1)
        if any(p.returncode != 0 for p in children) or not out.is_file():
            raise RuntimeError('replay did not produce successful evidence')
        print('TP4 replay complete: ' + str(out), flush=True)
    finally:
        # Only this reservation's containers; never stop a serving container.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda r: subprocess.run(command(r, ['docker', 'rm', '-f', name]),
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30), range(4)))
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.terminate()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
