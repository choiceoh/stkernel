"""Launch bounded private probes from srv1; never stop another container."""
import argparse
import shlex
import subprocess
import time

NODES = ['10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4']
SOURCE = '/home/choiceoh/st-engine-f4d7-graph-state'
ENV = {
    'MASTER_ADDR': NODES[0], 'MASTER_PORT': '29711', 'WORLD_SIZE': '4', 'LOCAL_RANK': '0',
    'PYTHONPATH': '/repo', 'MAX_JOBS': '2', 'OMP_NUM_THREADS': '2',
    'NCCL_NET': 'IB', 'NCCL_IB_DISABLE': '0',
    'NCCL_IB_HCA': 'rocep1s0f0,roceP2p1s0f0', 'NCCL_SOCKET_IFNAME': 'enp1s0f0np0',
    'GLOO_SOCKET_IFNAME': 'enP2p1s0f0np0', 'NCCL_CROSS_NIC': '1', 'NCCL_CUMEM_ENABLE': '0',
    'NCCL_IB_ROCE_VERSION_NUM': '2', 'NCCL_IB_ADDR_FAMILY': 'AF_INET', 'NCCL_NVLS_ENABLE': '0',
    'NCCL_DEBUG': 'WARN', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
    'NCCL_MIN_NCHANNELS': '16', 'NCCL_MAX_NCHANNELS': '16',
}


def run(ip, args):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', ip, shlex.join(args)],
                          check=True, text=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=['start', 'status', 'wait', 'logs', 'remove-finished'])
parser.add_argument('--name', default='st-graph-profile-fleet-f4d7')
parser.add_argument('--program', default='probes/engine_graph_profile.py')
parser.add_argument('--cpus', default='4')
args, extra = parser.parse_known_args()
owner = 'st-graph-state-f4d7/' + args.name


def release():
    # Never remove a lock acquired by another task or a later run.
    run(NODES[0], ['python3', '-c',
        'from pathlib import Path; import sys; p=Path("/home/choiceoh/st-fleet.lock"); '
        'p.unlink() if p.exists() and p.read_text().strip() == sys.argv[1] else None', owner])


if args.mode == 'start':
    # Performance qualification needs an idle fleet. The original exploratory
    # runs shared resident GLM workers; their noisy samples remain in evidence.
    for ip in NODES:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', ip,
                                 'docker ps --format "{{.Names}}"'],
                                text=True, capture_output=True, check=True)
        busy = [n for n in result.stdout.splitlines()
                if n.startswith(('glm53', 'q38', 'vllm', 'st-'))]
        if busy:
            raise SystemExit(f'{ip} is occupied by {busy}; wait for the fleet window')
    run(NODES[0], ['python3', '-c',
        'from pathlib import Path; import sys; '
        'p=Path("/home/choiceoh/st-fleet.lock"); '
        'f=p.open("x"); f.write(sys.argv[1]+"\\n"); f.close()', owner])
if args.mode == 'wait':
    deadline = time.monotonic() + 900
    previous = None
    while time.monotonic() < deadline:
        states = []
        for ip in NODES:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', ip, 'docker', 'inspect', args.name,
                                     '--format', '{{.State.Status}}:{{.State.ExitCode}}'],
                                    text=True, capture_output=True, check=True)
            states.append(result.stdout.strip())
        if states != previous:
            print(args.name, states, flush=True)
            previous = states
        if all(s.startswith('exited:') for s in states):
            release()
            raise SystemExit(0 if all(s == 'exited:0' for s in states) else 1)
        time.sleep(5)
    raise TimeoutError(args.name)
for rank, ip in enumerate(NODES):
    if args.mode == 'start':
        run(ip, ['mkdir', '-p', SOURCE + '/cache', SOURCE + '/evidence'])
        command = ['docker', 'run', '-d', '--name', args.name, '--gpus', 'all',
                   '--network', 'host', '--ipc', 'host', '--cpus', args.cpus, '--memory', '8g',
                   '--ulimit', 'memlock=-1:-1', '--cap-add', 'IPC_LOCK',
                   '--device', '/dev/infiniband:/dev/infiniband', '-w', '/repo']
        for key, value in dict(ENV, RANK=str(rank)).items():
            command += ['-e', f'{key}={value}']
        for src, dst, readonly in [
            (SOURCE, '/repo', True), (SOURCE + '/cache', '/cache', False),
            (SOURCE + '/evidence', '/evidence', False),
            ('/home/choiceoh/models/st-glm53-9391-up-gate-slice', '/ranks', True),
            (SOURCE + '/meta', '/meta', True),
        ]:
            command += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if readonly else '')]
        probe = ['python3', '-u', args.program, '--distributed', '--ranks', '/ranks',
                 '--ckpt-meta', '/meta'] + extra
        if args.program.endswith(('engine_graph_profile.py', 'engine_state_cache_bench.py')):
            probe += ['--output', f'/evidence/{args.name}-rank{rank}.json']
        shell = 'source /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; exec ' + shlex.join(probe)
        command += ['--entrypoint', 'bash', 'st-engine:9391', '-lc', shell]
        run(ip, command)
    elif args.mode == 'status':
        print(ip, flush=True)
        run(ip, ['docker', 'inspect', args.name, '--format', '{{.State.Status}} exit={{.State.ExitCode}}'])
        run(ip, ['docker', 'logs', '--tail', '5', args.name])
    elif args.mode == 'logs':
        with open(f'{SOURCE}/evidence/{args.name}-rank{rank}.log', 'w') as out:
            subprocess.run(['ssh', ip, 'docker', 'logs', args.name], stdout=out,
                           stderr=subprocess.STDOUT, check=True)
    else:
        run(ip, ['docker', 'rm', args.name])
if args.mode == 'remove-finished':
    release()
