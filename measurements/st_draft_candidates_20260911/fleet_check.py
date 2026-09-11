"""Run on srv1: bounded TP4 candidate correctness and selection timings.

Acquires the shared ST lock, never stops services, and removes only containers
created by this invocation. A busy GPU permits correctness, not clean timing.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

NODES = ['10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4']
SOURCE = '/home/choiceoh/st-draft-topk-fleet-f4d7'
NAME = 'st-draft-topk-f4d7-' + str(int(time.time()))
OWNER = NAME + ' bounded correctness probe'
ENV = dict(MASTER_ADDR=NODES[0], MASTER_PORT='29724', WORLD_SIZE='4', LOCAL_RANK='0',
           PYTHONPATH='/repo', OMP_NUM_THREADS='2', NCCL_P2P_LEVEL='SYS',
           NCCL_NET='IB', NCCL_IB_DISABLE='0', NCCL_IB_HCA='rocep1s0f0,roceP2p1s0f0',
           NCCL_SOCKET_IFNAME='enp1s0f0np0', GLOO_SOCKET_IFNAME='enP2p1s0f0np0',
           NCCL_CROSS_NIC='1', NCCL_PROTO='LL,LL128,Simple', NCCL_CUMEM_ENABLE='0',
           NCCL_IB_ROCE_VERSION_NUM='2', NCCL_IB_ADDR_FAMILY='AF_INET', NCCL_NVLS_ENABLE='0',
           NCCL_IGNORE_CPU_AFFINITY='1', NCCL_NCHANNELS_PER_NET_PEER='4',
           NCCL_MIN_NCHANNELS='16', NCCL_MAX_NCHANNELS='16',
           NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,NET,GRAPH,TUNING',
           TORCH_NCCL_ASYNC_ERROR_HANDLING='1')


def ssh(ip, args):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', ip, shlex.join(args)],
                          text=True, capture_output=True, check=True, timeout=240)


def snapshot(ip):
    names = ssh(ip, ['docker', 'ps', '--format', '{{.Names}}']).stdout.splitlines()
    services = []
    for name in names:
        data = json.loads(ssh(ip, ['docker', 'inspect', name]).stdout)[0]
        services.append(dict(name=name, id=data['Id'], started=data['State']['StartedAt']))
    return dict(node=ip, services=services, memory=ssh(ip, ['cat', '/proc/meminfo']).stdout,
                gpu=ssh(ip, ['nvidia-smi', '--query-gpu=name,utilization.gpu', '--format=csv,noheader']).stdout,
                processes=ssh(ip, ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader']).stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--real-weights', action='store_true')
    ap.add_argument('--unit-tests', action='store_true', help='run focused CUDA regressions on rank 1 after the probe')
    args = ap.parse_args()
    out = Path(SOURCE)/'collected'/NAME
    out.mkdir(parents=True, exist_ok=True)
    lock = '/home/choiceoh/st-fleet.lock'
    acquire = 'from pathlib import Path; import sys; p=Path(sys.argv[1]); f=p.open("x"); f.write(sys.argv[2]); f.close()'
    ssh(NODES[0], ['python3', '-c', acquire, lock, OWNER])
    print('acquired', OWNER, 'output', out, flush=True)
    started = []
    try:
        with ThreadPoolExecutor(4) as pool:
            before = list(pool.map(snapshot, NODES))
        (out/'before.json').write_text(json.dumps(before, indent=2)+'\n')
        for node in before:
            mem = dict(line.split(':', 1) for line in node['memory'].splitlines())
            assert int(mem['MemAvailable'].split()[0]) > 8*1024*1024, node['node']
            assert not any(s['name']=='st-glm53' for s in node['services']), 'ST service active'
        payload = Path('/tmp/st-draft-topk-fleet-f4d7.tar.gz').read_bytes()
        for ip in NODES:
            ssh(ip, ['mkdir', '-p', SOURCE+'/evidence'])
            subprocess.run(['ssh', ip, 'tar -xzf - -C '+shlex.quote(SOURCE)], input=payload, check=True, timeout=30)
        for rank, ip in enumerate(NODES):
            command = ['docker', 'run', '-d', '--name', NAME, '--gpus', 'all', '--network', 'host',
                       '--shm-size', '256m', '--cpus', '4', '--memory', '8g' if args.real_weights else '2g',
                       '--memory-swap', '8g' if args.real_weights else '2g',
                       '--ulimit', 'memlock=-1:-1', '--device', '/dev/infiniband:/dev/infiniband',
                       '-v', SOURCE+':/repo:ro', '-v', SOURCE+'/evidence:/evidence', '-w', '/repo']
            for key, value in dict(ENV, RANK=str(rank)).items(): command += ['-e', f'{key}={value}']
            probe = ['timeout', '--kill-after=5s', '210', 'python3', '-u', 'probes/engine_tp_draft_topk_check.py',
                     '--output', f'/evidence/{NAME}-rank{rank}.json']
            if args.real_weights:
                command += ['-v', '/home/choiceoh/models:/models:ro']
                probe = ['timeout', '--kill-after=5s', '210', 'python3', '-u', 'probes/engine_draft_candidate_check.py',
                         '--head', f'/models/st-glm53-9391-up-gate-slice/rank{rank}of4.safetensors',
                         '--drafter-dir', '/models/GLM-5.3-Flash-DFlash2', '--baseline', '/repo/baseline-drafter.py',
                         '--distributed', '--output', f'/evidence/{NAME}-rank{rank}.json']
            shell = 'source /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; exec '+shlex.join(probe)
            if args.unit_tests and rank == 1:
                unit = ['timeout', '60', 'python3', '-m', 'unittest', '-v',
                        'test_engine_vocab_topk', 'test_engine_vocab', 'test_engine_drafter']
                shell = ('source /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; '+shlex.join(probe)+
                         ' && PYTHONPATH=/repo:/repo/tests '+shlex.join(unit)+
                         ' > /evidence/'+NAME+'-focused.log 2>&1')
            command += ['--entrypoint', 'bash', 'st-engine:9391', '-lc', shell]
            ssh(ip, command)
            started.append(ip)
        def finish(item):
            rank, ip = item
            wait = ssh(ip, ['docker', 'wait', NAME])
            state = json.loads(ssh(ip, ['docker', 'inspect', NAME]).stdout)[0]
            log = ssh(ip, ['docker', 'logs', NAME])
            (out/f'rank{rank}.log').write_text(log.stdout+log.stderr)
            result = ssh(ip, ['bash', '-lc', 'cat '+shlex.quote(f'{SOURCE}/evidence/{NAME}-rank{rank}.json')+' 2>/dev/null || true']).stdout
            (out/f'rank{rank}.json').write_text(result)
            if args.unit_tests and rank == 1:
                unit = ssh(ip, ['cat', f'{SOURCE}/evidence/{NAME}-focused.log']).stdout
                (out/'focused-tests.log').write_text(unit)
            return dict(rank=rank, node=ip, exit=int(wait.stdout.strip()), state=state['State'], image=state['Image'])
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(finish, enumerate(NODES)))
        (out/'runs.json').write_text(json.dumps(results, indent=2)+'\n')
        print([(r['rank'], r['exit']) for r in results], flush=True)
        assert all(r['exit']==0 for r in results), 'rank failure; inspect retained logs'
    finally:
        try:
            for ip in started: ssh(ip, ['docker', 'rm', '-f', NAME])
            with ThreadPoolExecutor(4) as pool:
                after = list(pool.map(snapshot, NODES))
            (out/'after.json').write_text(json.dumps(after, indent=2)+'\n')
        finally:
            release = 'from pathlib import Path; import sys; p=Path(sys.argv[1]); assert p.read_text()==sys.argv[2], "owner changed"; p.unlink()'
            ssh(NODES[0], ['python3', '-c', release, lock, OWNER])
            print('removed owned containers and released owned lock', flush=True)


if __name__ == '__main__': main()
