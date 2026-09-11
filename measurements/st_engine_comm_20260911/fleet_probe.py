"""Run on srv1: bounded shared-machine communication diagnostics, no weights.

Owns the common ST launch lock. Existing services are recorded and never stopped.
This is a 2 GiB-per-node fabric diagnostic, not a second model service.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time

NODES = ['10.10.10.2','10.10.10.1','10.10.10.3','10.10.10.4']
SOURCE = '/home/choiceoh/st-tp4-lat-f4d7'
OWNER = 'st-tp4-lat-f4d7 bounded communication diagnostic'
ENV = dict(MASTER_ADDR=NODES[0], MASTER_PORT='29723', WORLD_SIZE='4', LOCAL_RANK='0',
           PYTHONPATH='/repo', OMP_NUM_THREADS='2', NCCL_P2P_LEVEL='SYS',
           NCCL_NET='IB', NCCL_IB_DISABLE='0', NCCL_IB_HCA='rocep1s0f0,roceP2p1s0f0',
           NCCL_SOCKET_IFNAME='enp1s0f0np0', GLOO_SOCKET_IFNAME='enP2p1s0f0np0',
           NCCL_CROSS_NIC='1', NCCL_PROTO='LL,LL128,Simple', NCCL_CUMEM_ENABLE='0',
           NCCL_IB_ROCE_VERSION_NUM='2', NCCL_IB_ADDR_FAMILY='AF_INET', NCCL_NVLS_ENABLE='0',
           NCCL_IGNORE_CPU_AFFINITY='1', NCCL_NCHANNELS_PER_NET_PEER='4',
           NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,NET,GRAPH,TUNING',
           TORCH_NCCL_ASYNC_ERROR_HANDLING='1')


def ssh(ip, args, **kwargs):
    return subprocess.run(['ssh','-o','BatchMode=yes',ip,shlex.join(args)],
                          text=True,capture_output=True,check=True,timeout=150,**kwargs)


def snapshot(ip):
    names=ssh(ip,['docker','ps','--format','{{.Names}}']).stdout.splitlines()
    services=[]
    for name in names:
        data=json.loads(ssh(ip,['docker','inspect',name]).stdout)[0]
        services.append(dict(name=name,id=data['Id'],started=data['State']['StartedAt']))
    return dict(node=ip,services=services,
                memory=ssh(ip,['cat','/proc/meminfo']).stdout,
                gpu=ssh(ip,['nvidia-smi','--query-gpu=name,utilization.gpu','--format=csv,noheader']).stdout)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds',type=int,default=3)
    ap.add_argument('--configs',nargs='+',default=['16','4','auto'])
    ap.add_argument('--batch',default='compare')
    args=ap.parse_args()
    assert 1<=args.rounds<=3 and set(args.configs)<={'16','4','auto'}
    assert args.batch.isalnum() and len(args.batch)<=12
    out=Path(SOURCE)/'evidence';out.mkdir(parents=True,exist_ok=True)
    lock=Path('/home/choiceoh/st-fleet.lock')
    code='from pathlib import Path; import sys; p=Path(sys.argv[1]); f=p.open("x"); f.write(sys.argv[2]+"\\n"); f.close()'
    ssh(NODES[0],['python3','-c',code,str(lock),OWNER])
    runs=[]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            before=list(pool.map(snapshot,NODES))
        (out/'before.json').write_text(json.dumps(before,indent=2)+'\n')
        for node in before:
            mem=dict(line.split(':',1) for line in node['memory'].splitlines())
            assert int(mem['MemAvailable'].split()[0])>8*1024*1024, node['node']
        payload=Path('/tmp/st-tp4-lat-source-f4d7.tar.gz').read_bytes()
        for ip in NODES:
            ssh(ip,['mkdir','-p',SOURCE+'/evidence'])
            subprocess.run(['ssh',ip,'tar -xzf - -C '+shlex.quote(SOURCE)],input=payload,check=True,timeout=30)
        for run in range(1,args.rounds+1):
            for config in args.configs if run%2 else args.configs[::-1]:
                name=f'st-tp4-lat-f4d7-{args.batch}-r{run}-{config}'
                started=[]
                results=[]
                try:
                    for rank,ip in enumerate(NODES):
                        env=dict(ENV,RANK=str(rank))
                        if config!='auto':env.update(NCCL_MIN_NCHANNELS=config,NCCL_MAX_NCHANNELS=config)
                        command=['docker','run','-d','--name',name,'--gpus','all','--network','host',
                                 '--shm-size','256m','--cpus','4','--memory','2g','--memory-swap','2g',
                                 '--ulimit','memlock=-1:-1','--device','/dev/infiniband:/dev/infiniband',
                                 '-v',SOURCE+':/repo:ro','-v',SOURCE+'/evidence:/evidence','-w','/repo']
                        for key,value in env.items():command+=['-e',f'{key}={value}']
                        probe=['timeout','--kill-after=5s','110','python3','-u','probes/engine_comm_profile.py',
                               '--output',f'/evidence/{name}-rank{rank}.json']
                        shell='source /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; exec '+shlex.join(probe)
                        command+=['--entrypoint','bash','st-engine:9391','-lc',shell]
                        ssh(ip,command);started.append(ip)
                    def finish(item):
                        rank,ip=item
                        wait=ssh(ip,['docker','wait',name])
                        state=json.loads(ssh(ip,['docker','inspect',name]).stdout)[0]
                        log=ssh(ip,['docker','logs',name])
                        (out/f'{name}-rank{rank}.log').write_text(log.stdout+log.stderr)
                        data=ssh(ip,['cat',f'{SOURCE}/evidence/{name}-rank{rank}.json']).stdout
                        # Rank 1 already wrote this exact path as container root.
                        # Avoid opening that same file for overwrite as the SSH user.
                        if ip!='10.10.10.1':
                            (out/f'{name}-rank{rank}.json').write_text(data)
                        return dict(rank=rank,node=ip,exit=int(wait.stdout.strip()),
                                    state=state['State'],image=state['Image'])
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        results=list(pool.map(finish,enumerate(NODES)))
                    print(name,[(r['rank'],r['exit']) for r in results],flush=True)
                    runs.append(dict(name=name,config=config,run=run,ranks=results))
                    (out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
                    assert all(r['exit']==0 for r in results), name
                finally:
                    for ip in started:
                        # Only this invocation's explicitly named diagnostic container.
                        ssh(ip,['docker','rm','-f',name])
    finally:
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                after=list(pool.map(snapshot,NODES))
            (out/'after.json').write_text(json.dumps(after,indent=2)+'\n')
        finally:
            code='from pathlib import Path; import sys; p=Path(sys.argv[1]); assert p.read_text()==sys.argv[2]+"\\n", "owner changed"; p.unlink()'
            ssh(NODES[0],['python3','-c',code,str(lock),OWNER])
            print('owned diagnostic containers removed; owned lock released',flush=True)


if __name__=='__main__':main()
