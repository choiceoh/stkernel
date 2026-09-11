"""Run only this task's named validation containers on the private fleet."""
import shlex
import subprocess
import sys

nodes=['10.10.10.2','10.10.10.1','10.10.10.3','10.10.10.4']
name='st-completion-9391'
source='/tmp/st-engine-completion'
env={
    'MASTER_ADDR':nodes[0], 'MASTER_PORT':'29691','WORLD_SIZE':'4','LOCAL_RANK':'0',
    'PYTHONPATH':'/repo','MAX_JOBS':'2','NCCL_NET':'IB','NCCL_IB_DISABLE':'0',
    'NCCL_IB_HCA':'rocep1s0f0,roceP2p1s0f0','NCCL_SOCKET_IFNAME':'enp1s0f0np0',
    'GLOO_SOCKET_IFNAME':'enP2p1s0f0np0','NCCL_CROSS_NIC':'1','NCCL_CUMEM_ENABLE':'0',
    'NCCL_IB_ROCE_VERSION_NUM':'2','NCCL_IB_ADDR_FAMILY':'AF_INET','NCCL_NVLS_ENABLE':'0',
    'NCCL_DEBUG':'WARN','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1',
    'NCCL_MIN_NCHANNELS':'16','NCCL_MAX_NCHANNELS':'16',
}
def run(ip,args):
    return subprocess.run(['ssh','-o','BatchMode=yes',ip,shlex.join(args)],check=True,text=True)

if sys.argv[1]=='start':
    for rank,ip in enumerate(nodes):
        cache='/tmp/st-engine-9391/cache' if rank==1 else '/home/choiceoh/.cache/st-completion'
        run(ip,['mkdir','-p',cache])
        args=['docker','run','-d','--name',name,'--gpus','all','--network','host','--ipc','host',
              '--cpus','4','--memory','80g','--ulimit','memlock=-1:-1','--cap-add','IPC_LOCK',
              '--device','/dev/infiniband:/dev/infiniband','-w','/repo']
        for key,value in dict(env,RANK=str(rank)).items(): args+=['-e',f'{key}={value}']
        for src,dst,ro in [(source,'/repo',True),('/home/choiceoh/models/st-glm53-9391-up-gate-slice','/ranks',True),(cache,'/cache',False)]:
            args+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
        command='source /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; exec '+shlex.join(['python3','-u','/repo/probes/engine_decode_graph_check.py','--distributed','--ranks','/ranks','--ckpt-meta','/repo/meta'])
        args+=['--entrypoint','bash','st-engine:9391','-lc',command]
        run(ip,args)
elif sys.argv[1]=='status':
    for ip in nodes:
        print(ip,flush=True)
        run(ip,['docker','inspect',name,'--format','{{.State.Status}} exit={{.State.ExitCode}}'])
        run(ip,['docker','logs','--tail','8',name])
elif sys.argv[1]=='logs':
    for rank,ip in enumerate(nodes):
        with open(f'{source}/graph-fleet-rank{rank}.log','w') as out:
            subprocess.run(['ssh',ip,'docker','logs',name],stdout=out,stderr=subprocess.STDOUT,check=True)
elif sys.argv[1]=='remove-finished':
    for ip in nodes:
        # docker rm refuses a running container; never use force here.
        run(ip,['docker','rm',name])
