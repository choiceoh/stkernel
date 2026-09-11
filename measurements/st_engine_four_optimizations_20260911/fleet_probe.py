import base64,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
NAME='st-four-tp-vocab-9391'
ROOT='/tmp/st-engine-four-9391'
ENV={'MASTER_ADDR':'10.10.10.2','MASTER_PORT':'29741','WORLD_SIZE':'4','LOCAL_RANK':'0','PYTHONPATH':'/repo','NCCL_NET':'IB','NCCL_IB_DISABLE':'0','NCCL_IB_HCA':'rocep1s0f0,roceP2p1s0f0','NCCL_SOCKET_IFNAME':'enp1s0f0np0','GLOO_SOCKET_IFNAME':'enP2p1s0f0np0','NCCL_CROSS_NIC':'1','NCCL_CUMEM_ENABLE':'0','NCCL_IB_ROCE_VERSION_NUM':'2','NCCL_IB_ADDR_FAMILY':'AF_INET','NCCL_NVLS_ENABLE':'0','NCCL_DEBUG':'WARN','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1','NCCL_MIN_NCHANNELS':'2','NCCL_MAX_NCHANNELS':'2'}
def run(rank, command, data=None):
    if rank==0: prefix=['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2']
    else: prefix=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','srv1' if rank==1 else 'srv4']
    if rank==2: command='ssh -o BatchMode=yes -o ConnectTimeout=5 10.10.10.3 '+shlex.quote(command)
    return subprocess.run(prefix+[command],input=data,capture_output=True,timeout=30)
mode=sys.argv[1]
if mode=='start':
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as arc:
        for path in ('launchers/lib/common-tp4.sh','probes/engine_tp_vocab_check.py'):
            arc.add(path)
    payload=base64.b64encode(buf.getvalue())
    for rank in range(4):
        result=run(rank,'base64 -d | tar -xzf - -C '+ROOT,payload); result.check_returncode()
        args=['docker','run','-d','--name',NAME,'--gpus','all','--network','host','--memory','2g','--cpus','1','--ulimit','memlock=-1:-1','--cap-add','IPC_LOCK','--device','/dev/infiniband:/dev/infiniband','-v',ROOT+':/repo:ro','-w','/repo']
        for key,value in dict(ENV,RANK=str(rank)).items(): args+=['-e',key+'='+value]
        command='source /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; exec timeout --kill-after=5s 80s python3 -u probes/engine_tp_vocab_check.py'
        args+=['--entrypoint','bash','st-engine:9391','-lc',command]
        result=run(rank,shlex.join(args)); print(rank,result.returncode,result.stdout.decode(),result.stderr.decode(),flush=True); result.check_returncode()
elif mode=='status':
    for rank in range(4):
        p=run(rank,'docker inspect '+NAME+' --format '+shlex.quote('{{.State.Status}} exit={{.State.ExitCode}}'))
        print(rank,p.stdout.decode(),p.stderr.decode(),flush=True)
        p=run(rank,'docker logs --tail 8 '+NAME); print(p.stdout.decode()+p.stderr.decode(),flush=True)
elif mode=='collect':
    out=Path('measurements/st_engine_four_optimizations_20260911'); out.mkdir(parents=True,exist_ok=True)
    for rank in range(4):
        p=run(rank,'docker logs '+NAME); (out/f'tp-vocab-rank{rank}.log').write_bytes(p.stdout+p.stderr)
        p=run(rank,'docker inspect '+NAME+' --format '+shlex.quote('{{.State.Status}} exit={{.State.ExitCode}}')); print(rank,p.stdout.decode(),flush=True)
        if p.returncode or p.stdout.strip()!=b'exited exit=0': raise RuntimeError('rank did not pass')
        p=run(rank,'docker rm '+NAME); p.check_returncode()
