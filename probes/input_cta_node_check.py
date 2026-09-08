#!/usr/bin/env python3
"""Exercise fresh-process MHC/GEMM initialization on a stopped serving node."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess

IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
ROOT = Path('/home/choiceoh/overlays/glm53')
CHECK = '''
import hashlib, importlib.util, json
from pathlib import Path
import torch
path=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_megakernel.py')
spec=importlib.util.spec_from_file_location('cta_node_driver',path)
mk=importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
with torch.inference_mode():
    ext=mk._build()
    assert ext.probe_device()[:3]==[12,1,48]
    ext.set_input_cta(0)
    assert mk._selftest_bf16_mhc(), 'MHC BF16 differential gate'
    assert mk._selftest_gemm(), 'GEMM independent oracle gate'
    for mode in (0,2,4):
        ext.set_input_cta(mode)
        assert ext.gemm_input_mode()==1
        assert mk._selftest_input_reuse(), ('input replay gate',mode)
    torch.cuda.synchronize()
print('NODE_CHECK='+json.dumps({'status':'PASS','torch':torch.__version__,
    'source_sha256':hashlib.sha256(path.with_suffix('.cu').read_bytes()).hexdigest(),
    'mhc_bf16':True,'gemm':True,'input_modes':[0,2,4]}),flush=True)
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--session',required=True)
    ap.add_argument('--source-sha256',required=True)
    a=ap.parse_args()
    assert re.fullmatch(r'[a-z0-9]+',a.session)
    assert re.fullmatch(r'[0-9a-f]{64}',a.source_sha256)
    actual=hashlib.sha256((ROOT/'glm53_megakernel.cu').read_bytes()).hexdigest()
    assert actual==a.source_sha256,(actual,a.source_sha256)
    names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
    assert not set(names)&{'glm53','glm53-worker'},'serving must be stopped under the fleet hold'
    old=None
    for name in ('glm53','glm53-worker'):
        r=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
        if r.returncode==0:
            old=json.loads(r.stdout)[0]; break
    assert old and old['Image']==IMAGE
    env=dict(s.split('=',1) for s in old['Config']['Env'])
    env={k:v for k,v in env.items() if k.startswith('VLLM_GLM53_MK_') or k in (
        'VLLM_GLM53_MEGAKERNEL','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','CUDA_MODULE_LOADING')}
    env.update(MAX_JOBS='1',VLLM_GLM53_MK_INPUT_CTA='0',VLLM_GLM53_MK_INPUT_REUSE='1')
    mounts=[m for m in old['Mounts'] if m['Source'].startswith(str(ROOT)+'/') or m['Destination']=='/cache']
    assert any(m['Destination']=='/cache' for m in mounts)
    container='ictacheck-'+a.session
    cmd=['docker','run','--rm','-i','--name',container,'--gpus','device=0','--network=none',
         '--cpuset-cpus=14-17','--memory=10g','--shm-size=1g']
    for k,v in sorted(env.items()): cmd+=['--env',k+'='+v]
    for m in mounts:
        cmd+=['--mount','type=bind,src='+m['Source']+',dst='+m['Destination']+
              (',readonly' if m['Destination']!='/cache' else '')]
    cmd+=['--entrypoint','python3',IMAGE,'-']
    print(json.dumps({'host':socket.gethostname(),'image':IMAGE,'source_sha256':actual,
                      'env':env,'status':'STARTING'}),flush=True)
    try:
        r=subprocess.run(cmd,input=CHECK,text=True,timeout=240)
        assert r.returncode==0,('node startup check failed',r.returncode)
    finally:
        subprocess.run(['docker','stop','-t','2',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=='__main__': main()
