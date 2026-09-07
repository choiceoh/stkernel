#!/usr/bin/env python3
"""Read actual CTA selection, source identity and capture on one rank."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ap=argparse.ArgumentParser();ap.add_argument('mode',type=int,choices=range(4));a=ap.parse_args()
names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
name=next(n for n in names if n in ('glm53','glm53-worker'))
o=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
env=dict(s.split('=',1) for s in o['Config']['Env'])
expected={'VLLM_GLM53_MK_INPUT_REUSE':'1','VLLM_GLM53_MK_INPUT_CTA':str(a.mode),
          'VLLM_GLM53_MK_FP8_PACK2':'1','VLLM_GLM53_MK_GEMM_TRANSPOSE_M8':'2',
          'VLLM_GLM53_MK_M8_FASTPATH':'1','VLLM_GLM53_MK_MHC_BF16':'1'}
root=Path('/home/choiceoh/overlays/glm53')
log=Path('/home/choiceoh/glm53-logs/glm53.log').read_text(errors='replace')
lines=[l for l in log.splitlines() if '[megakernel] input-cta CAPTURED' in l]
p={'host':socket.gethostname(),'mode':a.mode,'container':name,'image':o['Image'],
   'boot_id':o['Id']+'|'+o['State']['StartedAt'],'running':o['State']['Running'],
   'knobs':{k:env.get(k) for k in expected},'markers':lines,
   'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest()
                    for f in ('glm53_megakernel.cu','glm53_megakernel.py')}}
print(json.dumps(p,indent=2),flush=True)
assert p['running'] and p['knobs']==expected,p
assert p['image']=='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
assert '[megakernel] input-reuse CAPTURED M=6 N=6416 K=4096 split=8' in log
if a.mode:
    assert any(f'M=6 N=6416 K=4096 mode={a.mode} split=8' in line for line in lines),lines
else:
    assert not lines,lines
