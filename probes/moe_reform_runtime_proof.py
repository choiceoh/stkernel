#!/usr/bin/env python3
"""Prove each serving rank used the requested MoE source, lane and defaults."""
import argparse
import json
from pathlib import Path
import socket
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=('t', 't,r'))
ap.add_argument('v4_sha')
ap.add_argument('dispatch_sha')
ap.add_argument('v5_sha')
ap.add_argument('cta', choices=('2', '4'))
args = ap.parse_args()
names = subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
name = next(n for n in names if n in ('glm53', 'glm53-worker'))
obj = json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
env = dict(s.split('=',1) for s in obj['Config']['Env'])
expected = {'VLLM_GLM53_MK_INPUT_REUSE':'1', 'VLLM_GLM53_MK_INPUT_CTA':args.cta,
            'VLLM_GLM53_B12X_STATIC_V2':args.mode}
pkg = '/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/'
files = {pkg+'moe_static_kernel_v4.py':args.v4_sha, pkg+'moe_dispatch.py':args.dispatch_sha,
         pkg+'moe_static_kernel_v5.py':args.v5_sha}
hashes = {line.split()[1]:line.split()[0] for line in subprocess.check_output(
    ['docker','exec',name,'sha256sum',*files],text=True).splitlines()}
log = Path('/home/choiceoh/glm53-logs/glm53.log').read_text(errors='replace')
markers = [line for line in log.splitlines() if '[b12x static v2] lane serving:' in line]
lane = 'static2_m6_k4096_n512_t8_r'
suffix = 'tm32f2g2a32wut' + ('r16n128k256d256' if args.mode == 't,r' else '') + ' ('
report = dict(host=socket.gethostname(), mode=args.mode, image=obj['Image'],
              boot_id=obj['Id']+'|'+obj['State']['StartedAt'], running=obj['State']['Running'],
              source_sha256=hashes, knobs={k:env.get(k) for k in expected}, markers=markers)
print(json.dumps(report,indent=2),flush=True)
assert report['running'] and report['knobs'] == expected
assert hashes == files
assert report['image'] == 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
assert any(lane in line and suffix in line for line in markers), markers
assert '[megakernel] input-reuse CAPTURED M=6 N=6416 K=4096 split=8' in log
