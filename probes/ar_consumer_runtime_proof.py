#!/usr/bin/env python3
"""Read-only all-rank proof: exact mounted source, flags, and graph capture."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=('0', '1'))
ap.add_argument('expected', help='JSON mapping mounted paths to source SHA256')
args = ap.parse_args()
files = json.loads(args.expected)
names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
name = next(n for n in names if n in ('glm53', 'glm53-worker'))
obj = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
env = dict(s.split('=', 1) for s in obj['Config']['Env'])
expected = {'VLLM_GLM53_AR_CONSUMER_PDL': args.mode, 'VLLM_GLM53_MK_PDL': '1',
            'VLLM_GLM53_MK_MHC_BF16': '1', 'VLLM_GLM53_MK_INPUT_CTA': '4',
            'VLLM_GLM53_MK_INPUT_REUSE': '1', 'VLLM_GLM53_B12X_STATIC_V2': 't,r',
            'VLLM_GLM53_AR_PREFETCH': '0'}
hashes = {line.split()[1]: line.split()[0] for line in subprocess.check_output(
    ['docker', 'exec', name, 'sha256sum', *files], text=True).splitlines()}
log = Path('/home/choiceoh/glm53-logs/glm53.log').read_text(errors='replace')
required = ['[megakernel] input-reuse CAPTURED M=6 N=6416 K=4096 split=8']
candidate = ['[osar] consumer PDL self-test PASS', '[osar] consumer PDL CAPTURED',
             '[megakernel] AR consumer MHC self-test PASS',
             '[megakernel] AR consumer MHC CAPTURED T=6 bf16=True vec4=True']
if args.mode == '1':
    required += candidate
report = dict(host=socket.gethostname(), image=obj['Image'],
    boot_id=obj['Id'] + '|' + obj['State']['StartedAt'], running=obj['State']['Running'],
    source_sha256=hashes, knobs={k: env.get(k) for k in expected},
    log_sha256=hashlib.sha256(log.encode()).hexdigest(),
    markers={marker: marker in log for marker in required})
memory_fields = {'MemTotal', 'MemFree', 'MemAvailable', 'AnonPages', 'Shmem', 'Slab'}
report['host_memory_kib'] = {line.split(':', 1)[0]: int(line.split()[1])
    for line in Path('/proc/meminfo').read_text().splitlines()
    if line.split(':', 1)[0] in memory_fields}
print(json.dumps(report, indent=2), flush=True)
assert report['running'] and report['knobs'] == expected
assert hashes == files
assert report['image'] == 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
assert all(report['markers'].values()), report['markers']
assert args.mode == '1' or not any(marker in log for marker in candidate)
