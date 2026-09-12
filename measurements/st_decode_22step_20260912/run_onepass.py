"""Run the unmodified canonical CLI between all-rank ST identity receipts.

EVIDENCE_DIR names this pass's private directory. Arguments pass unchanged to
bench/onepass.py; its raw JSONL and workload remain unchanged. Identity is a
sidecar because the canonical build-label reader currently recognises vLLM.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['EVIDENCE_DIR'])
OUT.mkdir(parents=True, exist_ok=True)
# Run on srv2, the fleet's rank 0. The other homes are not shared.
HOSTS = (None, 'choiceoh@10.10.10.1', 'choiceoh@10.10.10.3', 'choiceoh@10.10.10.4')
CAPTURE = r'''
import hashlib,json,pathlib,re,subprocess
c=json.loads(subprocess.check_output(['docker','inspect','st-glm53'],text=True))[0]
assert c['State']['Running'] and not c['State']['Paused'] and not c['State']['Restarting']
env=dict(v.split('=',1) for v in c['Config']['Env'] if '=' in v)
code="import hashlib,json,pathlib; p=pathlib.Path('/repo'); print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for part in ('engine','launchers') for f in sorted((p/part).rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}))"
hashes=json.loads(subprocess.check_output(['docker','exec','st-glm53','python3','-c',code],text=True))
manifest=json.loads(subprocess.check_output(['docker','exec','st-glm53','cat','/opt/st/runtime-manifest.json'],text=True))
cmd=c['Config']['Cmd']
port=re.search(r'--port\s+(\d+)', ' '.join(cmd))
assert c['HostConfig']['NetworkMode']=='host' and port, 'unexpected ST listener configuration'
print(json.dumps(dict(rank=int(env['RANK']),boot_id=c['Id']+'|'+c['State']['StartedAt'],image=c['Image'],
    image_tag=c['Config']['Image'],command=cmd,port=int(port.group(1)),
    source_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
    files=hashes,runtime=manifest,mounts=c['Mounts'],
    environment={k:v for k,v in env.items() if k.startswith(('STK_','ST_LEASE_'))})))
'''


def capture():
    ranks = []
    for rank, host in enumerate(HOSTS):
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, 'python3', '-'] if host else ['python3', '-']
        value = json.loads(subprocess.check_output(cmd, input=CAPTURE, text=True, timeout=60))
        assert value['rank'] == rank and value['port'] == int(os.environ['GLM53_API_PORT'])
        assert value['environment']['ST_LEASE_OWNER'] == os.environ['LEASE_OWNER']
        ranks.append(value)
    assert len({r['source_sha256'] for r in ranks}) == 1, 'rank source mismatch'
    return ranks


before = capture()
(OUT / 'identity-before.json').write_text(json.dumps(before, indent=2) + '\n')
metrics_url = 'http://127.0.0.1:' + os.environ['GLM53_API_PORT'] + '/metrics'
with urllib.request.urlopen(metrics_url, timeout=10) as response:
    (OUT / 'metrics-before.prom').write_bytes(response.read())
command = [sys.executable, str(REPO / 'bench/onepass.py'), *sys.argv[1:]]
(OUT / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
try:
    result = subprocess.call(command)
finally:
    with urllib.request.urlopen(metrics_url, timeout=10) as response:
        (OUT / 'metrics-after.prom').write_bytes(response.read())
    after = capture()
    (OUT / 'identity-after.json').write_text(json.dumps(after, indent=2) + '\n')
    assert before == after, 'serving identity changed during onepass'
raise SystemExit(result)
