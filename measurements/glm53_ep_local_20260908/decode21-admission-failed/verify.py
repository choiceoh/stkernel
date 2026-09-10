#!/usr/bin/env python3
"""Pure archive verification, never launches a compiler/container/GPU."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
capture=json.loads((root/'capture.json').read_text())
submission=json.loads((root/'head/submission.json').read_text())
terminal=json.loads((root/'head/exit.json').read_text())
assert terminal['returncode']==terminal['payload_returncode']==2 and terminal['copy_returncode'] is None
assert capture['revision']==submission['revision']=='028f98167376f0a0857c20c7ec89a3505c1a000f'
assert not (root/'result.json').exists() and not (root/'evidence.tar.gz').exists()
for phase in ('before','after'):
    snap=capture[phase]
    assert snap['revision']==snap['worker']['revision']==capture['revision']
    assert snap['status']==snap['worker']['status']==''
    assert snap['worker']['evidence_exists'] is False
    assert len(snap['files'])==5
    for name,value in snap['files'].items():
        raw=(root/'head'/name).read_bytes()
        assert value==dict(size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
runner=(root/'frozen-runner.py').read_bytes()
runner_sha=hashlib.sha256(runner).hexdigest()
assert runner_sha==capture['before']['worker']['runner_sha256']==capture['after']['worker']['runner_sha256']
assert runner_sha==submission['worker_sources']['contract_sources']['probes/run_glm53_ep_short_decode_cpu.py']
fn=next(x for x in ast.parse(runner).body if isinstance(x,ast.FunctionDef) and x.name=='main')
guard=next(x for x in fn.body if isinstance(x,ast.If) and ast.unparse(x.test)=='available < 12 * 1024 * 1024')
assert len(guard.body)==1 and isinstance(guard.body[0],ast.Expr)
assert ast.unparse(guard.body[0].value)=="p.error('CPU compile needs 12 GiB available; serving memory is not reclaimed')"
mkdir=next(x for x in fn.body if isinstance(x,ast.Expr) and ast.unparse(x.value)=='output.mkdir(parents=True)')
docker=next(x for x in fn.body if isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='command')
run=next(x for x in fn.body if isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='completed')
assert guard.lineno<mkdir.lineno<docker.lineno<run.lineno
log=(root/'head/fleet.log').read_text()
assert 'CPU compile needs 12 GiB available; serving memory is not reclaimed' in log
assert (root/'head/driver.pid').read_text().strip()=='115895'
print(json.dumps(dict(verdict='PASS',scope='archive integrity and admission-refusal classification only',
    original_payload_returncode=2,compiler_result_present=False,compiler_started=False,
    container_started=False,actual_failed_memory_kib=None,original_job_files=5,
    original_job_bytes=sum(x['size'] for x in capture['before']['files'].values()),
    frozen_runner_sha256=runner_sha,source_revision=capture['revision'],
    guard_lines=[guard.lineno,mkdir.lineno,docker.lineno,run.lineno]),indent=2))
