#!/usr/bin/env python3
"""Read-only preservation of an already refused CPU21 admission."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

OUT=Path(__file__).resolve().parent
REVISION='028f98167376f0a0857c20c7ec89a3505c1a000f'
JOB='/tmp/glm53-ep-decode-cpu-0909-21'
SOURCE='/home/choiceoh/stkernel-ep-onepass-0909-21'
WORKER='choiceoh@10.10.10.1'

WORKER_CODE=r'''import base64,hashlib,json,pathlib,subprocess
source=pathlib.Path('/home/choiceoh/stkernel-ep-onepass-0909-21')
output=pathlib.Path('/home/choiceoh/glm53-ep-cpu-0909-21/evidence')
runner=source/'probes/run_glm53_ep_short_decode_cpu.py'
raw=runner.read_bytes()
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
print(json.dumps({'revision':git('rev-parse','HEAD'),'status':git('status','--porcelain'),
 'evidence_path':str(output),'evidence_exists':output.exists(),
 'runner_sha256':hashlib.sha256(raw).hexdigest(),'runner_base64':base64.b64encode(raw).decode()}))
'''

HEAD_CODE=r'''import base64,hashlib,json,pathlib,subprocess,sys,shlex,time
job=pathlib.Path('/tmp/glm53-ep-decode-cpu-0909-21')
source=pathlib.Path('/home/choiceoh/stkernel-ep-onepass-0909-21')
names=('driver.pid','driver.py','exit.json','fleet.log','submission.json')
assert sorted(p.name for p in job.iterdir())==sorted(names)
files={}
for name in names:
 p=job/name
 assert p.is_file() and not p.is_symlink()
 raw=p.read_bytes();files[name]={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
worker=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@10.10.10.1',shlex.join(['python3','-B','-c',sys.argv[1]])],capture_output=True,text=True,timeout=30)
if worker.returncode:raise RuntimeError(worker.stderr[-1000:])
print(json.dumps({'observed_at':time.time(),'revision':git('rev-parse','HEAD'),'status':git('status','--porcelain'),
 'files':files,'worker':json.loads(worker.stdout)}))
'''


def snapshot():
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2',
                           shlex.join(['python3','-B','-c',HEAD_CODE,WORKER_CODE])],
                          capture_output=True,timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
    value=json.loads(result.stdout)
    for node in (value,value['worker']):
        assert node['revision']==REVISION and node['status']==''
    assert value['worker']['evidence_exists'] is False
    return value


def main():
    assert not (OUT/'capture.json').exists()
    first=snapshot()
    (OUT/'head').mkdir(exist_ok=True)
    for name,value in first['files'].items():
        raw=base64.b64decode(value.pop('base64'),validate=True)
        assert len(raw)==value['size'] and hashlib.sha256(raw).hexdigest()==value['sha256']
        (OUT/'head'/name).write_bytes(raw)
    runner=base64.b64decode(first['worker'].pop('runner_base64'),validate=True)
    assert hashlib.sha256(runner).hexdigest()==first['worker']['runner_sha256']
    (OUT/'frozen-runner.py').write_bytes(runner)
    second=snapshot()
    for value in second['files'].values():value.pop('base64')
    assert base64.b64decode(second['worker'].pop('runner_base64'),validate=True)==runner
    assert first['files']==second['files'] and first['worker']==second['worker']
    submission=json.loads((OUT/'head/submission.json').read_text())
    assert submission['revision']==REVISION
    assert submission['worker_sources']['contract_sources']['probes/run_glm53_ep_short_decode_cpu.py']==first['worker']['runner_sha256']
    exit_receipt=json.loads((OUT/'head/exit.json').read_text())
    assert exit_receipt['returncode']==exit_receipt['payload_returncode']==2
    assert exit_receipt['copy_returncode'] is None
    assert b'CPU compile needs 12 GiB available; serving memory is not reclaimed' in (OUT/'head/fleet.log').read_bytes()
    capture=dict(scope='completed admission refusal only; no compiler result or GPU evidence',
                 revision=REVISION,head_job=JOB,source=SOURCE,worker=WORKER,
                 before=first,after=second,original_job_files=5,
                 original_job_bytes=sum(x['size'] for x in first['files'].values()),
                 compiler_started=False,container_started=False,
                 reason='frozen runner raises at unchanged memory guard before output mkdir and docker run',
                 memory_at_rejection_not_recorded=True)
    (OUT/'capture.json').write_text(json.dumps(capture,indent=2)+'\n')
    print(json.dumps({'original_job_files':5,'original_job_bytes':capture['original_job_bytes'],
                      'runner_sha256':first['worker']['runner_sha256']}))


if __name__=='__main__':main()
