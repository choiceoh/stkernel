#!/usr/bin/env python3
"""Reserved, bounded warp-consumer maintenance probe with fleet-owned idle recovery."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
OUT=Path(os.environ['GEMM_INPUT_WARP_OUT'])


def state():
    obj=json.loads(subprocess.check_output(['docker','inspect','glm53'],text=True))[0]
    return {'id':obj['Id'],'image':obj['Image'],'state':obj['State']}


def main():
    session=os.environ['FLEET_SESSION']
    assert re.fullmatch('[a-zA-Z0-9_-]+',session)
    holder=Path('/home/choiceoh/glm53-logs/fleet/holder').read_text().strip().split('|')
    assert holder[0]==session and holder[-1]=='boot',holder
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    assert not (OUT/'receipt.json').exists(), 'fresh evidence required'
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'build').mkdir(exist_ok=True)
    before=state();assert before['state']['Running'] and before['image']==IMAGE
    with urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5) as r:metrics=r.read().decode()
    for key in ('num_requests_running','num_requests_waiting'):
        vals=[float(line.rsplit(' ',1)[1]) for line in metrics.splitlines() if line.startswith('vllm:'+key+'{')]
        assert vals and sum(vals)==0,(key,vals)
    (OUT/'before-metrics.txt').write_text(metrics)
    (OUT/'before-head.log').write_bytes(Path('/home/choiceoh/glm53-logs/glm53.log').read_bytes())
    receipt={'started_utc':datetime.now(timezone.utc).isoformat(),'before':before,
             'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
             'probe_returncode':None,'public_recovery':'central idle controller','after':None}
    def save():(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    name='inputwarp-'+session
    try:
        subprocess.run(['docker','stop','-t','30','glm53'],check=True,timeout=40)
        assert not state()['state']['Running']
        available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
        assert available>=16*1024**3,available
        command=['docker','run','--rm','--name',name,'--gpus','device=0','--network=none',
                 '--cpuset-cpus=14-17','--memory=10g','--shm-size=1g',
                 '--mount',f'type=bind,src={ROOT},dst=/repo,readonly',
                 '--mount',f'type=bind,src={OUT},dst=/evidence',
                 '--mount',f'type=bind,src={OUT}/build,dst=/build',
                 '--workdir','/repo','--entrypoint','python3',IMAGE,'/repo/probes/gemm_input_warp.py']
        with (OUT/'probe.log').open('w') as log:
            receipt['probe_returncode']=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=360).returncode
        save()
    finally:
        subprocess.run(['docker','stop','-t','1',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        receipt['after']=state();receipt['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    assert receipt['probe_returncode']==0,receipt
    print('PASS bounded warp probe; fleet released for the next job',flush=True)


if __name__=='__main__':main()
