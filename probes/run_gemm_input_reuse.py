#!/usr/bin/env python3
"""Short fleet probe beside idle serving; never stop or modify the server."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
OUT=Path('/home/choiceoh/glm53-logs/INPUTREUSE0907')


def traffic():
    with urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=3) as response:
        lines=response.read().decode().splitlines()
    counters={k:0. for k in ('num_requests_running','num_requests_waiting','request_success_total')}
    seen=set()
    for line in lines:
        for key in counters:
            if line.startswith('vllm:'+key+'{'):
                counters[key]+=float(line.rsplit(' ',1)[1]);seen.add(key)
    assert seen==set(counters), ('missing traffic counters',seen)
    return counters


def inspect_server():
    obj=json.loads(subprocess.check_output(['docker','inspect','glm53'],text=True))[0]
    return {'boot_id':obj['Id'],'image':obj['Image'],'running':obj['State']['Running']}


def main():
    session=os.environ['FLEET_SESSION']
    assert re.fullmatch('[a-zA-Z0-9_-]+',session)
    holder=Path('/home/choiceoh/glm53-logs/fleet/holder').read_text().strip().split('|')
    assert holder[0]==session and holder[-1]=='probe', holder
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'result.json').exists(), 'fresh evidence required'
    (OUT/'build').mkdir(exist_ok=True)
    before=traffic();server=inspect_server()
    assert before['num_requests_running']==before['num_requests_waiting']==0
    assert server['running'] and server['image']==IMAGE
    name='inputreuse-'+session
    command=['docker','run','--rm','--name',name,'--gpus','device=0','--network=none',
             '--cpuset-cpus=14-17','--memory=10g','--shm-size=1g',
             '--mount',f'type=bind,src={ROOT},dst=/repo,readonly',
             '--mount',f'type=bind,src={OUT},dst=/evidence',
             '--mount',f'type=bind,src={OUT}/build,dst=/build',
             '--workdir','/repo','--entrypoint','python3',IMAGE,
             '/repo/probes/gemm_input_reuse.py']
    done=threading.Event();samples=[];issues=[]
    def watch():
        while not done.wait(.5):
            try:
                row=traffic();samples.append(row)
                if (row['num_requests_running'] or row['num_requests_waiting']
                    or row['request_success_total']!=before['request_success_total']):
                    issues.append('serving traffic arrived during probe')
            except Exception as exc: issues.append(type(exc).__name__)
            if issues:
                subprocess.run(['docker','stop','-t','1',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                return
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    rc=1
    try:
        with (OUT/'probe.log').open('w') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=360).returncode
    finally:
        done.set();thread.join(timeout=5)
        subprocess.run(['docker','stop','-t','1',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        after=traffic();server_after=inspect_server()
        receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),
                 'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                 'image':IMAGE,'before':before,'after':after,'traffic_samples':samples,
                 'issues':issues,'server_before':server,'server_after':server_after,'returncode':rc}
        (OUT/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert not issues and before==after and server==server_after, receipt
    assert rc==0, f'probe failed: {rc}; see {OUT}/probe.log'
    print('PASS: bounded GPU probe completed; server identity and traffic unchanged')


if __name__=='__main__': main()
