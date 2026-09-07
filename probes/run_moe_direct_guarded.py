#!/usr/bin/env python3
"""Fleet-owned kernel probe with continuous memory and service guards."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading

from run_gemm_input_reuse import IMAGE, ROOT, inspect_server, memory_available, traffic
from run_moe_direct_cpu import mounts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--maintenance',action='store_true')
    ap.add_argument('--variants',nargs='+',choices=('pair','vector','warp'),default=['pair','vector'])
    args=ap.parse_args()
    session=os.environ['FLEET_SESSION']
    assert re.fullmatch('[a-zA-Z0-9_-]+',session)
    holder=Path('/home/choiceoh/glm53-logs/fleet/holder').read_text().strip().split('|')
    assert holder[0]==session and holder[-1]==('boot' if args.maintenance else 'probe'),holder
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'source.commit').exists(),'fresh evidence required'
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    (out/'source.commit').write_text(commit+'\n')
    (out/'build').mkdir(exist_ok=True)
    name='moedirect-'+session
    done=threading.Event();samples=[];issues=[]
    before=server=None;rc=1;thread=None
    def stop_own():
        subprocess.run(['docker','stop','-t','1',name],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=10)
    def watch():
        while not done.wait(.5):
            try:
                row=traffic() if before is not None else None
                available=memory_available()
                samples.append({'traffic':row,'available_bytes':available})
                if inspect_server()!=server:issues.append('server identity or running state changed')
                if available<12*1024**3:issues.append('host headroom fell below 12 GiB')
                if row is not None and (row['num_requests_running'] or row['num_requests_waiting']
                    or row['request_success_total']!=before['request_success_total']):
                    issues.append('serving traffic arrived')
            except Exception as exc:issues.append(type(exc).__name__)
            if issues:
                try:stop_own()
                except Exception as exc:issues.append('stop: '+type(exc).__name__)
    try:
        server=inspect_server()
        assert server['image']==IMAGE,server
        assert server['running']==(not args.maintenance),server
        before=traffic() if server['running'] else None
        assert before is None or before['num_requests_running']==before['num_requests_waiting']==0
        available=memory_available()
        assert available>=16*1024**3,('need 16 GiB available before probe',available)
        common=['docker','run','--rm','--name',name,'--gpus','device=0','--network=none',
                '--cpuset-cpus=14-17','--memory=7g','--shm-size=1g',
                '--mount',f'type=bind,src={ROOT},dst=/repo,readonly',
                '--mount',f'type=bind,src={out},dst=/evidence',
                '--mount',f'type=bind,src={out}/build,dst=/build',
                '--mount','type=bind,src=/usr/local/cuda/compute-sanitizer,dst=/san,readonly',
                '-e','MK_PKG_PATH=/usr/local/lib/python3.12/dist-packages',
                *mounts(),'--workdir','/repo']
        thread=threading.Thread(target=watch,daemon=True);thread.start()
        for variant in args.variants:
            for tool in ('probe',):
                assert not issues,issues
                target=f'/repo/probes/moe_direct_scatter_ab.py'
                command=common+['--entrypoint','python3',IMAGE,target,'--variant',variant,
                    '--rounds','32','--out',f'/evidence/{variant}.json']
                with (out/(variant+'.log')).open('w') as log:
                    rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=480).returncode
                assert rc==0,(variant,rc)
                assert json.loads((out/(variant+'.json')).read_text())['status']=='PASS'
    except BaseException as exc:
        issues.append(str(exc));raise
    finally:
        done.set()
        if thread is not None:thread.join(timeout=10)
        stop_own()
        after=server_after=None
        try:
            server_after=inspect_server()
            after=traffic() if server_after['running'] else None
            if server!=server_after or before!=after:issues.append('final service or traffic mismatch')
        except Exception as exc:issues.append('final collection: '+type(exc).__name__)
        receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),'source_commit':commit,
                 'image':IMAGE,'before':before,'after':after,'traffic_samples':samples,
                 'issues':issues,'server_before':server,'server_after':server_after,'returncode':rc,
                 'admission_min_bytes':16*1024**3,'continuous_min_bytes':12*1024**3,
                 'maintenance':args.maintenance}
        (out/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert not issues,issues
    print('PASS numerical checks, graphs and continuous resource guard; sanitizer gate separate')


if __name__=='__main__':main()
