#!/usr/bin/env python3
"""Fleet-owned kernel probe with continuous memory and service guards."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading

from run_gemm_input_reuse import IMAGE, ROOT, inspect_server, memory_available, traffic
from run_moe_reform_cpu import mounts
from moe_reform_sanitizer import sanitizer_result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--maintenance',action='store_true')
    ap.add_argument('--resume-numerics',type=Path,
                    help='Reuse unchanged source numerical and API-only memcheck evidence; run racecheck next')
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
    reused = None
    sanitizer_reports = {}
    if args.resume_numerics:
        prior=args.resume_numerics
        old=(prior/'source.commit').read_text().strip()
        assert re.fullmatch('[0-9a-f]{40}',old)
        subprocess.run(['git','-C',str(ROOT),'diff','--exit-code',old,commit,'--',
                        'overlay/modules/glm53_moe','build/glm53',
                        'probes/moe_reform_ab.py','probes/moe_decode_stream_probe.py',
                        'probes/megakernel_glm53_bench.py'],check=True)
        receipt=json.loads((prior/'admission.json').read_text())
        assert receipt['image']==IMAGE and receipt['issues']==["('memcheck', 77)"]
        assert receipt['server_before']==receipt['server_after']
        assert receipt['before']==receipt['after']
        for report in ('bundle','memcheck'):
            assert json.loads((prior/(report+'.json')).read_text())['status']=='PASS'
            for suffix in ('.json','.log'):
                shutil.copy2(prior/(report+suffix),out/(report+suffix))
        sanitizer_reports['memcheck']=sanitizer_result((out/'memcheck.log').read_text(),77)
        reused=dict(source_commit=old,evidence=str(prior),reason='identical kernel and fixture sources')
        (out/'reused-evidence.json').write_text(json.dumps(reused,indent=2)+'\n')
    (out/'build').mkdir(exist_ok=True)
    name='moereform-'+session
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
        # One integrated numerical/graph campaign, then focused sanitizers.
        # No individual-feature benchmarks or micro-speed escalation gate.
        target='/repo/probes/moe_reform_ab.py'
        for tool in (('racecheck',) if reused else ('probe','memcheck','racecheck')):
            assert not issues, issues
            if tool == 'probe':
                command=common+['--entrypoint','python3',IMAGE,target,'--check-only',
                                '--out','/evidence/bundle.json']
                report='bundle'
            else:
                command=common+['--entrypoint','/san/compute-sanitizer',IMAGE,
                    '--tool',tool,'--target-processes','application-only',
                    '--error-exitcode','77','python3',target,'--check-only',
                    '--sanitizer-smoke','--out',f'/evidence/{tool}.json']
                report=tool
            with (out/(report+'.log')).open('w') as log:
                rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
            if tool == 'probe':
                assert rc==0,(tool,rc)
            else:
                sanitizer_reports[tool]=sanitizer_result((out/(report+'.log')).read_text(),rc)
            assert json.loads((out/(report+'.json')).read_text())['status']=='PASS'
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
        receipt['sanitizers']=sanitizer_reports
        receipt['reused']=reused
        (out/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert not issues,issues
    print('PASS integrated numerics, graphs, memcheck, racecheck and continuous resource guard')


if __name__=='__main__':main()
