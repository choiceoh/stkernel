"""Preserve owned probe exit and cgroup evidence before container cleanup."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def snapshot(name):
    result=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
    if result.returncode:
        return None  # The attached Docker client may not have created it yet.
    container=json.loads(result.stdout)[0]
    record=dict(container_id=container['Id'],state=container['State'],
        memory_limit_bytes=container['HostConfig']['Memory'])
    pid=container['State']['Pid']
    if pid:
        try:
            entries=Path(f'/proc/{pid}/cgroup').read_text().splitlines()
            relative=next(line.split(':',2)[2] for line in entries if line.startswith('0::'))
            root=Path('/sys/fs/cgroup')/relative.lstrip('/')
            record['memory']={key:(root/key).read_text().strip()
                for key in ('memory.current','memory.peak','memory.events','memory.max')}
        except (OSError,StopIteration) as exc:
            record['memory_unavailable']=type(exc).__name__
    return record


def run(command,name,out):
    report=dict(started=time.time(),name=name,command=command,exit_code=None,samples=0,
                observed_memory_peak_bytes=None,last_resource=None,final_container=None,
                sampler_errors=[],serving_gate=False,numerical_acceptance=False)
    process=None
    def sample():
        try:
            record=snapshot(name)
            if record is None:return
            report['samples']+=1
            report['final_container']=record
            memory=record.get('memory')
            if memory:
                report['last_resource']=record
                peak=int(memory['memory.peak'])
                prior=report['observed_memory_peak_bytes']
                report['observed_memory_peak_bytes']=peak if prior is None else max(prior,peak)
        except Exception as exc:
            # A missing resource sample is unknown, never a fabricated zero.
            error_type=type(exc).__name__
            if error_type not in report['sampler_errors']:report['sampler_errors'].append(error_type)
    try:
        process=subprocess.Popen(command)
        while process.poll() is None:
            sample();time.sleep(.25)
        report['exit_code']=process.returncode
        sample()
    finally:
        report['ended']=time.time()
        Path(out).write_text(json.dumps(report,indent=2)+'\n')
    return report['exit_code']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('command',nargs=argparse.REMAINDER)
    args=ap.parse_args()
    from glm53_offline_checks import check_holder
    check_holder()
    import os
    if not re.fullmatch('moe-m64-'+re.escape(os.environ['FLEET_SESSION'])+'-[A-Za-z0-9_-]+',args.name):
        ap.error('only this hold\'s container may be observed')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or command[0]!='timeout' or '--rm' in command:
        ap.error('a timeout-bounded persistent probe container is required')
    if command.count('--name')!=1 or command[command.index('--name')+1]!=args.name:
        ap.error('command container name mismatch')
    if args.out.exists() or snapshot(args.name) is not None:
        ap.error('fresh evidence and absent owned container required')
    return run(command,args.name,args.out)


if __name__=='__main__':raise SystemExit(main())
