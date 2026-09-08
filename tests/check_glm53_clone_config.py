#!/usr/bin/env python3
"""Create-only Docker config check against serving; never start or stop a model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'probes'))
import glm53_observation_host as host


def identity(container):
    return dict(id=container['Id'],image=container['Image'],config=host.digest(container['Config']),
        host_config=host.digest(host.host_config_identity(container['HostConfig'])),
        mounts=host.digest(sorted(container['Mounts'],key=lambda m:m['Destination'])),
        state={k:container['State'][k] for k in ('Running','StartedAt','Pid','OOMKilled')})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original',choices=('glm53','glm53-worker'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    for folder in ('prof','glmlogs'):(args.out/folder).mkdir()
    session='cpuconfig-'+uuid.uuid4().hex[:12];name='glm53-observe-'+session
    report=dict(complete=False,clone_started=False,gpu_work=False,source_sha256={
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__).resolve(),ROOT/'probes/glm53_observation_host.py')})
    cid=None
    try:
        before=host.inspect(args.original)
        if not before or not before['State']['Running']:raise RuntimeError('running original required')
        report['original_before']=identity(before)
        payload=host.clone_payload(before,directory=args.out,source=ROOT,session=session)
        cid=host.create(name,payload);clone=host.owned(name,session)
        if clone['Id']!=cid or clone['State']['Running'] or not clone['State']['StartedAt'].startswith('0001-'):
            raise RuntimeError('clone identity or never-started proof failed')
        config_diff=[k for k,v in payload.items() if k!='HostConfig' and clone['Config'].get(k)!=v]
        host_diff=host.host_config_differences(payload['HostConfig'],clone['HostConfig'])
        report.update(config_differences=config_diff,host_config_differences=host_diff,
            requested_oom_kill_disable=payload['HostConfig'].get('OomKillDisable'),
            observed_oom_kill_disable=clone['HostConfig'].get('OomKillDisable'),
            config_fields=len(clone['Config']),host_config_fields=len(clone['HostConfig']))
        if config_diff or host_diff:raise RuntimeError('cloned configuration changed')
        report['original_after']=identity(host.inspect(args.original))
        if report['original_before']!=report['original_after']:raise RuntimeError('original identity changed')
        report['complete']=True
    except Exception as exc:report['error']=repr(exc)
    finally:
        if cid:
            removed=subprocess.run(['docker','rm',cid],capture_output=True,text=True,timeout=75)
            report['clone_removed']=removed.returncode==0 and host.inspect(name) is None
            if not report['clone_removed']:report['complete']=False
        (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    return 0 if report['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
