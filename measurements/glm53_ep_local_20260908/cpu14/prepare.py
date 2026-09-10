#!/usr/bin/env python3
"""Prepare/submit CPU14 on the head only, after an explicit committed revision.

Safe to rerun after a memory refusal: no job is created below 12 GiB and an
identical clean frozen source can be reused. An existing job is reported, never
submitted again. No worker fallback, service reclaim or GPU queue mutation.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
BASE = '18148116a7abb242741d5b112c8d735353d9fc71'
BUNDLE = Path('/tmp/glm53-ep-local-0908-14.bundle')
REPORT = Path('/tmp/glm53-ep-local-cpu14-launch.json')

DRIVER = r'''import json,os,subprocess,time
from pathlib import Path
job=Path(__file__).resolve().parent
data=json.loads((job/'submission.json').read_text())
data['started']=time.time()
(job/'started.json').write_text(json.dumps(dict(pid=os.getpid(),started=data['started']))+'\n')
try:
    environment={k:v for k,v in os.environ.items() if not k.startswith('FLEET_')}
    environment['REPO']=data['scheduler']
    result=subprocess.run(data['command'],stdin=subprocess.DEVNULL,env=environment)
    data.update(returncode=result.returncode,complete=result.returncode==0)
except BaseException as exc:
    data.update(returncode=1,complete=False,error=type(exc).__name__+': '+str(exc)[:1000])
finally:
    data['ended']=time.time()
    (job/'exit.json').write_text(json.dumps(data,indent=2)+'\n')
'''

REMOTE = r'''import base64,fcntl,hashlib,json,pathlib,subprocess,sys,time
source=pathlib.Path('/home/choiceoh/stkernel-ep-local-0908-14')
previous=pathlib.Path('/home/choiceoh/stkernel-ep-local-0908-13')
job=pathlib.Path('/tmp/glm53-ep-local-compile0908-14-head')
bundle=pathlib.Path('/tmp/glm53-ep-local-0908-14.bundle')
scheduler=pathlib.Path('/home/choiceoh/stkernel')
mode,revision,driver64,bundle_sha=sys.argv[1:]

def run(command,label):
    result=subprocess.run(command,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(label+' failed: '+result.stderr[-1200:])
    return result.stdout.strip()

def available():
    return next(int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))

def frozen():
    if not source.exists():
        return False
    if run(['git','-C',str(source),'rev-parse','HEAD'],'frozen revision')!=revision:
        raise RuntimeError('existing CPU14 source has another revision; refusing replacement')
    if run(['git','-C',str(source),'status','--porcelain'],'frozen cleanliness'):
        raise RuntimeError('existing CPU14 source is dirty; refusing replacement')
    return True

def existing_job():
    if not job.exists():
        return None
    metadata=job/'submission.json'
    if not metadata.is_file():
        raise RuntimeError('CPU14 job path exists without submission metadata; inspect manually')
    data=json.loads(metadata.read_text())
    if data['revision']!=revision:
        raise RuntimeError('existing CPU14 job has another revision; refusing replacement')
    completed=job/'exit.json'
    return dict(submitted=False,already_submitted=True,source=str(source),job=str(job),revision=revision,
                completion=json.loads(completed.read_text()) if completed.is_file() else None,
                pid=(job/'driver.pid').read_text().strip() if (job/'driver.pid').exists() else None)

def prepare():
    prior=existing_job()
    if prior is not None:
        return prior
    reusable=frozen()
    free=available()
    info=dict(submitted=False,source=str(source),job=str(job),revision=revision,
              source_exists=reusable,host_mem_available_kib=free,
              bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest() if bundle.exists() else None)
    if free<12*1024*1024:
        return dict(info,reason='unchanged 12 GiB host memory guard; no reclaim or worker fallback')
    if mode=='inspect':
        return dict(info,ready=True)
    if mode!='submit':
        raise RuntimeError('unsupported preparation mode')
    if not reusable:
        if not bundle_sha or not bundle.is_file() or hashlib.sha256(bundle.read_bytes()).hexdigest()!=bundle_sha:
            raise RuntimeError('CPU14 bundle is missing or changed')
        run(['git','-C',str(previous),'cat-file','-e','18148116a7abb242741d5b112c8d735353d9fc71^{commit}'],'CPU13 base prerequisite')
        run(['git','clone','--quiet','--shared','--no-checkout',str(previous),str(source)],'CPU14 clone')
        run(['git','-C',str(source),'fetch','--quiet',str(bundle),'HEAD'],'CPU14 bundle fetch')
        run(['git','-C',str(source),'checkout','--quiet','--detach','FETCH_HEAD'],'CPU14 checkout')
        run(['git','-C',str(source),'remote','set-url','origin','https://github.com/choiceoh/stkernel.git'],'CPU14 origin')
        frozen()
    free=available()
    if free<12*1024*1024:
        return dict(info,source_exists=True,host_mem_available_kib=free,
                    reason='12 GiB guard after source freeze; frozen source reusable, no job started')
    if run(['git','-C',str(scheduler),'status','--porcelain'],'scheduler cleanliness'):
        raise RuntimeError('normal scheduler is dirty; no CPU14 job started')
    scheduler_revision=run(['git','-C',str(scheduler),'rev-parse','HEAD'],'scheduler revision')
    scheduler_approved_main=run(['git','-C',str(scheduler),'rev-parse','origin/main'],'approved main')
    run(['git','-C',str(scheduler),'merge-base','--is-ancestor',
         scheduler_revision,scheduler_approved_main],'approved scheduler ancestry')
    # Other holders keep the operational checkout fixed during their run.
    # An upstream-only change to an unrelated probe need not move that tree:
    # require every scheduler, launcher, profile and test byte to match main.
    run(['git','-C',str(scheduler),'diff','--quiet',scheduler_revision,
         scheduler_approved_main,'--','bench','launchers','profiles','tests'],
        'current approved scheduler/support content')
    run(['git','-C',str(scheduler),'merge-base','--is-ancestor',
         'd95a2cdb46f2446bd4b58f7d5969b42c895f0b3d','HEAD'],'approved restore fixture ancestry')
    image='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
    command=['bash',str(scheduler/'bench/fleet.sh'),'run','--cpu','eplocalcpu0908v14head','6',
             'CPU14 committed EP local kernel follow-up; compare CPU13; runc no devices 4g 2cpu','--',
             'python3',str(source/'probes/run_glm53_ep_local_cpu_compile.py'),
             '--image',image,'--arm','local','--output',str(job/'local')]
    metadata=dict(source=str(source),revision=revision,scheduler=str(scheduler),
                  scheduler_revision=scheduler_revision,
                  scheduler_approved_main=scheduler_approved_main,
                  approved_scheduler_ancestor=True,
                  approved_identical_directories=['bench','launchers','profiles','tests'],
                  command=command,created=time.time(),host_mem_available_kib=free,
                  existing_frozen_source_reused=reusable,serving_memory_reclaimed=False,
                  reason='CPU14 committed EP local kernel candidate, current CPU-only wrapper and source contracts; compare CPU13')
    raw=base64.b64decode(driver64,validate=True)
    compile(raw,'CPU14 driver','exec')
    job.mkdir()
    (job/'submission.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (job/'driver.py').write_bytes(raw)
    with (job/'fleet.log').open('x') as log:
        process=subprocess.Popen(['python3','-B',str(job/'driver.py')],stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (job/'driver.pid').write_text(str(process.pid)+'\n')
    return dict(submitted=True,source=str(source),job=str(job),revision=revision,pid=process.pid,
                host_mem_available_kib=free,scheduler_revision=metadata['scheduler_revision'])

try:
    if mode=='submit':
        with pathlib.Path(str(job)+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            result=prepare()
    else:
        result=prepare()
    print(json.dumps(result))
except BaseException as exc:
    print(json.dumps(dict(submitted=False,error=type(exc).__name__+': '+str(exc)[:1800])),file=sys.stderr)
    raise SystemExit(1)
'''


def run(command, label, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(label + ' failed: ' + (result.stderr or result.stdout)[-1800:])
    return result.stdout.strip()


def remote(mode, revision, bundle_sha=''):
    command = ['python3', '-B', '-c', REMOTE, mode, revision, base64.b64encode(DRIVER.encode()).decode(), bundle_sha]
    value = run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'choiceoh@srv2', shlex.join(command)],
                'CPU14 remote ' + mode, stdin=subprocess.DEVNULL)
    return json.loads(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True, help='Exact committed future HEAD approved by parent')
    args = parser.parse_args()
    if re.fullmatch('[a-f0-9]{40}', args.revision) is None:
        parser.error('--revision must be a full commit SHA')
    revision = run(['git', 'rev-parse', 'HEAD'], 'local HEAD', cwd=ROOT)
    if revision != args.revision:
        raise RuntimeError('requested revision does not equal current HEAD')
    if run(['git', 'status', '--porcelain', '--untracked-files=no'], 'local cleanliness', cwd=ROOT):
        raise RuntimeError('tracked source changes are not committed')
    run(['git', 'merge-base', '--is-ancestor', BASE, revision], 'CPU13 base ancestry', cwd=ROOT)
    info = remote('inspect', revision)
    if info.get('already_submitted') or not info.get('ready'):
        REPORT.write_text(json.dumps(info, indent=2) + '\n')
        print(json.dumps(info))
        return
    digest = ''
    if not info['source_exists']:
        if not BUNDLE.exists():
            run(['git', 'bundle', 'create', str(BUNDLE), 'HEAD', '^' + BASE], 'CPU14 bundle create', cwd=ROOT)
        heads = run(['git', 'bundle', 'list-heads', str(BUNDLE)], 'CPU14 bundle identity', cwd=ROOT).splitlines()
        if heads != [revision + ' HEAD']:
            raise RuntimeError('CPU14 bundle revision differs; refusing path replacement')
        digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
        if info['bundle_sha256'] is None:
            run(['scp', '-q', str(BUNDLE), 'choiceoh@srv2:' + str(BUNDLE)], 'CPU14 bundle transfer')
        elif info['bundle_sha256'] != digest:
            raise RuntimeError('remote CPU14 bundle differs; refusing path replacement')
    info = remote('submit', revision, digest)
    REPORT.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info))


if __name__ == '__main__':
    try:
        ast.parse(DRIVER)
        ast.parse(REMOTE)
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(type(exc).__name__ + ': ' + str(exc)[:2000], file=sys.stderr)
        raise SystemExit(1)
