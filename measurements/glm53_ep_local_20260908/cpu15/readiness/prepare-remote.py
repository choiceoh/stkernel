import base64,fcntl,hashlib,json,pathlib,subprocess,sys,time
source=pathlib.Path('/home/choiceoh/stkernel-ep-local-0908-15')
previous=pathlib.Path('/home/choiceoh/stkernel-ep-local-0908-14')
job=pathlib.Path('/tmp/glm53-ep-local-compile0908-15-head')
bundle=pathlib.Path('/tmp/glm53-ep-local-0908-15.bundle')
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
        raise RuntimeError('existing CPU15 source has another revision; refusing replacement')
    if run(['git','-C',str(source),'status','--porcelain'],'frozen cleanliness'):
        raise RuntimeError('existing CPU15 source is dirty; refusing replacement')
    return True

def existing_job():
    if not job.exists():
        return None
    metadata=job/'submission.json'
    if not metadata.is_file():
        raise RuntimeError('CPU15 job path exists without submission metadata; inspect manually')
    data=json.loads(metadata.read_text())
    if data['revision']!=revision:
        raise RuntimeError('existing CPU15 job has another revision; refusing replacement')
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
            raise RuntimeError('CPU15 bundle is missing or changed')
        run(['git','-C',str(previous),'cat-file','-e','881456a1f43fcf61de1bb5822b883dd2fd32e693^{commit}'],'CPU14 base prerequisite')
        run(['git','clone','--quiet','--shared','--no-checkout',str(previous),str(source)],'CPU15 clone')
        run(['git','-C',str(source),'fetch','--quiet',str(bundle),'HEAD'],'CPU15 bundle fetch')
        run(['git','-C',str(source),'checkout','--quiet','--detach','FETCH_HEAD'],'CPU15 checkout')
        run(['git','-C',str(source),'remote','set-url','origin','https://github.com/choiceoh/stkernel.git'],'CPU15 origin')
        frozen()
    free=available()
    if free<12*1024*1024:
        return dict(info,source_exists=True,host_mem_available_kib=free,
                    reason='12 GiB guard after source freeze; frozen source reusable, no job started')
    if run(['git','-C',str(scheduler),'status','--porcelain'],'scheduler cleanliness'):
        raise RuntimeError('normal scheduler is dirty; no CPU15 job started')
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
    command=['bash',str(scheduler/'bench/fleet.sh'),'run','--cpu','eplocalcpu0908v15head','6',
             'CPU15 capsule13.0.3 full CuTe compile and identity; runc no devices 4g2CPU','--',
             'python3','-B',str(source/'probes/run_glm53_ep_local_cpu_compile.py'),
             '--image',image,'--arm','local','--output',str(job/'local'),
             '--capsule-root','/tmp/glm53-bindings-capsule-cpu0908-2/capsule',
             '--manifest-sha256','b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab']
    metadata=dict(source=str(source),revision=revision,scheduler=str(scheduler),
                  scheduler_revision=scheduler_revision,
                  scheduler_approved_main=scheduler_approved_main,
                  approved_scheduler_ancestor=True,
                  approved_identical_directories=['bench','launchers','profiles','tests'],
                  command=command,created=time.time(),host_mem_available_kib=free,
                  existing_frozen_source_reused=reusable,serving_memory_reclaimed=False,
                  reason='CPU15 committed EP local kernel candidate, current CPU-only wrapper and source contracts; compare CPU14')
    raw=base64.b64decode(driver64,validate=True)
    compile(raw,'CPU15 driver','exec')
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
