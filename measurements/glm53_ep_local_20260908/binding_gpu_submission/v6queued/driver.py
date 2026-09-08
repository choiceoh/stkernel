import hashlib,json,os,subprocess,time
from pathlib import Path
job=Path(__file__).resolve().parent
data=json.loads((job/'submission.json').read_text())
data['started']=time.time()
data['exit_code']=1
def git(*args):
    return subprocess.check_output(['git','-C',data['scheduler'],*args],text=True).strip()
try:
    assert not git('status','--porcelain'), 'scheduler became dirty before submission'
    assert git('rev-parse','HEAD')==data['scheduler_revision'], 'scheduler changed before submission'
    assert git('rev-parse','origin/main')==data['scheduler_revision'], 'scheduler is not approved main'
    for revision in data['approved_fixes']:
        subprocess.run(['git','-C',data['scheduler'],'merge-base','--is-ancestor',revision,'HEAD'],check=True)
    source=Path(data['source'])
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==data['revision']
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    assert hashlib.sha256((job/'cpu-evidence.json').read_bytes()).hexdigest()==data['cpu_evidence_sha256']
    for name,digest in data['capsule_cpu_artifacts'].items():
        assert hashlib.sha256((job/name).read_bytes()).hexdigest()==digest
    for name,digest in data['pair_sources'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    import sys
    sys.path.insert(0,str(source))
    from probes.glm53_ep_bindings_capsule import validate_capsule
    validate_capsule(Path(data['capsule_root']),data['capsule_manifest_sha256'])
    # Do not inherit an enclosing reservation, prepared manifest, priority,
    # bypass, or same-session override. fleet.sh creates normal fresh state.
    removed=sorted(k for k in os.environ if k.startswith('FLEET_'))
    environment={k:v for k,v in os.environ.items() if not k.startswith('FLEET_')}
    environment['REPO']=data['scheduler']
    data['removed_inherited_fleet_environment_names']=removed
    (job/'driver-start.json').write_text(json.dumps(data,indent=2)+'\n')
    result=subprocess.run(data['command'],stdin=subprocess.DEVNULL,env=environment)
    data['exit_code']=result.returncode
except BaseException as exc:
    data['driver_error']=str(exc)
finally:
    data['ended']=time.time()
    data['complete']=data['exit_code']==0
    (job/'exit.json').write_text(json.dumps(data,indent=2)+'\n')
