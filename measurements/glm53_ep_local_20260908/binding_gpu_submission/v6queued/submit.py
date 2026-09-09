#!/usr/bin/env python3
"""Freeze and normally enqueue binding diagnostic v6 only when explicitly run.

Prepared for review; importing this module performs no action. The diagnostic uses a frozen CPU2 capsule source with unchanged CPU13 kernel contracts.
Neither submission nor this script relaxes the GO-time lifecycle checks.
"""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

LOCAL_REPO = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
REVISION = '63f56a54dbb91e994456b519a53b679963a93583'
PROOF_PATH = 'measurements/glm53_ep_local_20260908/cpu13/local/result.json'
SESSION = 'epbindinggpu0908v6'
APPROVED_FIXES = (
    'd95a2cdb46f2446bd4b58f7d5969b42c895f0b3d',
    '4a2c960ad632596ca3224ead88da43160012d393',
)

DRIVER = r'''import hashlib,json,os,subprocess,time
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
'''

REMOTE = r'''import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
revision,proof_path,session,fixes_json,proof_b64,driver_b64=sys.argv[1:]
fixes=json.loads(fixes_json)
source=Path('/home/choiceoh/stkernel-ep-binding-gpu0908-6b')
job=Path('/tmp/glm53-ep-binding-gpu0908-6')
scheduler=Path('/home/choiceoh/stkernel')
fleet=Path('/home/choiceoh/glm53-logs/fleet')
def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def read_optional(path):
    return path.read_text() if path.exists() else None
def no_duplicate():
    queue=read_optional(fleet/'queue') or ''
    holder=read_optional(fleet/'holder') or ''
    assert not any(len(row.split('|'))>1 and row.split('|')[1]==session
                   for row in queue.splitlines()), 'session already queued'
    assert not holder or holder.split('|')[0]!=session, 'session already holds fleet'

assert source.is_dir() and not job.exists(), 'frozen source missing or v6 job exists; do not resubmit'
no_duplicate()
assert not git(scheduler,'status','--porcelain'), 'scheduler must be clean'
scheduler_revision=git(scheduler,'rev-parse','HEAD')
assert scheduler_revision==git(scheduler,'rev-parse','origin/main'), 'scheduler must equal origin/main'
# Read the upstream ref without mutating the running scheduler checkout.
advertised=git(scheduler,'ls-remote','origin','refs/heads/main').splitlines()
assert len(advertised)==1 and advertised[0].split()[0]==scheduler_revision, 'scheduler is not current approved main'
for fix in fixes:
    subprocess.run(['git','-C',str(scheduler),'merge-base','--is-ancestor',fix,'HEAD'],check=True)
assert git(source,'rev-parse','HEAD')==revision and not git(source,'status','--porcelain')
proof_bytes=base64.b64decode(proof_b64,validate=True)
driver_bytes=base64.b64decode(driver_b64,validate=True)
compile(driver_bytes,'driver.py','exec')

# mkdir is exclusive: concurrent/repeated invocations cannot both submit.
job.mkdir()
(job/'driver.py').write_bytes(driver_bytes)
assert (source/proof_path).read_bytes()==proof_bytes, 'local and frozen CPU13 receipts differ'
(job/'cpu-evidence.json').write_bytes(proof_bytes)
sys.path.insert(0,str(source))
from probes.glm53_ep_local_evidence import validate_compile_evidence
proof=validate_compile_evidence(source,job/'cpu-evidence.json')
assert proof['contracts']['tests_run']==68
assert proof['contracts']['failures']==proof['contracts']['errors']==proof['contracts']['skips']==0
assert proof['cuda_initialized'] is False
assert len(proof['mounted_sources'])==13 and len(proof['contracts']['files'])==18
assert len(proof['remap_compilation'])==24
proof_sha=hashlib.sha256(proof_bytes).hexdigest()
cpu=Path('/tmp/glm53-bindings-capsule-cpu0908-2')
capsule=cpu/'capsule'
manifest_sha='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
from probes.glm53_ep_bindings_capsule import validate_capsule
validate_capsule(capsule,manifest_sha)
receipt=json.loads((cpu/'receipt.json').read_text())
inner=json.loads((cpu/'result.json').read_text())
assert receipt['verdict']==inner['verdict']=='PASS'
assert receipt['exit_code']==0
assert receipt['image']==inner['image']=='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
assert receipt['result_sha256']==hashlib.sha256((cpu/'result.json').read_bytes()).hexdigest()
assert receipt['log_sha256']==hashlib.sha256((cpu/'container.log').read_bytes()).hexdigest()
assert receipt['capsule_manifest_sha256']==inner['capsule_manifest_sha256']==manifest_sha
assert receipt['sources']==inner['sources'] and len(receipt['sources'])==3
assert all(hashlib.sha256((source/p).read_bytes()).hexdigest()==s for p,s in receipt['sources'].items())
assert inner['exposed_device_nodes']==[] and inner['probe_cuda_api_calls'] is False
assert inner['cuda_context_queried'] is False and inner['torch_imported'] is False
assert inner['dependencies']['compatible'] is True and inner['imports']
for key in ('base_distributions','distribution_resolution'):
 assert hashlib.sha256((cpu/Path(inner[key]['path']).name).read_bytes()).hexdigest()==inner[key]['sha256']
artifacts={}
for name in ('receipt.json','result.json','container.log','base-distributions.json','distribution-resolution.json'):
 blob=(cpu/name).read_bytes(); target='capsule-cpu-'+name
 (job/target).write_bytes(blob); artifacts[target]=hashlib.sha256(blob).hexdigest()
pair_files={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in (
 'probes/glm53_ep_bindings_pair_check.py','probes/run_glm53_ep_bindings_pair_offline.py')}
(job/'capsule-cpu-binding.json').write_text(json.dumps(dict(cpu=str(cpu),sources=receipt['sources'],pair_sources=pair_files,manifest_sha256=manifest_sha,artifacts=artifacts,verified=time.time()),indent=2)+'\n')
(job/'freeze.json').write_text(json.dumps(dict(
    source=str(source),revision=revision,clean=True,cpu_tests=68,
    cpu_proof_source_match=True,cpu_evidence_sha256=proof_sha,
    mounted_source_count=13,contract_source_count=18,remap_compile_count=24,
    purpose='paired 13.3.1/13.0.3 post-context API diagnostic only; exact original lifecycle restore',
    created=time.time()),indent=2)+'\n')

# This inventory is evidence only. Another holder may be booting or serving
# requests now. Full lifecycle validation and idleness are enforced at GO.
sys.path.insert(0,str(source/'probes'))
import glm53_probe_lifecycle as life
inventory=dict(captured=time.time(),admission_gate=False)
try:
    inventory['nodes']=life.snapshot()
except Exception as exc:
    inventory['snapshot_error']=str(exc)
(job/'inventory-at-submission.json').write_text(json.dumps(inventory,indent=2)+'\n')
state=dict(path=str(scheduler),revision=scheduler_revision,
           approved_fixes=fixes,approved_main_verified=True,captured=time.time(),
           queue=read_optional(fleet/'queue'),holder=read_optional(fleet/'holder'),
           restore_debt=read_optional(fleet/'restore-debt.json'))
(job/'scheduler.json').write_text(json.dumps(state,indent=2)+'\n')
no_duplicate()
assert not git(source,'status','--porcelain')
assert not git(scheduler,'status','--porcelain') and git(scheduler,'rev-parse','HEAD')==scheduler_revision
command=['bash',str(scheduler/'bench/fleet.sh'),'run','--gpu',session,'8',
         'Fresh binding 13.3.1 vs capsule13.0.3; CPU2 import PASS, exact source/capsule and incoming restore',
         '--','python3','-B',str(source/'probes/run_glm53_ep_bindings_pair_offline.py'),
         '--revision',revision,'--out',str(job/'capture'),'--capsule-root',str(capsule),'--manifest-sha256',manifest_sha]
metadata=dict(source=str(source),scheduler=str(scheduler),scheduler_revision=scheduler_revision,
              revision=revision,approved_fixes=fixes,cpu_evidence_sha256=proof_sha,
              capsule_root=str(capsule),capsule_manifest_sha256=manifest_sha,capsule_cpu_artifacts=artifacts,pair_sources=pair_files,
              command=command,created=time.time(),diagnostic_only=True,
              performance_acceptance=False,full_gpu_acceptance=False,enqueue_while_other_holder_allowed=True,
              go_time_lifecycle_guards_unchanged=True,
              supersedes='v5 completed with34 API lookup errors and exact restoration; v6 is fresh same-image pinned binding A/B, baseline failure remains failure')
(job/'submission.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (job/'fleet.log').open('x') as log:
    process=subprocess.Popen(['python3','-B',str(job/'driver.py')],stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(job/'driver.pid').write_text(str(process.pid)+'\n')
print(json.dumps(dict(source=str(source),revision=revision,job=str(job),pid=process.pid,
                      session=session,scheduler_revision=scheduler_revision,
                      driver_bytes=len(driver_bytes),cpu_tests=68)))
'''


def main():
    proof = subprocess.check_output(
        ['git', '-C', str(LOCAL_REPO), 'show', REVISION + ':' + PROOF_PATH])
    parsed = json.loads(proof)
    assert parsed['contracts']['tests_run'] == 68
    assert parsed['cuda_initialized'] is False
    compile(DRIVER, 'remote-driver.py', 'exec')
    compile(REMOTE, 'remote-prepare.py', 'exec')
    arguments = [
        'python3', '-B', '-c', REMOTE, REVISION, PROOF_PATH, SESSION,
        json.dumps(APPROVED_FIXES), base64.b64encode(proof).decode(),
        base64.b64encode(DRIVER.encode()).decode(),
    ]
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'choiceoh@srv2', shlex.join(arguments)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True)
    log = Path('/tmp/glm53-ep-binding-gpu0908-6-launch.json')
    if result.returncode:
        Path('/tmp/glm53-ep-binding-gpu0908-6-preparation-error.txt').write_text(
            result.stdout + result.stderr)
        raise SystemExit('v6 preparation/submission failed; inspect preparation-error.txt; do not blindly rerun')
    log.write_text(result.stdout)
    print(result.stdout, end='')


if __name__ == '__main__':
    main()
