#!/usr/bin/env python3
"""Freeze and normally enqueue CPU16-matched full MoE GPU v5 when explicitly run.

Prepared for parent review only. Importing or parsing this file performs no
remote action. --revision must name the clean committed source plus actual
CPU16 receipt. No CPU/GPU test runs here; the normal fleet wrapper owns GO-time
admission, per-cell isolation, sanitizer checks and exact service restoration.
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
import uuid

ROOT = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
PROOF = 'measurements/glm53_ep_local_20260908/cpu16/local/result.json'
BUNDLE = Path('/tmp/glm53-ep-local-gpu-0908-5.bundle')
REPORT = Path('/tmp/glm53-ep-local-gpu-0908-5b-launch.json')
HOST = 'choiceoh@srv2'

# Shared read-only validation is repeated by the detached driver immediately
# before entering fleet. It does not inspect serving idleness or pause a node.
GUARDS = r'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
SOURCE=Path('/home/choiceoh/stkernel-ep-local-gpu-0908-5b')
CPU_SOURCE=Path('/home/choiceoh/stkernel-ep-local-0908-16')
CPU_JOB=Path('/tmp/glm53-ep-local-compile0908-16-head')
JOB=Path('/tmp/glm53-ep-local-gpu-0908-5')
SCHEDULER=Path('/home/choiceoh/stkernel')
FLEET=Path('/home/choiceoh/glm53-logs/fleet')
CAPSULE=Path('/tmp/glm53-bindings-capsule-cpu0908-2/capsule')
MANIFEST='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
PROOF='measurements/glm53_ep_local_20260908/cpu16/local/result.json'
SESSION='eplocal0908v5'
FIXES=('d95a2cdb46f2446bd4b58f7d5969b42c895f0b3d','4a2c960ad632596ca3224ead88da43160012d393')
sys.dont_write_bytecode=True

def require(value,message):
    if not value:
        raise RuntimeError(message)

def run(command,label):
    result=subprocess.run(command,text=True,capture_output=True,stdin=subprocess.DEVNULL)
    require(result.returncode==0,label+' failed: '+(result.stderr or result.stdout)[-1500:])
    return result.stdout.strip()

def git(root,*args):
    return run(['git','-C',str(root),*args],'git '+args[0])

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def read_optional(path):
    return path.read_text() if path.exists() else None

def no_duplicate():
    queue=read_optional(FLEET/'queue') or ''
    holder=read_optional(FLEET/'holder') or ''
    require(not any(len(row.split('|'))>1 and row.split('|')[1]==SESSION
                    for row in queue.splitlines()),'full GPU v5 session already queued')
    require(not holder or holder.split('|')[0]!=SESSION,'full GPU v5 session already holds fleet')

def scheduler_state(expected=None):
    require(not git(SCHEDULER,'status','--porcelain'),'normal scheduler must be clean')
    revision=git(SCHEDULER,'rev-parse','HEAD')
    approved=git(SCHEDULER,'rev-parse','origin/main')
    advertised=git(SCHEDULER,'ls-remote','origin','refs/heads/main').splitlines()
    require(len(advertised)==1 and advertised[0].split()[0]==approved,
            'origin/main is not the current approved remote main; no scheduler mutation attempted')
    git(SCHEDULER,'merge-base','--is-ancestor',revision,approved)
    git(SCHEDULER,'diff','--quiet',revision,approved,'--','bench','launchers','profiles','tests')
    for fix in FIXES:
        git(SCHEDULER,'merge-base','--is-ancestor',fix,revision)
    value=dict(revision=revision,approved_main=approved,approved_fixes=list(FIXES),
               approved_identical_directories=['bench','launchers','profiles','tests'])
    if expected is not None:
        require(value==expected,'approved scheduler changed since preparation')
    return value

def cpu_state(proof_bytes,expected=None):
    submission_bytes=(CPU_JOB/'submission.json').read_bytes()
    exit_bytes=(CPU_JOB/'exit.json').read_bytes()
    submission=json.loads(submission_bytes)
    completed=json.loads(exit_bytes)
    revision=submission['revision']
    require(len(revision)==40 and all(c in '0123456789abcdef' for c in revision),'invalid CPU16 revision')
    expected_command=['bash',str(SCHEDULER/'bench/fleet.sh'),'run','--cpu','eplocalcpu0908v16head','6',
        'CPU16 cached SFA row bases and unsigned prefix; capsule13.0.3 no devices 4g2CPU','--',
        'python3','-B',str(CPU_SOURCE/'probes/run_glm53_ep_local_cpu_compile.py'),
        '--image',IMAGE,'--arm','local','--output',str(CPU_JOB/'local'),
        '--capsule-root',str(CAPSULE),'--manifest-sha256',MANIFEST]
    require(submission.get('source')==str(CPU_SOURCE) and submission.get('scheduler')==str(SCHEDULER)
            and submission.get('command')==expected_command,'CPU16 did not use the expected normal no-device command')
    require(completed.get('complete') is True and completed.get('returncode')==0
            and 'error' not in completed,'actual CPU16 normal-fleet job did not complete successfully')
    for key,value in submission.items():
        require(completed.get(key)==value,'CPU16 completion differs from submission: '+key)
    require(git(CPU_SOURCE,'rev-parse','HEAD')==revision and not git(CPU_SOURCE,'status','--porcelain'),
            'actual CPU16 source is no longer the clean frozen revision')
    require((CPU_JOB/'local/result.json').read_bytes()==proof_bytes,
            'committed CPU16 receipt differs from actual successful job result')
    value=dict(revision=revision,submission_sha256=sha(submission_bytes),exit_sha256=sha(exit_bytes),
               result_sha256=sha(proof_bytes))
    if expected is not None:
        require(value==expected,'CPU16 source or receipt identity changed since preparation')
    return value

def proof_and_capsule(root,proof_path,expected_summary):
    sys.path.insert(0,str(root/'probes'))
    from glm53_ep_local_evidence import validate_compile_evidence,CONTRACT_PATHS,compile_cases
    from glm53_ep_capsule_runtime import validate_capsule_input,validate_runtime_receipt,CAPSULE_SHA256
    require(CAPSULE_SHA256==MANIFEST,'source runtime expects a different capsule')
    proof=validate_compile_evidence(root,proof_path)
    require(set(proof['contracts']['files'])==set(CONTRACT_PATHS),'CPU contract file set differs')
    summary=dict(tests_run=proof['contracts']['tests_run'],mounted_source_count=len(proof['mounted_sources']),
                 contract_source_count=len(proof['contracts']['files']),remap_compile_count=len(proof['remap_compilation']),
                 binding_runtime=validate_runtime_receipt(proof['binding_runtime']))
    require(summary==expected_summary,'validated CPU16 proof differs from committed receipt summary')
    require(type(summary['tests_run']) is int and summary['tests_run']>0,'invalid CPU16 test count')
    require(summary['mounted_source_count']==13 and summary['contract_source_count']==27
            and summary['remap_compile_count']==len(compile_cases())==24,'incomplete current CPU16 compile coverage')
    require(validate_capsule_input(CAPSULE,MANIFEST)==CAPSULE,'pinned capsule path identity changed')
    return proof

def frozen(revision):
    require(SOURCE.is_dir(),'full GPU v5 source is missing')
    require(git(SOURCE,'rev-parse','HEAD')==revision and not git(SOURCE,'status','--porcelain'),
            'full GPU v5 source revision or cleanliness differs')

def full_command(revision):
    return ['bash',str(SCHEDULER/'bench/fleet.sh'),'run','--gpu',SESSION,'45',
            'CPU16-matched full MoE remap/numerics/streams/sanitizers; capsule13.0.3 and exact incoming restore','--',
            'python3','-B',str(SOURCE/'probes/run_glm53_ep_local_offline.py'),
            '--revision',revision,'--out',str(JOB/'capture'),
            '--capsule-root',str(CAPSULE),'--manifest-sha256',MANIFEST]
'''

DRIVER = GUARDS + r'''
data=json.loads((JOB/'submission.json').read_text())
data.update(started=time.time(),exit_code=1,complete=False)
try:
    require(Path(__file__).resolve()==JOB/'driver.py','unexpected detached driver path')
    require(sha((JOB/'driver.py').read_bytes())==data['driver_sha256'],'driver bytes changed')
    scheduler_state(data['scheduler_state'])
    no_duplicate()
    frozen(data['revision'])
    proof_bytes=(JOB/'cpu-evidence.json').read_bytes()
    require(sha(proof_bytes)==data['cpu_evidence_sha256'],'copied CPU16 receipt changed')
    require((SOURCE/PROOF).read_bytes()==proof_bytes,'frozen receipt changed')
    cpu_state(proof_bytes,data['cpu_state'])
    proof_and_capsule(SOURCE,SOURCE/PROOF,data['cpu_summary'])
    require(data['command']==full_command(data['revision']),'normal full-suite command changed')
    removed=sorted(k for k in os.environ if k.startswith('FLEET_'))
    environment={k:v for k,v in os.environ.items() if not k.startswith('FLEET_')}
    environment['REPO']=str(SCHEDULER)
    environment['PYTHONDONTWRITEBYTECODE']='1'
    data['removed_inherited_fleet_environment_names']=removed
    with (JOB/'driver-start.json').open('x') as handle:
        handle.write(json.dumps(data,indent=2)+'\n')
    process=subprocess.run(data['command'],stdin=subprocess.DEVNULL,env=environment)
    data['exit_code']=process.returncode
    data['complete']=process.returncode==0
except BaseException as exc:
    data['driver_error']=type(exc).__name__+': '+str(exc)[:1800]
finally:
    data['ended']=time.time()
    with (JOB/'exit.json').open('x') as handle:
        handle.write(json.dumps(data,indent=2)+'\n')
'''

REMOTE = GUARDS + r'''
import base64,fcntl
mode,payload64=sys.argv[1:]
payload=json.loads(base64.b64decode(payload64,validate=True))
revision=payload['revision']
proof_bytes=base64.b64decode(payload['proof_b64'],validate=True)
bundle=Path('/tmp/glm53-ep-local-gpu-0908-5.bundle')

def prepare():
    require(not JOB.exists(),'full GPU v5 job path already exists; never resubmit or overwrite')
    no_duplicate()
    approved=scheduler_state()
    cpu=cpu_state(proof_bytes)
    proof_and_capsule(CPU_SOURCE,CPU_JOB/'local/result.json',payload['cpu_summary'])
    if SOURCE.exists():
        frozen(revision)
    info=dict(ready=True,submitted=False,source_exists=SOURCE.exists(),cpu_state=cpu,
              scheduler_state=approved,bundle_sha256=sha(bundle.read_bytes()) if bundle.exists() else None)
    if mode=='inspect':
        return info
    require(mode=='submit','unsupported preparation mode')
    require(cpu==payload['cpu_state'] and approved==payload['scheduler_state'],
            'CPU16 or approved scheduler changed after initial inspection')
    require(bundle.is_file() and sha(bundle.read_bytes())==payload['bundle_sha256'],'immutable bundle missing or changed')
    require(git(CPU_SOURCE,'bundle','list-heads',str(bundle)).splitlines()==[revision+' HEAD'],
            'bundle does not contain the requested exact HEAD')
    if not SOURCE.exists():
        # A fresh object store avoids the exhausted CPU-source alternates chain.
        # The explicit full revision is d53fd44f2b8a69869cddc9df10fa4cbb21bc56d9
        # for this preparation; depth 2 also fetches its CPU16 parent. The
        # immutable local bundle remains independently validated above.
        run(['git','init','--quiet',str(SOURCE)],'full GPU v5b fresh init')
        git(SOURCE,'remote','add','origin','https://github.com/choiceoh/stkernel.git')
        git(SOURCE,'fetch','--quiet','--depth=2','origin',revision)
        git(SOURCE,'checkout','--quiet','--detach','FETCH_HEAD')
    frozen(revision)
    git(SOURCE,'merge-base','--is-ancestor',cpu['revision'],revision)
    require((SOURCE/PROOF).read_bytes()==proof_bytes,'bundle CPU16 receipt differs')
    proof_and_capsule(SOURCE,SOURCE/PROOF,payload['cpu_summary'])
    # The production runner must itself consume CPU16, not a stale evidence path.
    import run_glm53_ep_local_offline as offline
    require(offline.CPU_EVIDENCE==Path(PROOF),'full runner does not select CPU16 evidence')
    require(Path(offline.__file__).resolve()==SOURCE/'probes/run_glm53_ep_local_offline.py',
            'runner import did not come from frozen GPU source')
    driver_bytes=base64.b64decode(payload['driver_b64'],validate=True)
    compile(driver_bytes,'full-gpu-v5-driver.py','exec')
    no_duplicate()
    scheduler_state(approved)
    cpu_state(proof_bytes,cpu)
    frozen(revision)
    proof_and_capsule(SOURCE,SOURCE/PROOF,payload['cpu_summary'])
    # Exclusive mkdir plus the preparation lock prevents duplicate local drivers.
    JOB.mkdir()
    (JOB/'driver.py').write_bytes(driver_bytes)
    (JOB/'cpu-evidence.json').write_bytes(proof_bytes)
    metadata=dict(source=str(SOURCE),revision=revision,source_clean=True,scheduler=str(SCHEDULER),
        scheduler_state=approved,cpu_state=cpu,cpu_evidence_sha256=sha(proof_bytes),
        cpu_summary=payload['cpu_summary'],capsule_root=str(CAPSULE),capsule_manifest_sha256=MANIFEST,
        bundle_sha256=payload['bundle_sha256'],driver_sha256=sha(driver_bytes),
        command=full_command(revision),created=time.time(),session=SESSION,
        performance_acceptance=False,full_gpu_acceptance=False,
        enqueue_while_other_holder_allowed=True,go_time_lifecycle_guards_unchanged=True,
        queue_before=read_optional(FLEET/'queue'),holder_before=read_optional(FLEET/'holder'),
        restore_debt_before=read_optional(FLEET/'restore-debt.json'))
    (JOB/'submission.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (JOB/'fleet.log').open('x') as log:
        process=subprocess.Popen(['python3','-B',str(JOB/'driver.py')],stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (JOB/'driver.pid').write_text(str(process.pid)+'\n')
    return dict(submitted=True,job=str(JOB),source=str(SOURCE),revision=revision,session=SESSION,
                pid=process.pid,cpu_tests=payload['cpu_summary']['tests_run'],scheduler_state=approved)

try:
    if mode=='submit':
        with Path(str(JOB)+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            result=prepare()
    else:
        result=prepare()
    print(json.dumps(result))
except BaseException as exc:
    print(json.dumps(dict(submitted=False,error=type(exc).__name__+': '+str(exc)[:2000])),file=sys.stderr)
    raise SystemExit(1)
'''


def run(command, label, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL, **kwargs)
    if result.returncode:
        raise RuntimeError(label + ' failed: ' + (result.stderr or result.stdout)[-2200:])
    return result.stdout.strip()


def remote(mode, payload):
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    command = ['python3', '-B', '-c', REMOTE, mode, encoded]
    return json.loads(run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST,
                          shlex.join(command)], 'full GPU v5 remote ' + mode))


def transfer_bundle(digest):
    # Upload to a fresh name, then publish by exclusive hard link. Existing
    # immutable bundle bytes are only compared, never replaced by scp.
    incoming = str(BUNDLE) + '.upload-' + uuid.uuid4().hex
    run(['scp', '-q', str(BUNDLE), HOST + ':' + incoming], 'full GPU v5 bundle upload')
    publish = r'''import hashlib,os,sys
from pathlib import Path
incoming,target,digest=sys.argv[1:]
incoming,target=Path(incoming),Path(target)
if hashlib.sha256(incoming.read_bytes()).hexdigest()!=digest:
    raise RuntimeError('uploaded immutable bundle hash mismatch')
try:
    os.link(incoming,target)
except FileExistsError:
    if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('existing immutable bundle differs; no replacement')
incoming.unlink()
'''
    run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST,
         shlex.join(['python3', '-B', '-c', publish, incoming, str(BUNDLE), digest])], 'exclusive bundle publish')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True, help='Exact clean committed HEAD including actual CPU16 PASS receipt')
    args = parser.parse_args()
    if re.fullmatch('[0-9a-f]{40}', args.revision) is None:
        parser.error('--revision must be a full commit SHA')
    if REPORT.exists():
        raise RuntimeError('launch report already exists; inspect it instead of resubmitting')
    if run(['git', 'rev-parse', 'HEAD'], 'local HEAD', cwd=ROOT) != args.revision:
        raise RuntimeError('requested revision does not equal current local HEAD')
    if run(['git', 'status', '--porcelain'], 'local cleanliness', cwd=ROOT):
        raise RuntimeError('local source/evidence must be clean and committed')
    committed = subprocess.check_output(['git', '-C', str(ROOT), 'show', args.revision + ':' + PROOF])
    if (ROOT / PROOF).read_bytes() != committed:
        raise RuntimeError('current CPU16 receipt differs from committed HEAD')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'probes'))
    from glm53_ep_local_evidence import validate_compile_evidence, CONTRACT_PATHS, compile_cases
    from glm53_ep_capsule_runtime import validate_runtime_receipt
    proof = validate_compile_evidence(ROOT, ROOT / PROOF)
    if set(proof['contracts']['files']) != set(CONTRACT_PATHS):
        raise RuntimeError('CPU16 contract file set differs from current source')
    summary = dict(tests_run=proof['contracts']['tests_run'], mounted_source_count=len(proof['mounted_sources']),
                   contract_source_count=len(proof['contracts']['files']), remap_compile_count=len(proof['remap_compilation']),
                   binding_runtime=validate_runtime_receipt(proof['binding_runtime']))
    if (type(summary['tests_run']) is not int or summary['tests_run'] <= 0
            or summary['mounted_source_count'] != 13 or summary['contract_source_count'] != 27
            or summary['remap_compile_count'] != len(compile_cases()) or len(compile_cases()) != 24):
        raise RuntimeError('incomplete current CPU16 proof coverage')
    payload = dict(revision=args.revision, proof_b64=base64.b64encode(committed).decode(), cpu_summary=summary)
    inspected = remote('inspect', payload)
    base = inspected['cpu_state']['revision']
    run(['git', 'merge-base', '--is-ancestor', base, args.revision], 'CPU16 source ancestry', cwd=ROOT)
    if not BUNDLE.exists():
        run(['git', 'bundle', 'create', str(BUNDLE), 'HEAD', '^' + base], 'immutable GPU v5 bundle create', cwd=ROOT)
    if run(['git', 'bundle', 'list-heads', str(BUNDLE)], 'bundle identity', cwd=ROOT).splitlines() != [args.revision + ' HEAD']:
        raise RuntimeError('existing GPU v5 bundle differs; no replacement')
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    if inspected['bundle_sha256'] is None:
        transfer_bundle(digest)
    elif inspected['bundle_sha256'] != digest:
        raise RuntimeError('existing remote GPU v5 bundle differs; no replacement')
    payload.update(bundle_sha256=digest, cpu_state=inspected['cpu_state'], scheduler_state=inspected['scheduler_state'],
                   driver_b64=base64.b64encode(DRIVER.encode()).decode())
    result = remote('submit', payload)
    with REPORT.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        ast.parse(GUARDS)
        ast.parse(DRIVER)
        ast.parse(REMOTE)
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(type(exc).__name__ + ': ' + str(exc)[:2200], file=sys.stderr)
        raise SystemExit(1)
