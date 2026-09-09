import hashlib,json,os,subprocess,sys,time
from pathlib import Path
SOURCE=Path('/home/choiceoh/stkernel-ep-local-diag-0908-1')
CPU_SOURCE=Path('/home/choiceoh/stkernel-ep-local-0908-17')
CPU_JOB=Path('/tmp/glm53-ep-local-compile0908-17-head')
JOB=Path('/tmp/glm53-ep-local-diag-gpu-0908-1')
SCHEDULER=Path('/home/choiceoh/stkernel-ep-local-scheduler-0909')
FLEET=Path('/home/choiceoh/glm53-logs/fleet')
CAPSULE=Path('/tmp/glm53-bindings-capsule-cpu0908-2/capsule')
MANIFEST='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
PROOF='measurements/glm53_ep_local_20260908/cpu17/local/result.json'
SESSION='eplocaldiag0908v1'
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
                    for row in queue.splitlines()),'diagnostic GPU v1 session already queued')
    require(not holder or holder.split('|')[0]!=SESSION,'diagnostic GPU v1 session already holds fleet')

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
    require(len(revision)==40 and all(c in '0123456789abcdef' for c in revision),'invalid CPU17 revision')
    expected_command=['bash',str(SCHEDULER/'bench/fleet.sh'),'run','--cpu','eplocalcpu0908v17head','6',
        'CPU17 bounded numerical failure diagnostics; unchanged kernel, capsule13.0.3 no devices 4g2CPU','--',
        'python3','-B',str(CPU_SOURCE/'probes/run_glm53_ep_local_cpu_compile.py'),
        '--image',IMAGE,'--arm','local','--output',str(CPU_JOB/'local'),
        '--capsule-root',str(CAPSULE),'--manifest-sha256',MANIFEST]
    require(submission.get('source')==str(CPU_SOURCE) and submission.get('scheduler')==str(SCHEDULER)
            and submission.get('command')==expected_command,'CPU17 did not use the expected normal no-device command')
    require(completed.get('complete') is True and completed.get('returncode')==0
            and 'error' not in completed,'actual CPU17 normal-fleet job did not complete successfully')
    for key,value in submission.items():
        require(completed.get(key)==value,'CPU17 completion differs from submission: '+key)
    require(git(CPU_SOURCE,'rev-parse','HEAD')==revision and not git(CPU_SOURCE,'status','--porcelain'),
            'actual CPU17 source is no longer the clean frozen revision')
    require((CPU_JOB/'local/result.json').read_bytes()==proof_bytes,
            'committed CPU17 receipt differs from actual successful job result')
    value=dict(revision=revision,submission_sha256=sha(submission_bytes),exit_sha256=sha(exit_bytes),
               result_sha256=sha(proof_bytes))
    if expected is not None:
        require(value==expected,'CPU17 source or receipt identity changed since preparation')
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
    require(summary==expected_summary,'validated CPU17 proof differs from committed receipt summary')
    require(type(summary['tests_run']) is int and summary['tests_run']==146,'invalid CPU17 test count')
    require(summary['mounted_source_count']==13 and summary['contract_source_count']==29
            and summary['remap_compile_count']==len(compile_cases())==24,'incomplete current CPU17 compile coverage')
    require(validate_capsule_input(CAPSULE,MANIFEST)==CAPSULE,'pinned capsule path identity changed')
    return proof

def frozen(revision):
    require(SOURCE.is_dir(),'diagnostic GPU v1 source is missing')
    require(git(SOURCE,'rev-parse','HEAD')==revision and not git(SOURCE,'status','--porcelain'),
            'diagnostic GPU v1 source revision or cleanliness differs')
    require(not (SOURCE/'.git/objects/info/alternates').exists(),
            'diagnostic source must use an independent object store')

def receipt_parent(revision,cpu_revision):
    require(git(SOURCE,'rev-list','--parents','-n','1',revision).split()==[revision,cpu_revision],
            'depth-2 diagnostic source requires the receipt commit to be the immediate single child of CPU17 source')

def diagnostic_command(revision):
    return ['bash',str(SCHEDULER/'bench/fleet.sh'),'run','--gpu',SESSION,'8',
            'CPU17-matched concentrated6912 failure diagnostics only; unchanged fixture and exact incoming restore','--',
            'python3','-B',str(SOURCE/'probes/run_glm53_ep_local_offline.py'),
            '--revision',revision,'--out',str(JOB/'capture'),
            '--capsule-root',str(CAPSULE),'--manifest-sha256',MANIFEST,
            '--diagnose-case','concentrated6912']

data=json.loads((JOB/'submission.json').read_text())
data.update(started=time.time(),exit_code=1,complete=False)
try:
    require(Path(__file__).resolve()==JOB/'driver.py','unexpected detached driver path')
    require(sha((JOB/'driver.py').read_bytes())==data['driver_sha256'],'driver bytes changed')
    scheduler_state(data['scheduler_state'])
    no_duplicate()
    frozen(data['revision'])
    proof_bytes=(JOB/'cpu-evidence.json').read_bytes()
    require(sha(proof_bytes)==data['cpu_evidence_sha256'],'copied CPU17 receipt changed')
    require((SOURCE/PROOF).read_bytes()==proof_bytes,'frozen receipt changed')
    cpu_state(proof_bytes,data['cpu_state'])
    receipt_parent(data['revision'],data['cpu_state']['revision'])
    proof_and_capsule(SOURCE,SOURCE/PROOF,data['cpu_summary'])
    require(data['command']==diagnostic_command(data['revision']),'normal single-case diagnostic command changed')
    require(data.get('diagnostic_only') is True and data.get('mode')=='diagnostic'
            and data.get('diagnose_case')=='concentrated6912'
            and data.get('performance_acceptance') is False and data.get('full_gpu_acceptance') is False
            and data.get('default_promotion') is False,'diagnostic scope metadata changed')
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
