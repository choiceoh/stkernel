"""Create one bounded no-GPU readiness controller for an immutable CPU15 bundle."""
import base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
REV='30608530f0c908bc5f81db6bdb69fd70ccee9a24'
spec=importlib.util.spec_from_file_location('prepare','/tmp/glm53_prepare_cpu15.py');prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==REV
assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip()
bundle=prepare.BUNDLE
assert not bundle.exists()
subprocess.run(['git','-C',str(ROOT),'bundle','create',str(bundle),'HEAD','^'+prepare.BASE],check=True)
bundle_sha=hashlib.sha256(bundle.read_bytes()).hexdigest()
controller=r'''import base64,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
config=json.loads((root/'config.json').read_text())
assert hashlib.sha256(Path(config['bundle']).read_bytes()).hexdigest()==config['bundle_sha256']
env={k:v for k,v in os.environ.items() if not k.startswith('FLEET_')}
state=dict(revision=config['revision'],started=time.time(),deadline=time.time()+3600,phase='WAITING_MEMORY',submitted=False,attempts=0)
def save():
 temp=root/'state.next.json';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'state.json')
def call(mode):
 command=['python3','-B',str(root/'prepare-remote.py'),mode,config['revision'],config['driver_b64'],config['bundle_sha256']]
 result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
 with (root/'events.jsonl').open('a') as f:
  f.write(json.dumps(dict(captured=time.time(),mode=mode,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))+'\n')
 if result.returncode:raise RuntimeError('guarded preparation failed; inspect events.jsonl')
 return json.loads(result.stdout)
try:
 while time.time()<state['deadline']:
  state['attempts']+=1;info=call('inspect');state.update(checked=time.time(),last_inspection=info)
  if info.get('already_submitted'):
   state.update(phase='ALREADY_SUBMITTED',submitted=False);break
  if info.get('ready'):
   result=call('submit');state['submission']=result
   if result.get('submitted') or result.get('already_submitted'):
    state.update(phase='SUBMITTED' if result.get('submitted') else 'ALREADY_SUBMITTED',submitted=bool(result.get('submitted')));break
   if '12 GiB' not in result.get('reason',''):raise RuntimeError('unexpected non-admission; refusing retries')
  elif '12 GiB' not in info.get('reason',''):
   raise RuntimeError('unexpected readiness state; refusing retries')
  save();time.sleep(20)
 else:state['phase']='EXPIRED'
except BaseException as exc:
 state.update(phase='FAILED',error=repr(exc))
finally:
 state['ended']=time.time();save()
'''
compile(controller,'controller.py','exec')
remote=r'''import base64,hashlib,json,subprocess,sys,time
from pathlib import Path
revision,bsha,blob,remote_b64,driver_b64,controller_b64=sys.argv[1:]
root=Path('/tmp/glm53-cpu15-ready-0908');bundle=Path('/tmp/glm53-ep-local-0908-15.bundle')
assert not root.exists() and not bundle.exists() and not Path('/tmp/glm53-ep-local-compile0908-15-head').exists(), 'CPU15 state exists; do not duplicate'
raw=base64.b64decode(blob,validate=True);assert hashlib.sha256(raw).hexdigest()==bsha
root.mkdir();bundle.write_bytes(raw)
(root/'prepare-remote.py').write_bytes(base64.b64decode(remote_b64,validate=True))
(root/'controller.py').write_bytes(base64.b64decode(controller_b64,validate=True))
config=dict(revision=revision,bundle=str(bundle),bundle_sha256=bsha,driver_b64=driver_b64,created=time.time(),scope='bounded memory readiness only; normal --cpu submission once; no GPU queue or serving mutation')
(root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
with (root/'controller.log').open('x') as log:
 process=subprocess.Popen(['python3','-B',str(root/'controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'controller.pid').write_text(str(process.pid)+'\n')
print(json.dumps(dict(root=str(root),pid=process.pid,revision=revision,bundle_sha256=bsha,created=config['created'])))
'''
# Send bundle via stdin-safe base64 argument (small repository delta only).
# ssh command argument limits are avoided by an exclusive scp staging file.
stage=Path('/tmp/glm53-cpu15-ready-launch-payload.json')
payload=[REV,bundle_sha,base64.b64encode(bundle.read_bytes()).decode(),base64.b64encode(prepare.REMOTE.encode()).decode(),base64.b64encode(prepare.DRIVER.encode()).decode(),base64.b64encode(controller.encode()).decode()]
stage.write_text(json.dumps(payload))
subprocess.run(['scp','-q',str(stage),'choiceoh@srv2:/tmp/glm53-cpu15-ready-launch-payload.json'],check=True)
entry="import json,sys;sys.argv=['prepare']+json.load(open('/tmp/glm53-cpu15-ready-launch-payload.json'));exec("+repr(remote)+")"
r=subprocess.run(['ssh','-o','BatchMode=yes','choiceoh@srv2',shlex.join(['python3','-B','-c',entry])],capture_output=True,text=True)
Path('/tmp/glm53-cpu15-ready-launch.stdout').write_text(r.stdout);Path('/tmp/glm53-cpu15-ready-launch.stderr').write_text(r.stderr)
print(r.stdout,r.stderr);raise SystemExit(r.returncode)
