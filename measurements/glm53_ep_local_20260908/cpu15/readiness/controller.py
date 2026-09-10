import base64,hashlib,json,os,subprocess,time
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
