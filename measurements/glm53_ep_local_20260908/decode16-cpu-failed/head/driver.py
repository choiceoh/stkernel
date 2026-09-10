import json,os,subprocess,time
from pathlib import Path
job=Path(__file__).resolve().parent; data=json.loads((job/'submission.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith('FLEET_')};env['REPO']=data['source']
started=time.time();p=subprocess.run(data['command'],cwd=data['source'],env=env,stdin=subprocess.DEVNULL)
copy_rc=None
if p.returncode==0:
 copy_rc=subprocess.run(['rsync','-a','--',data['worker']+':'+data['worker_root']+'/evidence/',str(job/'evidence')+'/']).returncode
rc=p.returncode if p.returncode else copy_rc
(job/'exit.json').write_text(json.dumps(dict(returncode=rc,payload_returncode=p.returncode,copy_returncode=copy_rc,started=started,ended=time.time()))+'\n')
