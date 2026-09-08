import json,os,subprocess,time
from pathlib import Path
source='/home/choiceoh/stkernel-cpu-vote-memory-0908-1'
revision='6e693f53a118bce9fc0a8b457943c3880a9fb59b'
job=Path('/tmp/glm53-cpu-vote-memory-0908-1')
session='glm53cpuvotemem0908v1'
env=dict(os.environ,REPO=source,FLEET_OBSERVATION_CLONES='1')
command=['bash',source+'/bench/fleet.sh','run','--gpu',session,'65',
 'PR470 warm-cache GPU/CPU/GPU readiness-vote memory bracket; no model requests, no promotion','--',
 'python3',source+'/bench/glm53_cpu_vote_memory_pair.py','--revision',revision,'--out',str(job/'capture')]
result=dict(session=session,source=source,revision=revision,command=command,environment={'REPO':source,'FLEET_OBSERVATION_CLONES':'1'},started=time.time())
(job/'submitted.json').write_text(json.dumps(result,indent=2)+'\n')
with (job/'fleet.log').open('x') as log:result['exit_code']=subprocess.call(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
result['ended']=time.time();(job/'exit.json').write_text(json.dumps(result,indent=2)+'\n')
