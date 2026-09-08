import json,os,subprocess,time
from pathlib import Path
source='/home/choiceoh/stkernel-prefill-observation-memory-0908-1'
revision='37db5eb1fb17b1ed03d5116f02e317fc64ffee4c'
job=Path('/tmp/glm53-prefill-observation-memory-0908-1')
session='glm53observemem0908v1'
env=dict(os.environ,REPO=source,FLEET_OBSERVATION_CLONES='1')
command=['bash',source+'/bench/fleet.sh','run','--gpu',session,'65',
 'Diagnose/reclaim unused host RAM; unchanged 12GiB guard before default TTFT/quality/profile/routes; no promotion','--',
 'python3',source+'/bench/prefill_observation_run.py','run','--reclaim-host-memory','--revision',revision,'--out',str(job/'capture')]
result=dict(session=session,source=source,revision=revision,command=command,environment={'REPO':source,'FLEET_OBSERVATION_CLONES':'1'},started=time.time())
(job/'submitted.json').write_text(json.dumps(result,indent=2)+'\n')
with (job/'fleet.log').open('x') as log:result['exit_code']=subprocess.call(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
result['ended']=time.time();(job/'exit.json').write_text(json.dumps(result,indent=2)+'\n')
