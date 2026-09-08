import json,os,subprocess,time
from pathlib import Path
source='/home/choiceoh/stkernel-prefill-observation-0908-1'
revision='75686447d5cca72904b7050e333f8c319884a28d'
job=Path('/tmp/glm53-prefill-observation-0908-1')
session='glm53observe0908v1'
env=dict(os.environ,REPO=source,FLEET_OBSERVATION_CLONES='1')
command=['bash',source+'/bench/fleet.sh','run','--gpu',session,'65',
 'Default prefill TTFT/quality plus separate all-rank profile and model routes; no candidate promotion','--',
 'python3',source+'/bench/prefill_observation_run.py','run','--revision',revision,'--out',str(job/'capture')]
result=dict(session=session,source=source,revision=revision,command=command,started=time.time())
(job/'submitted.json').write_text(json.dumps(result,indent=2)+'\n')
with (job/'fleet.log').open('x') as log:result['exit_code']=subprocess.call(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
result['ended']=time.time();(job/'exit.json').write_text(json.dumps(result,indent=2)+'\n')
