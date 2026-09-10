import json,os,subprocess,time
from pathlib import Path
job=Path('/tmp/glm53-ep-local-compile0908-9')
data=json.loads((job/'submission.json').read_text())
env=dict(os.environ,REPO=data['source'])
results=[]
for command in data['commands']:
 start=time.time()
 result=subprocess.run(command,env=env)
 results.append(dict(command=command,started=start,ended=time.time(),returncode=result.returncode))
 (job/'exit.json').write_text(json.dumps(dict(results=results,complete=len(results)==len(data['commands']) and all(r['returncode']==0 for r in results)),indent=2)+'\n')
 if result.returncode:break
