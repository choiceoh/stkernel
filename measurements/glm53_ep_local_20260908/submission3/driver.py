import json,os,subprocess,time
from pathlib import Path
job=Path('/tmp/glm53-ep-local-gpu-0908-3')
data=json.loads((job/'submission.json').read_text())
started=time.time()
result=subprocess.run(data['command'],env=dict(os.environ,REPO=data['source']))
data.update(started=started,ended=time.time(),exit_code=result.returncode)
(job/'exit.json').write_text(json.dumps(data,indent=2)+'\n')
