import os,subprocess,time,json
from pathlib import Path
job=Path(__file__).resolve().parent
data=json.loads((job/'submission.json').read_text())
data['started']=time.time()
p=subprocess.run(data['command'],stdin=subprocess.DEVNULL,env=dict(os.environ,REPO=data['scheduler']))
data.update(ended=time.time(),exit_code=p.returncode)
(job/'exit.json').write_text(json.dumps(data,indent=2)+'\n')
