import os,json,subprocess,time
from pathlib import Path
job=Path(__file__).resolve().parent
data=json.loads((job/'submission.json').read_text())
data['started']=time.time()
p=subprocess.run(data['command'],stdin=subprocess.DEVNULL,env=dict(os.environ,REPO=data['scheduler']))
data.update(ended=time.time(),returncode=p.returncode,complete=p.returncode==0)
(job/'exit.json').write_text(json.dumps(data,indent=2)+'\n')
