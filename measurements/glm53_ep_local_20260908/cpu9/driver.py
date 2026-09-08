import json,subprocess,os,time
from pathlib import Path
job=Path(__file__).resolve().parent
data=json.loads((job/'submission.json').read_text())
started=time.time()
r=subprocess.run(data['commands'][0],env=dict(os.environ,REPO=data['source']))
(job/'exit.json').write_text(json.dumps(dict(complete=r.returncode==0,results=[dict(command=data['commands'][0],started=started,ended=time.time(),returncode=r.returncode)]),indent=2)+'\n')
