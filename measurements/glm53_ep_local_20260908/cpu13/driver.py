import json,os,subprocess,time
from pathlib import Path
job=Path(__file__).resolve().parent
data=json.loads((job/'submission.json').read_text())
data['started']=time.time()
(job/'started.json').write_text(json.dumps(dict(pid=os.getpid(),started=data['started']))+'\n')
try:
    result=subprocess.run(data['command'],stdin=subprocess.DEVNULL,env=dict(os.environ,REPO=data['scheduler']))
    data.update(returncode=result.returncode,complete=result.returncode==0)
except BaseException as exc:
    data.update(returncode=1,complete=False,error=type(exc).__name__+': '+str(exc)[:1000])
finally:
    data['ended']=time.time()
    (job/'exit.json').write_text(json.dumps(data,indent=2)+'\n')
