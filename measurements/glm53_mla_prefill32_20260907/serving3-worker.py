import datetime,json,os,pathlib,subprocess,sys
job=pathlib.Path(__file__).resolve().parent
request=json.loads((job/'request.json').read_text())
source=pathlib.Path(request['source'])
sys.path.insert(0,str(source/'bench'))
import prefill_serving
result=dict(started=datetime.datetime.now().isoformat(),exit_code=3)
try:
    if (job/'submitted.json').exists():raise RuntimeError('already submitted; inspect existing job')
    prefill_serving.pinned(str(source),request['revision'])
    assert request['candidate']=='mla' and '--refresh-gate' in request['command']
    subprocess.run(['git','-C',str(source),'fetch','--quiet','origin','main'],check=True)
    subprocess.run(['git','-C',str(source),'merge-base','--is-ancestor','origin/main',request['revision']],check=True)
    (job/'submitted.json').write_text(json.dumps(dict(submitted=datetime.datetime.now().isoformat(),revision=request['revision']))+'\n')
    result['exit_code']=subprocess.call(request['command'],cwd=source,env=dict(os.environ,REPO=str(source)))
except Exception as exc:
    result['error']=repr(exc)
finally:
    result['ended']=datetime.datetime.now().isoformat()
    (job/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
raise SystemExit(result['exit_code'])
