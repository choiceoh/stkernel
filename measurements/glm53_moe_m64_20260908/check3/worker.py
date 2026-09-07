import datetime,json,os,pathlib,subprocess,sys
job=pathlib.Path(__file__).resolve().parent
q=json.loads((job/'request.json').read_text());source=pathlib.Path(q['source'])
sys.path.insert(0,str(source/'probes'));import glm53_offline_checks as m
result=dict(started=datetime.datetime.now().isoformat(),exit_code=3)
try:
    if (job/'submitted.json').exists():raise RuntimeError('already submitted')
    m.pinned(str(source),q['revision'])
    subprocess.run(['git','-C',str(source),'fetch','--quiet','origin','main'],check=True)
    subprocess.run(['git','-C',str(source),'merge-base','--is-ancestor','origin/main',q['revision']],check=True)
    preflight=m.probe_api_preflight(source,q['revision'])
    (job/'submitted.json').write_text(json.dumps(dict(submitted=datetime.datetime.now().isoformat(),revision=q['revision'],api_preflight=preflight),indent=2)+'\n')
    result['exit_code']=subprocess.call(q['command'],cwd=source,env=dict(os.environ,REPO=str(source),OFFLINE_SOURCE_REV=q['revision']))
except Exception as exc:result['error']=repr(exc)
finally:
    result['ended']=datetime.datetime.now().isoformat()
    (job/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
raise SystemExit(result['exit_code'])
