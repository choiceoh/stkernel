"""Read-only terminal collection. This script does not run GPU workloads."""
import base64, datetime, gzip, hashlib, json, os, pathlib, runpy, subprocess
P=pathlib.Path
ROOT=P('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass6-completed'
REV='e2a54cff881465c2bb7dbbd3f5ec39ca240c7f74'
SESSION='eplocalonepass0909v6'; TICKET='17889071393096545'
NODES=('local','10.10.10.1','10.10.10.3','10.10.10.4')
sha=lambda b:hashlib.sha256(b).hexdigest()
provenance={}
def save(name,data,origin,**extra):
 p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_bytes()==data,name
 else: p.write_bytes(data)
 provenance[name]={'origin':origin,'bytes':len(data),'sha256':sha(data),**extra}
def pack(name,data,origin):
 save(name,gzip.compress(data,mtime=0),origin,original_bytes=len(data),original_sha256=sha(data))
def js(name,obj,origin): save(name,(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode(),origin)
code=r'''
import pathlib,subprocess,json,os,hashlib,base64,time
P=pathlib.Path; root=P('/home/choiceoh/stkernel-ep-onepass-0909-6');job=P('/tmp/glm53-ep-onepass-0909-6')
def identity():
 def git(*a):return subprocess.check_output(['git','-C',str(root),*a],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).decode().strip()
 return dict(head=git('rev-parse','HEAD'),status=git('status','--porcelain'),shallow=git('rev-parse','--is-shallow-repository'))
def show():
 p=subprocess.run(['bash',str(root/'bench/fleet.sh'),'show','eplocalonepass0909v6','--ticket','17889071393096545','--json'],capture_output=True,text=True);assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
r={'captured_at':time.time(),'source_before':identity(),'fleet_before':show(),'files':{},'absent':[]}
assert r['fleet_before']['state'] not in ('queued','running'),r['fleet_before']['state']
paths=[job/'onepass.jsonl',job/'verdicts.jsonl',P('/home/choiceoh/glm53-logs/fleet/run-logs/594ee7cb485f192ff57f2480a7e7ac7b172048ffe26acbae80d18e9c96403760.log')]
paths += [P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS6'+a+'.log') for a in ('B1','A','B2')]
for p in paths:
 if not p.exists():r['absent'].append(str(p));continue
 a=p.stat();raw=p.read_bytes();b=p.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns) and len(raw)<128*2**20
 r['files'][str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()}
r.update(source_after=identity(),fleet_after=show());print(json.dumps(r))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','choiceoh@srv2','python3 -B -'],input=code.encode(),capture_output=True);assert p.returncode==0,p.stderr.decode()
r=json.loads(p.stdout)
assert r['source_before']==r['source_after']==dict(head=REV,status='',shallow='false')
assert r['fleet_after']['state']==r['fleet_before']['state']
for source,item in r.pop('files').items():
 raw=base64.b64decode(item['data']);assert sha(raw)==item['sha256'];name=P(source).name
 if name=='onepass.jsonl':
  rows=[json.loads(l) for l in raw.splitlines()];assert [x['name'] for x in rows]==['EPONEPASS6'+a for a in ('B1','A','B2')][:len(rows)]
  assert raw.startswith(P('/tmp/glm53-onepass6-through-B1.jsonl').read_bytes())
  save('records/onepass.jsonl',raw,source)
 elif name=='verdicts.jsonl':save('records/verdicts.jsonl',raw,source)
 elif name.startswith('boot-'):pack('boot/'+name+'.gz',raw,source)
 else:pack('fleet/terminal-run.log.gz',raw,source)
js('terminal-capture.json',r,'read-only terminal fleet/source checks')
def gitfile(name):return subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT)
manifest=gitfile('build/glm53/manifest.tsv');expected=b'# source_commit='+REV.encode()+b'\n'+manifest
mounts={line.split('\t')[1]:sha(gitfile('build/glm53/'+line.split('\t')[0])) for line in manifest.decode().splitlines() if line and not line.startswith('#')}
save('source/frozen-manifest.tsv',manifest,'git '+REV)
js('source/mounted-hashes.json',mounts,'git '+REV)
for arm in ('B1','A','B2'):
 private=P('/tmp/glm53-onepass6-live-'+arm+'-observer')
 if not (private/'identity.json').exists():continue
 rawid=(private/'identity.json').read_bytes();identity=json.loads(rawid)
 assert (identity['revision'],identity['arm'],identity['session'])==(REV,arm,SESSION)
 parserbytes=(private/'launch-parser.py').read_bytes();assert sha(parserbytes)==identity['parser_sha256'] and parserbytes==gitfile('bench/glm53_launch_metadata.py')
 parser=runpy.run_path(str(private/'launch-parser.py'))
 def stable(c):
  return {k:c[k] for k in ('Id','Image','Config','HostConfig','RestartCount')}|{'Mounts':sorted(c['Mounts'],key=lambda m:json.dumps(m,sort_keys=True)),'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
 summary={k:identity[k] for k in ('schema','arm','revision','source','session','captured_at','parser_sha256')}
 summary.update(private_identity_sha256=sha(rawid),raw_inspect_archived=False,nodes={})
 for rank,node in enumerate(NODES):
  s=identity['nodes'][node]
  beforepack=(private/(node+'.inspect.before.json.gz')).read_bytes();afterpack=(private/(node+'.inspect.after.json.gz')).read_bytes()
  before_raw=gzip.decompress(beforepack);after_raw=gzip.decompress(afterpack);before=json.loads(before_raw)[0];after=json.loads(after_raw)[0]
  assert stable(before)==stable(after) and before['State']['Running'] and after['State']['Running']
  assert before['Id']==s['id'] and before['Image']==s['image'] and before['State']['StartedAt']==s['started_at']
  assert before['Image']=='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
  top=parser['launch_parallelism'](before['Config']['Cmd']);assert top==s['topology'] and top['node_rank']==rank and top['enabled']==(arm=='A')
  env={}
  for item in before['Config']['Env']:
   k,v=item.split('=',1);assert k not in env;env[k]=v
  flags={k:env[k] for k in ('VLLM_GLM53_EP_PREFILL_LOCAL','VLLM_B12X_EP_WARM_COMPACT','VLLM_B12X_EP_ZERO_WEIGHT_MICRO','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE')}
  assert flags==s['flags']=={k:('1' if arm=='A' or k=='VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE' else '0') for k in flags}
  assert s['source']=={'manifest_sha256':sha(expected),'mounts':mounts}
  safe={k:s[k] for k in ('node','id','image','started_at','capture_started_at','capture_finished_at','running_start_config_stable','environment_sha256','flags','endpoint','source','log_source')}
  safe['topology']={k:top[k] for k in ('schema','source','enabled','nnodes','node_rank','tensor_parallel_size','command_sha256','prelude_sha256','serve_argv_sha256','serve_argv_without_ep_sha256')}
  safe['private_inspect_hashes']={'before_stored':sha(beforepack),'before_original':sha(before_raw),'after_stored':sha(afterpack),'after_original':sha(after_raw)}
  summary['nodes'][node]=safe
  for suffix in ('serving.log.gz','docker.log.gz','manifest.tsv.gz'):
   name=node+'.'+suffix;raw=(private/name).read_bytes();meta=identity['files'][name]
   assert sha(raw)==meta['stored_sha256'] and sha(gzip.decompress(raw))==meta['original_sha256']
   if suffix=='manifest.tsv.gz':assert gzip.decompress(raw)==expected
   save(arm+'-identity/'+name,raw,str(private/name),original_sha256=meta['original_sha256'],original_bytes=meta['original_bytes'])
 js(arm+'-identity/allowlisted-summary.json',summary,'independently checked private before/after captures')
 save(arm+'-identity/launch-parser.py',parserbytes,str(private/'launch-parser.py'))
S=P('/tmp/glm53-onepass6-streams');raw=(S/'events.jsonl').read_bytes();events=[json.loads(l) for l in raw.splitlines()]
assert events[-1]['kind']=='observer_finished'
assert any(e['kind']=='stream_end' for e in events)
pid=int((S/'observer.pid').read_text())
try:os.kill(pid,0)
except ProcessLookupError:pass
else:raise RuntimeError('observer still alive')
pack('streams/events.jsonl.gz',raw,str(S/'events.jsonl'))
ends={};chunks={}
for e in events:
 if e['kind']!='chunk':continue
 key=e['node'],e['channel'];assert e['offset']==ends.get(key,0)
 ends[key]=e['offset']+e['bytes'];chunks.setdefault(key,[]).append(e)
for (node,channel),end in ends.items():
 p=S/(node+'.'+channel+'.raw');raw=p.read_bytes();assert len(raw)==end
 for e in chunks[(node,channel)]:assert sha(raw[e['offset']:e['offset']+e['bytes']])==e['sha256']
 pack('streams/'+p.name+'.gz',raw,str(p))
js('streams/closure.json',{'observer_pid':pid,'alive':False,'closure_events':[e for e in events if e['kind'] in ('stream_end','observer_stop','observer_finished')],'scope':'Chunk hashes and offsets verified. Raw streams span arms and remain UNASSIGNED until separate container/time/worker-prefix attribution.'},'passive observer originals')
save('collect-evidence.py',P(__file__).read_bytes(),str(P(__file__)))
js('originals.json',provenance.copy(),'original byte provenance')
print(json.dumps({'archive':str(OUT),'rows':[x['name'] for x in rows],'state':r['fleet_after']['state'],'files':len(provenance)}))
