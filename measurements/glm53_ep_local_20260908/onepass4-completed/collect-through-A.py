import base64,datetime,gzip,hashlib,json,pathlib,shlex,subprocess,runpy
ROOT=pathlib.Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass4-candidate-completed'
QUEUED=ROOT/'measurements/glm53_ep_local_20260908/onepass4-queued'
REV='96cb599d8816ee2585988fc7b750a21e2eb66a0b'
CUTOFF=datetime.datetime.fromisoformat('2026-09-09T06:10:00+09:00').timestamp()
NODES=('local','10.10.10.1','10.10.10.3','10.10.10.4')
sha=lambda b:hashlib.sha256(b).hexdigest()
provenance={}
def save(name,data,origin,**details):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_bytes()==data
 else: p.write_bytes(data)
 assert sha(p.read_bytes())==sha(data)
 provenance[name]={'origin':origin,'bytes':len(data),'sha256':sha(data),**details}
def pack(name,data,origin,**details):
 packed=gzip.compress(data,mtime=0)
 save(name,packed,origin,original_bytes=len(data),original_sha256=sha(data),**details)
def jsonsave(name,obj,origin): save(name,(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode(),origin)
for name,expected,count in [('through-A.jsonl','8d11ccadc4cd3191e9f90fad920897fcb2a0e9b35dafdea4cef8e666065433a0',2),('through-B1.jsonl','489a5478eb8bfbe096d521fb762b198dd2878bf11e2c63081790cbc1dec30e48',1)]:
 data=(OUT/'records'/name).read_bytes();assert sha(data)==expected and len(data.splitlines())==count
 save('records/'+name,data,'local original copied immediately at task start')
rows=[json.loads(line) for line in (OUT/'records/through-A.jsonl').read_bytes().splitlines()]
assert [r['name'] for r in rows]==['EPONEPASS4B1','EPONEPASS4A']
# Existing private capture is only read locally. Never copy its raw inspect/Env/Cmd.
private=pathlib.Path('/tmp/glm53-onepass4-live-A-manualfix')
identity_bytes=(private/'identity.json').read_bytes();identity=json.loads(identity_bytes)
assert identity['revision']==REV and identity['arm']=='A' and identity['session']=='eplocalonepass0909v4'
parser_bytes=(private/'launch-parser.py').read_bytes();assert sha(parser_bytes)==identity['parser_sha256']
parser=runpy.run_path(str(private/'launch-parser.py'))
expected_sources=json.loads((QUEUED/'source/hashes.json').read_text())['files']
expected_manifest=b'# source_commit='+REV.encode()+b'\n'+(QUEUED/'source/frozen-manifest.tsv').read_bytes()
expected_mounts={line.split('\t')[1]:expected_sources['build/glm53/'+line.split('\t')[0]]['sha256'] for line in (QUEUED/'source/frozen-manifest.tsv').read_text().splitlines() if line and not line.startswith('#')}
summary={'schema':1,'arm':'A','revision':REV,'source':identity['source'],'session':identity['session'],'captured_at':identity['captured_at'],'private_identity_sha256':sha(identity_bytes),'parser_sha256':sha(parser_bytes),'raw_inspect_archived':False,'nodes':{}}
for node in NODES:
 s=identity['nodes'][node]
 before_pack=(private/(node+'.inspect.before.json.gz')).read_bytes();after_pack=(private/(node+'.inspect.after.json.gz')).read_bytes()
 before_raw=gzip.decompress(before_pack);after_raw=gzip.decompress(after_pack)
 before=json.loads(before_raw)[0];after=json.loads(after_raw)[0]
 def stable(c):
  return {k:c[k] for k in ('Id','Image','Config','HostConfig','RestartCount')} | {'Mounts':sorted(c['Mounts'],key=lambda m:json.dumps(m,sort_keys=True)),'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
 assert stable(before)==stable(after)
 assert before['Id']==s['id'] and before['Image']==s['image'] and before['State']['StartedAt']==s['started_at']
 assert before['State']['Running'] and after['State']['Running']
 assert parser['launch_parallelism'](before['Config']['Cmd'])==s['topology']
 env={}
 for item in before['Config']['Env']:
  k,v=item.split('=',1);assert k not in env;env[k]=v
 flags={k:env[k] for k in ('VLLM_GLM53_EP_PREFILL_LOCAL','VLLM_B12X_EP_WARM_COMPACT','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE')}
 assert flags==s['flags']=={k:'1' for k in flags}
 assert s['source']['mounts']==expected_mounts
 assert s['source']['manifest_sha256']==sha(expected_manifest)
 assert s['running_start_config_stable'] is True
 safe={k:s[k] for k in ('node','id','image','started_at','capture_started_at','capture_finished_at','running_start_config_stable','environment_sha256')}
 safe['flags']=flags
 safe['endpoint']={k:s['endpoint'][k] for k in ('--host','--port') if k in s['endpoint']}
 safe['topology']={k:s['topology'][k] for k in ('schema','source','enabled','nnodes','node_rank','tensor_parallel_size','command_sha256','prelude_sha256','serve_argv_sha256','serve_argv_without_ep_sha256')}
 safe['source']={'manifest_sha256':s['source']['manifest_sha256'],'mounts':s['source']['mounts']}
 safe['log_source']={k:s['log_source'][k] for k in ('path','device','inode','captured_prefix_bytes','size_after','mtime_ns_before','mtime_ns_after')}
 safe['private_inspect_hashes']={'before_stored':sha(before_pack),'before_original':sha(before_raw),'after_stored':sha(after_pack),'after_original':sha(after_raw)}
 summary['nodes'][node]=safe
 for suffix in ('serving.log.gz','docker.log.gz','manifest.tsv.gz'):
  name=node+'.'+suffix;raw=(private/name).read_bytes();meta=identity['files'][name]
  assert sha(raw)==meta['stored_sha256'] and sha(gzip.decompress(raw))==meta['original_sha256']
  if suffix=='manifest.tsv.gz': assert gzip.decompress(raw)==expected_manifest
  save('A-identity/'+name,raw,str(private/name),original_sha256=meta['original_sha256'],original_bytes=meta['original_bytes'])
jsonsave('A-identity/allowlisted-summary.json',summary,'allowlisted and independently verified existing private capture')
save('A-identity/launch-parser.py',parser_bytes,str(private/'launch-parser.py'))
# Preserve immutable arrival-time prefix with its original event/chunk bytes.
streams=pathlib.Path('/tmp/glm53-onepass4-streams')
lines=(streams/'events.jsonl').read_bytes().splitlines(keepends=True)
selected=[];events=[]
for raw in lines:
 try: event=json.loads(raw)
 except json.JSONDecodeError: break
 if event['local_at']>CUTOFF: break
 assert not ({'Env','Cmd','command','argv'} & event.keys())
 selected.append(raw);events.append(event)
assert events and all(e['local_at']<=CUTOFF for e in events)
ends={};chunks={}
for e in events:
 if e['kind']!='chunk': continue
 key=(e['node'],e['channel']);assert e['node'] in NODES and e['channel'] in ('stdout','stderr')
 assert e['offset']==ends.get(key,0)
 ends[key]=e['offset']+e['bytes'];chunks.setdefault(key,[]).append(e)
pack('streams/events-through-A-arrival-cutoff.jsonl.gz',b''.join(selected),str(streams/'events.jsonl'),cutoff_epoch=CUTOFF)
stream_summary={'cutoff_kst':'2026-09-09T06:10:00+09:00','cutoff_epoch':CUTOFF,'selection':'original contiguous events prefix with local_at <= cutoff; stream data remains UNASSIGNED across B1/A, not per-arm identity proof','last_local_at':events[-1]['local_at'],'files':{}}
for (node,channel),end in sorted(ends.items()):
 p=streams/(node+'.'+channel+'.raw')
 with p.open('rb') as f: data=f.read(end)
 assert len(data)==end
 for e in chunks[(node,channel)]: assert sha(data[e['offset']:e['offset']+e['bytes']])==e['sha256']
 name='streams/'+p.name+'.through-A.gz';pack(name,data,str(p),prefix_bytes=end)
 stream_summary['files'][p.name]={'prefix_bytes':end,'prefix_sha256':sha(data),'chunks':len(chunks[(node,channel)]),'first_chunk_at':chunks[(node,channel)][0]['at'],'last_chunk_at':chunks[(node,channel)][-1]['at']}
jsonsave('streams/boundary.json',stream_summary,'verified original passive stream chunks')
# Small private failure logs inspected before inclusion: no raw Env or Cmd.
for label in ('B1','A'):
 for source,name in [(streams/(label+'.snapshot.private.log'),'observer-'+label+'.log'),(pathlib.Path('/tmp/glm53-onepass4-live-'+label+'-observer/failure.private.log'),'coordinator-'+label+'.log')]:
  data=source.read_bytes();assert b'Config' not in data and b'Env' not in data and b'Cmd' not in data
  save('snapshot-failures/'+name,data,str(source))
for name in ('snapshot-mount-order-fix.json','snapshot-mount-order-fix.completed.json'):
 data=(streams/name).read_bytes();assert b'Config' not in data and b'Env' not in data and b'Cmd' not in data
 save('snapshot-failures/'+name,data,str(streams/name))
# Read only the completed head boot logs, verdict snapshot and saved failure logs remotely.
code=r'''import base64,datetime,hashlib,json,pathlib,subprocess,os
root=pathlib.Path('/home/choiceoh/stkernel-ep-onepass-0909-4');job=pathlib.Path('/tmp/glm53-ep-onepass-0909-4')
def ident():
 return {'head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip(),'status':subprocess.check_output(['git','-C',str(root),'status','--porcelain'],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).decode()}
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before':ident(),'files':{}}
paths=[pathlib.Path('/home/choiceoh/glm53-logs/boot-EPONEPASS4'+a+'.log') for a in ('B1','A')]
paths += [job/'verdicts.jsonl',job/'onepass.jsonl']
paths += [job/('live-'+a+'-observer')/(node+'.failure.private.log') for a in ('B1','A') for node in ('local','10.10.10.1','10.10.10.3','10.10.10.4')]
for p in paths:
 a=p.stat();raw=p.read_bytes();b=p.stat();assert len(raw)<=16*2**20
 assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 out['files'][str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'data':base64.b64encode(raw).decode()}
out['after']=ident();print(json.dumps(out))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','choiceoh@srv2','python3 -c '+shlex.quote(code)],capture_output=True,text=True)
if p.returncode: raise RuntimeError(p.stderr)
r=json.loads(p.stdout)
for when in ('before','after'): assert r[when]=={'head':REV,'status':''}
for source,item in r.pop('files').items():
 raw=base64.b64decode(item['data']);assert sha(raw)==item['sha256']
 p=pathlib.Path(source)
 if p.name.startswith('boot-'): pack('boot/'+p.name+'.gz',raw,source)
 elif p.name=='verdicts.jsonl':
  prefix=[]
  for line in raw.splitlines(keepends=True):
   record=json.loads(line)
   if record.get('t','')>'2026-09-09 06:10:00': break
   prefix.append(line)
  save('records/verdict-through-A.jsonl',b''.join(prefix),source,captured_full_sha256=sha(raw),captured_full_bytes=len(raw))
 elif p.name=='onepass.jsonl':
  prefix=b''.join(raw.splitlines(keepends=True)[:2]);assert prefix==(OUT/'records/through-A.jsonl').read_bytes()
  r['remote_onepass']={'capture_rows':len(raw.splitlines()),'full_sha256':sha(raw),'first_two_rows_exactly_match':True,'later_rows_excluded_from_this_archive':len(raw.splitlines())>2}
 else:
  assert len(raw)==168 and b'Env' not in raw and b'Cmd' not in raw
  save('snapshot-failures/'+p.parent.name+'-'+p.name,raw,source)
jsonsave('remote-capture.json',r,'read-only completed logs and frozen-source identity')
jsonsave('originals.json',provenance,'archive originals and byte-prefix provenance')
print(json.dumps({'originals':len(provenance),'stream_files':len(ends),'events':len(events),'remote_capture':r['captured_utc'],'record_rows':2,'later_remote_rows_excluded':r['remote_onepass']['later_rows_excluded_from_this_archive']},indent=2))
