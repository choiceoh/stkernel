import pathlib,json,base64,datetime,gzip,hashlib,shlex,subprocess
ROOT=pathlib.Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
partial=ROOT/'measurements/glm53_ep_local_20260908/onepass4-candidate-completed'
complete=ROOT/'measurements/glm53_ep_local_20260908/onepass4-completed'
assert partial.is_dir() and not complete.exists();partial.rename(complete)
original_collector=pathlib.Path('/tmp/glm53_archive_onepass4_candidate.py').read_text()
# Reuse the same verified, allowlisting-only reader for the B2 private capture.
helpers=original_collector.split('for name,expected,count in ')[0].replace('onepass4-candidate-completed','onepass4-completed')
exec(helpers)
provenance=json.loads((OUT/'originals.json').read_text())
identity_block=original_collector.split('# Existing private capture is only read locally. Never copy its raw inspect/Env/Cmd.\n',1)[1].split('# Preserve immutable arrival-time prefix with its original event/chunk bytes.\n',1)[0]
identity_block=identity_block.replace('/tmp/glm53-onepass4-live-A-manualfix','/tmp/glm53-onepass4-live-B2-observer').replace("identity['arm']=='A'", "identity['arm']=='B2'").replace("'arm':'A'", "'arm':'B2'").replace("{k:'1' for k in flags}", "{k:('1' if k=='VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE' else '0') for k in flags}").replace("'A-identity/", "'B2-identity/")
exec(identity_block)
# Terminal fleet state, records and completed logs from one read-only capture.
code=r'''import base64,datetime,hashlib,json,pathlib,subprocess,os
root=pathlib.Path('/home/choiceoh/stkernel-ep-onepass-0909-4');job=pathlib.Path('/tmp/glm53-ep-onepass-0909-4')
def ident():
 return {'head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip(),'status':subprocess.check_output(['git','-C',str(root),'status','--porcelain'],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).decode()}
def show():
 p=subprocess.run(['bash',str(root/'bench/fleet.sh'),'show','eplocalonepass0909v4'],capture_output=True,text=True)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before':ident(),'fleet_before':show(),'files':{}}
paths=[job/'onepass.jsonl',job/'verdicts.jsonl',pathlib.Path('/home/choiceoh/glm53-logs/boot-EPONEPASS4B2.log'),pathlib.Path('/home/choiceoh/glm53-logs/fleet/run-logs/684fe9015a6e3b720a4d8d163b6af8ad220c85c1d4f0276261551889262aecbc.log')]
for p in paths:
 a=p.stat();raw=p.read_bytes();b=p.stat();assert len(raw)<=32*2**20
 assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 out['files'][str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'data':base64.b64encode(raw).decode()}
out['after']=ident();out['fleet_after']=show();print(json.dumps(out))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','choiceoh@srv2','python3 -c '+shlex.quote(code)],capture_output=True,text=True)
if p.returncode: raise RuntimeError(p.stderr)
r=json.loads(p.stdout)
for when in ('before','after'): assert r[when]=={'head':REV,'status':''}
for when in ('fleet_before','fleet_after'):
 state=r[when];assert state['returncode']==0 and 'eplocalonepass0909v4: succeeded' in state['stdout'] and 'payload_returncode: 0' in state['stdout']
for source,item in r.pop('files').items():
 raw=base64.b64decode(item['data']);assert sha(raw)==item['sha256']
 p=pathlib.Path(source)
 if p.name=='onepass.jsonl':
  rows=[json.loads(l) for l in raw.splitlines()];assert [row['name'] for row in rows]==['EPONEPASS4B1','EPONEPASS4A','EPONEPASS4B2']
  assert b''.join(raw.splitlines(keepends=True)[:2])==(OUT/'records/through-A.jsonl').read_bytes()
  save('records/final-three-rows.jsonl',raw,source)
 elif p.name=='verdicts.jsonl': save('records/final-verdicts.jsonl',raw,source)
 elif p.name.startswith('boot-'): pack('boot/'+p.name+'.gz',raw,source)
 else: pack('fleet/terminal-run.log.gz',raw,source)
jsonsave('terminal-capture.json',r,'read-only terminal fleet/source state before and after collection')
# The stream subprocess has closed. Preserve its complete contiguous event prefix.
streams=pathlib.Path('/tmp/glm53-onepass4-streams');event_raw=(streams/'events.jsonl').read_bytes()
event_lines=event_raw.splitlines(keepends=True);events=[json.loads(line) for line in event_lines]
end_index=next(i for i,e in enumerate(events) if e['kind']=='stream_end')
selected=events[:end_index+1];selected_raw=b''.join(event_lines[:end_index+1])
assert not any(e['kind']=='chunk' for e in events[end_index+1:])
pack('streams/events-through-stream-end.jsonl.gz',selected_raw,str(streams/'events.jsonl'))
ends={};chunks={}
for e in selected:
 if e['kind']!='chunk': continue
 key=e['node'],e['channel'];assert e['offset']==ends.get(key,0)
 ends[key]=e['offset']+e['bytes'];chunks.setdefault(key,[]).append(e)
full={'stream_end':selected[-1],'outer_observer_end_at_capture':[e for e in events if e['kind']=='observer_end'],'outer_observer_still_polling_at_capture':not any(e['kind']=='observer_end' for e in events),'scope':'Full terminal data streams are UNASSIGNED across B1/A/B2. Arrival time alone is not strict per-arm identity. Outer status polling may continue after stream subprocess closes.','files':{}}
for (node,channel),end in sorted(ends.items()):
 p=streams/(node+'.'+channel+'.raw');a=p.stat();raw=p.read_bytes();b=p.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns) and len(raw)==end
 for e in chunks[(node,channel)]: assert sha(raw[e['offset']:e['offset']+e['bytes']])==e['sha256']
 pack('streams/'+p.name+'.terminal.gz',raw,str(p))
 full['files'][p.name]={'bytes':len(raw),'sha256':sha(raw),'chunks':len(chunks[(node,channel)])}
jsonsave('streams/terminal-boundary.json',full,'verified terminal passive data streams; no process signalling by collector')
jsonsave('originals.json',provenance,'original provenance merged after terminal extension')
print(json.dumps({'archive':str(OUT),'final_record_rows':len(rows),'terminal_source_clean':True,'full_stream_files':len(ends),'outer_observer_closed':not full['outer_observer_still_polling_at_capture'],'captured_utc':r['captured_utc']},indent=2))
