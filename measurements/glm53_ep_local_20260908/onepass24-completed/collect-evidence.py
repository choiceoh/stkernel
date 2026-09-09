#!/usr/bin/env python3
"""Archive the completed TP A0 quality-gate failure: remote reads and fresh archive only."""
import base64,datetime,gzip,hashlib,importlib.util,json,os,re,subprocess
from pathlib import Path
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass24-completed'
REV='82ac3c34173ae63b3dd0a42c49f8421097e96a1a';SESSION='eplocalonepass0909v24';TICKET='1788928530449824';PID=449824
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
NODES=('local','10.10.10.1','10.10.10.3','10.10.10.4');items={};originals={}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
 path=Path(path);assert path.is_file() and not path.is_symlink();a=path.stat();raw=path.read_bytes();b=path.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
 assert len(raw)==a.st_size and len(raw)<128*2**20
 return raw
def save(name,raw,origin,**metadata):
 assert name not in items and not Path(name).is_absolute() and '..' not in Path(name).parts
 items[name]=raw;originals[name]=dict(origin=origin,bytes=len(raw),sha256=sha(raw),**metadata)
def packed(name,raw,origin):save(name,gzip.compress(raw,mtime=0),origin,original_bytes=len(raw),original_sha256=sha(raw))
def record(name,value,origin):save(name,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode(),origin)
def gitfile(name):return subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT)
def date(value):return datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
assert not OUT.exists(),'refuse existing archive'
REMOTE=r'''
import base64,hashlib,json,os,pathlib,subprocess,time
P=pathlib.Path;root=P('/home/choiceoh/stkernel-ep-onepass-0909-24');job=P('/tmp/glm53-ep-onepass-0909-24')
session='eplocalonepass0909v24';ticket='1788928530449824'
log=P('/home/choiceoh/glm53-logs/fleet/run-logs/b7faec2bc52ee6533d85aca96a99a57430d74039fdd05d890aebc1e11fd45d78.log')
def source():
 def git(*args):return subprocess.check_output(['git','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).decode().strip()
 return dict(head=git('rev-parse','HEAD'),status=git('status','--porcelain'))
def fleet():
 d=json.loads(subprocess.check_output(['bash',str(root/'bench/fleet.sh'),'show',session,'--ticket',ticket,'--json']))
 keys=('session','ticket','state','phase','started_at','payload_finished_at','finished_at','payload_returncode','returncode','outcome','log_path','recovery_policy','recovery_deferred','supervisor_alive','payload_seconds')
 return {k:d[k] for k in keys if k in d}
def own():
 p=P('/home/choiceoh/glm53-logs/fleet/holder');return p.exists() and p.read_text().split('|',1)[0]==session
r=dict(captured_at=time.time(),source_before=source(),fleet_before=fleet(),own_holder_before=own(),files={},absent=[],cancellation_absent={})
f=r['fleet_before'];assert f['session']==session and str(f['ticket'])==ticket and f['log_path']==str(log)
assert f['phase']=='finished' and f['state']=='failed' and f['payload_returncode']==4 and f['returncode']==4 and not f['supervisor_alive'] and not r['own_holder_before']
paths=[job/name for name in ('submission.json','submit.exit.json','submit.stdout','submit.stderr','onepass.jsonl','verdicts.jsonl')]+[log,P('/tmp/leg.457771')]
paths += [P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS24'+arm+'.log') for arm in ('A0','B1','A','B2','B3')]
for p in paths:
 if not p.exists():r['absent'].append(str(p));continue
 assert p.is_file() and not p.is_symlink()
 a=p.stat();raw=p.read_bytes();b=p.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==a.st_size and len(raw)<128*2**20
 r['files'][str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mtime_ns':b.st_mtime_ns,'data':base64.b64encode(raw).decode()}
for name in ('cancel-request.json','failfast-first-fixed-A0','failfast-first-fixed-A'):
 r['cancellation_absent'][name]=not (job/name).exists()
assert all(r['cancellation_absent'].values())
r.update(source_after=source(),fleet_after=fleet(),own_holder_after=own())
assert r['source_before']==r['source_after'] and r['fleet_before']==r['fleet_after'] and not r['own_holder_after']
print(json.dumps(r))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','choiceoh@srv2','python3 -B -'],input=REMOTE.encode(),capture_output=True,timeout=60)
assert p.returncode==0,p.stderr.decode();remote=json.loads(p.stdout)
assert remote['source_before']==dict(head=REV,status='')
for origin,descriptor in remote['files'].items():
 raw=base64.b64decode(descriptor.pop('data'));assert sha(raw)==descriptor['sha256'] and len(raw)==descriptor['bytes']
 name=Path(origin).name
 if name.startswith('boot-'):packed('boot/'+name+'.gz',raw,origin)
 elif origin==remote['fleet_after']['log_path']:packed('fleet/terminal-run.log.gz',raw,origin)
 elif name=='leg.457771':packed('failure/leg.457771.raw.gz',raw,origin)
 else:save('job/'+name,raw,origin)
record('terminal-capture.json',remote,'read-only terminal fleet/source/own-holder and cancellation-receipt absence capture')
rows=[json.loads(line) for line in items['job/onepass.jsonl'].splitlines()];assert [r['name'] for r in rows]==['EPONEPASS24A0']
verdicts=[json.loads(line) for line in items['job/verdicts.jsonl'].splitlines()]
assert len(verdicts)==1 and verdicts[0]['cand']=='EPONEPASS24A0' and verdicts[0]['status']=='invalid' and verdicts[0]['gates']==['korean 2/8'] and verdicts[0]['verdict']=='GATE FAIL: korean 2/8'
assert rows[0]['quality']=={'ok':18,'total':18} and rows[0]['korean']=={'dirty':2,'n':8,'kinds':{'replacement':0,'lone_jamo':0,'cjk_mixed':4,'control':0},'hits':[['fixed2K rep0',{'cjk_mixed':2}],['fixed2K rep2',{'cjk_mixed':2}]]}
terminal_log=gzip.decompress(items['fleet/terminal-run.log.gz'])
assert terminal_log.count('Halvorsen博士'.encode())>=2
assert len(re.findall(rb'fixed2K rep=\d+ tokens=1024/1024 decode=',terminal_log))==3
assert b'GATE FAIL: korean 2/8' in terminal_log
leg_name='failure/leg.457771.raw.gz'
if leg_name in items:
 leg=gzip.decompress(items[leg_name]);assert len(re.findall(rb'fixed2K rep=\d+ tokens=1024/1024 decode=',leg))==3 and leg.count('Halvorsen博士'.encode())>=2
else:
 assert '/tmp/leg.457771' in remote['absent'];leg=None
record('failure/leg-binding.json',dict(path='/tmp/leg.457771',available_at_collection=leg is not None,sha256=sha(leg) if leg is not None else None,source=REV,session=SESSION,ticket=TICKET,owner_pid=PID,onepass_child_pid=494734,lever_pid=457771,scope='Parent supplied live process ancestry; completed temporary leg was absent at collection. Full canonical terminal log retains all three fixed decode lines and both failure excerpts. No reconstruction is called an original leg.'),'parent live observation plus terminal file existence check')
record('collection/first-attempt.json',dict(verdict='REFUSED_BEFORE_ARCHIVE_WRITE',reason='required temporary leg absent',collector_sha256='6382fee61ea1b38ceeecf001de4a234e69f4f6a959b5a2c70c2b2c28ad23dc31',runtime_source_changed=False),'first collector terminal result')
packed('collection/first-attempt.log.gz',read('/tmp/glm53-onepass24-archive-requiredleg.log'),'/tmp/glm53-onepass24-archive-requiredleg.log')


manifest_raw=gitfile('build/glm53/manifest.tsv');manifest=('# source_commit='+REV+'\n').encode()+manifest_raw
mounts={line.split('\t')[1]:sha(gitfile('build/glm53/'+line.split('\t')[0])) for line in manifest.decode().splitlines() if line and not line.startswith('#')}
packed('source/onepass.py.gz',gitfile('bench/onepass.py'),'git '+REV+':bench/onepass.py')
save('source/git-manifest.tsv',manifest_raw,'git '+REV);save('source/frozen-manifest.tsv',manifest,'normal source_commit header + git '+REV)
record('source/mounted-hashes.json',mounts,'frozen build '+REV)
for name in ('glm53_tp_sf6_q0_selftest.py','glm53_ep_local_selftest.py','moe_dynamic_gated_sf6_q0.py','moe_dynamic_gated_sf6.py','moe_dispatch.py','flashinfer_b12x_moe.py','gpu_worker.py'):
 packed('source/'+name+'.gz',gitfile('build/glm53/'+name),'git '+REV+':build/glm53/'+name)
identities={}
def fixed(c):return {k:c[k] for k in ('Id','Created','Image','Config','HostConfig','RestartCount')}|{'Mounts':sorted(c['Mounts'],key=lambda m:json.dumps(m,sort_keys=True)),'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
for arm in ('A0',):
 private=Path('/tmp/glm53-onepass24-live-'+arm+'-observer');iraw=read(private/'identity.json');identity=json.loads(iraw);identities[arm]=identity
 assert (identity['revision'],identity['session'],identity['ticket'],identity['owner_pid'],identity['arm'])==(REV,SESSION,TICKET,str(PID),arm)
 assert set(identity['nodes'])==set(NODES)
 for name,d in identity['files'].items():
  stored=read(private/name);original=gzip.decompress(stored)
  assert len(stored)==d['stored_bytes'] and sha(stored)==d['stored_sha256'] and len(original)==d['original_bytes'] and sha(original)==d['original_sha256']
  if '.inspect.' not in name:save('snapshot/'+arm+'/'+name,stored,str(private/name),original_bytes=len(original),original_sha256=sha(original))
 for rank,node in enumerate(NODES):
  d=identity['nodes'][node];before=json.loads(gzip.decompress(read(private/(node+'.inspect.before.json.gz'))))[0];after=json.loads(gzip.decompress(read(private/(node+'.inspect.after.json.gz'))))[0]
  assert fixed(before)==fixed(after) and before['State']['Running'] and after['State']['Running']
  assert before['Id']==d['id'] and before['Image']==IMAGE and d['image']==IMAGE
  assert before['State']['StartedAt']==d['started_at'] and before['Created']==d['created_at']
  assert date(d['started_at'])>=date(d['created_at'])>=identity['arm_event']['started_at']
  assert d['source']==dict(manifest_sha256=sha(manifest),mounts=mounts)
  assert all(d['topology'][key]==value for key,value in dict(enabled=False,tensor_parallel_size=4,nnodes=4,node_rank=rank).items())
  env=dict(x.split('=',1) for x in before['Config']['Env'])
  expected={'VLLM_B12X_EP_ZERO_WEIGHT_MICRO':'0','VLLM_GLM53_EP_PREFILL_LOCAL':'0','VLLM_B12X_EP_WARM_COMPACT':'0','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE':'1','VLLM_GLM53_STARTUP_TRIM':'1','VLLM_GLM53_TP_SF6_Q0':'0' if arm=='B1' else '1'}
  assert d['flags']==expected and all(env[key]==value for key,value in expected.items())
  assert d['environment_sha256']==sha(json.dumps(env,sort_keys=True,separators=(',',':')).encode())
  assert d['mm_limit']=={'image':4,'video':0}
  readiness=d['readiness'];assert readiness['graph_finished'] and readiness['candidate']==(arm!='B1')
  assert len(readiness['tp_sf6_q0_pass_records'])==(0 if arm=='B1' else 1)
  trim=readiness['startup_trim_records'];assert len(trim)==1;tr=trim[0]['receipt']
  assert tr['verdict']=='COMPLETE' and tr['rank']==rank and tr['measurement_errors']==[]
  assert [x['stage'] for x in tr['stages']]==['synchronize','gc_collect','empty_cache','malloc_trim'] and all(x['status']=='COMPLETE' for x in tr['stages'])
  assert date(d['started_at'])<=tr['started_at']<=tr['completed_at']<=d['capture_finished_at'] and tr['before']['allocated']==tr['after']['allocated']
  log=gzip.decompress(items['snapshot/'+arm+'/'+node+'.serving.log.gz']);trim_lines=[line for line in log.splitlines() if b'[glm53-startup-trim] ' in line]
  assert len(trim_lines)==1 and sha(trim_lines[0])==trim[0]['line_sha256'] and json.loads(trim_lines[0].split(b'[glm53-startup-trim] ',1)[1])==tr
  graph=re.search(rb'Graph capturing finished in [0-9]+ secs, took ',log);assert graph and graph.start()<log.index(trim_lines[0])
 save('snapshot/'+arm+'/allowlisted-identity.json',iraw,str(private/'identity.json'))
 parser=read(private/'launch-parser.py');assert sha(parser)==identity['parser_sha256'];save('snapshot/'+arm+'/launch-parser.py',parser,str(private/'launch-parser.py'))
completed=[]
for row in rows:
 arm=row['name'].removeprefix('EPONEPASS24');identity=identities[arm]
 assert row['session']==SESSION and row['git']==REV[:8] and row['overlay']==sha(manifest)[:12]
 assert row['boot_id']==identity['nodes']['local']['id']+'|'+identity['nodes']['local']['started_at']
 fixed_requests=[request for request in row['requests'] if request.get('fixed_decode')]
 assert len(row['requests'])==8 and len(fixed_requests)==3 and [r['rep'] for r in fixed_requests]==[0,1,2]
 assert all(r['completion_tokens']==r['min_tokens']==r['max_tokens']==1024 and r['finish_reason']=='length' for r in fixed_requests)
 pooled=sum(r['completion_tokens']-1 for r in fixed_requests)/sum(r['decode_s'] for r in fixed_requests)
 completed.append(dict(arm=arm,fixed_decode_tok_s=[r['decode_tok_s'] for r in fixed_requests],fixed_pooled_tok_s=pooled,prefill=row['prefill'],quality=row['quality'],korean=row['korean'],proof_ok=row['proof_ok'],onepass_record_sha256=sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode())))

stream_root=Path('/tmp/glm53-onepass24-streams');eraw=read(stream_root/'events.jsonl');events=[json.loads(line) for line in eraw.splitlines()]
assert events[-1]['kind']=='observer_finished' and events[-1]['attempted_arms']==['A0'] and events[-1]['owned_go_seen']
ends={};chunks={};streams={}
for event in events:
 if event['kind']!='chunk':continue
 key=event['node'],event['channel'];assert event['offset']==ends.get(key,0)
 ends[key]=event['offset']+event['bytes'];chunks.setdefault(key,[]).append(event)
for key,end in ends.items():
 path=stream_root/(key[0]+'.'+key[1]+'.raw');raw=read(path);assert len(raw)==end
 for event in chunks[key]:assert sha(raw[event['offset']:event['offset']+event['bytes']])==event['sha256']
 streams[key]=raw;packed('streams/'+path.name+'.gz',raw,str(path))
assert all((node,'stdout') in streams for node in NODES)
packed('streams/events.jsonl.gz',eraw,str(stream_root/'events.jsonl'))
observer_pid=int(read(stream_root/'observer.pid'))
try:os.kill(observer_pid,0)
except ProcessLookupError:alive=False
else:alive=True
assert not alive,'observer still alive'
record('streams/closure.json',dict(observer_pid=observer_pid,alive=False,events=[e for e in events if e['kind'] in ('stream_end','observer_stop','observer_finished')],scope='All original chunks/offsets/hashes verified; bytes unassigned by observer remain unassigned.'),'closed passive observer')

helper=Path('/tmp/glm53_extract_onepass24_canary.py');helper_raw=read(helper)
assert sha(helper_raw)=='983183d400ca6d86032e7b485c60a9183fe1ecc42235b7eb6811a44757fd92df'
spec=importlib.util.spec_from_file_location('canary_validator',helper);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
expected={role:dict(path=next(path for path in mounts if Path(path).name==filename),sha256=sha(gitfile('build/glm53/'+filename))) for role,filename in validator.FILES.items()}
expected['stock']=dict(path='/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py',sha256=validator.STOCK_SHA)
canary_summary={}
for occurrence,arm in enumerate(('A0',)):
 canary_root=Path('/tmp/glm53-onepass24-canary-'+arm);validation_raw=read(canary_root/'validation.json');validation=json.loads(validation_raw);identity=identities[arm]
 assert validation['verdict']=='RAW_CANARY_SOURCE_VALIDATED' and validation['revision']==REV and validation['cpu24_result_sha256']==validator.CPU_SHA and validation['occurrence']==occurrence
 assert validation['arm_container_proof'] is False
 if arm=='A':assert validation['within_rank_continuity']['match'] and validation['within_rank_continuity']['difference_count']==0
 save('canary/'+arm+'/validation-original.json',validation_raw,str(canary_root/'validation.json'));canary_summary[arm]={}
 for node in NODES:
  path=canary_root/(node+'.json');raw=read(path);receipt=json.loads(raw);details=validator.validate(receipt,expected)
  line=read(canary_root/(node+'.receipt-line.raw'));prov=validation['nodes'][node]['provenance'];stream=streams[node,'stdout']
  assert stream[prov['raw_offset']:prov['raw_end']]==line and sha(line)==prov['line_sha256']
  assert sha(stream[:prov['raw_end']])==prov['prefix_sha256'] and sha(raw.rstrip(b'\n'))==prov['json_sha256']
  assert stream.count(b'[tp-sf6-q0-selftest] PASS ')==1 and b'[tp-sf6-q0-selftest] FAIL ' not in stream
  log=gzip.decompress(items['snapshot/'+arm+'/'+node+'.serving.log.gz']);marker=b'[tp-sf6-q0-selftest] PASS ';assert log.count(marker)==1
  logged,_=json.JSONDecoder().raw_decode(log.split(marker,1)[1].decode());assert logged==receipt
  assert date(identity['nodes'][node]['started_at'])<=receipt['started_at']<=receipt['completed_at']<=identity['nodes'][node]['capture_finished_at']
  assert receipt['completed_at']<=identity['nodes'][node]['readiness']['startup_trim_records'][0]['receipt']['started_at']
  save('canary/'+arm+'/'+node+'.json',raw,str(path),raw_stream_match=prov);save('canary/'+arm+'/'+node+'.receipt-line.raw',line,str(canary_root/(node+'.receipt-line.raw')))
  details.pop('versions');canary_summary[arm][node]=dict(details,verdict='PASS',strict_source_container_bound=True)
record('canary/verified-summary.json',canary_summary,'4 TP PASS receipts source-bound and matched to strict A0 logs and original streams')
for source,name in [('/tmp/glm53-onepass24-submit.json','submission/request.json'),('/tmp/glm53-onepass24-submit-receipt.txt','submission/receipt.json'),('/tmp/glm53_onepass24_observer.py','tools/passive-observer.py'),('/tmp/glm53_onepass24_snapshot.py','tools/snapshot-helper.py'),(str(helper),'tools/canary-extractor-validator.py'),('/tmp/glm53-onepass24-A0-verified.json','canary/A0/independent-identity-trim-mm-verification.json'),('/tmp/glm53_onepass24_failfast.py','tools/failfast-helper.py')]:save(name,read(source),source)
submission=json.loads(items['submission/receipt.json']);assert (submission['session'],submission['ticket'],submission['pid'])==(SESSION,TICKET,PID)
cpu_path=ROOT/'measurements/glm53_ep_local_20260908/decode24-cpu/result.json';cpu_raw=read(cpu_path);assert sha(cpu_raw)==validator.CPU_SHA
record('source/cpu24-reference.json',dict(path=str(cpu_path.relative_to(ROOT)),sha256=sha(cpu_raw),scope='CPU capsule13.0.3 versus serving image13.3.1; matching kernel/source bytes only'),'existing original CPU24 archive')
record('failure/failfast-observation.json',dict(scope='Parent-supplied tool stdout observation, not a captured raw stdout file',exec_session=93829,exit_code=0,action='REPORT_FIRST_RESULT_NOT_BELOW_THRESHOLD',signal_sent=False,observed_rounded_decode_tok_s=81.29,raw_stdout_archived=False,signal_paths_absent=remote['cancellation_absent']),'parent message plus independently collected terminal path absence')
independent=json.loads(items['canary/A0/independent-identity-trim-mm-verification.json'])
assert independent['verdict']=='STRICT_A0_IDENTITY_NUMERICS_TRIM_MM_VERIFIED' and independent['revision']==REV and independent['arm']=='A0'
summary=dict(schema=1,verdict='A0_KOREAN_GATE_FAILURE_CHANNEL_UNKNOWN',revision=REV,session=SESSION,ticket=TICKET,supervisor_pid=PID,
 completed_arms=['A0'],completed=completed,onepass_records=1,partial_arm=None,unrun_arms=['B1','A','B2','B3'],payload_returncode=4,returncode=4,own_holder_after=False,
 termination=dict(source='canonical judge',reason='GATE FAIL: korean 2/8',failfast_signal=False,failfast_scope='Parent observed report-only result; terminal cancel-request and all failfast signal-evidence paths absent'),
 full_canary_pass_receipts=4,canary_cases=16,candidate_comparisons=96,stock_control_comparisons=96,q0_comparisons=96,
 canonical_verdicts=verdicts,affected_output_channel='UNKNOWN',literal_korean_hits='Halvorsen博士',historical_similarity_scope='Same literal CJK mixed-token gate failure documented in MEASUREMENTS input-reuse campaign; no causal attribution',
 performance_acceptance=False,matched_baseline_record=False,adoption_acceptance=False,full_sanitizer_acceptance=False,
 head_post_trim_memory_comparison=independent['head_post_trim_memory_comparison'],
 recovery_policy=remote['fleet_after'].get('recovery_policy'),recovery_deferred=remote['fleet_after'].get('recovery_deferred'))
record('result-summary.json',summary,'Completed A0 original record, unchanged Korean gate verdict and source-bound TP canaries')
readme="""# onepass24: A0 completed; Korean gate failed

Frozen source `82ac3c34173ae63b3dd0a42c49f8421097e96a1a`; normal fleet session `eplocalonepass0909v24`, ticket `1788928530449824`, supervisor `449824`. The one completed canonical A0 record remains unmodified in `job/onepass.jsonl`, including all requests/output hashes/metrics. B1, warm A, B2 and B3 never ran. There is no matched baseline or performance/default-adoption verdict.

A0 completed 128K single-request prefill at3127.225tok/s with41.1096s TTFT. Fixed1024 decode repetitions measured81.288/69.274/65.466tok/s (pooled71.407tok/s). These are standalone candidate measurements. Facts passed18/18 and serving proof3/3, but the existing Korean gate failed2/8: four mixed CJK characters in the two literal `Halvorsen博士` excerpts from fixed repetitions0 and2. Replacement/jamo/control counts were zero. The original canonical judge verdict is retained as `GATE FAIL: korean 2/8`; combined reasoning/content logging leaves the affected channel UNKNOWN. Similar literal failures were documented in the earlier input-reuse campaign; that observation does not establish a cause or waive this failure.

Canonical payload/supervisor returned4 and the owned holder/supervisor were absent at collection. No failfast signal was sent; cancellation and failfast signal-evidence paths were absent. The completed first repetition exceeded the65tok/s stop threshold. Original full fleet/boot logs and closed four-node passive streams are preserved. Temporary `/tmp/leg.457771` was already absent at collection; it is not reconstructed or claimed as archived. The full terminal log retains all three fixed result lines and both failure excerpts. Normal idle recovery is recorded without claiming public restoration.

All four TP Q0 startup canary receipts pass:16 fixture passes,96 candidate comparisons,96 stock-control checks and96 Q0 checks. Source/CPU24 hashes, exact marker bytes and strict A0 snapshots bind the numerical receipts. All four snapshots preserve pinned image/source/TP4/4, EP/local/warm/zero0, Q01, skip1/trim1 and actual image4/video0. Every graph precedes a complete four-stage trim receipt with unchanged allocated bytes. Head post-trim available RAM is13.638GiB versus GPU23 A0's9.577GiB (+4.060GiB across boots); this is not isolated video-memory attribution. Raw Env/Cmd/inspect remain private and are excluded.

CPU24 used an isolated13.0.3 bindings capsule on srv1 under the normal head CPU fleet; serving used image13.3.1. Matching mounted source and CPU artifacts do not establish CUDA binary equality, all-layer numerics or sanitizer acceptance. Observer chunks/offsets/hashes and terminal closure are verified; unassigned bytes keep that attribution. Logs/streams/frozen Python use deterministic gzip with original/stored hashes. Collection performed no GPU/HTTP/queue/service or runtime-source mutation.
"""
save('README.md',readme.encode(),'bounded evidence scope');save('collect-evidence.py',read(__file__),'this collector')
record('originals.json',originals.copy(),'original/stored byte provenance')
checks=''.join(f'{sha(raw)}  {name}\n' for name,raw in sorted(items.items()));items['SHA256SUMS']=checks.encode()
OUT.mkdir(parents=False)
for name,raw in items.items():
 path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for name,raw in items.items():assert sha(read(OUT/name))==sha(raw)
print(json.dumps(dict(archive=str(OUT),files=len(items),bytes=sum(map(len,items.values())),summary_sha256=sha(items['result-summary.json']),sums_sha256=sha(items['SHA256SUMS'])),sort_keys=True))
