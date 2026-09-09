#!/usr/bin/env python3
"""Archive the naturally failed TP A0 run: remote reads and fresh archive only."""
import base64,datetime,gzip,hashlib,importlib.util,json,os,re,subprocess
from pathlib import Path
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass23-completed'
REV='cbf1c7916247f946167c15ff74d92261588e8cea';SESSION='eplocalonepass0909v23';TICKET='1788926263262324';PID=262324
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
P=pathlib.Path;root=P('/home/choiceoh/stkernel-ep-onepass-0909-23');job=P('/tmp/glm53-ep-onepass-0909-23')
session='eplocalonepass0909v23';ticket='1788926263262324'
log=P('/home/choiceoh/glm53-logs/fleet/run-logs/a6f6a9232220a5509b815e9cdc74886dd40a8cc81dcce4d4cb5c0e74dca438ce.log')
def source():
 def git(*args):return subprocess.check_output(['git','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).decode().strip()
 return dict(head=git('rev-parse','HEAD'),status=git('status','--porcelain'))
def fleet():
 d=json.loads(subprocess.check_output(['bash',str(root/'bench/fleet.sh'),'show',session,'--ticket',ticket,'--json']))
 keys=('session','ticket','state','phase','started_at','payload_finished_at','finished_at','payload_returncode','returncode','outcome','log_path','recovery_policy','recovery_deferred','supervisor_alive','payload_seconds')
 return {k:d[k] for k in keys if k in d}
def own():
 p=P('/home/choiceoh/glm53-logs/fleet/holder');return p.exists() and p.read_text().split('|',1)[0]==session
r=dict(captured_at=time.time(),source_before=source(),fleet_before=fleet(),own_holder_before=own(),files={},absent=[],journal={})
f=r['fleet_before'];assert f['session']==session and str(f['ticket'])==ticket and f['log_path']==str(log)
assert f['phase']=='finished' and f['state']=='failed' and f['payload_returncode']==1 and f['returncode']==1 and not f['supervisor_alive'] and not r['own_holder_before']
paths=[job/name for name in ('submission.json','submit.exit.json','submit.stdout','submit.stderr','onepass.jsonl','verdicts.jsonl')]+[log,P('/tmp/leg.357316')]
paths += [P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS23'+arm+'.log') for arm in ('A0','B1','A','B2','B3')]
for p in paths:
 if not p.exists():r['absent'].append(str(p));continue
 assert p.is_file() and not p.is_symlink()
 a=p.stat();raw=p.read_bytes();b=p.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==a.st_size and len(raw)<128*2**20
 r['files'][str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mtime_ns':b.st_mtime_ns,'data':base64.b64encode(raw).decode()}
for key,scope in (('earlyoom',['-u','earlyoom']),('kernel',['-k'])):
 argv=['sudo','-n','journalctl',*scope,'--utc','--since','2026-09-09 04:16:00 UTC','--until','2026-09-09 04:24:10 UTC','--no-pager','-o','short-iso']
 p=subprocess.run(argv,capture_output=True);assert p.returncode==0
 raw=p.stdout;assert len(raw)<2**20
 r['journal'][key]=dict(argv=argv,returncode=p.returncode,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),data=base64.b64encode(raw).decode())
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
 elif name=='leg.357316':packed('failure/leg.357316.raw.gz',raw,origin)
 else:save('job/'+name,raw,origin)
for key,d in remote['journal'].items():
 raw=base64.b64decode(d.pop('data'));assert sha(raw)==d['sha256'] and len(raw)==d['bytes'];packed('failure/'+key+'.journal.raw.gz',raw,'head: '+' '.join(d['argv']))
record('terminal-capture.json',remote,'read-only terminal fleet/source/own-holder and fixed-window journal capture')
rows=[json.loads(line) for line in items['job/onepass.jsonl'].splitlines()];assert [r['name'] for r in rows]==['EPONEPASS23A0','EPONEPASS23B1']
verdicts=[json.loads(line) for line in items.get('job/verdicts.jsonl',b'').splitlines()]
leg=gzip.decompress(items['failure/leg.357316.raw.gz'])
assert b'500 Internal Server Error' in leg or b'500 Server Error' in leg or b'HTTP Error 500' in leg
assert not re.search(rb'fixed2K rep=\d+ tokens=1024/1024 decode=',leg),'unexpected fixed result'
early=gzip.decompress(items['failure/earlyoom.journal.raw.gz'])
assert b'2026-09-09T04:23:49' in early and b'365084' in early and b'SIGTERM' in early and b'5858' in early
assert b'exited' in early

manifest_raw=gitfile('build/glm53/manifest.tsv');manifest=('# source_commit='+REV+'\n').encode()+manifest_raw
mounts={line.split('\t')[1]:sha(gitfile('build/glm53/'+line.split('\t')[0])) for line in manifest.decode().splitlines() if line and not line.startswith('#')}
save('source/git-manifest.tsv',manifest_raw,'git '+REV);save('source/frozen-manifest.tsv',manifest,'normal source_commit header + git '+REV)
record('source/mounted-hashes.json',mounts,'frozen build '+REV)
for name in ('glm53_tp_sf6_q0_selftest.py','glm53_ep_local_selftest.py','moe_dynamic_gated_sf6_q0.py','moe_dynamic_gated_sf6.py','moe_dispatch.py','flashinfer_b12x_moe.py','gpu_worker.py'):
 packed('source/'+name+'.gz',gitfile('build/glm53/'+name),'git '+REV+':build/glm53/'+name)
identities={}
def fixed(c):return {k:c[k] for k in ('Id','Created','Image','Config','HostConfig','RestartCount')}|{'Mounts':sorted(c['Mounts'],key=lambda m:json.dumps(m,sort_keys=True)),'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
for arm in ('A0','B1','A'):
 private=Path('/tmp/glm53-onepass23-live-'+arm+'-observer');iraw=read(private/'identity.json');identity=json.loads(iraw);identities[arm]=identity
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
 arm=row['name'].removeprefix('EPONEPASS23');identity=identities[arm]
 assert row['session']==SESSION and row['git']==REV[:8] and row['overlay']==sha(manifest)[:12]
 assert row['boot_id']==identity['nodes']['local']['id']+'|'+identity['nodes']['local']['started_at']
 fixed_requests=[request for request in row['requests'] if request.get('fixed_decode')]
 assert len(row['requests'])==8 and len(fixed_requests)==3 and [r['rep'] for r in fixed_requests]==[0,1,2]
 assert all(r['completion_tokens']==r['min_tokens']==r['max_tokens']==1024 and r['finish_reason']=='length' for r in fixed_requests)
 pooled=sum(r['completion_tokens']-1 for r in fixed_requests)/sum(r['decode_s'] for r in fixed_requests)
 completed.append(dict(arm=arm,fixed_decode_tok_s=[r['decode_tok_s'] for r in fixed_requests],fixed_pooled_tok_s=pooled,prefill=row['prefill'],quality=row['quality'],korean=row['korean'],proof_ok=row['proof_ok'],onepass_record_sha256=sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode())))

stream_root=Path('/tmp/glm53-onepass23-streams');eraw=read(stream_root/'events.jsonl');events=[json.loads(line) for line in eraw.splitlines()]
assert events[-1]['kind']=='observer_finished' and events[-1]['attempted_arms']==['A','A0','B1'] and events[-1]['owned_go_seen']
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

helper=Path('/tmp/glm53_extract_onepass23_canary.py');helper_raw=read(helper)
assert sha(helper_raw)=='faf539dff9dfe3eb9dfff0046d7f5f74869a6c392b1cf59ade7217c4d82f557b'
spec=importlib.util.spec_from_file_location('canary_validator',helper);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
expected={role:dict(path=next(path for path in mounts if Path(path).name==filename),sha256=sha(gitfile('build/glm53/'+filename))) for role,filename in validator.FILES.items()}
expected['stock']=dict(path='/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py',sha256=validator.STOCK_SHA)
canary_summary={}
for occurrence,arm in enumerate(('A0','A')):
 canary_root=Path('/tmp/glm53-onepass23-canary-'+arm);validation_raw=read(canary_root/'validation.json');validation=json.loads(validation_raw);identity=identities[arm]
 assert validation['verdict']=='RAW_CANARY_SOURCE_VALIDATED' and validation['revision']==REV and validation['cpu23_result_sha256']==validator.CPU_SHA and validation['occurrence']==occurrence
 assert validation['arm_container_proof'] is False
 if arm=='A':assert validation['within_rank_continuity']['match'] and validation['within_rank_continuity']['difference_count']==0
 save('canary/'+arm+'/validation-original.json',validation_raw,str(canary_root/'validation.json'));canary_summary[arm]={}
 for node in NODES:
  path=canary_root/(node+'.json');raw=read(path);receipt=json.loads(raw);details=validator.validate(receipt,expected)
  line=read(canary_root/(node+'.receipt-line.raw'));prov=validation['nodes'][node]['provenance'];stream=streams[node,'stdout']
  assert stream[prov['raw_offset']:prov['raw_end']]==line and sha(line)==prov['line_sha256']
  assert sha(stream[:prov['raw_end']])==prov['prefix_sha256'] and sha(raw.rstrip(b'\n'))==prov['json_sha256']
  assert stream.count(b'[tp-sf6-q0-selftest] PASS ')==2 and b'[tp-sf6-q0-selftest] FAIL ' not in stream
  log=gzip.decompress(items['snapshot/'+arm+'/'+node+'.serving.log.gz']);marker=b'[tp-sf6-q0-selftest] PASS ';assert log.count(marker)==1
  logged,_=json.JSONDecoder().raw_decode(log.split(marker,1)[1].decode());assert logged==receipt
  assert date(identity['nodes'][node]['started_at'])<=receipt['started_at']<=receipt['completed_at']<=identity['nodes'][node]['capture_finished_at']
  assert receipt['completed_at']<=identity['nodes'][node]['readiness']['startup_trim_records'][0]['receipt']['started_at']
  save('canary/'+arm+'/'+node+'.json',raw,str(path),raw_stream_match=prov);save('canary/'+arm+'/'+node+'.receipt-line.raw',line,str(canary_root/(node+'.receipt-line.raw')))
  details.pop('versions');canary_summary[arm][node]=dict(details,verdict='PASS',strict_source_container_bound=True)
record('canary/verified-summary.json',canary_summary,'8 TP PASS receipts source-bound and matched to strict A0/A logs and original streams')
for source,name in [('/tmp/glm53-onepass23-submit.json','submission/request.json'),('/tmp/glm53-onepass23-submit-receipt.txt','submission/receipt.json'),('/tmp/glm53_onepass23_observer.py','tools/passive-observer.py'),('/tmp/glm53_onepass23_snapshot.py','tools/snapshot-helper.py'),(str(helper),'tools/canary-extractor-validator.py'),('/tmp/glm53-onepass23-A-verified.json','canary/A/independent-identity-trim-verification.json'),('/tmp/glm53-onepass23-A0-vs21-continuity.json','canary/A0/versus21-TP-content-continuity.json')]:save(name,read(source),source)
submission=json.loads(items['submission/receipt.json']);assert (submission['session'],submission['ticket'],submission['pid'])==(SESSION,TICKET,PID)
cpu_path=ROOT/'measurements/glm53_ep_local_20260908/decode23-cpu/result.json';cpu_raw=read(cpu_path);assert sha(cpu_raw)==validator.CPU_SHA
record('source/cpu23-reference.json',dict(path=str(cpu_path.relative_to(ROOT)),sha256=sha(cpu_raw),scope='CPU capsule13.0.3 versus serving image13.3.1; matching kernel/source bytes only'),'existing original CPU23 archive')
record('failure/partial-leg-binding.json',dict(path='/tmp/leg.357316',sha256=sha(leg),source=REV,session=SESSION,ticket=TICKET,owner_pid=PID,onepass_child_pid=389865,lever_pid=357316,parent_observed_at='2026-09-09T13:23:38+09:00',scope='Parent observed exact process ancestry/cwd before death; collector hashes terminal original file. No claim processes are still alive.'),'parent supplied live lineage observation plus terminal raw file')
summary=dict(schema=1,verdict='WARM_A_INFRASTRUCTURE_FAILURE_BEFORE_FIXED_DECODE',revision=REV,session=SESSION,ticket=TICKET,supervisor_pid=PID,
 completed_arms=['A0','B1'],completed=completed,onepass_records=2,partial_arm='A',unrun_arms=['B2','B3'],payload_returncode=1,returncode=1,own_holder_after=False,
 warm_A_fixed_decode_tok_s=[],warm_A_fixed_pooled_tok_s=None,warm_A_quality=None,warm_A_korean=None,warm_A_proof_ok=None,
 termination=dict(source='head earlyoom journal',action='SIGTERM',worker_pid=365084,at_utc='2026-09-09T04:23:49Z',available_memory_mib=5858,available_percent=4.78),
 full_canary_pass_receipts=8,canary_cases=32,candidate_comparisons=192,stock_control_comparisons=192,q0_comparisons=192,
 canonical_verdicts=verdicts,performance_acceptance=False,matched_warm_A_record=False,adoption_acceptance=False,full_sanitizer_acceptance=False,
 recovery_policy=remote['fleet_after'].get('recovery_policy'),recovery_deferred=remote['fleet_after'].get('recovery_deferred'))
record('result-summary.json',summary,'A0/B1 completed records, exact partial warm A leg, earlyoom journal and source-bound TP canaries')
readme="""# onepass23: A0/B1 completed, warm A stopped by earlyoom

Frozen source `cbf1c7916247f946167c15ff74d92261588e8cea`; normal fleet session `eplocalonepass0909v23`, ticket `1788926263262324`, supervisor `262324`. Completed canonical A0 and B1 records (all original requests, output hashes, prefill, quality, Korean checks and fixed decode) remain unmodified in `job/onepass.jsonl`. Their parsed fixed repetitions and pooled decode values are in `result-summary.json`. The intended matched warm A record was never completed; B2 and B3 did not run. No warm A decode value or complete matched performance/adoption verdict is claimed.

Warm A's128K request died before a completed decode measurement and its subsequent fixed request returnedHTTP500. Head earlyoom sentSIGTERM to worker365084 at04:23:49UTC with5858MiB available (4.78%). The original partial leg357316, closed four-node streams and fixed-window earlyoom/kernel journals are preserved. This was an infrastructure termination, with no failfast cancellation requested by this collector. Payload/supervisor returned1 and the owned holder/supervisor were absent at collection. Normal idle recovery is recorded without claiming public restoration.

A0 and warm A each produced four full TP Q0 canary PASS receipts:32 fixture passes and192 candidate comparisons overall, plus192 stock-control and192 Q0 checks. The same-rank actual TP weight/input contents match between A0 and A. Source/CPU23 checks, exact marker bytes and per-arm strict snapshots bind the numerical receipts. A0/B1/A snapshots independently preserve all four identities, pinned image, frozen mounted sources, TP4/4 withEP/local/warm/zero disabled, Q0 candidate1/baseline0, and common startup trim1. Every graph finished before a complete four-stage trim receipt with unchanged allocated bytes. Raw Env/Cmd/inspect remain private and are not copied here.

CPU23 used isolated bindings13.0.3 while serving used image13.3.1; CUDA binary equality between those environments is not asserted. These bounded canaries are not all-layer/full-sanitizer proof. The TP canary content comparison against GPU21 is separate from performance. Original observer chunk hashes and terminal closure are verified; unassigned bytes keep that attribution. Logs, streams and frozen Python files use deterministic gzip, and original/stored hashes are recorded. Collection used read-only observations and fresh archive writes only, with no GPU/HTTP/queue/service or runtime-source mutation.
"""
save('README.md',readme.encode(),'bounded evidence scope');save('collect-evidence.py',read(__file__),'this collector')
record('originals.json',originals.copy(),'original/stored byte provenance')
checks=''.join(f'{sha(raw)}  {name}\n' for name,raw in sorted(items.items()));items['SHA256SUMS']=checks.encode()
OUT.mkdir(parents=False)
for name,raw in items.items():
 path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for name,raw in items.items():assert sha(read(OUT/name))==sha(raw)
print(json.dumps(dict(archive=str(OUT),files=len(items),bytes=sum(map(len,items.values())),summary_sha256=sha(items['result-summary.json']),sums_sha256=sha(items['SHA256SUMS'])),sort_keys=True))
