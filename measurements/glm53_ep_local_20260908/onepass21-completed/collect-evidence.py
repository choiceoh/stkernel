#!/usr/bin/env python3
"""Archive the naturally failed TP A0 run: remote reads and fresh archive only."""
import base64,datetime,gzip,hashlib,importlib.util,json,os,re,subprocess
from pathlib import Path
ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass21-completed'
REV='028f98167376f0a0857c20c7ec89a3505c1a000f';SESSION='eplocalonepass0909v21';TICKET='1788924164133801';PID=133801
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
P=pathlib.Path;root=P('/home/choiceoh/stkernel-ep-onepass-0909-21');job=P('/tmp/glm53-ep-onepass-0909-21')
session='eplocalonepass0909v21';ticket='1788924164133801'
log=P('/home/choiceoh/glm53-logs/fleet/run-logs/cbb5f99d8948d25415f18beb687c09e88198d62405c948c0c7a342e808bd4abd.log')
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
paths=[job/name for name in ('submission.json','submit.exit.json','submit.stdout','submit.stderr','onepass.jsonl','verdicts.jsonl')]+[log,P('/tmp/leg.134506')]
paths += [P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS21'+arm+'.log') for arm in ('A0','B1','A','B2','B3')]
for p in paths:
 if not p.exists():r['absent'].append(str(p));continue
 assert p.is_file() and not p.is_symlink()
 a=p.stat();raw=p.read_bytes();b=p.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==a.st_size and len(raw)<128*2**20
 r['files'][str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mtime_ns':b.st_mtime_ns,'data':base64.b64encode(raw).decode()}
for key,scope in (('earlyoom',['-u','earlyoom']),('kernel',['-k'])):
 argv=['sudo','-n','journalctl',*scope,'--utc','--since','2026-09-09 03:25:00 UTC','--until','2026-09-09 03:34:00 UTC','--no-pager','-o','short-iso']
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
 elif name=='leg.134506':packed('failure/leg.134506.raw.gz',raw,origin)
 else:save('job/'+name,raw,origin)
for key,d in remote['journal'].items():
 raw=base64.b64decode(d.pop('data'));assert sha(raw)==d['sha256'] and len(raw)==d['bytes'];packed('failure/'+key+'.journal.raw.gz',raw,'head: '+' '.join(d['argv']))
record('terminal-capture.json',remote,'read-only terminal fleet/source/own-holder and fixed-window journal capture')
assert not items.get('job/onepass.jsonl',b'').strip() and not items.get('job/verdicts.jsonl',b'').strip(),'unexpected completed record'
leg=gzip.decompress(items['failure/leg.134506.raw.gz'])
assert b'500 Internal Server Error' in leg or b'500 Server Error' in leg or b'HTTP Error 500' in leg
assert not re.search(rb'fixed2K rep=\d+ tokens=1024/1024 decode=',leg),'unexpected fixed result'
early=gzip.decompress(items['failure/earlyoom.journal.raw.gz'])
assert b'2026-09-09T03:30:59' in early and b'141595' in early and b'SIGTERM' in early and b'6011' in early
assert b'2026-09-09T03:31:06' in early and b'exited' in early

manifest_raw=gitfile('build/glm53/manifest.tsv');manifest=('# source_commit='+REV+'\n').encode()+manifest_raw
mounts={line.split('\t')[1]:sha(gitfile('build/glm53/'+line.split('\t')[0])) for line in manifest.decode().splitlines() if line and not line.startswith('#')}
save('source/git-manifest.tsv',manifest_raw,'git '+REV);save('source/frozen-manifest.tsv',manifest,'normal source_commit header + git '+REV)
record('source/mounted-hashes.json',mounts,'frozen build '+REV)
for name in ('glm53_tp_sf6_q0_selftest.py','glm53_ep_local_selftest.py','moe_dynamic_gated_sf6_q0.py','moe_dynamic_gated_sf6.py','moe_dispatch.py','flashinfer_b12x_moe.py'):
 packed('source/'+name+'.gz',gitfile('build/glm53/'+name),'git '+REV+':build/glm53/'+name)
private=Path('/tmp/glm53-onepass21-live-A0-observer');iraw=read(private/'identity.json');identity=json.loads(iraw)
assert (identity['revision'],identity['session'],identity['ticket'],identity['owner_pid'],identity['arm'])==(REV,SESSION,TICKET,str(PID),'A0')
assert set(identity['nodes'])==set(NODES)
for name,d in identity['files'].items():
 stored=read(private/name);original=gzip.decompress(stored)
 assert len(stored)==d['stored_bytes'] and sha(stored)==d['stored_sha256'] and len(original)==d['original_bytes'] and sha(original)==d['original_sha256']
 if '.inspect.' not in name:save('snapshot/'+name,stored,str(private/name),original_bytes=len(original),original_sha256=sha(original))
def fixed(c):return {k:c[k] for k in ('Id','Created','Image','Config','HostConfig','RestartCount')}|{'Mounts':sorted(c['Mounts'],key=lambda m:json.dumps(m,sort_keys=True)),'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
for node,d in identity['nodes'].items():
 before=json.loads(gzip.decompress(read(private/(node+'.inspect.before.json.gz'))))[0]
 after=json.loads(gzip.decompress(read(private/(node+'.inspect.after.json.gz'))))[0]
 assert fixed(before)==fixed(after) and before['State']['Running'] and after['State']['Running']
 assert before['Id']==d['id'] and before['Image']==IMAGE and d['image']==IMAGE
 assert before['State']['StartedAt']==d['started_at'] and before['Created']==d['created_at']
 assert date(d['started_at'])>=date(d['created_at'])>=identity['arm_event']['started_at']
 assert d['source']==dict(manifest_sha256=sha(manifest),mounts=mounts)
 assert d['topology']['enabled'] is False and d['topology']['tensor_parallel_size']==4 and d['topology']['nnodes']==4
 env=dict(x.split('=',1) for x in before['Config']['Env'])
 for key,value in {'VLLM_B12X_EP_ZERO_WEIGHT_MICRO':'0','VLLM_GLM53_EP_PREFILL_LOCAL':'0','VLLM_B12X_EP_WARM_COMPACT':'0','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE':'1','VLLM_GLM53_TP_SF6_Q0':'1'}.items():assert env[key]==value
 assert d['environment_sha256']==sha(json.dumps(env,sort_keys=True,separators=(',',':')).encode())
 assert d['readiness']['graph_finished'] and d['readiness']['candidate'] and len(d['readiness']['tp_sf6_q0_pass_records'])==1
save('snapshot/allowlisted-identity.json',iraw,str(private/'identity.json'))
parser=read(private/'launch-parser.py');assert sha(parser)==identity['parser_sha256'];save('snapshot/launch-parser.py',parser,str(private/'launch-parser.py'))

stream_root=Path('/tmp/glm53-onepass21-streams');eraw=read(stream_root/'events.jsonl');events=[json.loads(line) for line in eraw.splitlines()]
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

helper=Path('/tmp/glm53_extract_onepass21_canary.py');helper_raw=read(helper)
assert sha(helper_raw)=='722b3551fb5c8a0eb04619d2eb7c614b560a5835d056333550e3c58b37581b09'
spec=importlib.util.spec_from_file_location('canary_validator',helper);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
expected={role:dict(path=next(path for path in mounts if Path(path).name==filename),sha256=sha(gitfile('build/glm53/'+filename))) for role,filename in validator.FILES.items()}
expected['stock']=dict(path='/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py',sha256=validator.STOCK_SHA)
canary_root=Path('/tmp/glm53-onepass21-canary-verified');validation_raw=read(canary_root/'validation.json');validation=json.loads(validation_raw)
assert validation['verdict']=='RAW_CANARY_SOURCE_VALIDATED' and validation['revision']==REV and validation['cpu21_result_sha256']==validator.CPU_SHA
assert validation['arm_container_proof'] is False
save('canary/validation-original.json',validation_raw,str(canary_root/'validation.json'))
canary_summary={}
for node in NODES:
 path=canary_root/(node+'.json');raw=read(path);receipt=json.loads(raw);details=validator.validate(receipt,expected)
 line=read(canary_root/(node+'.receipt-line.raw'));prov=validation['nodes'][node]['provenance'];stream=streams[node,'stdout']
 assert stream[prov['raw_offset']:prov['raw_end']]==line and sha(line)==prov['line_sha256']
 assert sha(stream[:prov['raw_end']])==prov['prefix_sha256'] and sha(raw.rstrip(b'\n'))==prov['json_sha256']
 assert stream.count(b'[tp-sf6-q0-selftest] PASS ')==1 and b'[tp-sf6-q0-selftest] FAIL ' not in stream
 log=gzip.decompress(items['snapshot/'+node+'.serving.log.gz']);marker=b'[tp-sf6-q0-selftest] PASS '
 assert log.count(marker)==1
 logged,_=json.JSONDecoder().raw_decode(log.split(marker,1)[1].decode());assert logged==receipt
 assert date(identity['nodes'][node]['started_at'])<=receipt['started_at']<=receipt['completed_at']<=identity['nodes'][node]['capture_finished_at']
 save('canary/'+node+'.json',raw,str(path),raw_stream_match=prov)
 save('canary/'+node+'.receipt-line.raw',line,str(canary_root/(node+'.receipt-line.raw')))
 details.pop('versions');canary_summary[node]=dict(details,verdict='PASS',strict_A0_source_container_bound=True)
record('canary/verified-summary.json',canary_summary,'source-bound 4 TP receipts revalidated and matched to strict A0 logs and original stream bytes')
old=Path('/tmp/glm53-onepass21-canary');rejected=json.loads(read(old/'validation.json'))
assert rejected['verdict']=='INCOMPLETE_OR_REJECTED'
for node in NODES:assert read(old/(node+'.json'))==items['canary/'+node+'.json']
for path in sorted(old.iterdir()):
 assert path.name in {*(node+'.json' for node in NODES),*(node+'.receipt-line.raw' for node in NODES),'validation.json'}
 save('collection-error/'+path.name,read(path),str(path))
for source,name in [('/tmp/glm53-onepass21-submit.json','submission/request.json'),('/tmp/glm53-onepass21-submit-receipt.txt','submission/receipt.json'),('/tmp/glm53_onepass21_observer.py','tools/passive-observer.py'),('/tmp/glm53_onepass21_snapshot.py','tools/snapshot-helper.py'),(str(helper),'tools/canary-extractor-validator.py'),('/tmp/glm53-onepass21-canary-format-check-v2.log','tools/canary-format-check-v2.log')]:save(name,read(source),source)
submission=json.loads(items['submission/receipt.json']);assert (submission['session'],submission['ticket'],submission['pid'])==(SESSION,TICKET,PID)
cpu_path=ROOT/'measurements/glm53_ep_local_20260908/decode21-cpu/result.json';cpu_raw=read(cpu_path);assert sha(cpu_raw)==validator.CPU_SHA
record('source/cpu21-reference.json',dict(path=str(cpu_path.relative_to(ROOT)),sha256=sha(cpu_raw),scope='CPU capsule13.0.3 versus serving image13.3.1; matching kernel/source bytes only'),'existing original CPU21 archive')
summary=dict(schema=1,verdict='A0_INFRASTRUCTURE_FAILURE_BEFORE_FIXED_DECODE',revision=REV,session=SESSION,ticket=TICKET,supervisor_pid=PID,
 completed_arms=[],onepass_records=0,unrun_arms=['B1','A','B2','B3'],payload_returncode=1,returncode=1,own_holder_after=False,
 fixed_decode_tok_s=[],fixed_pooled_tok_s=None,quality=None,korean=None,proof_ok=None,
 warm_prefill_stdout_only=dict(tokens_2k_per_s=2431,tokens_32k_per_s=3022,request_128k_completion_tokens=0,request_128k_elapsed_s=24.10,valid_128k_measurement=False),
 termination=dict(source='head earlyoom journal',action='SIGTERM',worker_pid=141595,at_utc='2026-09-09T03:30:59Z',available_memory_mib=6011,available_percent=4.90,worker_exited_utc='2026-09-09T03:31:06Z'),
 full_canary_pass_receipts=4,canary_cases=16,candidate_comparisons=96,stock_control_comparisons=96,q0_comparisons=96,
 collection_error='initial safe literal parser rejected torch.int32 repr; identical original PASS receipts preserved then revalidated',
 performance_acceptance=False,matched_baseline=False,adoption_acceptance=False,full_sanitizer_acceptance=False,
 recovery_policy=remote['fleet_after'].get('recovery_policy'),recovery_deferred=remote['fleet_after'].get('recovery_deferred'))
# These are only quoted rounded stdout observations, tied to the raw leg.
for text in (b'2431',b'3022',b'24.10'):assert text in leg
record('result-summary.json',summary,'terminal records, exact raw leg, earlyoom journal and source-bound TP canary receipts')
readme='''# onepass21: TP canary PASS, serving run ended before fixed decode

Frozen source `028f98167376f0a0857c20c7ec89a3505c1a000f`; normal fleet session `eplocalonepass0909v21`, ticket `1788924164133801`, supervisor `133801`. A0 was TP4/4 with EP/local/warm/zero disabled, Q0 enabled and skip-unused-graph enabled. All four strict startup snapshots preserve source/image/container/config identity and graph completion.

All four ranks published full TP Q0 canary PASS receipts: four fixtures per rank, initial/changed inputs with B1/B2/B3 controls and eager/current-graph/side-graph candidates (96 candidate comparisons and 96 control comparisons). Same-address changed inputs, real packed weight backing before/after, Q0 route metadata and bounded A/SFA samples were checked. Receipts exactly match frozen source, CPU21 mounted source hashes, strict A0 logs and original stream byte ranges. CPU used isolated CUDA bindings13.0.3 while serving used image13.3.1; runtime binary equality is not asserted. The canary does not cover all layers, full sanitizer behavior or throughput.

The head earlyoom journal recorded memory6011MiB (4.90%) and SIGTERM to worker141595 at03:30:59UTC, then exit at03:31:06UTC. Original journal windows and all four terminal passive streams are preserved. Rounded onepass stdout showed 2K/32K warm prefill2431/3022tok/s, then a128K request with zero completion tokens in24.10s and fixed-request HTTP500. The empty128K response is not a valid prefill result. No fixed-decode result, completed onepass record, aggregate quality/Korean score, matched baseline or adoption verdict exists. This is an infrastructure failure, not a measured decode regression or numerical rejection.

The first offline receipt extraction safely rejected `torch.int32` cache-key syntax. Its identical four PASS receipts and rejection report remain under `collection-error/`; the corrected strict AST parser only normalizes that exact dtype attribute. This collection error is separate from numerics.

Payload and supervisor returned1; source remained frozen/clean and the owned supervisor/holder were absent. Passive streams ended normally. Recovery follows the normal idle policy; this archive does not claim public restoration. Only allowlisted snapshot identity and safe logs are archived, excluding private Env/Cmd/raw inspect. Raw logs/streams and frozen Python source use deterministic gzip; `originals.json` preserves original/stored hashes and sizes. No GPU, HTTP, queue, deployment or source mutation was performed by this collector.
'''
save('README.md',readme.encode(),'bounded evidence scope');save('collect-evidence.py',read(__file__),'this collector')
record('originals.json',originals.copy(),'original/stored byte provenance')
checks=''.join(f'{sha(raw)}  {name}\n' for name,raw in sorted(items.items()));items['SHA256SUMS']=checks.encode()
OUT.mkdir(parents=False)
for name,raw in items.items():
 path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for name,raw in items.items():assert sha(read(OUT/name))==sha(raw)
print(json.dumps(dict(archive=str(OUT),files=len(items),bytes=sum(map(len,items.values())),summary_sha256=sha(items['result-summary.json']),sums_sha256=sha(items['SHA256SUMS'])),sort_keys=True))
