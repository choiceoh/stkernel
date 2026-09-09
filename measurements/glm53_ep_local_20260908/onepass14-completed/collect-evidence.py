#!/usr/bin/env python3
"""Archive closed A0 observations; only remote reads and a fresh local archive."""
import base64
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass14-completed'
REV='c3696a76c6674ad38e01967fa587b8d31a6d121e'
SESSION='eplocalonepass0909v14'; TICKET='17889166573854513'; PID=3854513
NODES=('local','10.10.10.1','10.10.10.3','10.10.10.4')
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
items={}; originals={}
def read(path):
    path=Path(path); a=path.stat(); raw=path.read_bytes(); b=path.stat()
    assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
    assert len(raw)==a.st_size and len(raw)<128*2**20
    return raw

def save(name,raw,origin,**meta):
    assert name not in items and not Path(name).is_absolute() and '..' not in Path(name).parts
    items[name]=raw; originals[name]=dict(origin=origin,bytes=len(raw),sha256=sha(raw),**meta)
def packed(name,raw,origin):
    save(name,gzip.compress(raw,mtime=0),origin,original_bytes=len(raw),original_sha256=sha(raw))
def record(name,value,origin):save(name,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode(),origin)
assert not OUT.exists(),'refuse existing archive'
REMOTE=r'''
import base64,hashlib,json,os,pathlib,subprocess,time
P=pathlib.Path
root=P('/home/choiceoh/stkernel-ep-onepass-0909-14');job=P('/tmp/glm53-ep-onepass-0909-14')
session='eplocalonepass0909v14';ticket='17889166573854513'
log=P('/home/choiceoh/glm53-logs/fleet/run-logs/5b492c9f661652e31f4f60aedcba8bea339b973d116cbba3f1526b6e0e7a1a4c.log')
def identity():
 def git(*args):return subprocess.check_output(['git','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).decode().strip()
 return dict(head=git('rev-parse','HEAD'),status=git('status','--porcelain'))
def fleet():
 d=json.loads(subprocess.check_output(['bash',str(root/'bench/fleet.sh'),'show',session,'--ticket',ticket,'--json']))
 keys=('session','ticket','state','phase','started_at','payload_finished_at','finished_at','payload_returncode','returncode','outcome','log_path','recovery_policy','recovery_deferred','supervisor_alive','payload_seconds')
 return {k:d[k] for k in keys if k in d}
def own():
 p=P('/home/choiceoh/glm53-logs/fleet/holder'); return p.exists() and p.read_text().split('|',1)[0]==session
r=dict(captured_at=time.time(),source_before=identity(),fleet_before=fleet(),own_holder_before=own(),files={},absent=[])
f=r['fleet_before'];assert f['session']==session and str(f['ticket'])==ticket and f['log_path']==str(log)
assert f['phase']=='finished' and f['state']=='cancelled' and f['payload_returncode']==143 and f['returncode']==143 and not f['supervisor_alive'] and not r['own_holder_before']
paths=[job/name for name in ('submission.json','submit.exit.json','submit.stdout','submit.stderr','cancel-request.json','pre-cancel-records.jsonl','onepass.jsonl','verdicts.jsonl')]+[log]
paths += [P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS14'+arm+'.log') for arm in ('A0','B1','A','B2','B3')]
for path in paths:
 if not path.exists():r['absent'].append(str(path));continue
 a=path.stat();raw=path.read_bytes();b=path.stat()
 assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==a.st_size and len(raw)<128*2**20
 r['files'][str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mtime_ns':b.st_mtime_ns,'data':base64.b64encode(raw).decode()}
r.update(source_after=identity(),fleet_after=fleet(),own_holder_after=own())
assert r['source_before']==r['source_after'] and r['fleet_before']==r['fleet_after'] and not r['own_holder_after']
print(json.dumps(r))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','choiceoh@srv2','python3 -B -'],input=REMOTE.encode(),capture_output=True,timeout=60)
assert p.returncode==0,p.stderr.decode(); remote=json.loads(p.stdout)
assert remote['source_before']==dict(head=REV,status='')
for origin,d in remote['files'].items():
    raw=base64.b64decode(d.pop('data'));assert sha(raw)==d['sha256'] and len(raw)==d['bytes']
    name=Path(origin).name
    if name.startswith('boot-'):packed('boot/'+name+'.gz',raw,origin)
    elif name==Path(remote['fleet_after']['log_path']).name:packed('fleet/terminal-run.log.gz',raw,origin)
    else:save('job/'+name,raw,origin)
record('terminal-capture.json',remote,'read-only terminal fleet/source/own-holder verification')
raw=items['job/onepass.jsonl'];assert raw==items['job/pre-cancel-records.jsonl']
rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)==1
row=rows[0];assert row['name']=='EPONEPASS14A0' and row['session']==SESSION and row['git']==REV[:8]
cancel=json.loads(items['job/cancel-request.json']);assert (cancel['session'],cancel['ticket'],cancel['pid'],cancel['signal'])==(SESSION,TICKET,PID,'SIGTERM')
assert cancel['records_sha256']==sha(raw)
verdicts=[json.loads(line) for line in items['job/verdicts.jsonl'].splitlines()]
assert len(verdicts)==1 and verdicts[0]['base'] is None and verdicts[0]['status']=='incomplete'
assert row['quality']=={'ok':18,'total':18} and row['korean']['dirty']==0 and row['korean']['n']==8 and row['proof_ok']=='4/4'
assert len(row['requests'])==8 and row['traffic']['issues']==[]

def gitfile(name):return subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT)
git_manifest=gitfile('build/glm53/manifest.tsv')
manifest=('# source_commit='+REV+'\n').encode()+git_manifest
mounts={l.split('\t')[1]:sha(gitfile('build/glm53/'+l.split('\t')[0])) for l in manifest.decode().splitlines() if l and not l.startswith('#')}
save('source/git-manifest.tsv',git_manifest,'git '+REV);save('source/frozen-manifest.tsv',manifest,'normal deploy source_commit header plus exact git '+REV+' manifest');record('source/mounted-hashes.json',mounts,'git '+REV+' build bytes')
for name in ('glm53_ep_local_selftest.py','flashinfer_b12x_moe.py'):
    packed('source/'+name+'.gz',gitfile('build/glm53/'+name),'git '+REV+':build/glm53/'+name)
private=Path('/tmp/glm53-onepass14-live-A0-observer'); iraw=read(private/'identity.json'); identity=json.loads(iraw)
assert (identity['revision'],identity['session'],identity['ticket'],identity['owner_pid'],identity['arm'])==(REV,SESSION,TICKET,str(PID),'A0')
assert set(identity['nodes'])==set(NODES)
for name,d in identity['files'].items():
    stored=read(private/name);original=gzip.decompress(stored)
    assert len(stored)==d['stored_bytes'] and sha(stored)==d['stored_sha256'] and len(original)==d['original_bytes'] and sha(original)==d['original_sha256']
    if '.inspect.' not in name:save('snapshot/'+name,stored,str(private/name),original_bytes=len(original),original_sha256=sha(original))
def fixed(c):
    return {k:c[k] for k in ('Id','Created','Image','Config','HostConfig','RestartCount')} | {'Mounts':sorted(c['Mounts'],key=lambda m:json.dumps(m,sort_keys=True)),'StartedAt':c['State']['StartedAt'],'Pid':c['State']['Pid']}
def date(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
for node,d in identity['nodes'].items():
    before=json.loads(gzip.decompress(read(private/(node+'.inspect.before.json.gz'))))[0]
    after=json.loads(gzip.decompress(read(private/(node+'.inspect.after.json.gz'))))[0]
    assert fixed(before)==fixed(after) and before['State']['Running'] and after['State']['Running']
    assert before['Id']==d['id'] and before['Image']==IMAGE and d['image']==IMAGE
    assert before['State']['StartedAt']==d['started_at'] and before['Created']==d['created_at']
    assert date(d['started_at'])>=date(d['created_at'])>=identity['arm_event']['started_at']
    assert d['source']==dict(manifest_sha256=sha(manifest),mounts=mounts) and d['topology']['enabled']
    env=dict(x.split('=',1) for x in before['Config']['Env'])
    assert all(env[k]=='1' for k in ('VLLM_B12X_EP_NO_DUMMY','VLLM_B12X_EP_ZERO_WEIGHT_MICRO','VLLM_GLM53_EP_PREFILL_LOCAL','VLLM_B12X_EP_WARM_COMPACT','VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE'))
    assert d['environment_sha256']==sha(json.dumps(env,sort_keys=True,separators=(',',':')).encode())
assert row['boot_id']==identity['nodes']['local']['id']+'|'+identity['nodes']['local']['started_at']
assert row['overlay']==sha(manifest)[:12]
save('snapshot/allowlisted-identity.json',iraw,str(private/'identity.json'))
parser=read(private/'launch-parser.py');assert sha(parser)==identity['parser_sha256'];save('snapshot/launch-parser.py',parser,str(private/'launch-parser.py'))

stream_root=Path('/tmp/glm53-onepass14-streams');eraw=read(stream_root/'events.jsonl');events=[json.loads(line) for line in eraw.splitlines()]
assert events[-1]['kind']=='observer_finished' and events[-1]['attempted_arms']==['A0'] and events[-1]['owned_go_seen']
ends={};chunks={};streams={}
for e in events:
    if e['kind']!='chunk':continue
    key=e['node'],e['channel'];assert e['offset']==ends.get(key,0)
    ends[key]=e['offset']+e['bytes'];chunks.setdefault(key,[]).append(e)
for key,end in ends.items():
    path=stream_root/(key[0]+'.'+key[1]+'.raw');r=read(path);assert len(r)==end
    for e in chunks[key]:assert sha(r[e['offset']:e['offset']+e['bytes']])==e['sha256']
    streams[key]=r;packed('streams/'+path.name+'.gz',r,str(path))
assert all((n,'stdout') in streams for n in NODES)
assert not any(b'[ep-local-selftest] PASS ' in r or b'[ep-local-selftest] FAIL ' in r for (n,ch),r in streams.items() if ch=='stdout')
packed('streams/events.jsonl.gz',eraw,str(stream_root/'events.jsonl'))
observer_pid=int(read(stream_root/'observer.pid'))
try:os.kill(observer_pid,0)
except ProcessLookupError:alive=False
else:alive=True
assert not alive,'observer still alive'
record('streams/closure.json',dict(observer_pid=observer_pid,alive=False,events=[e for e in events if e['kind'] in ('stream_end','observer_stop','observer_finished')],scope='All original chunks/offsets/hashes verified; unassigned attribution retained. A0 was strictly captured; later B1 boot bytes are not candidate traffic.'),'closed passive observer')
inferred_raw=read('/tmp/glm53_onepass14_inferred_canary_gate.json');inferred=json.loads(inferred_raw)
assert inferred['revision']==REV and inferred['private_snapshot_sha256']==sha(iraw) and not inferred['full_pass_json_captured']
for node,d in inferred['nodes'].items():
    assert d['container_id']==identity['nodes'][node]['id']
    for line in d['markers'].values():assert line.encode() in streams[node,'stdout']
save('canary/inferred-gate-original.json',inferred_raw,'/tmp/glm53_onepass14_inferred_canary_gate.json')
for source,name in [('/tmp/glm53-onepass14-submit.json','submission/request.json'),('/tmp/glm53-onepass14-submit-receipt.txt','submission/receipt.json'),('/tmp/glm53_onepass14_observer.py','tools/passive-observer.py'),('/tmp/glm53_onepass14_snapshot.py','tools/snapshot-helper.py')]:save(name,read(source),source)
submission=json.loads(items['submission/receipt.json']);assert (submission['session'],submission['ticket'],submission['pid'])==(SESSION,TICKET,PID)
summary=dict(schema=1,verdict='A0_OBSERVED_CANCELLED_BEFORE_MATCHED_BASELINE',revision=REV,session=SESSION,ticket=TICKET,supervisor_pid=PID,
    completed_arms=['A0'],onepass_records=1,requests=8,boot_only_arms=['B1'],unrun_arms=['A','B2','B3'],payload_returncode=143,returncode=143,own_holder_after=False,
    fixed_decode_tok_s=cancel['fixed_decode_tok_s'],prefill=row['prefill'],quality=row['quality'],korean=row['korean'],proof=row['proof'],proof_ok=row['proof_ok'],
    canonical_verdict=verdicts[0],cold_compile=row['cold_compile'],mandatory_canary_gate='PASS inferred from mandatory fail-closed source and post-load graph progress on all four strictly attributed workers',
    full_canary_pass_receipts=0,per_case_numeric_metrics_available=False,case_input_hash_continuity_verified=False,effective_logger_level_verified=False,
    performance_acceptance=False,matched_baseline=False,adoption_acceptance=False,full_sanitizer_acceptance=False,
    recovery_policy=remote['fleet_after']['recovery_policy'],recovery_deferred=remote['fleet_after']['recovery_deferred'])
record('result-summary.json',summary,'unchanged canonical A0 record, cancellation and strict capture evidence')
readme='''# onepass14: A0 observed, comparison cancelled

Frozen source `c3696a76c6674ad38e01967fa587b8d31a6d121e`, session `eplocalonepass0909v14`, ticket `17889166573854513`, supervisor `3854513`.

A0 completed the unchanged canonical onepass workload: fixed decode 54.4293 / 57.5863 / 64.2312 tok/s, quality 18/18, Korean contamination 0/8 and knob proof 4/4. Full original prefill, TTFT, decode intervals and eight request records remain in `job/onepass.jsonl`; the pre-cancel bytes match it exactly. There is no matched baseline. The canonical verdict is incomplete / no baseline on this build. These observations do not establish a measured regression percentage or performance/default-adoption acceptance.

The owner stopped the remaining arms after the A0 result. B1 had started booting but produced no measurement record; A/B2/B3 did not run. Cancellation is pinned to the exact supervisor/session/ticket and original record hash. Payload and supervisor returned 143; the owned holder was absent at terminal capture. Recovery was deferred to the normal idle controller; this archive does not claim public restoration completed.

All four A0 containers were strictly captured before traffic finished. Original image, config, full mounts, start/creation identity and frozen mounted bytes were rechecked locally. Raw inspect/Env/Cmd remains in the private `/tmp` capture and is not copied here; the allowlisted identity records their hashes. The original observer chunk hashes and offsets were checked through closure. Streams include explicitly unassigned and later B1-boot bytes, which must not be counted as A0 activity.

The mandatory startup self-test returned successfully on every worker by inference from its exact fail-closed source, enabled flags, and the same worker's subsequent model-load and graph-completion markers. No full PASS JSON was published by the INFO-level logger. Individual numerical values, input-hash continuity, and the effective runtime logger level therefore remain unobserved. The original inference receipt contains an earlier, unapplied WARNING-level logging proposal; it is historical evidence only. A later source change uses explicit flushed stdout for future positive receipts and was not applied to this frozen run.

Logs, streams and frozen Python source use deterministic gzip without content normalization. `originals.json` preserves original/stored byte counts and SHA256; `SHA256SUMS` covers the archive. Collection performed only read-only remote observations and fresh local archive writes, with no GPU/test/queue/deployment action.
'''
save('README.md',readme.encode(),'bounded evidence explanation')
save('collect-evidence.py',read(__file__),'this collector')
record('originals.json',originals.copy(),'original/stored byte provenance')
items['SHA256SUMS']=''.join(sha(raw)+'  '+name+'\n' for name,raw in sorted(items.items())).encode()
OUT.mkdir()
for name,raw in items.items():
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
print(json.dumps(dict(archive=str(OUT),files=len(items),bytes=sum(map(len,items.values())),summary_sha256=sha(items['result-summary.json']),sha256sums_sha256=sha(items['SHA256SUMS']),verdict=summary['verdict'])))
