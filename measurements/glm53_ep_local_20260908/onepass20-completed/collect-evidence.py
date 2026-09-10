#!/usr/bin/env python3
"""Archive closed A0 observations; only remote reads and a fresh local archive."""
import base64
import datetime
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess

ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=ROOT/'measurements/glm53_ep_local_20260908/onepass20-completed'
REV='82d81d65b6a74e09abd9be0a8f05508e4855f6f1'
SESSION='eplocalonepass0909v20'; TICKET='178892234520953'; PID=20953
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
root=P('/home/choiceoh/stkernel-ep-onepass-0909-20');job=P('/tmp/glm53-ep-onepass-0909-20')
session='eplocalonepass0909v20';ticket='178892234520953'
log=P('/home/choiceoh/glm53-logs/fleet/run-logs/1f424a07447b41543d9790ac53f288e1ea5f81806c7182664316655dd5b5722e.log')
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
paths += [job/'failfast-first-fixed-A0'/name for name in ('leg.raw','pre-signal.json','result.json')]
paths += [P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS20'+arm+'.log') for arm in ('A0','B1','A','B2','B3')]
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
    if '/failfast-first-fixed-A0/' in origin:save('job/failfast-first-fixed-A0/'+name,raw,origin)
    elif name.startswith('boot-'):packed('boot/'+name+'.gz',raw,origin)
    elif name==Path(remote['fleet_after']['log_path']).name:packed('fleet/terminal-run.log.gz',raw,origin)
    else:save('job/'+name,raw,origin)
record('terminal-capture.json',remote,'read-only terminal fleet/source/own-holder verification')
raw=items.get('job/onepass.jsonl',b'')
rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)<=1
row=rows[0] if rows else None
verdicts=[json.loads(line) for line in items.get('job/verdicts.jsonl',b'').splitlines()]
fixed_requests=[];fixed_pooled_tok_s=None;partial=None;cancel=None
if row is not None:
    assert row['name']=='EPONEPASS20A0' and row['session']==SESSION and row['git']==REV[:8]
    assert raw==items['job/pre-cancel-records.jsonl']
    cancel=json.loads(items['job/cancel-request.json'])
    assert (cancel['session'],str(cancel['ticket']),cancel['owner_pid'],str(cancel['owner_start_tick']))==(SESSION,TICKET,PID,'42473355')
    assert cancel['method']=='SIGTERM exact owned fleet_boot supervisor' and cancel['arm']=='B1'
    assert cancel['records_sha256']==sha(raw)
    assert len(verdicts)==1 and verdicts[0]['cand']=='EPONEPASS20A0' and verdicts[0]['base'] is None and verdicts[0]['status']=='incomplete'
    assert row['quality']['total']==18 and 0<=row['quality']['ok']<=18
    assert row['korean']['n']==8 and 0<=row['korean']['dirty']<=8
    assert len(row['requests'])==8 and isinstance(row['traffic']['issues'],list)
    fixed_requests=[r for r in row['requests'] if r.get('fixed_decode')]
    assert len(fixed_requests)==3 and [r['rep'] for r in fixed_requests]==[0,1,2]
    assert all(r['completion_tokens']==r['min_tokens']==r['max_tokens']==1024 and r['finish_reason']=='length' for r in fixed_requests)
    assert [r['decode_tok_s'] for r in fixed_requests]==cancel['fixed_decode_tok_s']
    fixed_pooled_tok_s=sum(r['completion_tokens']-1 for r in fixed_requests)/sum(r['decode_s'] for r in fixed_requests)
    assert abs(fixed_pooled_tok_s-cancel['fixed_pooled_tok_s'])<1e-10
else:
    assert not verdicts and not items.get('job/pre-cancel-records.jsonl',b'')
    base='job/failfast-first-fixed-A0/'
    partial=json.loads(items[base+'result.json']);prior=json.loads(items[base+'pre-signal.json']);leg=items[base+'leg.raw']
    assert (partial['session'],partial['ticket'],partial['owner_pid'],partial['owner_start_tick'],partial['revision'])==(SESSION,TICKET,PID,'42473355',REV)
    assert partial['action']=='SIGTERM_SENT' and partial['signal_sent'] is True
    assert prior['action']=='SIGNAL_REQUESTED' and prior['signal_sent'] is False
    assert {k:v for k,v in partial.items() if k not in ('action','signal_sent','signalled_at')}=={k:v for k,v in prior.items() if k not in ('action','signal_sent')}
    assert partial['arm']['arm']=='A0' and partial['completed_records']==[] and partial['records_sha256']==sha(b'')
    assert partial['threshold_tok_s']==65. and 0<partial['observed']['decode_tok_s']<65.
    assert partial['observed']['sha256']==sha(leg) and partial['observed']['bytes']==len(leg)
    observed=partial['observed'];line=observed['line'].encode();assert leg[observed['offset']:observed['offset']+len(line)]==line
    assert sha(line)==observed['line_sha256']
    matches=list(re.finditer(rb'(?m)^\s*fixed2K rep=(\d+) tokens=1024/1024 decode=([0-9]+\.[0-9]+) tok/s[^\r\n]*\n',leg))
    assert len(matches)==1 and matches[0][1]==b'0' and float(matches[0][2])==observed['decode_tok_s']
    chain=partial['processes'];assert len(chain)>=3 and chain[-1]['pid']==PID and chain[-1]['start']=='42473355'
    assert all(p['cwd']=='/home/choiceoh/stkernel-ep-onepass-0909-20' for p in chain)
    assert all(a['ppid']==b['pid'] for a,b in zip(chain,chain[1:]))
    assert Path(observed['path'])==Path('/tmp')/('leg.'+str(chain[1]['pid']))
    assert observed['observed_at']<=partial['signalled_at']<=remote['fleet_after']['finished_at']
    assert partial['signalled_at']-observed['observed_at']<5.

def gitfile(name):return subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT)
git_manifest=gitfile('build/glm53/manifest.tsv')
manifest=('# source_commit='+REV+'\n').encode()+git_manifest
mounts={l.split('\t')[1]:sha(gitfile('build/glm53/'+l.split('\t')[0])) for l in manifest.decode().splitlines() if l and not l.startswith('#')}
save('source/git-manifest.tsv',git_manifest,'git '+REV);save('source/frozen-manifest.tsv',manifest,'normal deploy source_commit header plus exact git '+REV+' manifest');record('source/mounted-hashes.json',mounts,'git '+REV+' build bytes')
for name in ('glm53_ep_local_selftest.py','flashinfer_b12x_moe.py','moe_micro_kernel.py','moe_dispatch.py','moe_dynamic_ep_local.py','glm53_ep_route_remap.py'):
    packed('source/'+name+'.gz',gitfile('build/glm53/'+name),'git '+REV+':build/glm53/'+name)
private=Path('/tmp/glm53-onepass20-live-A0-observer'); iraw=read(private/'identity.json'); identity=json.loads(iraw)
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
if row is not None:
    assert row['boot_id']==identity['nodes']['local']['id']+'|'+identity['nodes']['local']['started_at']
    assert row['overlay']==sha(manifest)[:12]
else:
    assert partial['arm']['started_at']==identity['arm_event']['started_at']
    assert partial['arm']['line']==identity['arm_event']['line']
    assert partial['arm']['offset']==identity['arm_event']['offset']
save('snapshot/allowlisted-identity.json',iraw,str(private/'identity.json'))
parser=read(private/'launch-parser.py');assert sha(parser)==identity['parser_sha256'];save('snapshot/launch-parser.py',parser,str(private/'launch-parser.py'))

stream_root=Path('/tmp/glm53-onepass20-streams');eraw=read(stream_root/'events.jsonl');events=[json.loads(line) for line in eraw.splitlines()]
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
assert not any(b'[ep-local-selftest] FAIL ' in r for (n,ch),r in streams.items() if ch=='stdout')
packed('streams/events.jsonl.gz',eraw,str(stream_root/'events.jsonl'))
observer_pid=int(read(stream_root/'observer.pid'))
try:os.kill(observer_pid,0)
except ProcessLookupError:alive=False
else:alive=True
assert not alive,'observer still alive'
record('streams/closure.json',dict(observer_pid=observer_pid,alive=False,events=[e for e in events if e['kind'] in ('stream_end','observer_stop','observer_finished')],scope='All original chunks/offsets/hashes verified; unassigned attribution retained. A0 was strictly captured; bytes without strict attribution remain explicitly unassigned.'),'closed passive observer')
continuity_raw=read('/tmp/glm53-onepass20-continuity.json');continuity=json.loads(continuity_raw)
provided_raw=read('/tmp/glm53-onepass20-mounted-hashes.json');provided=json.loads(provided_raw)
assert all(mounts.get(k)==v for k,v in provided.items())
assert continuity['new_source_manifest_sha256']==sha(provided_raw)
assert continuity['verdict']=='MATCH_AND_CANARY_PASS' and continuity['identity_match'] and continuity['all_four_new_canaries_pass']
old_root=ROOT/'measurements/glm53_ep_local_20260908/onepass9-startup-failed'
assert continuity['old_source_manifest_sha256']==sha(read(old_root/'source/mounted-hashes.json'))
save('canary/continuity-original.json',continuity_raw,'/tmp/glm53-onepass20-continuity.json')
save('source/continuity-input-mounted-hashes.json',provided_raw,'/tmp/glm53-onepass20-mounted-hashes.json')
cases=('concentrated6912','balanced4096','remote4096','duplicate4096','zeros4097','balanced8192','short6')
canary_summary={}
def tensor_equal(a,b):return all(a[k]==b[k] for k in ('dtype','shape','sha256'))
for node in NODES:
    path=Path('/tmp/glm53-onepass20-canary')/(node+'.json');rraw=read(path);r=json.loads(rraw)
    old_raw=read(old_root/'failures'/path.name);old=json.loads(old_raw);d=continuity['nodes'][node]
    assert sha(rraw)==d['new_receipt_sha256'] and sha(old_raw)==d['old_receipt_sha256']
    assert d['identity_match'] and d['within_boot_storage_reused'] and d['difference_count']==0
    assert r['verdict']=='PASS' and r['seed']==905308 and r['scratch_cache_restored'] and r['caller_scale_storage_unchanged']
    assert not r['performance_acceptance'] and not r['full_sanitizer_acceptance'] and 'error' not in r and 'cleanup_error' not in r
    assert tuple(c['case'] for c in r['cases'])==cases
    assert all(c['verdict']=='PASS' and c['phase']=='complete' and len(c['candidate'])==6 and len(c['controls'])==2 and len(c['inputs'])==2 for c in r['cases'])
    for c in r['cases']:
        assert all(v['bad_rows']==0 for v in c['candidate'])
        assert all(v['bad_rows']==0 for control in c['controls'] for v in control)
    assert set(r['weights'])==set(old['weights'])
    assert all(tensor_equal(t,old['weights'][name]) for name,t in r['weights'].items())
    first=r['cases'][0];prior=old['cases'][0]
    for stage in range(2):
        assert set(first['inputs'][stage])==set(prior['inputs'][stage])
        for name,t in first['inputs'][stage].items():assert tensor_equal(t,prior['inputs'][stage][name])
    assert all(first['inputs'][0][name]['data_ptr']==first['inputs'][1][name]['data_ptr'] for name in first['inputs'][0])
    for role,src in r['provenance']['source'].items():
        expected='993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445' if role=='stock' else mounts[src['path']]
        assert src['sha256']==expected
    assert all("'glm53_ep_micro_scatter_fp32_v1', 'glm53_ep_micro_direct_scatter_v1', 'glm53_ep_micro_shared_fc1_a_v1', 'glm53_ep_micro_m16_v1')" in key for key in r['micro_keys']['candidate']) and len(r['micro_keys']['candidate'])==1
    assert len(r['micro_keys']['control'])==1 and r['micro_keys']['control'][0].endswith("'glm53_ep_micro_scatter_fp32_v1')")
    prep=r['cases'][-1]['preparation'];assert len(prep)==4
    for index,entry in enumerate(prep):
        values=entry if index in (0,3) else entry['tensors'];assert set(values)=={'X','ids','weights'}
        for v in values.values():assert v['exact'] and v['actual_sha256']==v['reference_sha256']
    assert prep[1]['input_weights_dtype']=='torch.float32' and prep[2]['input_weights_dtype']=='torch.float16'
    stream=streams[node,'stdout'];matches=[];offset=0
    for line in stream.splitlines(keepends=True):
        if b'[ep-local-selftest] PASS ' in line:
            got,_=json.JSONDecoder().raw_decode(line.split(b'[ep-local-selftest] PASS ',1)[1].decode())
            if got==r:matches.append(dict(line_sha256=sha(line),stream_byte_offset=offset))
        offset+=len(line)
    assert len(matches)==1
    log=gzip.decompress(items['snapshot/'+node+'.serving.log.gz'])
    marker=b'[ep-local-selftest] PASS '; assert marker in log
    assert str(r['pid']).encode() in log.split(marker,1)[0].splitlines()[-1]
    logged,_=json.JSONDecoder().raw_decode(log.split(marker,1)[1].decode());assert logged==r
    assert date(identity['nodes'][node]['started_at'])<=r['started_at']<=r['completed_at']<=identity['nodes'][node]['capture_finished_at']
    save('canary/'+path.name,rraw,str(path),raw_stream_match=matches[0])
    canary_summary[node]=dict(pid=r['pid'],verdict='PASS',source_and_live_snapshot_bound=True,old_onepass9_weight_and_concentrated_input_identity=True,within_boot_concentrated_storage_reused=True,
        case_pass_count=7,candidate_comparisons=42,short6_preparation_variants=4,raw_stream_match=matches[0],cases=[dict(case=c['case'],max_row_relative_abs=max(v['max_row_relative_abs'] for v in c['candidate']),max_row_relative_l2=max(v['max_row_relative_l2'] for v in c['candidate']),duration_s=c['duration_s']) for c in r['cases']])
record('canary/verified-summary.json',canary_summary,'full original PASS receipts bound to frozen source, strict snapshots, original streams and old failure data')
for source,name in [('/tmp/glm53-onepass20-submit.json','submission/request.json'),('/tmp/glm53-onepass20-submit-receipt.txt','submission/receipt.json'),('/tmp/glm53_onepass20_observer.py','tools/passive-observer.py'),('/tmp/glm53_onepass20_snapshot.py','tools/snapshot-helper.py')]:save(name,read(source),source)
submission=json.loads(items['submission/receipt.json']);assert (submission['session'],submission['ticket'],submission['pid'])==(SESSION,TICKET,PID)
helper=read('/tmp/glm53_onepass20_failfast.py');save('tools/failfast-observer.py',helper,'/tmp/glm53_onepass20_failfast.py')
if partial is not None:assert partial['helper_sha256']==sha(helper),'running failfast helper bytes changed; archive exact executed copy instead'
summary=dict(schema=1,verdict='A0_OBSERVED_CANCELLED_BEFORE_MATCHED_BASELINE' if row is not None else 'A0_CANCELLED_AFTER_FIRST_FIXED_RESULT',
    revision=REV,session=SESSION,ticket=TICKET,supervisor_pid=PID,
    completed_arms=['A0'] if row is not None else [],onepass_records=len(rows),requests=8 if row is not None else None,
    boot_only_arms=['B1'] if row is not None else [],unrun_arms=['A','B2','B3'] if row is not None else ['B1','A','B2','B3'],
    payload_returncode=143,returncode=143,own_holder_after=False,
    fixed_decode_tok_s=[r['decode_tok_s'] for r in fixed_requests] if row is not None else [partial['observed']['decode_tok_s']],
    fixed_pooled_tok_s=fixed_pooled_tok_s,rounded_stdout_only=row is None,
    prefill=row['prefill'] if row is not None else None,quality=row['quality'] if row is not None else None,korean=row['korean'] if row is not None else None,
    proof_ok=row['proof_ok'] if row is not None else None,canonical_verdict=verdicts[0] if verdicts else None,
    full_canary_pass_receipts=4,canary_case_passes=28,candidate_comparisons=168,
    old_onepass9_weight_and_concentrated_input_identity=True,within_boot_concentrated_storage_reused=True,
    performance_acceptance=False,matched_baseline=False,adoption_acceptance=False,full_sanitizer_acceptance=False,
    complete_request_hashes_available=row is not None,
    recovery_policy=remote['fleet_after']['recovery_policy'],recovery_deferred=remote['fleet_after']['recovery_deferred'])
record('result-summary.json',summary,'actual completed record or exact first-fixed cancellation, full canary receipts and strict identity')
if row is not None:
    observation='A0 completed the canonical workload. Fixed decode values were '+str(summary['fixed_decode_tok_s'])+' tok/s, pooled '+str(fixed_pooled_tok_s)+' tok/s; quality '+str(row['quality'])+', Korean '+str(row['korean'])+' and proof '+str(row['proof_ok'])+'. The full eight-request record and the byte-identical pre-cancel copy are preserved. B1 had begun booting but produced no measurement. The canonical verdict remains incomplete / no baseline on this build.'
else:
    observation='The first fixed1024 result was '+str(summary['fixed_decode_tok_s'][0])+' tok/s in rounded canonical stdout. The reviewed observer sent SIGTERM to the exact supervisor through pidfd after checking its reservation, candidate arm, process ancestry and fresh first-result bytes. The raw leg, pre-signal receipt and sent receipt are preserved. No completed onepass JSON, aggregate quality/Korean result, full request/output hash record or matched baseline exists. Later repetitions were not accepted as evidence.'
readme='# onepass20: terminal candidate observations, no matched baseline\n\nFrozen source `82d81d65b6a74e09abd9be0a8f05508e4855f6f1`, normal fleet session `eplocalonepass0909v20`, ticket `178892234520953`, supervisor `20953`, start tick `42473355`.\n\n'+observation+'\n\nAll four workers published one full PASS JSON from the mandatory startup canary: seven cases per rank (28 cases, 168 candidate comparisons), including short6 numerics and preparation bytes. Original weights and concentrated6912 initial/changed tensor dtype/shape/hashes match onepass9, with storage reuse verified within this boot. Each receipt matches frozen build source, strict A0 container identity, its stream exactly once and the full JSON in its snapshot log. Candidate keys carry FP32/direct/shared-FC1-A/M16 tags. This does not prove full sanitizer coverage or arbitrary in-place alpha mutability.\n\nPayload and supervisor returned 143, and the owned holder/supervisor were absent at collection. Recovery was deferred to the normal idle controller; public restoration is not asserted. This archive makes no matched performance, regression-percentage or default-adoption claim.\n\nOnly allowlisted snapshot identities, hashes and safe logs are archived; raw Env/Cmd/inspect remain private. Original observer chunk offsets/hashes are verified through closure, and unattributed bytes remain unattributed. Original/stored SHA256 and sizes are in `originals.json`; logs, streams and frozen Python sources use deterministic gzip. The collection performs read-only remote observations and fresh archive writes, without GPU/HTTP/queue/deployment actions.\n'
save('README.md',readme.encode(),'bounded evidence explanation')
save('collect-evidence.py',read(__file__),'this collector')
record('originals.json',originals.copy(),'original/stored byte provenance')
items['SHA256SUMS']=''.join(sha(raw)+'  '+name+'\n' for name,raw in sorted(items.items())).encode()
OUT.mkdir()
for name,raw in items.items():
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
print(json.dumps(dict(archive=str(OUT),files=len(items),bytes=sum(map(len,items.values())),summary_sha256=sha(items['result-summary.json']),sha256sums_sha256=sha(items['SHA256SUMS']),verdict=summary['verdict'])))
