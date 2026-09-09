#!/usr/bin/env python3
"""Archive closed onepass11 A0 failure: read-only SSH; new archive writes only."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT = ROOT/'measurements/glm53_ep_local_20260908/onepass11-startup-failed'
REV = 'c7dec80a0f73d4a2b683ce2c4813978938694095'
SESSION = 'eplocalonepass0909v11'
TICKET = '17889136023671585'
NODES = ('local', '10.10.10.1', '10.10.10.3', '10.10.10.4')
PREFILL = ('concentrated6912','balanced4096','remote4096','duplicate4096','zeros4097','balanced8192')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
assert not OUT.exists(), 'refuse existing archive'
items, originals = {}, {}
def save(name, raw, origin, **meta):
    assert name not in items and not Path(name).is_absolute() and '..' not in Path(name).parts
    items[name] = raw
    originals[name] = dict(origin=origin, bytes=len(raw), sha256=sha(raw), **meta)
def packed(name, raw, origin):
    save(name, gzip.compress(raw,mtime=0), origin, original_bytes=len(raw), original_sha256=sha(raw))
def record(name, value, origin):
    save(name,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode(),origin)
def read(path):
    path = Path(path)
    before=path.stat();raw=path.read_bytes();after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert len(raw)==before.st_size and len(raw)<128*2**20
    return raw

REMOTE = r'''
import base64,hashlib,json,os,pathlib,subprocess,time
P=pathlib.Path
root=P('/home/choiceoh/stkernel-ep-onepass-0909-11')
job=P('/tmp/glm53-ep-onepass-0909-11')
session='eplocalonepass0909v11';ticket='17889136023671585'
log=P('/home/choiceoh/glm53-logs/fleet/run-logs/42754a82ba632ea85f7ca1804f833f45fc3fd6bd1b10cdb06346c8b7171c22d0.log')
def identity():
 def git(*args):return subprocess.check_output(['git','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).decode().strip()
 return dict(head=git('rev-parse','HEAD'),status=git('status','--porcelain'))
def fleet():
 p=subprocess.run(['bash',str(root/'bench/fleet.sh'),'show',session,'--ticket',ticket,'--json'],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 d=json.loads(p.stdout)
 keys=('session','ticket','pid','supervisor_pid','state','phase','started_at','payload_finished_at','finished_at','payload_returncode','returncode','outcome','log_path','recovery_policy','recovery_deferred','supervisor_alive','payload_seconds')
 return {k:d[k] for k in keys if k in d}
def own():
 p=P('/home/choiceoh/glm53-logs/fleet/holder')
 return p.exists() and p.read_text().split('|',1)[0]==session
r=dict(captured_at=time.time(),source_before=identity(),fleet_before=fleet(),own_holder_before=own(),files={},absent=[])
f=r['fleet_before']
assert f['session']==session and str(f['ticket'])==ticket and f['log_path']==str(log)
assert f['phase']=='finished' and f['payload_returncode']==1 and f['returncode']==1 and not r['own_holder_before']
paths=[job/'onepass.jsonl',job/'verdicts.jsonl',log]
paths += [P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS11'+arm+'.log') for arm in ('A0','B1','A','B2','B3')]
for path in paths:
 if not path.exists():r['absent'].append(str(path));continue
 before=path.stat();raw=path.read_bytes();after=path.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and len(raw)==before.st_size and len(raw)<128*2**20
 if path.name in ('onepass.jsonl','verdicts.jsonl'):assert not raw.strip(), 'unexpected measurement records'
 if path.name.startswith('boot-') and path.name!='boot-EPONEPASS11A0.log':raise AssertionError('later arm unexpectedly started')
 r['files'][str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mtime_ns':after.st_mtime_ns,'data':base64.b64encode(raw).decode()}
r.update(source_after=identity(),fleet_after=fleet(),own_holder_after=own())
assert r['source_before']==r['source_after'] and r['fleet_before']==r['fleet_after'] and not r['own_holder_after']
print(json.dumps(r))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','choiceoh@srv2','python3 -B -'],input=REMOTE.encode(),capture_output=True,timeout=60)
assert r.returncode==0,r.stderr.decode()
remote=json.loads(r.stdout)
assert remote['source_before']==dict(head=REV,status='')
for origin,detail in remote['files'].items():
    raw=base64.b64decode(detail.pop('data'));assert sha(raw)==detail['sha256'] and len(raw)==detail['bytes']
    name=Path(origin).name
    if name in ('onepass.jsonl','verdicts.jsonl'):save('records/'+name,raw,origin)
    elif name.startswith('boot-'):packed('boot/'+name+'.gz',raw,origin)
    else:packed('fleet/terminal-run.log.gz',raw,origin)
record('terminal-capture.json',remote,'read-only normal fleet/source/own-holder terminal checks')

def gitfile(name):return subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT)
manifest=gitfile('build/glm53/manifest.tsv')
mounts={line.split('\t')[1]:sha(gitfile('build/glm53/'+line.split('\t')[0])) for line in manifest.decode().splitlines() if line and not line.startswith('#')}
save('source/frozen-manifest.tsv',manifest,'git '+REV)
record('source/mounted-hashes.json',mounts,'git '+REV+' build bytes')
provided_raw=read('/tmp/glm53-onepass11-mounted-hashes.json');provided=json.loads(provided_raw)
assert all(mounts.get(k)==v for k,v in provided.items())
save('source/continuity-input-mounted-hashes.json',provided_raw,'/tmp/glm53-onepass11-mounted-hashes.json')
continuity_raw=read('/tmp/glm53-onepass11-continuity.json');continuity=json.loads(continuity_raw)
assert continuity['verdict']=='MATCH_BUT_CANARY_FAIL' and continuity['identity_match'] is True
assert continuity['new_source_manifest_sha256']==sha(provided_raw)
old=ROOT/'measurements/glm53_ep_local_20260908/onepass9-startup-failed'
assert continuity['old_source_manifest_sha256']==sha(read(old/'source/mounted-hashes.json'))
save('continuity.json',continuity_raw,'/tmp/glm53-onepass11-continuity.json')

stream_root=Path('/tmp/glm53-onepass11-streams')
event_raw=read(stream_root/'events.jsonl');events=[json.loads(line) for line in event_raw.splitlines()]
assert events[-1]['kind']=='observer_finished' and events[-1]['attempted_arms']==[] and events[-1]['owned_go_seen']
ends,chunks={},{}
for e in events:
    if e['kind']!='chunk':continue
    key=e['node'],e['channel'];assert e['offset']==ends.get(key,0)
    ends[key]=e['offset']+e['bytes'];chunks.setdefault(key,[]).append(e)
streams={}
for (node,channel),end in ends.items():
    path=stream_root/(node+'.'+channel+'.raw');raw=read(path);assert len(raw)==end
    for e in chunks[node,channel]:assert sha(raw[e['offset']:e['offset']+e['bytes']])==e['sha256']
    streams[node,channel]=raw;packed('streams/'+path.name+'.gz',raw,str(path))
assert all((node,'stdout') in streams for node in NODES)
packed('streams/events.jsonl.gz',event_raw,str(stream_root/'events.jsonl'))
pid=int(read(stream_root/'observer.pid'))
try:os.kill(pid,0)
except ProcessLookupError:alive=False
else:alive=True
assert not alive,'observer still running'
record('streams/closure.json',dict(observer_pid=pid,alive=alive,events=[e for e in events if e['kind'] in ('stream_end','observer_stop','observer_finished')],scope='All chunk hashes/offsets checked. Streams started after owned GO; unassigned bytes retain original attribution, no strict container snapshot exists.'),'closed passive observer')

failures={}
for node in NODES:
    path=Path('/tmp/glm53-onepass11-canary')/(node+'.json');raw=read(path);receipt=json.loads(raw)
    assert receipt['verdict']=='FAIL' and receipt['seed']==905308 and len(receipt['cases'])==7
    assert sha(raw)==continuity['nodes'][node]['new_receipt_sha256']
    assert sha(read(old/'failures'/path.name))==continuity['nodes'][node]['old_receipt_sha256']
    assert continuity['nodes'][node]['identity_match'] and continuity['nodes'][node]['within_boot_storage_reused'] and continuity['nodes'][node]['difference_count']==0
    for role,src in receipt['provenance']['source'].items():
        expected='993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445' if role=='stock' else mounts[src['path']]
        assert src['sha256']==expected
    assert tuple(c['case'] for c in receipt['cases'][:6])==PREFILL
    for case in receipt['cases'][:6]:assert case['verdict']=='PASS' and case['phase']=='complete' and len(case['candidate'])==6 and len(case['inputs'])==2
    c=receipt['cases'][-1];assert c['case']=='short6' and len(c['inputs'])==1 and len(c['preparation'])==3
    assert c['inputs'][0]['weights']['dtype']=='torch.bfloat16'
    for index,p in enumerate(c['preparation']):
        if index:assert p['input_weights_dtype']==('torch.float32','torch.float16')[index-1]
        tensors=p if index==0 else p['tensors'];assert set(tensors)=={'X','ids','weights'}
        for proof in tensors.values():assert proof['exact'] and proof['actual_sha256']==proof['reference_sha256']
    assert ('CANDIDATE_NUMERICS_FAIL' if node=='local' else 'UNSTABLE_STOCK_CONTROL') in receipt['error']
    stream=streams[node,'stdout'];matches=[];offset=0
    for line in stream.splitlines(keepends=True):
        marker=b'[ep-local-selftest] FAIL '
        if marker in line:
            observed,_=json.JSONDecoder().raw_decode(line.split(marker,1)[1].decode())
            if observed==receipt:matches.append(dict(line_sha256=sha(line),stream_byte_offset=offset))
        offset+=len(line)
    assert len(matches)==1,(node,'nonunique or missing stream receipt')
    save('failures/'+path.name,raw,str(path),raw_stream_match=matches[0])
    failures[node]=dict(error=receipt['error'],phase=c['phase'],prefill_pass_cases=list(PREFILL),short6_initial_preparation_dtypes=['torch.bfloat16','torch.float32','torch.float16'],short6_changed_reached=False,candidate_first_failure=c.get('candidate_first_failure'),candidate_pass_count=len(c['candidate']),duration_s=c['duration_s'],raw_stream_match=matches[0])

for source,name in [('/tmp/glm53-onepass11-submit.json','submission/request.json'),('/tmp/glm53-onepass11-submit-receipt.txt','submission/receipt.json'),('/tmp/glm53_onepass11_observer.py','tools/passive-observer.py'),('/tmp/glm53_onepass11_snapshot.py','tools/snapshot-helper.py'),('/tmp/glm53_extract_onepass11_canary.py','tools/extract-canary.py'),('/tmp/glm53_onepass10_compare_canary.py','tools/compare-canary.py')]:save(name,read(source),source)
submission=json.loads(items['submission/receipt.json'])
assert (submission['session'],submission['ticket'],submission['pid'],submission['log_path'])==(SESSION,TICKET,3671585,remote['fleet_after']['log_path'])
summary=dict(schema=1,verdict='STARTUP_CANARY_FAIL',revision=REV,session=SESSION,ticket=TICKET,supervisor_pid=3671585,arm='A0',payload_returncode=1,returncode=1,own_holder_after=False,onepass_request_records=0,strict_arm_snapshots=0,later_arms_started=[],prefill_case_pass_count=24,canary_pass_count=0,canary_failure_count=4,same_old_weight_and_concentrated_input_identity=True,short6_initial_preparation_exact_variants=12,short6_changed_reached=False,failures=failures,performance_acceptance=False,adoption_acceptance=False,full_sanitizer_acceptance=False,scope='A0 startup self-test only. Normal onepass HTTP workload did not start. Source bytes are independently bound to frozen Git revision; no running-container snapshot/image attribution was completed.')
record('failure-summary.json',summary,'unchanged original receipts and closed normal fleet evidence')
readme='''# onepass11: A0 startup canary failed

Frozen source `c7dec80a0f73d4a2b683ce2c4813978938694095`, normal fleet session `eplocalonepass0909v11`, ticket `17889136023671585`, supervisor `3671585`.

All four ranks passed all six prefill fixtures: concentrated6912, balanced4096, remote4096, duplicate4096, zeros4097, balanced8192 (24 case passes). The archived continuity report binds all four synthetic weights and concentrated initial/changed inputs to the original onepass9 failure receipts; within each boot storage was reused. It reports MATCH_BUT_CANARY_FAIL, not adoption acceptance.

Initial T6 preparation bytes matched the reference for BF16, FP32 and FP16 weights on all four ranks. Changed T6 was not reached. Short6 numerical validation failed: rank0 at initial-C3 (1 bad row, normalized peak 0.0410628021); ranks1/2/3 stopped on unstable stock controls (normalized peaks 0.0649606287, 0.0549019612, 0.1175889298). Original errors, controls and diagnostics remain in all four complete receipts.

A0 failed before readiness and before the normal onepass HTTP workload. B1/A/B2/B3 never started; no measurement records or strict arm snapshots exist. Both payload and supervisor returned 1, and the owned holder was absent at terminal collection. This is no performance, decode, full-sanitizer or default-adoption acceptance.

Original streams retain their UNASSIGNED attribution where no strict container snapshot completed. Each receipt was found exactly once in its named node stream; original chunk offsets/hashes and closed observer state were verified. Source receipt hashes were independently compared with frozen Git build bytes. This does not create missing live container/image identity evidence.

Logs and streams use deterministic gzip (mtime 0); originals.json records original and stored SHA256/lengths. Four canary receipts, continuity report and submission receipts preserve original bytes. SHA256SUMS covers every archived file other than itself. Collection reads remote terminal/source state only; no GPU, queue, deployment or test action was performed.
'''
save('README.md',readme.encode(),'archive result explanation')
save('collect-evidence.py',Path(__file__).read_bytes(),'this read-only collector')
record('originals.json',originals.copy(),'original and stored SHA256 provenance')
checks=''.join(sha(raw)+'  '+name+'\n' for name,raw in sorted(items.items()))
items['SHA256SUMS']=checks.encode()
OUT.mkdir()
for name,raw in items.items():
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
print(json.dumps(dict(archive=str(OUT),files=len(items),summary_sha256=sha(items['failure-summary.json']),sha256sums_sha256=sha(items['SHA256SUMS']),verdict=summary['verdict'])))
