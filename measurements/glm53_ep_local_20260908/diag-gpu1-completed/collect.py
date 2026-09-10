#!/usr/bin/env python3
"""Read-only terminal archive for the single concentrated6912 diagnostic.

Preparation only. No action on import. Full GPU/performance/default acceptance
remain false even when this one diagnostic passes. No tests or device calls.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile

ROOT=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
HOST='choiceoh@srv2'
SOURCE='/home/choiceoh/stkernel-ep-local-diag-0908-1'
JOB='/tmp/glm53-ep-local-diag-gpu-0908-1'
CPU_SOURCE='/home/choiceoh/stkernel-ep-local-0908-17'
CPU_JOB='/tmp/glm53-ep-local-compile0908-17-head'
SCHEDULER='/home/choiceoh/stkernel-ep-local-scheduler-0909'
SESSION='eplocaldiag0908v1'
PROOF='measurements/glm53_ep_local_20260908/cpu17/local/result.json'
CAPSULE='/tmp/glm53-bindings-capsule-cpu0908-2/capsule'
MANIFEST='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
LIMIT=128*2**20

def require(value,message):
    if not value: raise ValueError(message)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def safe(name):
    p=PurePosixPath(name)
    require(p.parts and not p.is_absolute() and '..' not in p.parts and '\\' not in name
            and str(p)==name and name!='.', 'unsafe archive path: '+name)
    return name

def gpu_command(revision):
    return ['bash',SCHEDULER+'/bench/fleet.sh','run','--gpu',SESSION,'8',
        'CPU17-matched concentrated6912 failure diagnostics only; unchanged fixture and exact incoming restore','--',
        'python3','-B',SOURCE+'/probes/run_glm53_ep_local_offline.py',
        '--revision',revision,'--out',JOB+'/capture','--capsule-root',CAPSULE,
        '--manifest-sha256',MANIFEST,'--diagnose-case','concentrated6912']
def cpu_command():
    return ['bash',SCHEDULER+'/bench/fleet.sh','run','--cpu','eplocalcpu0908v17head','6',
        'CPU17 bounded numerical failure diagnostics; unchanged kernel, capsule13.0.3 no devices 4g2CPU','--',
        'python3','-B',CPU_SOURCE+'/probes/run_glm53_ep_local_cpu_compile.py',
        '--image',IMAGE,'--arm','local','--output',CPU_JOB+'/local',
        '--capsule-root',CAPSULE,'--manifest-sha256',MANIFEST]

# Filesystem reads and read-only git only. Logs are filtered from bounded tails;
# other sessions' moving global records are never shipped or archived wholesale.
REMOTE=r"""
import base64,hashlib,io,json,os,stat,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
C=json.loads(base64.b64decode(sys.argv[1],validate=True))
source=Path(C['source']);job=Path(C['job']);cpu_source=Path(C['cpu_source']);cpu_job=Path(C['cpu_job'])
fleet=Path('/home/choiceoh/glm53-logs/fleet')
def need(v,m):
    if not v: raise RuntimeError(m)
def sha(b): return hashlib.sha256(b).hexdigest()
def regular(p):
    for parent in (p,*p.parents):need(not parent.is_symlink(),'symlink input: '+str(p))
    s=p.stat();need(stat.S_ISREG(s.st_mode),'nonregular input: '+str(p));return s
def blob(p):
    s=regular(p);need(s.st_size<=64*2**20,'input exceeds 64 MiB')
    b=p.read_bytes();e=regular(p)
    need((s.st_ino,s.st_size,s.st_mtime_ns)==(e.st_ino,e.st_size,e.st_mtime_ns) and len(b)==s.st_size,'input changed: '+str(p))
    return b,dict(bytes=len(b),sha256=sha(b),source=str(p),mtime_ns=s.st_mtime_ns)
def tree(p):
    need(p.is_dir() and not p.is_symlink(),'regular directory required')
    result=[]
    for parent,dirs,names in os.walk(p,followlinks=False):
        for n in dirs:need(not (Path(parent)/n).is_symlink(),'symlink directory')
        result.extend(Path(parent)/n for n in names)
    need(len(result)<=4096,'too many files');return sorted(result)
def git(p,*args):return subprocess.check_output(['git','--no-optional-locks','-C',str(p),*args],text=True).strip()
def frozen(p,revision):
    need(git(p,'rev-parse','HEAD')==revision and not git(p,'status','--porcelain'),'frozen source changed: '+str(p))
    need(not (p/'.git/objects/info/alternates').exists(),'unexpected shared object store')
def tail_lines(p):
    s=regular(p);start=max(0,s.st_size-2*2**20)
    with p.open('rb') as f:
        f.seek(start);b=f.read(s.st_size-start)
        f.seek(start);again=f.read(s.st_size-start)
    e=regular(p)
    need(e.st_ino==s.st_ino and e.st_size>=s.st_size and b==again,'log tail changed while reading')
    if start:b=b.partition(b'\n')[2]
    return b.splitlines(keepends=True),dict(source=str(p),snapshot_size=s.st_size,tail_start=start,
        scope='Selected exact lines from a bounded 2 MiB log tail; not a whole-file hash')
def selected(p,predicate):
    lines,info=tail_lines(p);kept=b''.join(line for line in lines if predicate(line))
    return kept,dict(info,bytes=len(kept),sha256=sha(kept))
outer=json.loads(blob(job/'exit.json')[0]);done=json.loads(blob(job/'capture/completion.json')[0])
submission=json.loads(blob(job/'submission.json')[0])
need(type(outer.get('exit_code')) is int and type(done.get('exit_code')) is int and 'ended' in outer and 'ended' in done,'diagnostic not terminal')
need(outer['revision']==submission['revision']==done['source_revision']==C['revision'],'revision differs')
need(outer['session']==submission['session']==C['session'] and done['ended']<=outer['ended'],'terminal session/times differ')
need(all(outer.get(k)==v for k,v in submission.items()) and 'driver_error' not in outer,'outer driver failed or submission changed')
need(outer['exit_code']==done['exit_code'] and outer['complete'] is (outer['exit_code']==0),'payload/fleet exit differs')
release={}
for name in ('log','events.log'):
    release[name]=selected(fleet/name,lambda line:line.decode().split()[1:3]==['release',C['session']])
    need(len(release[name][0].splitlines())==1,'normal release missing/ambiguous in bounded tail: '+name)
life=selected(fleet/'lifecycle.jsonl',lambda line:json.loads(line).get('session')==C['session'])
events=[json.loads(line) for line in life[0].splitlines()]
payload=[x for x in events if x['event']=='payload-finished']
finished=[x for x in events if (x['event']=='restore-finished' and x.get('rc')==0) or x['event']=='handoff-accepted']
need(len(payload)==1 and payload[0]['rc']==done['exit_code'],'normal payload receipt differs')
need(finished and done['ended']<=payload[0]['t']<=finished[-1]['t']<=outer['ended'],'normal restore/handoff not complete')
holder=blob(fleet/'holder') if (fleet/'holder').exists() else (b'',dict(bytes=0,sha256=sha(b''),source=str(fleet/'holder'),absent=True))
queue=blob(fleet/'queue');debt=blob(fleet/'restore-debt.json') if (fleet/'restore-debt.json').exists() else None
need(not any(line.split('|')[0]==C['session'] for line in holder[0].decode().splitlines()),'diagnostic still holds fleet')
need(not any(len(line.split('|'))>1 and line.split('|')[1]==C['session'] for line in queue[0].decode().splitlines()),'diagnostic still queued')
if debt:
    d=json.loads(debt[0]);need(not d or d.get('owner',{}).get('session')!=C['session'],'diagnostic still owns restore debt')
frozen(source,C['revision'])
state=submission['cpu_state'];frozen(cpu_source,state['revision'])
need(git(source,'rev-list','--parents','-n','1',C['revision']).split()==[C['revision'],state['revision']],'receipt is not immediate CPU17 child')
cpu_submission_raw=blob(cpu_job/'submission.json')[0];cpu_exit_raw=blob(cpu_job/'exit.json')[0]
proof_raw=blob(cpu_job/'local/result.json')[0]
need(sha(cpu_submission_raw)==state['submission_sha256'] and sha(cpu_exit_raw)==state['exit_sha256']
     and sha(proof_raw)==state['result_sha256']==C['proof_sha']==submission['cpu_evidence_sha256'],'original CPU17 receipt hash differs')
cpu_submission=json.loads(cpu_submission_raw);cpu_exit=json.loads(cpu_exit_raw);proof=json.loads(proof_raw)
need(cpu_submission['revision']==state['revision'] and cpu_submission['source']==str(cpu_source)
     and cpu_submission['scheduler']==C['scheduler'] and cpu_submission['command']==C['cpu_command'],'CPU17 command/source differs')
need(cpu_exit.get('complete') is True and cpu_exit.get('returncode')==0 and 'error' not in cpu_exit
     and all(cpu_exit.get(k)==v for k,v in cpu_submission.items()),'actual CPU17 job was not successful')
need(blob(source/C['proof'])[0]==blob(job/'cpu-evidence.json')[0]==proof_raw,'committed/copied CPU17 receipt differs')
need(proof['verdict']=='PASS' and proof['phase']=='complete' and proof['binding_runtime_rechecked'] is True
     and proof['cuda_initialized'] is False and 'error' not in proof and 'binding_runtime_recheck_error' not in proof,'CPU17 incomplete')
contracts=proof['contracts'];need(contracts['tests_run']==146 and all(contracts[k]==0 for k in ('errors','failures','skips'))
     and len(contracts['files'])==29 and len(proof['mounted_sources'])==13 and len(proof['remap_compilation'])==24,'CPU17 coverage differs')
for rel,want in contracts['files'].items():
    p=PurePosixPath(rel);need(not p.is_absolute() and '..' not in p.parts,'unsafe contract path')
    need(sha(blob(source/rel)[0])==sha(blob(cpu_source/rel)[0])==want,'CPU17 contract source changed: '+rel)
files={};metadata={};inputs={};total=0
def add(name,path=None,value=None):
    global total
    need(name not in files,'duplicate archive file')
    raw,info=blob(path) if path is not None else value
    total+=len(raw);need(total<=C['limit'],'archive exceeds 128 MiB')
    files[name]=raw;metadata[name]=info
    if path is not None:inputs[name]=path
job_paths=tree(job)
for p in job_paths:add('job/'+p.relative_to(job).as_posix(),p)
need(files['job/cpu-evidence.json']==proof_raw,'CPU17 proof changed during snapshot')
need(json.loads(files['job/submission.json'])==submission and json.loads(files['job/exit.json'])==outer
     and json.loads(files['job/capture/completion.json'])==done,'terminal receipts changed during snapshot')
selected_sources={C['proof'],'build/glm53/manifest.tsv',*contracts['files']}
mounted={}
for line in blob(source/'build/glm53/manifest.tsv')[0].decode().splitlines():
    filename,target,*_=line.split('\t')
    need(Path(filename).name==filename,'invalid manifest name')
    if '/flashinfer/' in target or filename=='flashinfer_b12x_moe.py':
        need(target not in mounted,'duplicate mount');mounted[target]=filename
        selected_sources.add('build/glm53/'+filename)
need(set(mounted)==set(proof['mounted_sources']),'mounted source set differs')
for target,filename in mounted.items():
    rel='build/glm53/'+filename
    need(sha(blob(source/rel)[0])==sha(blob(cpu_source/rel)[0])==proof['mounted_sources'][target],'compiled source changed: '+filename)
for rel in sorted(selected_sources):add('source/'+rel,source/rel)
for filename in ('submission.json','exit.json','local/result.json'):add('cpu17/'+filename,cpu_job/filename)
cap=Path(C['capsule']);manifest_raw=blob(cap/'capsule-manifest.json')[0]
need(sha(manifest_raw)==C['manifest'],'capsule manifest changed');manifest=json.loads(manifest_raw)
need({p.relative_to(cap).as_posix() for p in tree(cap)}==set(manifest['files'])|{'capsule-manifest.json'},'capsule file set changed')
for rel,info in manifest['files'].items():
    raw,_=blob(cap/rel);need(sha(raw)==info['sha256'] and len(raw)==info['size'],'capsule file changed: '+rel)
add('runtime/capsule-manifest.json',cap/'capsule-manifest.json')
for name,value in release.items():add('fleet-session/'+name,value=value)
add('fleet-session/lifecycle.jsonl',value=life)
# Record only this session's absence, avoiding other holders/queue payloads.
absence=dict(session=C['session'],session_holder_absent=True,session_queue_absent=True,session_restore_debt_absent=True,captured=time.time())
raw=(json.dumps(absence,indent=2)+'\n').encode();add('fleet-session/absence.json',value=(raw,dict(bytes=len(raw),sha256=sha(raw),scope='Read-only namespace snapshot')))
need(tree(job)==job_paths,'terminal job file set changed')
for name,path in inputs.items():need(blob(path)[0]==files[name],'immutable input changed: '+name)
frozen(source,C['revision']);frozen(cpu_source,state['revision'])
identity=dict(revision=C['revision'],source=str(source),job=str(job),cpu17_revision=state['revision'],
    source_clean=True,captured_epoch=time.time(),files=metadata,lifecycle_events=events,
    scope='Terminal diagnostic snapshot; no live public-service equality after handoff')
files['source-identity.json']=(json.dumps(identity,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for name,raw in sorted(files.items()):
        item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.mtime=0
        archive.addfile(item,io.BytesIO(raw))
"""

def fetch(revision,proof_sha):
    config=dict(source=SOURCE,job=JOB,cpu_source=CPU_SOURCE,cpu_job=CPU_JOB,scheduler=SCHEDULER,
        revision=revision,session=SESSION,proof=PROOF,proof_sha=proof_sha,cpu_command=cpu_command(),
        capsule=CAPSULE,manifest=MANIFEST,limit=LIMIT)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,
        shlex.join(['python3','-B','-c',REMOTE,base64.b64encode(json.dumps(config).encode()).decode()])],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
    require(result.returncode==0,'terminal collection refused: '+result.stderr.decode(errors='replace')[-2200:])
    require(len(result.stdout)<=LIMIT+8*2**20,'oversized archive transport');return result.stdout

def unpack(payload,staging):
    raw={};records=[];targets=set();total=0
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
        for member in archive:
            name=safe(member.name)
            require(member.isfile() and name not in raw and member.size<=64*2**20,'invalid archive member')
            data=archive.extractfile(member).read();total+=len(data)
            require(len(data)==member.size and total<=LIMIT,'invalid archive size');raw[name]=data
            compress=Path(name).suffix in ('.log','.ptx','.cubin') or name.startswith('fleet-session/') or name.startswith('source/') and name.endswith('.py')
            target=name+'.gz' if compress else name;require(target not in targets,'stored path collision');targets.add(target)
            stream=io.BytesIO()
            if compress:
                with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0,compresslevel=9) as f:f.write(data)
            stored=stream.getvalue() if compress else data
            require((gzip.decompress(stored) if compress else stored)==data,'gzip roundtrip differs')
            p=staging/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(stored)
            records.append(dict(original_path=name,stored_path=target,original_bytes=len(data),stored_bytes=len(stored),original_sha256=sha(data),stored_sha256=sha(stored)))
    identity=json.loads(raw['source-identity.json'])
    require(set(identity['files'])==set(raw)-{'source-identity.json'},'snapshot file set differs')
    for name,info in identity['files'].items():require(sha(raw[name])==info['sha256'] and len(raw[name])==info['bytes'],'snapshot hash differs')
    (staging/'archive-manifest.json').write_text(json.dumps(dict(host=HOST,records=records),indent=2)+'\n')
    return raw,identity

def summarize(raw,identity,revision,proof_raw):
    read=lambda name:json.loads(raw[name])
    done=read('job/capture/completion.json');submission=read('job/submission.json');outer=read('job/exit.json')
    require(submission['command']==gpu_command(revision) and submission['scheduler']==SCHEDULER,'diagnostic command/scheduler differs')
    require(submission['diagnostic_only'] is True and submission['mode']=='diagnostic' and submission['diagnose_case']=='concentrated6912'
        and submission['performance_acceptance'] is False and submission['full_gpu_acceptance'] is False and submission['default_promotion'] is False,'submission is not diagnostic-only')
    require(done['mode']=='diagnostic' and done['diagnose_case']=='concentrated6912' and done['performance_acceptance'] is False
        and done['full_gpu_acceptance'] is False,'completion is not diagnostic-only')
    require(sha(raw['job/driver.py'])==submission['driver_sha256'],'driver bytes changed')
    require(raw['job/cpu-evidence.json']==raw['source/'+PROOF]==raw['cpu17/local/result.json']==proof_raw,'CPU17 proof differs')
    runtime=json.loads(proof_raw)['binding_runtime']
    require(runtime==submission['cpu_summary']['binding_runtime'] and runtime['capsule_manifest_sha256']==MANIFEST,'CPU17 runtime differs')
    if 'binding_runtime' in done:require(done['binding_runtime']==runtime,'completion runtime differs')
    cells=done['cells'];require(len(cells)<=1 and all(c['case']=='concentrated6912' and c['sanitizer'] is None for c in cells),'unexpected diagnostic cells')
    evidence=read('job/capture/concentrated6912.json') if 'job/capture/concentrated6912.json' in raw else {}
    if evidence:
        require(evidence['case']=='concentrated6912' and evidence['sanitize'] is False and evidence['performance_acceptance'] is False,'unexpected inner diagnostic scope')
        if 'binding_runtime' in evidence:require(evidence['binding_runtime']==runtime,'inner runtime differs')
    passed=bool(len(cells)==1 and cells[0].get('exit_code')==0 and evidence.get('verdict')=='PASS')
    if done['exit_code']==0:
        require(passed and evidence.get('phase')=='complete' and evidence.get('binding_runtime_rechecked') is True
            and cells[0].get('binding_runtime')==runtime and 'error' not in evidence and 'binding_runtime_recheck_error' not in evidence
            and 'candidate_first_failure' not in evidence,'incomplete diagnostic success')
    first=evidence.get('candidate_first_failure');diagnostics=evidence.get('candidate_failure_diagnostics')
    if first:require(first['verdict']=='CANDIDATE_NUMERICS_FAIL' and first['bad_rows']>0 and evidence['verdict']=='FAIL','first failure lost')
    if diagnostics:
        rows=diagnostics['rows'];require(first and diagnostics['diagnostic_only'] is True and diagnostics['first_failure_preserved'] is True
            and diagnostics['raw_dtype']=='bfloat16' and 0<len(rows)<=8 and diagnostics['captured_bad_rows']==len(rows)
            and diagnostics['total_bad_rows']==first['bad_rows'],'invalid bounded diagnostic record')
        for row in rows:
            require(len(row['routes'])==8 and 0<len(row['worst_columns'])<=8 and all(math.isfinite(v) for v in row['metrics'].values()),'unbounded/invalid diagnostic row')
            for column in row['worst_columns']:
                require(set(column['raw_bf16_u16'])=={'B1','B2','B3','C1','X'} and all(type(v) is int and 0<=v<=65535 for v in column['raw_bf16_u16'].values()),'invalid raw BF16 words')
    restoration=dict(scope='Normal supervisor closure; no live public identity after handoff',original_snapshot_equal=None)
    if 'job/capture/before.json' in raw:
        before=read('job/capture/before.json');require(len(before)==4,'incoming set incomplete')
        if all(v is None for v in before.values()):
            require(done['incoming_mode']=='absent','incoming mode differs');restoration['incoming_mode']='absent'
        else:
            stopped=read('job/capture/stopped.json');restored=read('job/capture/restored.json')
            require(set(before)==set(stopped)==set(restored) and all(before.values()) and done.get('restored_original') is True,'original restoration missing')
            immutable=lambda item:{k:v for k,v in item.items() if k not in ('running','started')}
            for node,original in before.items():
                require(original['image']==IMAGE and original['auto_remove'] is False and original['overlays'],'incoming identity missing')
                require(immutable(original)==immutable(stopped[node])==immutable(restored[node]) and stopped[node]['running'] is False
                    and restored[node]['running']==original['running'],'original restoration differs')
            if not any(v['running'] for v in before.values()):require(before==stopped==restored==read('job/capture/stopped-restored.json'),'stopped original fields differ')
            restoration.update(incoming_mode=done['incoming_mode'],original_snapshot_equal=before==restored,immutable_and_running_equal=True)
    else:require(not cells and done['exit_code']!=0,'before snapshot missing after a cell')
    return dict(verdict='PASS_DIAGNOSTIC_ONLY' if done['exit_code']==0 else 'FAIL_DIAGNOSTIC',diagnostic_only=True,
        performance_acceptance=False,full_gpu_acceptance=False,default_promotion=False,session=SESSION,revision=revision,
        case='concentrated6912',observed_cells=len(cells),inner_exit_code=done['exit_code'],outer_exit_code=outer['exit_code'],
        error=done.get('error'),probe_verdict=evidence.get('verdict','NOT_RUN'),phase=evidence.get('phase'),
        candidate_first_failure=first,candidate_failure_diagnostics=diagnostics,
        candidate_failure_diagnostics_error=evidence.get('candidate_failure_diagnostics_error'),
        probe_error=evidence.get('error'),controls=evidence.get('controls',[]),candidate=evidence.get('candidate',[]),
        timing=evidence.get('timing',{}),timing_scope=evidence.get('timing_scope'),rows=evidence.get('rows'),
        max_allocated_bytes=evidence.get('max_allocated_bytes'),max_reserved_bytes=evidence.get('max_reserved_bytes'),
        restoration=restoration,lifecycle_events=identity['lifecycle_events'],cpu17_proof_sha256=sha(proof_raw),
        limitations=['One unchanged synthetic fixture; no full GPU, sanitizer-suite or performance acceptance.',
                     'No full-model TTFT or serving-quality proof. The original GPU v5 failure is preserved.',
                     'Raw BF16 words and original comparison limits are diagnostic evidence, not a new acceptance rule.'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--revision',required=True)
    args=parser.parse_args();require(re.fullmatch('[0-9a-f]{40}',args.revision) is not None,'full revision required')
    out=ROOT/'measurements/glm53_ep_local_20260908/diag-gpu1-completed';staging=out.with_name(out.name+'.collecting')
    require(not out.exists() and not staging.exists(),'output/staging exists; no overwrite')
    proof=(ROOT/PROOF).read_bytes()
    committed=subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'show',args.revision+':'+PROOF])
    require(proof==committed,'local CPU17 proof differs from requested committed receipt')
    payload=fetch(args.revision,sha(proof));staging.mkdir(parents=True,exist_ok=False)
    try:
        raw,identity=unpack(payload,staging);summary=summarize(raw,identity,args.revision,proof)
        (staging/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (staging/'README.md').write_text(summary['verdict']+'; concentrated6912 diagnostic only.\n\n'
            'Normal outer exit, successful supervisor restore/handoff and exact session release were required. Original job files preserve failure details and before/stopped/restored records when present; this is not a live service equality check after handoff.\n\n'
            'CPU17 actual source and receipt matched the frozen committed diagnostic source. Logs and source Python snapshots use deterministic gzip; archive-manifest.json records original/stored bytes and SHA256. Fleet evidence contains only this session selected from bounded log tails, plus a namespace absence snapshot.\n\n'
            +'\n'.join(summary['limitations'])+'\n')
        (staging/'collect.py').write_bytes(Path(__file__).read_bytes())
        files=sorted(p for p in staging.rglob('*') if p.is_file())
        (staging/'SHA256SUMS').write_text(''.join(sha(p.read_bytes())+'  '+p.relative_to(staging).as_posix()+'\n' for p in files))
        for line in (staging/'SHA256SUMS').read_text().splitlines():
            want,name=line.split('  ',1);require(sha((staging/safe(name)).read_bytes())==want,'stored hash mismatch')
        require(not out.exists(),'output appeared during collection');staging.rename(out)
        print(json.dumps(dict(archive=str(out),verdict=summary['verdict'],diagnostic_only=True,full_gpu_acceptance=False)))
    except BaseException as exc:
        (staging/'VERIFICATION_ERROR.txt').write_text(repr(exc)+'\n');raise

if __name__=='__main__':main()
