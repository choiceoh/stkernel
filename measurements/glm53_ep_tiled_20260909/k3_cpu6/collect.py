#!/usr/bin/env python3
"""Read-only collection of completed EP K3 native-shape CPU originals, not a test run."""
import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

REPO = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT = REPO/'measurements/glm53_ep_tiled_20260909/k3_cpu6'
REV = '6977199cf699f82696f925f77bc930d500135532'
RESULT = '93e28079eff1ec4493633ef253c00442cf3f960f0060731ed348b7eadb547348'
EVIDENCE = '/home/choiceoh/glm53-ep-tiled-k3-6-cpu-evidence'
LIMIT = 32*1024*1024

READ = r'''
import hashlib,io,json,pathlib,subprocess,sys,tarfile,time
root=pathlib.Path('/home/choiceoh/glm53-ep-tiled-k3-6-cpu-evidence')
def inventory():
 result={}
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink(),'symlink in original evidence'
  if p.is_file():result[str(p.relative_to(root))]=p
 assert len(result)==29 and 'result.json' in result and 'contracts.json' in result
 out={}; raw={}
 for name,p in result.items():
  assert not p.is_symlink() and p.is_file()
  a=p.stat();data=p.read_bytes();b=p.stat()
  assert (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
  out[name]=dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),size=len(data),mtime_ns=a.st_mtime_ns,inode=a.st_ino)
  raw[name]=data
 assert sum(len(v) for v in raw.values())<=32*1024*1024
 return out,raw
before,raw=inventory()
if sys.argv[1]=='inventory':
 source='/home/choiceoh/stkernel-ep-tiled-0909-ring'
 def run(args):return subprocess.check_output(args,text=True,timeout=15).strip()
 capsule=pathlib.Path('/home/choiceoh/glm53-ep-cpu-0909-21/capsule/capsule-manifest.json')
 print(json.dumps(dict(captured_at=time.time(),files=before,source=source,
  head=run(['git','-C',source,'rev-parse','HEAD']),status=run(['git','-C',source,'status','--porcelain']),
  image=run(['docker','image','inspect','--format','{{.Id}}','sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211']),
  capsule_manifest_sha256=hashlib.sha256(capsule.read_bytes()).hexdigest()),sort_keys=True));raise SystemExit
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as t:
 for name,data in raw.items():
  m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o600;t.addfile(m,io.BytesIO(data))
assert inventory()[0]==before,'originals changed while copying'
sys.stdout.buffer.write(buffer.getvalue())
'''


def sha(raw): return hashlib.sha256(raw).hexdigest()


def remote(mode):
    worker = shlex.join(['python3','-B','-c',READ,mode])
    head = shlex.join(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',
                      'choiceoh@10.10.10.4',worker])
    r = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',
                        'choiceoh@srv2',head],capture_output=True,timeout=90)
    if r.returncode: raise RuntimeError('read-only collection failed, rc='+str(r.returncode))
    assert len(r.stdout)<=LIMIT
    return r.stdout


def git_bytes(path):
    return subprocess.check_output(['git','show',REV+':'+path],cwd=REPO)


assert not OUT.exists(),'fresh archive required'
before_raw=remote('inventory');before=json.loads(before_raw)
bundle=remote('bundle')
after_raw=remote('inventory');after=json.loads(after_raw)
assert before['files']==after['files'],'originals changed before/after transfer'
for key in ('source','head','status','image','capsule_manifest_sha256'):assert before[key]==after[key],key
assert before['head']==REV and before['status']==''
assert before['image']=='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
assert before['capsule_manifest_sha256']=='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
raw={}
with tarfile.open(fileobj=io.BytesIO(bundle),mode='r:') as t:
    assert len(t.getmembers())==len(before['files'])
    for item in t.getmembers():
        name=item.name;p=Path(name)
        assert item.isfile() and not p.is_absolute() and '..' not in p.parts
        assert name not in raw and name in before['files']
        data=t.extractfile(item).read();row=before['files'][name]
        assert len(data)==row['size'] and sha(data)==row['sha256']
        raw[name]=data
assert set(raw)==set(before['files']) and sha(raw['result.json'])==RESULT
r=json.loads(raw['result.json']);c=json.loads(raw['contracts.json'])
for result in (r,c):
    assert result['verdict']=='PASS' and result['phase']=='complete'
    assert result['cuda_initialized'] is False and result['binding_runtime_rechecked'] is True
    assert result['compile_only'] is True
    assert not any(k in result for k in ('error','cleanup_error','recheck_error'))
    assert result['contracts']==dict(tests_run=119,failures=0,errors=0,skips=0)
    assert sum(result['selected_test_counts'].values())==119 and len(result['selected_test_counts'])==12
    assert result['gpu_numerics_acceptance'] is False and result['performance_acceptance'] is False
assert r['contracts_process_isolated'] is True
previous=json.loads((REPO/'measurements/glm53_ep_tiled_20260909/bf16_cpu5/result.json').read_text())
assert r['binding_runtime']==previous['binding_runtime'],'immutable capsule runtime identity changed'
for key in ('contracts','selected_test_counts','binding_runtime','mounted_sources','contract_sources'):
    assert r[key]==c[key],key
expected=set();resources=[]
for kind,rows in (('static',(4,6,8,12,16,24,32)),('dynamic',(33,8192))):
    passes=r[kind+'_passes'];assert [p['arm'] for p in passes]==[kind+'/M'+str(m) for m in rows]
    for m,p in zip(rows,passes):
        key=p['cache_key']
        if kind=='static':
            ring=m<=8
            assert key[:4]==['glm53_ep_static_tiled_fp32_v1',m,256,48] and key[10]=='sf6_v1'
            assert len(key)==(19 if ring else 16)
            if ring:assert key[-4:]==['bf16_scatter','glm53_ep_static_sf6_a_ring_v1','glm53_ep_static_sf6_word_unpack_v1','glm53_ep_static_bf16_scatter_v1']
            assert key[-1]==('glm53_ep_static_bf16_scatter_v1' if ring else 'fp32_scatter')
            assert p['specialization']==dict(a_ring=ring,word_unpack=ring,scatter_bf16=ring,output_dtype='bfloat16' if ring else 'float32',scale_mode='sf6_v1',cache_tag=key[-1])
        else:
            assert key[3:7]==[72,4096,2048,8] and key[17] is True
            assert key[-2:]==['glm53_ep_prefill_local_fp32_v2','glm53_ep_tiled_sf6_v1']
        for field,suffix in (('artifacts','.ptx'),('resources','.cubin')):
            assert len(p[field])==1
            entry=p[field][0];name=entry['path']
            assert name.startswith(p['arm']+'/') and name.endswith(suffix) and name not in expected
            assert sha(raw[name])==entry['sha256'];expected.add(name)
            if suffix=='.cubin':
                logfile=str(Path(name).with_suffix('.resources.log'))
                assert raw[logfile].decode()==entry['resources']
                expected.add(logfile)
                metrics={k:int(v) for k,v in re.findall(r'\b(REG|STACK|LOCAL|SHARED):(\d+)',entry['resources'])}
                wanted=dict(REG=123 if m<=8 else 118,STACK=0,LOCAL=0,SHARED=1024) if kind=='static' else dict(REG=168,STACK=112,LOCAL=0,SHARED=1024)
                assert metrics==wanted,(p['arm'],metrics)
                resources.append(dict(arm=p['arm'],**metrics))
assert expected|{'result.json','contracts.json'}==set(raw)
manifest=git_bytes('build/glm53/manifest.tsv').decode()
targets={x.split('\t')[1]:x.split('\t')[0] for x in manifest.splitlines() if x and not x.startswith('#')}
source_checks={}
for target,want in r['mounted_sources'].items():
    path='build/glm53/'+targets[target];assert sha(git_bytes(path))==want,target
    source_checks[path]=want
for path,want in r['contract_sources'].items():
    assert sha(git_bytes(path))==want,path;source_checks[path]=want
assert len(r['mounted_sources'])==22 and len(r['contract_sources'])==26
log=Path('/tmp/glm53-ep-k3-cpu6.log').read_bytes()
assert b'Ran 119 tests' in log and b'\nOK\n' in log
raw['fleet.log']=log
execution_raw=Path('/tmp/glm53-ep-k3-cpu6-execution.json').read_bytes()
execution=json.loads(execution_raw)
source='/home/choiceoh/stkernel-ep-tiled-0909-ring'
assert execution['revision']==REV and execution['source']==source
assert execution['session']=='eptiledcpu0910k36' and execution['output']==EVIDENCE
assert execution['returncode']==0 and execution['finished']>=execution['started']
inner=['nice','-n19','python3','-B',source+'/probes/run_glm53_ep_tiled_cpu.py',
 '--image','sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211',
 '--output',EVIDENCE,'--capsule-root','/home/choiceoh/glm53-ep-cpu-0909-21/capsule',
 '--manifest-sha256','b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab']
command=['env','REPO='+source,'bash',source+'/bench/fleet.sh','run','--cpu','eptiledcpu0910k36','5',
 'EP SF6 K3 shapes and live speculation proof: nine real lowerings and source-bound CPU contracts; no GPU devices','--',
 'ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@10.10.10.4',shlex.join(inner)]
assert execution['command']==command
raw['execution.json']=execution_raw
helper_identity=json.loads(git_bytes('measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json'))
assert r['scatter_helper']==dict(path=helper_identity['source_path'],sha256=helper_identity['source_sha256'],size=helper_identity['source_bytes'],helper='scatter_add_v4_bf16x2')
raw['helper-proof.json']=(json.dumps(dict(helper=r['scatter_helper'],source=REV,result_sha256=RESULT,
 scope='actual imported helper source path, complete file SHA and size from CPU result; source-bound probe also asserts imported helper object identity before lowering; not a host filesystem substitution'),indent=2)+'\n').encode()
raw['worker-before.json']=before_raw;raw['worker-after.json']=after_raw
raw['source-verification.json']=(json.dumps(dict(source=REV,verified=source_checks,
    scope='all mounted and contract source hashes equal the exact immutable git CPU source 6977199c; no source rebind required'),indent=2)+'\n').encode()
summary=dict(verdict='ORIGINAL_CPU_EVIDENCE_VERIFIED',source=REV,result_sha256=RESULT,
    original_worker_files=29,original_worker_bytes=sum(v['size'] for v in before['files'].values()),
    execution_sha256=sha(execution_raw),stdout_sha256=sha(log),stdout_bytes=len(log),mounted_sources=len(r['mounted_sources']),contract_sources=len(r['contract_sources']),
    contracts=r['contracts'],scatter_helper=r['scatter_helper'],resources=resources,compile_only=True,cuda_initialized=False,
    gpu_numerics_acceptance=False,performance_acceptance=False,source_rebind_required=False)
raw['verification.json']=(json.dumps(summary,indent=2)+'\n').encode()
OUT.mkdir(parents=True)
entries={}
for name,data in raw.items():
    compressed=name.endswith(('.log','.ptx','.cubin'))
    stored=name+'.gz' if compressed else name
    contents=gzip.compress(data,mtime=0) if compressed else data
    dest=OUT/stored;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(contents)
    entries[name]=dict(path=stored,encoding='gzip' if compressed else 'raw',original_sha256=sha(data),
                       original_size=len(data),stored_sha256=sha(contents),stored_size=len(contents))
readme=f'''# EP tiled K3 shape CPU evidence

Normal CPU fleet session: `eptiledcpu0910k36`, worker `10.10.10.4` (srv4).
Original output: `{EVIDENCE}`. Collection used only read-only archive operations, with no tests or GPU work.

The original receipt reports **119 tests, zero failures/errors/skips**, **seven static and two dynamic CuTe lowerings**, and CUDA uninitialized. Runtime identity was rechecked and CPU contracts ran in a separate process. Original no-device, no-network, 4 GiB/2 CPU and 12 GiB host-availability constraints remain.

| Variant | A-ring / word unpack | BF16 scatter | Output dtype | Registers | Stack bytes | Local bytes |
|---|---:|---:|---|---:|---:|---:|
| Static M4, M6, M8 / SF6 | true | true | BF16 | 123 | 0 | 0 |
| Static M12, M16, M24, M32 / SF6 | false | false | FP32 | 118 | 0 | 0 |
| Dynamic M33, M8192 / SF6 | unchanged | unchanged | FP32 | 168 | 112 | 0 |

Actual constructor flags and fake output element types were checked before and after lowering. M4/M6/M8 have 19-field keys with `bf16_scatter` at index 15 and the ordered A-ring, word-unpack, BF16-scatter suffixes. Other static keys retain 16 fields and FP32 output. New K3 request batches use native rows M4/M8/M12/M16; this CPU receipt proves compilation of those shapes, not that a serving process selected K3. The source adds actual-weight canary M4/M8/M16 after all nine prior cases, preserving their input seeds. GPU execution of all twelve cases and live speculation configuration/counters remain separate gates.

The real CPU process read imported `flashinfer.cute_dsl.fp4_common` and checked its complete SHA, size and path against the pinned oracle; the source-bound probe also checked imported helper object identity. Receipt: SHA `a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1`, 87,909 bytes. `helper-proof.json` preserves this scope. Capsule runtime identity exactly matches the prior CPU5 actual receipt and the fixed manifest; the worker image, manifest and clean source were checked before and after copying.

Exact compiled source: `{REV}`. All **22 mounted sources and 26 contracts** match immutable committed bytes. No source rebind was needed. Original result SHA: `{RESULT}`.

All **29 original worker files**, before/after inventories, exact execution metadata and CPU stdout are preserved. `execution.json` came from the process executing the normal fleet command and records rc0; elapsed time was {execution["finished"]-execution["started"]:.3f} seconds. `verification.json` binds that execution and stdout. Logs/PTX/cubins use deterministic gzip; original and stored hashes/sizes are in `manifest.json`. Raw JSON remains unchanged. Resource numbers describe compiled artifacts, not runtime timing or occupancy. SHARED1024 is static resource reporting, not total dynamic shared allocation; LOCAL0 alone is not a runtime spill-performance claim.

This archive proves CPU compilation and contracts only. GPU numerical correctness, graph behavior, throughput, quality, the 67 tok/s target and default adoption are **not accepted by this evidence**.
'''
(OUT/'README.md').write_text(readme)
collector=Path(__file__).read_bytes();(OUT/'collect.py').write_bytes(collector)
for name in ('README.md','collect.py'):
    data=(OUT/name).read_bytes();entries[name]=dict(path=name,encoding='raw',original_sha256=sha(data),original_size=len(data),stored_sha256=sha(data),stored_size=len(data),scope='post-measurement archive utility or documentation')
(OUT/'manifest.json').write_text(json.dumps(dict(version=1,files=entries),indent=2)+'\n')
checks={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'SHA256SUMS').write_text(''.join(value+'  '+name+'\n' for name,value in checks.items()))
for name,row in entries.items():
    data=(OUT/row['path']).read_bytes();assert sha(data)==row['stored_sha256']
    original=gzip.decompress(data) if row['encoding']=='gzip' else data
    assert len(original)==row['original_size'] and sha(original)==row['original_sha256'],name
print(json.dumps(dict(archive=str(OUT),files=len(entries),stored_bytes=sum(v['stored_size'] for v in entries.values()),
    manifest_sha256=sha((OUT/'manifest.json').read_bytes()),verification=summary),sort_keys=True))
