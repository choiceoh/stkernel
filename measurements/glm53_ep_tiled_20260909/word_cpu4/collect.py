#!/usr/bin/env python3
"""Read-only collection of completed EP A-ring CPU originals, not a test run."""
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
OUT = REPO/'measurements/glm53_ep_tiled_20260909/word_cpu4'
REV = '549fd55319c3379438c3cb99694df9f28c01aefb'
RESULT = '453d1dd74340e9bcdb1be67ff2fd5f202c3cf5ab649cc708038a65395e990078'
EVIDENCE = '/home/choiceoh/glm53-ep-tiled-word4-cpu-evidence'
LIMIT = 32*1024*1024

READ = r'''
import hashlib,io,json,pathlib,subprocess,sys,tarfile,time
root=pathlib.Path('/home/choiceoh/glm53-ep-tiled-word4-cpu-evidence')
def inventory():
 result={}
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink(),'symlink in original evidence'
  if p.is_file():result[str(p.relative_to(root))]=p
 assert len(result)==20 and 'result.json' in result and 'contracts.json' in result
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
    assert result['contracts']==dict(tests_run=107,failures=0,errors=0,skips=0)
    assert sum(result['selected_test_counts'].values())==107 and len(result['selected_test_counts'])==11
    assert result['gpu_numerics_acceptance'] is False and result['performance_acceptance'] is False
assert r['contracts_process_isolated'] is True
for key in ('contracts','selected_test_counts','binding_runtime','mounted_sources','contract_sources'):
    assert r[key]==c[key],key
expected=set();resources=[]
for kind,rows in (('static',(6,12,24,32)),('dynamic',(33,8192))):
    passes=r[kind+'_passes'];assert [p['arm'] for p in passes]==[kind+'/M'+str(m) for m in rows]
    for m,p in zip(rows,passes):
        key=p['cache_key']
        if kind=='static':
            ring=m<=8
            assert key[:4]==['glm53_ep_static_tiled_fp32_v1',m,256,48] and key[10]=='sf6_v1'
            assert len(key)==(18 if ring else 16)
            if ring:assert key[-2:]==['glm53_ep_static_sf6_a_ring_v1','glm53_ep_static_sf6_word_unpack_v1']
            assert key[-1]==('glm53_ep_static_sf6_word_unpack_v1' if ring else 'fp32_scatter')
            assert p['specialization']==dict(a_ring=ring,word_unpack=ring,scale_mode='sf6_v1',cache_tag=key[-1])
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
                wanted=dict(REG=123 if m==6 else 118,STACK=0,LOCAL=0,SHARED=1024) if kind=='static' else dict(REG=168,STACK=112,LOCAL=0,SHARED=1024)
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
assert len(r['mounted_sources'])==22 and len(r['contract_sources'])==23
log=Path('/tmp/glm53-ep-word-cpu4.log').read_bytes()
assert b'Ran 107 tests' in log and b'\nOK\n' in log
raw['fleet.log']=log
execution_raw=Path('/tmp/glm53-ep-word-cpu4-execution.json').read_bytes()
execution=json.loads(execution_raw)
source='/home/choiceoh/stkernel-ep-tiled-0909-ring'
assert execution['revision']==REV and execution['source']==source
assert execution['session']=='eptiledcpu0909word4' and execution['output']==EVIDENCE
assert execution['returncode']==0 and execution['finished']>=execution['started']
inner=['nice','-n19','python3','-B',source+'/probes/run_glm53_ep_tiled_cpu.py',
 '--image','sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211',
 '--output',EVIDENCE,'--capsule-root','/home/choiceoh/glm53-ep-cpu-0909-21/capsule',
 '--manifest-sha256','b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab']
command=['env','REPO='+source,'bash',source+'/bench/fleet.sh','run','--cpu','eptiledcpu0909word4','5',
 'EP SF6 word unpack: 6 real lowerings and 107 CPU contracts; no GPU devices','--',
 'ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@10.10.10.4',shlex.join(inner)]
assert execution['command']==command
assert execution['transport_command']==['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2',shlex.join(command)]
raw['execution.json']=execution_raw
raw['ptx-comparison.json']=Path('/tmp/glm53-ep-word-cpu4-ptx-comparison.json').read_bytes()
comparison=json.loads(raw['ptx-comparison.json'])
assert comparison['new_result_sha256']==RESULT
assert comparison['new']['sha256']==r['static_passes'][0]['artifacts'][0]['sha256']
raw['compare.py']=Path('/tmp/glm53_compare_ep_word4_ptx.py').read_bytes()
raw['worker-before.json']=before_raw;raw['worker-after.json']=after_raw
raw['source-verification.json']=(json.dumps(dict(source=REV,verified=source_checks,
    scope='all mounted and contract source hashes equal the exact immutable git CPU source 549fd553; no source rebind required'),indent=2)+'\n').encode()
summary=dict(verdict='ORIGINAL_CPU_EVIDENCE_VERIFIED',source=REV,result_sha256=RESULT,
    original_worker_files=20,original_worker_bytes=sum(v['size'] for v in before['files'].values()),
    execution_sha256=sha(execution_raw),stdout_sha256=sha(log),stdout_bytes=len(log),mounted_sources=len(r['mounted_sources']),contract_sources=len(r['contract_sources']),
    contracts=r['contracts'],resources=resources,compile_only=True,cuda_initialized=False,
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
readme=f'''# EP tiled SF6 word-unpack CPU evidence

Normal CPU fleet session: `eptiledcpu0909word4`, worker `10.10.10.4` (srv4).
Original output: `{EVIDENCE}`. This collection performed read-only archive operations, no tests or GPU work.

The original CPU receipt reports **107 tests, zero failures/errors/skips**, four static and two dynamic CuTe lowerings, and CUDA uninitialized. Runtime identity was rechecked and CPU contracts ran in a separate process. The runner kept the original no-device, no-network, 4 GiB/2 CPU and 12 GiB host-availability constraints.

| Variant | A-ring | Word unpack | Registers | Stack bytes | Local bytes |
|---|---:|---:|---:|---:|---:|
| Static M6 / SF6 | true | true | 123 | 0 | 0 |
| Static M12, M24, M32 / SF6 | false | false | 118 | 0 | 0 |
| Dynamic M33, M8192 / SF6 | unchanged | unchanged | 168 | 112 | 0 |

The actual M6 constructor flags were checked before and after lowering; its 18-field key ends with `glm53_ep_static_sf6_a_ring_v1`, then `glm53_ep_static_sf6_word_unpack_v1`. Other static keys retain 16 fields. Source `{REV}` is unchanged; all **22 mounted sources and 23 contracts** match its immutable committed bytes. No source rebind was needed. Original result SHA: `{RESULT}`.

Against `../ring_cpu3`, M6's first emitted PTX SF6 restore region, strictly between its read/write `bar.sync 3,128`, changes integer ALU/address instructions **130 to 80**, and total instructions **134 to 84**. Four shared stores remain. Each of the six emitted restore regions has 50 fewer instructions; all 12 barrier-3 instructions remain. REG123/STACK0/LOCAL0 is unchanged. These are static PTX counts, excluding the preceding packed loads; they are not SASS timing, numerical or performance acceptance. Full hashes, line ranges, opcode counts and the first-region excerpts are in `ptx-comparison.json`. `compare.py` is a post-measurement read-only utility; reproduce with `python3 compare.py --old-root ../ring_cpu3 --new-root .` from this directory.

All **20 original worker files**, before/after inventories, exact execution metadata, CPU stdout and compiler outputs are retained. `execution.json` was written by the process that executed the exact normal fleet command, with rc0 and start/end times; total execution was {execution["finished"]-execution["started"]:.3f} seconds. Its stdout hash is bound by `verification.json`. Logs/PTX/cubins use deterministic gzip; original and stored hashes/sizes are in `manifest.json`. Raw JSON contents are unchanged. The archive adds no runtime-source changes.

This is CPU compilation and contract evidence only. GPU numerical correctness, graph behavior, throughput, quality and default-adoption acceptance remain separate; none is claimed here.
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
