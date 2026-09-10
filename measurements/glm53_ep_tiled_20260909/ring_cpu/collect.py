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
OUT = REPO/'measurements/glm53_ep_tiled_20260909/ring_cpu'
REV = '5663b1beb13e97116e6244af02a22ecc5074a263'
RESULT = '30a05469b124b567573d873ec03e27954398d732ba63879d04fced72a80b565e'
EVIDENCE = '/home/choiceoh/glm53-ep-tiled-ring-cpu-evidence'
REBIND = '/home/choiceoh/glm53-ep-tiled-ring-cpu-rebind.json'
LIMIT = 32*1024*1024

READ = r'''
import hashlib,io,json,pathlib,subprocess,sys,tarfile,time
root=pathlib.Path('/home/choiceoh/glm53-ep-tiled-ring-cpu-evidence')
rebind=pathlib.Path('/home/choiceoh/glm53-ep-tiled-ring-cpu-rebind.json')
def inventory():
 result={}
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink(),'symlink in original evidence'
  if p.is_file():result[str(p.relative_to(root))]=p
 assert len(result)==20 and 'result.json' in result and 'contracts.json' in result
 result['rebind.json']=rebind
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
r=json.loads(raw['result.json']);c=json.loads(raw['contracts.json']);rebind=json.loads(raw['rebind.json'])
for result in (r,c):
    assert result['verdict']=='PASS' and result['phase']=='complete'
    assert result['cuda_initialized'] is False and result['binding_runtime_rechecked'] is True
    assert result['compile_only'] is True
    assert not any(k in result for k in ('error','cleanup_error','recheck_error'))
    assert result['contracts']==dict(tests_run=101,failures=0,errors=0,skips=0)
    assert sum(result['selected_test_counts'].values())==101 and len(result['selected_test_counts'])==10
    assert result['gpu_numerics_acceptance'] is False and result['performance_acceptance'] is False
assert r['contracts_process_isolated'] is True
for key in ('contracts','selected_test_counts','binding_runtime','mounted_sources','contract_sources'):
    assert r[key]==c[key],key
assert rebind['source']==REV and rebind['cpu_result_sha256']==RESULT
assert rebind['mounted_and_contract_source_bytes_match'] is True and rebind['tests_run']==101
expected=set();resources=[]
for kind,rows in (('static',(6,12,24,32)),('dynamic',(33,8192))):
    passes=r[kind+'_passes'];assert [p['arm'] for p in passes]==[kind+'/M'+str(m) for m in rows]
    for m,p in zip(rows,passes):
        key=p['cache_key']
        if kind=='static':
            ring=m<=8
            assert key[:4]==['glm53_ep_static_tiled_fp32_v1',m,256,48] and key[10]=='sf6_v1'
            assert len(key)==(17 if ring else 16)
            assert key[-1]==('glm53_ep_static_sf6_a_ring_v1' if ring else 'fp32_scatter')
            assert p['specialization']==dict(a_ring=ring,scale_mode='sf6_v1',cache_tag=key[-1])
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
assert expected|{'result.json','contracts.json','rebind.json'}==set(raw)
manifest=git_bytes('build/glm53/manifest.tsv').decode()
targets={x.split('\t')[1]:x.split('\t')[0] for x in manifest.splitlines() if x and not x.startswith('#')}
source_checks={}
for target,want in r['mounted_sources'].items():
    path='build/glm53/'+targets[target];assert sha(git_bytes(path))==want,target
    source_checks[path]=want
for path,want in r['contract_sources'].items():
    assert sha(git_bytes(path))==want,path;source_checks[path]=want
log=Path('/tmp/glm53-ep-ring-cpu2.log').read_bytes()
assert b'Ran 101 tests' in log and b'\nOK\n' in log
raw['fleet.log']=log
raw['failed-first-attempt.log']=Path('/tmp/glm53-ep-ring-cpu.log').read_bytes()
source='/home/choiceoh/stkernel-ep-tiled-0909-ring'
payload=['nice','-n19','python3','-B',source+'/probes/run_glm53_ep_tiled_cpu.py',
    '--image',before['image'],'--output',EVIDENCE,'--capsule-root','/home/choiceoh/glm53-ep-cpu-0909-21/capsule',
    '--manifest-sha256',before['capsule_manifest_sha256']]
submission=dict(scope='reconstructed from parent execution-tool record; not original submission stdout',
    command=['env','REPO='+source,'bash',source+'/bench/fleet.sh','run','--cpu','eptiledcpu0909ring2','5',
        'EP SF6 A-ring 6 real lowerings and 101 CPU contracts; no GPU devices','--','ssh','-o','BatchMode=yes',
        '-o','ConnectTimeout=8','choiceoh@10.10.10.4',shlex.join(payload)],
    returncode=0,returncode_scope='parent execution-tool report',first_attempt=dict(session='eptiledcpu0909ring',
        returncode=2,reason='worker source directory missing; no compilation or GPU execution',scope='parent execution-tool report; original failed log preserved separately'))
raw['submission-reconstructed.json']=(json.dumps(submission,indent=2)+'\n').encode()
raw['worker-before.json']=before_raw;raw['worker-after.json']=after_raw
raw['source-verification.json']=(json.dumps(dict(source=REV,verified=source_checks,
    scope='all mounted and contract source hashes equal immutable git 5663b1b; original result and explicit worker rebind retained'),indent=2)+'\n').encode()
summary=dict(verdict='ORIGINAL_CPU_EVIDENCE_VERIFIED',source=REV,result_sha256=RESULT,
    original_worker_files=20,original_worker_bytes=sum(v['size'] for k,v in before['files'].items() if k!='rebind.json'),
    rebind_sha256=sha(raw['rebind.json']),mounted_sources=len(r['mounted_sources']),contract_sources=len(r['contract_sources']),
    contracts=r['contracts'],resources=resources,compile_only=True,cuda_initialized=False,
    gpu_numerics_acceptance=False,performance_acceptance=False,source_rebound_without_new_compile=True)
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
readme=f'''# EP tiled SF6 A-ring CPU evidence

Normal CPU fleet session: `eptiledcpu0909ring2`, worker `10.10.10.4` (srv4).
Original output: `{EVIDENCE}`. This archive validates completed originals; collection ran no tests or GPU work.

The original CPU receipt reports **101 tests, zero failures/errors/skips**, four static and two dynamic CuTe lowerings, and CUDA uninitialized. Runtime identity was rechecked and CPU contracts ran in a separate process.

| Variant | A-ring | Registers | Stack bytes | Local bytes |
|---|---:|---:|---:|---:|
| Static M6 / SF6 | true | 123 | 0 | 0 |
| Static M12, M24, M32 / SF6 | false | 118 | 0 | 0 |
| Dynamic M33, M8192 / SF6 | unchanged | 168 | 112 | 0 |

M6 carries `glm53_ep_static_sf6_a_ring_v1`; the other static keys retain their previous 16 fields. Resource values describe compiled artifacts, not measured runtime occupancy or performance.

Final source `{REV}` committed a generated build snapshot with the bytes already compiled. `rebind.json` preserves the worker's explicit rebind, and `source-verification.json` independently checks every mounted and contract hash against that immutable commit. This was **not another compile**. The original result SHA is `{RESULT}` and is unchanged.

All 20 original worker files, their before/after inventory, original rebind, and local captured CPU stdout are retained. The original first-attempt log is separate: that attempt returned 2 because the worker source directory was absent, before compilation. `submission-reconstructed.json` explicitly reconstructs the successful normal fleet command and reported exit 0 from the parent's execution-tool record; it is not original submission stdout. Logs/PTX/cubins use deterministic gzip; original and stored hashes/sizes are in `manifest.json`. Raw JSON receipt contents are unchanged. No runtime files or historical archives were edited.

This is CPU compilation and contract evidence only. GPU numerical correctness, graph behavior, throughput, quality, and default adoption acceptance remain separate. No such acceptance is claimed here.
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
