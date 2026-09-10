#!/usr/bin/env python3
"""Read-only CPU10 original evidence collection; writes only private /tmp staging."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

REPO=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
OUT=Path('/tmp/glm53-ep-prep-cpu10-archive')
REV='388aabdd79e874808262e15c0b974dc8ccca6f07'
REVISIONS={10:REV}
SOURCE='/home/choiceoh/stkernel-ep-tiled-0909-ring'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
CAPSULE='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
LIMIT=128*1024*1024
READ=r'''
import hashlib,io,json,pathlib,subprocess,sys,tarfile,time
roots={n:pathlib.Path('/home/choiceoh/glm53-ep-prep-%d-cpu-evidence'%n) for n in (10,)}
def inventory():
 files={};raw={};presence={}
 for n,root in roots.items():
  assert root.is_dir() and not root.is_symlink()
  names={str(p.relative_to(root)):p for p in sorted(root.rglob('*')) if p.is_file()}
  assert all(not p.is_symlink() for p in root.rglob('*'))
  presence[str(n)]=dict(original_file_count=len(names),result='result.json' in names,contracts='contracts.json' in names)
  selected=names
  for name,p in selected.items():
   prefix='cpu10/'
   a=p.stat();data=p.read_bytes();b=p.stat()
   assert (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
   key=prefix+name
   files[key]=dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),size=len(data),mtime_ns=a.st_mtime_ns,inode=a.st_ino)
   raw[key]=data
 assert sum(map(len,raw.values()))<=128*1024*1024
 return files,raw,presence
files,raw,presence=inventory()
if sys.argv[1]=='inventory':
 def run(args):return subprocess.check_output(args,text=True,timeout=15).strip()
 source='/home/choiceoh/stkernel-ep-tiled-0909-ring'
 capsule=pathlib.Path('/home/choiceoh/glm53-ep-cpu-0909-21/capsule/capsule-manifest.json')
 print(json.dumps(dict(captured_at=time.time(),files=files,presence=presence,source=source,
  head=run(['git','-C',source,'rev-parse','HEAD']),status=run(['git','-C',source,'status','--porcelain']),
  image=run(['docker','image','inspect','--format','{{.Id}}','sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211']),
  capsule_manifest_sha256=hashlib.sha256(capsule.read_bytes()).hexdigest()),sort_keys=True));raise SystemExit
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as tar:
 for name,data in raw.items():
  entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o600
  tar.addfile(entry,io.BytesIO(data))
assert inventory()[0]==files,'originals changed during read'
sys.stdout.buffer.write(buffer.getvalue())
'''

def sha(raw):return hashlib.sha256(raw).hexdigest()
def remote(mode):
 worker=shlex.join(['python3','-B','-c',READ,mode])
 head=shlex.join(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@10.10.10.4',worker])
 result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@srv2',head],capture_output=True,timeout=90)
 if result.returncode:raise RuntimeError('read-only evidence read failed rc='+str(result.returncode)+' '+result.stderr.decode()[-2000:])
 assert len(result.stdout)<=LIMIT
 return result.stdout

def git_bytes(path,revision=REV):
 return subprocess.check_output(['git','show',revision+':'+path],cwd=REPO)

def load_pure(path,functions,assignments=()):
 raw=git_bytes(path);tree=ast.parse(raw)
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions
  or isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in assignments]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),VALIDATOR)
 return raw

def main():
 assert not OUT.exists(),'fresh private archive required'
 execution_raw=Path('/tmp/glm53-ep-prep-cpu10-execution.json').read_bytes()
 execution=json.loads(execution_raw)
 assert execution.get('returncode')==0 and execution.get('finished',0)>=execution['started'],'CPU10 not terminal PASS'
 before_raw=remote('inventory');before=json.loads(before_raw)
 bundle=remote('bundle');after_raw=remote('inventory');after=json.loads(after_raw)
 assert before['files']==after['files'] and before['presence']==after['presence']
 for name in ('source','head','status','image','capsule_manifest_sha256'):assert before[name]==after[name]
 assert before['source']==SOURCE and before['head']==REV and before['status']==''
 assert before['image']==IMAGE and before['capsule_manifest_sha256']==CAPSULE
 originals={}
 with tarfile.open(fileobj=io.BytesIO(bundle),mode='r:') as tar:
  assert len(tar.getmembers())==len(before['files'])
  for entry in tar.getmembers():
   name=entry.name;p=Path(name)
   assert entry.isfile() and not p.is_absolute() and '..' not in p.parts
   assert name not in originals and name in before['files']
   raw=tar.extractfile(entry).read();record=before['files'][name]
   assert len(raw)==record['size'] and sha(raw)==record['sha256']
   originals[name]=raw
 assert set(originals)==set(before['files'])
 worker={k.removeprefix('cpu10/'):v for k,v in originals.items() if k.startswith('cpu10/')}
 r=json.loads(worker['result.json']);c=json.loads(worker['contracts.json'])
 for receipt in (r,c):
  assert receipt['verdict']=='PASS' and receipt['phase']=='complete'
  assert receipt['cuda_initialized'] is False and receipt['binding_runtime_rechecked'] is True
  assert receipt['compile_only'] is True and receipt['gpu_numerics_acceptance'] is False and receipt['performance_acceptance'] is False
  assert not any(k in receipt for k in ('error','cleanup_error','recheck_error'))
  assert receipt['contracts']==dict(tests_run=158,failures=0,errors=0,skips=0)
 assert r['contracts_process_isolated'] is True
 for key in ('contracts','selected_test_counts','binding_runtime','mounted_sources','contract_sources'):assert r[key]==c[key]
 probe=load_pure('probes/glm53_ep_tiled_compile.py',('static_specialization','global_static_specialization'),
  ('STATIC_ROWS','DYNAMIC_ROWS','GLOBAL_STATIC_CASES','CPU_TESTS','CPU_TEST_COUNTS','EXPECTED_CPU_TESTS','CONTRACT_PATHS'))
 assert r['selected_test_counts']==VALIDATOR['CPU_TEST_COUNTS'] and VALIDATOR['EXPECTED_CPU_TESTS']==158
 load_pure('probes/glm53_ep_capsule_runtime.py',('expected_runtime_receipt','validate_runtime_receipt'),
  ('CAPSULE_MOUNT','CAPSULE_SHA256','SITE','PATHFINDER_FILE','PATHFINDER_METADATA'))
 VALIDATOR['validate_runtime_receipt'](r['binding_runtime'])
 expected=set();resources=[]
 groups=(('static',VALIDATOR['STATIC_ROWS']),('global_static',VALIDATOR['GLOBAL_STATIC_CASES']),('dynamic',VALIDATOR['DYNAMIC_ROWS']))
 for kind,cases in groups:
  passes=r[kind+'_passes']
  arms=['global-static/'+case[0] for case in cases] if kind=='global_static' else [kind+'/M'+str(m) for m in cases]
  assert [p['arm'] for p in passes]==arms
  for case,p in zip(cases,passes):
   key=p['cache_key']
   if kind in ('static','global_static'):
    sp=p['specialization'];args=(case,key,sp['a_ring'],sp['word_unpack'],sp['scatter_bf16'],sp['output_dtype'])
    actual=VALIDATOR['global_static_specialization'](*args,sp['route']) if kind=='global_static' else VALIDATOR['static_specialization'](*args)
    assert sp==actual
   else:
    assert key[3:7]==[72,4096,2048,8] and key[17] is True
    assert key[-2:]==['glm53_ep_prefill_local_fp32_v2','glm53_ep_tiled_sf6_v1']
   for field,suffix in (('artifacts','.ptx'),('resources','.cubin')):
    assert len(p[field])==1
    row=p[field][0];name=row['path'];rel=Path(name)
    assert str(rel.parent)==p['arm'] and rel.suffix==suffix and name not in expected
    assert name in worker and sha(worker[name])==row['sha256'];expected.add(name)
    if suffix=='.cubin':
     logfile=str(rel.with_suffix('.resources.log'))
     assert worker[logfile].decode()==row['resources'];expected.add(logfile)
     metrics={k:int(v) for k,v in re.findall(r'\b(REG|STACK|LOCAL|SHARED):(\d+)',row['resources'])}
     assert set(metrics)=={'REG','STACK','LOCAL','SHARED'}
     resources.append(dict(arm=p['arm'],**metrics))
 assert expected|{'result.json','contracts.json'}==set(worker) and len(worker)==59
 assert len(resources)==19
 source_checks={};manifest=git_bytes('build/glm53/manifest.tsv').decode()
 targets={x.split('\t')[1]:x.split('\t')[0] for x in manifest.splitlines() if x and not x.startswith('#')}
 for target,digest in r['mounted_sources'].items():
  path='build/glm53/'+targets[target];assert sha(git_bytes(path))==digest;source_checks[path]=digest
 assert set(r['contract_sources'])==set(VALIDATOR['CONTRACT_PATHS'])
 for path,digest in r['contract_sources'].items():
  assert sha(git_bytes(path))==digest;source_checks[path]=digest
 assert len(r['mounted_sources'])==22 and len(r['contract_sources'])==39
 identity=json.loads(git_bytes('measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json'))
 assert r['scatter_helper']==dict(path=identity['source_path'],sha256=identity['source_sha256'],size=identity['source_bytes'],helper='scatter_add_v4_bf16x2')
 raw={};raw.update(worker)
 ex=execution_raw;meta=execution
 log=Path('/tmp/glm53-ep-prep-cpu10.log').read_bytes()
 assert meta['revision']==REV and meta['source']==SOURCE and meta['output']=='/home/choiceoh/glm53-ep-prep-10-cpu-evidence'
 assert meta['session']=='eptiledcpu0910prep10' and meta['finished']>=meta['started'] and meta['returncode']==0
 assert meta['command'][:7]==['env','REPO='+SOURCE,'bash',SOURCE+'/bench/fleet.sh','run','--cpu',meta['session']]
 assert meta['command'][10:]==['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','choiceoh@10.10.10.4',
   shlex.join(['nice','-n19','python3','-B',SOURCE+'/probes/run_glm53_ep_tiled_cpu.py','--image',IMAGE,'--output',meta['output'],
    '--capsule-root','/home/choiceoh/glm53-ep-cpu-0909-21/capsule','--manifest-sha256',CAPSULE])]
 raw['execution.json']=ex;raw['fleet.log']=log
 assert b'Ran 158 tests' in log and b'\nOK\n' in log
 raw['worker-before.json']=before_raw;raw['worker-after.json']=after_raw
 summary=dict(verdict='ORIGINAL_CPU_EVIDENCE_VERIFIED',source=REV,result_sha256=sha(worker['result.json']),
  original_worker_files=59,original_worker_bytes=sum(map(len,worker.values())),lowerings=19,contracts=r['contracts'],
  mounted_sources=22,contract_sources=39,cuda_initialized=False,binding_runtime_rechecked=True,
  scatter_helper=r['scatter_helper'],resources=resources,
  compile_only=True,gpu_numerics_acceptance=False,performance_acceptance=False)
 raw['verification.json']=(json.dumps(summary,indent=2)+'\n').encode()
 raw['source-verification.json']=(json.dumps(dict(source=REV,verified=source_checks),indent=2)+'\n').encode()
 OUT.mkdir()
 entries={}
 def store(name,data):
  compressed=name.endswith(('.log','.ptx','.cubin'))
  stored=name+'.gz' if compressed else name
  content=gzip.compress(data,mtime=0) if compressed else data
  path=OUT/stored;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
  entries[name]=dict(path=stored,encoding='gzip' if compressed else 'raw',original_sha256=sha(data),original_size=len(data),
    stored_sha256=sha(content),stored_size=len(content))
 for name,data in raw.items():store(name,data)
 # Replay the frozen actual artifact validator over locally decoded originals;
 # this is file validation only, with no compiler/runtime/GPU imports.
 load_pure('probes/run_glm53_ep_tiled_cpu.py',('validate_artifacts',))
 validation_root=OUT/'_verification_raw';validation_root.mkdir()
 for name,data in worker.items():
  path=validation_root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 VALIDATOR['validate_artifacts'](validation_root,r)
 # Remove only our temporary decoded copies within this freshly-owned archive.
 for path in sorted(validation_root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if path.is_file():path.unlink()
  elif path.is_dir():path.rmdir()
 validation_root.rmdir()
 table='\n'.join('| {arm} | {REG} | {STACK} | {LOCAL} | {SHARED} |'.format(**x) for x in resources)
 readme=f'''# EP native route fusion + PREP_FUSED checkpoint logging CPU10 originals

Source `{REV}`. Normal no-device CPU session `eptiledcpu0910prep10` ran through the head fleet controller with a bounded SSH CPU payload on srv4. The original receipt reports **158 tests, zero failures/errors/skips**, and **19 actual CuTe lowerings**: seven local static, ten global-route static and two dynamic. CUDA stayed uninitialized; runtime identity was rechecked and CPU contracts ran in a fresh process.

All **59 original worker files** are preserved: result/contracts JSON plus 19 PTX, 19 cubins and 19 resource logs. The source receipt's **22 mounted sources and 39 contracts** match the exact immutable git source. Actual global constructor/map fake ABI and every artifact path/hash were checked using the frozen pure validators. The pinned imported BF16 helper source identity and capsule receipt match. Before/after worker inventories confirm byte stability; image ID and current worker source HEAD/clean status are retained.

| Variant | Registers | Stack bytes | Local bytes | Static shared bytes |
|---|---:|---:|---:|---:|
{table}

Resource values are compiler metadata, not measured timing or occupancy. `LOCAL=0` alone does not establish absence of stack spills. This archive makes no GPU numerics, graph, performance, quality or adoption claim.

The earlier CPU7/8 failures remain preserved in the separate committed CPU9 archive; they are not duplicated or reinterpreted here.

All logs/PTX/cubins use deterministic gzip with both original and stored hashes/sizes in `manifest.json`; raw JSON bytes are unchanged. Collection involved file reads and private `/tmp` writes only. No tests, compilation, fleet submission, GPU or service action was performed.
'''
 store('README.md',readme.encode());store('collect.py',Path(__file__).read_bytes())
 (OUT/'manifest.json').write_text(json.dumps(dict(schema=1,files=entries),indent=2)+'\n')
 sums={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file()}
 (OUT/'SHA256SUMS').write_text(''.join(digest+'  '+name+'\n' for name,digest in sums.items()))
 for name,row in entries.items():
  data=(OUT/row['path']).read_bytes();assert sha(data)==row['stored_sha256']
  original=gzip.decompress(data) if row['encoding']=='gzip' else data
  assert len(original)==row['original_size'] and sha(original)==row['original_sha256'],name
 print(json.dumps(dict(archive=str(OUT),stored_files=len(sums)+1,manifest_sha256=sha((OUT/'manifest.json').read_bytes()),verification=summary),sort_keys=True))

VALIDATOR=dict(Path=Path,hashlib=hashlib)
if __name__=='__main__':main()
