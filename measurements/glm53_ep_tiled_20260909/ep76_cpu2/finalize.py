#!/usr/bin/env python3
import ast,gzip,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path('/tmp/glm53-ep76-cpu2-archive')
REPO=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
REV='b41d0da24059379d41c079626cc67c3e83cd14e3'
def sha(data):return hashlib.sha256(data).hexdigest()
def git(path):return subprocess.check_output(['git','show',REV+':'+path],cwd=REPO)
raw={str(p.relative_to(ROOT/'originals')):p.read_bytes() for p in (ROOT/'originals').rglob('*') if p.is_file()}
r=json.loads(raw['result.json']);c=json.loads(raw['contracts.json'])
for result in (r,c):
 assert result['verdict']=='PASS' and result['phase']=='complete'
 assert result['contracts']==dict(tests_run=181,failures=0,errors=0,skips=0)
 assert result['compile_only'] is True and result['cuda_initialized'] is False
 assert result['gpu_numerics_acceptance'] is False and result['performance_acceptance'] is False
 assert result['binding_runtime_rechecked'] is True
 assert not any(k in result for k in ('error','cleanup_error','recheck_error'))
for k in ('binding_runtime','contract_sources','mounted_sources','selected_test_counts','contracts'):assert r[k]==c[k]
ns=dict(Path=Path,hashlib=hashlib,re=re)
def load(path,functions,assignments=()):
 tree=ast.parse(git(path));nodes=[]
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in functions:nodes.append(n)
  elif isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in assignments:nodes.append(n)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
load('probes/glm53_ep_tiled_compile.py',('static_specialization','global_static_specialization','opt_static_specialization','opt_shared_capacity'),('STATIC_ROWS','DYNAMIC_ROWS','GLOBAL_STATIC_CASES','OPT_STATIC_CASES','CPU_TESTS','CPU_TEST_COUNTS','EXPECTED_CPU_TESTS','CONTRACT_PATHS'))
assert ns['EXPECTED_CPU_TESTS']==181 and r['selected_test_counts']==ns['CPU_TEST_COUNTS']
load('probes/run_glm53_ep_tiled_cpu.py',('validate_artifacts',))
ns['validate_artifacts'](ROOT/'originals',r)
load('probes/glm53_ep_capsule_runtime.py',('expected_runtime_receipt','validate_runtime_receipt'),('CAPSULE_MOUNT','CAPSULE_SHA256','SITE','PATHFINDER_FILE','PATHFINDER_METADATA'))
ns['validate_runtime_receipt'](r['binding_runtime'])
manifest=git('build/glm53/manifest.tsv').decode();targets={x.split('\t')[1]:x.split('\t')[0] for x in manifest.splitlines() if x and not x.startswith('#')}
source={}
for target,digest in r['mounted_sources'].items():
 p='build/glm53/'+targets[target];assert sha(git(p))==digest;source[p]=digest
assert set(r['contract_sources'])==set(ns['CONTRACT_PATHS'])
for p,digest in r['contract_sources'].items():assert sha(git(p))==digest;source[p]=digest
assert len(r['mounted_sources'])==22 and len(r['contract_sources'])==43
oracle=json.loads(git('measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json'))
assert r['scatter_helper']==dict(path=oracle['source_path'],sha256=oracle['source_sha256'],size=oracle['source_bytes'],helper='scatter_add_v4_bf16x2')
resources=[];files={'result.json','contracts.json'}
for group in ('static','global_static','opt_static','dynamic'):
 for p in r[group+'_passes']:
  row=p['resources'][0];met={k:int(v) for k,v in re.findall(r'\b(REG|STACK|LOCAL|SHARED):(\d+)',row['resources'])}
  resources.append(dict(arm=p['arm'],**met,**({'capacity':p['shared_capacity']} if group=='opt_static' else {})))
  files.update(x['path'] for x in p['artifacts']);files.add(row['path']);files.add(str(Path(row['path']).with_suffix('.resources.log')))
assert set(raw)==files and len(raw)==71 and len(resources)==23
for x in resources:
 if x['arm'].startswith('opt-static/'):
  assert {k:x[k] for k in ('REG','STACK','LOCAL','SHARED')}==dict(REG=123,STACK=0,LOCAL=0,SHARED=1024)
  assert x['capacity']==dict(dynamic_bytes=100352,static_bytes=1024,total_bytes=101376,block_limit_bytes=101376)
ex=json.loads((ROOT/'execution.json').read_bytes());log=gzip.decompress((ROOT/'fleet.log.gz').read_bytes())
assert ex['revision']==REV and ex['returncode']==0 and b'Ran 181 tests' in log and b'\nOK\n' in log
assert r['contracts_process_isolated'] is True
before=json.loads((ROOT/'worker-before.json').read_bytes());after=json.loads((ROOT/'worker-after.json').read_bytes());assert before['files']==after['files']
for p,data in raw.items():assert sha(data)==before['files'][p]['sha256']
summary=dict(verdict='PASSED_CPU_ORIGINALS_VERIFIED',source=REV,result_sha256=sha(raw['result.json']),contracts_sha256=sha(raw['contracts.json']),
 original_files=71,original_bytes=sum(map(len,raw.values())),lowerings=23,contracts=r['contracts'],mounted_sources=22,contract_sources=43,
 cuda_initialized=False,runtime_post_recheck_recorded=True,elapsed_seconds=ex['finished']-ex['started'],resources=resources,
 artifact_validator='frozen source pure validator replayed over original files; no tests/compiler/GPU',
 source_binding='original receipt compared against immutable git b41d0da2; worker before/after HEAD and clean state independently retained',gpu_numerics_acceptance=False,performance_acceptance=False)
(ROOT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'source-verification.json').write_text(json.dumps(dict(source=REV,verified=source),indent=2)+'\n')
rows='\n'.join('| {arm} | {REG} | {STACK} | {LOCAL} | {SHARED} |'.format(**x) for x in resources)
(ROOT/'README.md').write_text(f'''# EP76 CPU2 PASS originals

Source `{REV}`, normal fleet CPU session `epdecode76cpu0910v2`, worker srv4. Terminal rc0 after {summary['elapsed_seconds']:.3f}s. The original CPU gate reports **181 tests, zero failures/errors/skips**, isolated CPU contracts and **23 actual CuTe lowerings**: baseline seven local, ten global, two dynamic and four optimized static.

The four optimized M6 local/global M4/M6/M8 variants record actual CuTe storage100352B plus cubin static shared1024B, total101376B equal to the asserted block limit. All four use123 registers and zero stack/local bytes. Cache lengths are20 local and24 global. The immutable source validators were replayed over every original artifact path/hash and specialization. Resource values are compiler metadata, not runtime timing or measured occupancy.

| Variant | Registers | Stack bytes | Local bytes | Static shared bytes |
|---|---:|---:|---:|---:|
{rows}

All71 original worker files remain unchanged after decompression. Original22 mounted source hashes and43 contract hashes match immutable source `{REV}`. The imported stock BF16 helper identity and binding capsule receipt validate. Both receipts record CUDA uninitialized and successful final runtime recheck. Before/after inventories bind all files; image and worker HEAD/clean state are retained. CPU1's four test fixture errors remain a separate failed prerequisite and are never reinterpreted as PASS.

Collection used remote file reads and private `/tmp` writes only. The frozen pure artifact validator performed local file checks; no tests, compilation, GPU or service operations were run. This archive establishes CPU compilation and contracts only. GPU numerics, graph behavior, throughput, quality and adoption acceptance remain separate.
''')
# Keep source scripts for audit; collector-reference provides its original helper dependency.
(ROOT/'collect.py').write_bytes(Path('/tmp/glm53_archive_ep76_cpu2.py').read_bytes())
(ROOT/'collector-reference.py').write_bytes(Path('/tmp/glm53_archive_prep_cpu10.py').read_bytes())
(ROOT/'finalize.py').write_bytes(Path(__file__).read_bytes())
entries={}
for name,data in raw.items():
 p=ROOT/'originals'/name
 compress=p.suffix in ('.ptx','.cubin','.log');dest=p.with_name(p.name+'.gz') if compress else p
 stored=gzip.compress(data,mtime=0) if compress else data
 if compress:dest.write_bytes(stored);p.unlink()
 entries['originals/'+name]=dict(path=str(dest.relative_to(ROOT)),encoding='gzip' if compress else 'raw',original_sha256=sha(data),original_size=len(data),stored_sha256=sha(stored),stored_size=len(stored))
for p in sorted(ROOT.iterdir()):
 if not p.is_file():continue
 data=p.read_bytes();original=gzip.decompress(data) if p.suffix=='.gz' else data
 entries[p.name]=dict(path=p.name,encoding='gzip' if p.suffix=='.gz' else 'raw',original_sha256=sha(original),original_size=len(original),stored_sha256=sha(data),stored_size=len(data))
(ROOT/'manifest.json').write_text(json.dumps(dict(schema=1,files=entries),indent=2)+'\n')
sums={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(ROOT.rglob('*')) if p.is_file()}
(ROOT/'SHA256SUMS').write_text(''.join(digest+'  '+p+'\n' for p,digest in sums.items()))
for name,row in entries.items():
 data=(ROOT/row['path']).read_bytes();assert sha(data)==row['stored_sha256']
 original=gzip.decompress(data) if row['encoding']=='gzip' else data
 assert sha(original)==row['original_sha256'] and len(original)==row['original_size']
print(json.dumps(dict(archive=str(ROOT),stored_files=len(sums)+1,manifest_sha256=sha((ROOT/'manifest.json').read_bytes()),result_sha256=summary['result_sha256'],contracts=summary['contracts'],opt_resources=[x for x in resources if x['arm'].startswith('opt-static/')]),sort_keys=True))
