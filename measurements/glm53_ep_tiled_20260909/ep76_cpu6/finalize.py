#!/usr/bin/env python3
"""Verify and freeze CPU6 PASS originals with actual Q1 lowering evidence."""
import ast,gzip,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path('/tmp/glm53-ep76-cpu6-archive')
REPO=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
REV='4618859c90131b33c5d9ebd85a67f1537497a357'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
CAPSULE='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
def sha(data):return hashlib.sha256(data).hexdigest()
def git(path):return subprocess.check_output(['git','show',REV+':'+path],cwd=REPO)
raw={str(p.relative_to(ROOT/'originals')):p.read_bytes() for p in (ROOT/'originals').rglob('*') if p.is_file()}
r=json.loads(raw['result.json']);c=json.loads(raw['contracts.json'])
for result in (r,c):
 assert result['verdict']=='PASS' and result['phase']=='complete'
 assert result['contracts']==dict(tests_run=183,failures=0,errors=0,skips=0)
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
load('probes/glm53_ep_tiled_compile.py',('static_specialization','global_static_specialization','opt_static_specialization','opt_shared_capacity','validate_q1_pair_layout'),('STATIC_ROWS','DYNAMIC_ROWS','GLOBAL_STATIC_CASES','OPT_STATIC_CASES','CPU_TESTS','CPU_TEST_COUNTS','EXPECTED_CPU_TESTS','CONTRACT_PATHS'))
assert ns['EXPECTED_CPU_TESTS']==183 and r['selected_test_counts']==ns['CPU_TEST_COUNTS']
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
assert len(r['mounted_sources'])==22 and len(r['contract_sources'])==46
oracle=json.loads(git('measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json'))
assert r['scatter_helper']==dict(path=oracle['source_path'],sha256=oracle['source_sha256'],size=oracle['source_bytes'],helper='scatter_add_v4_bf16x2')
resources=[];files={'result.json','contracts.json'};ptx_proof=[]
for group in ('static','global_static','opt_static','dynamic'):
 for passed in r[group+'_passes']:
  row=passed['resources'][0];met={k:int(v) for k,v in re.findall(r'\b(REG|STACK|LOCAL|SHARED):(\d+)',row['resources'])}
  entry=dict(arm=passed['arm'],**met)
  if group=='opt_static':
   entry.update(capacity=passed['shared_capacity'],q1_pair_layout=passed['q1_pair_layout'])
   assert met==dict(REG=128,STACK=8 if passed['arm']=='opt-static/M6-local' else 0,LOCAL=0,SHARED=1024)
   assert entry['capacity']==dict(dynamic_bytes=98304,static_bytes=1024,total_bytes=99328,block_limit_bytes=101376)
   assert entry['q1_pair_layout']==ns['validate_q1_pair_layout'](entry['q1_pair_layout'],fast_math=passed['cache_key'][6])
   ptxrow=passed['artifacts'][0];data=raw[ptxrow['path']];lines=data.decode().splitlines()
   matches=[(i+1,s.strip()) for i,s in enumerate(lines) if 'shfl.sync' in s]
   assert len(matches)==6
   q1=matches[-3:]
   assert all(re.fullmatch(r'shfl.sync.idx.b32\s+%r\d+, %r\d+, %r\d+, 31, -1;',s) for _,s in q1)
   lo=max(0,q1[0][0]-10);hi=q1[-1][0]+4
   ptx_proof.append(dict(arm=passed['arm'],path=ptxrow['path'],sha256=sha(data),total_shuffle_count=6,
     preexisting_shuffle_count=3,q1_shuffle_count=3,op='shfl.sync.idx.b32',mask_and_clamp=31,membermask_hex='0xffffffff',
     q1_instructions=[dict(line=i,text=s) for i,s in q1],context_first_line=lo+1,
     context='\n'.join(lines[lo:hi]),ptx_local_loads=len(re.findall(r'\bld\.local\.',data.decode())),ptx_local_stores=len(re.findall(r'\bst\.local\.',data.decode()))))
  resources.append(entry)
  files.update(x['path'] for x in passed['artifacts']);files.add(row['path']);files.add(str(Path(row['path']).with_suffix('.resources.log')))
assert set(raw)==files and len(raw)==71 and len(resources)==23
ex=json.loads((ROOT/'execution.json').read_bytes());log=gzip.decompress((ROOT/'fleet.log.gz').read_bytes())
assert ex['revision']==REV and ex['returncode']==0 and b'Ran 183 tests' in log and b'\nOK\n' in log
assert IMAGE in ex['command'][-1] and CAPSULE in ex['command'][-1]
assert r['contracts_process_isolated'] is True
before=json.loads((ROOT/'worker-before.json').read_bytes());after=json.loads((ROOT/'worker-after.json').read_bytes())
assert before['files']==after['files'] and before['head']==after['head']==REV and before['status']==after['status']==''
assert before['capsule_manifest_sha256']==after['capsule_manifest_sha256']==CAPSULE
for p,data in raw.items():assert sha(data)==before['files'][p]['sha256']
summary=dict(verdict='PASSED_CPU_ORIGINALS_VERIFIED',source=REV,result_sha256=sha(raw['result.json']),contracts_sha256=sha(raw['contracts.json']),
 original_files=71,original_bytes=sum(map(len,raw.values())),lowerings=23,contracts=r['contracts'],mounted_sources=22,contract_sources=46,
 cuda_initialized=False,runtime_post_recheck_recorded=True,elapsed_seconds=ex['finished']-ex['started'],resources=resources,
 artifact_validator='frozen source pure validator replayed over original files; no tests/compiler/GPU',
 source_binding='original receipt compared against immutable git 4618859c; worker before/after HEAD and clean state independently retained',
 image=dict(id=IMAGE,evidence='original submission command; no Docker re-inspection during collection'),
 cpu5_artifacts_identical=69,
 gpu_numerics_acceptance=False,performance_acceptance=False)
(ROOT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'source-verification.json').write_text(json.dumps(dict(source=REV,verified=source),indent=2)+'\n')
(ROOT/'q1-ptx-verification.json').write_text(json.dumps(dict(source=REV,variants=ptx_proof,
 scope='PTX opcode, operands, compiler resources and source-bound layout witness only; no GPU numerical or performance claim'),indent=2)+'\n')
rows='\n'.join('| {arm} | {REG} | {STACK} | {LOCAL} | {SHARED} |'.format(**x) for x in resources)
(ROOT/'README.md').write_text(f'''# EP76 CPU6 PASS originals

Source `{REV}`, normal fleet CPU session `epdecode76cpu0910v6`, worker srv4. Terminal rc0 after {summary['elapsed_seconds']:.3f}s. **183 tests PASS, zero failures/errors/skips; 23 actual CuTe lowerings**: seven local, ten global, four optimized static, two dynamic. Frozen artifact and physical mapping receipt validators were replayed against all original files locally. This repeats file validation only, not tests or lowering.

All **69 PTX/cubin/resource files are byte-identical to CPU5** (23 of each type), including the four Q1 pair variants. Full immutable git comparison changes only `tests/test_glm53_ep_tiled_static.py`, fixing the retired AST branch extractor. CPU5 remains a separate original FAIL. All22 mounted source hashes, binding runtime and imported helper identities are unchanged; among46 contract sources only that test hash differs. Exact artifact hashes and before/after source diff are preserved alongside this README.

The four optimized variants carry `glm53_ep_static_sf6_q1_pair_v4`, actual fast-math mode and source-bound R0..8 address/ownership digests. Each records dynamic shared98,304B plus cubin static shared1,024B =99,328B within the101,376B cap. Actual Q1 PTX contains three `shfl.sync.idx.b32` instructions, clamp/mask field31 and member mask-1 (`0xffffffff`). The first peer uses warp-lane XOR1; the next two lane AND30. Exact paths, hashes, lines and context are in `q1-ptx-verification.json`. All four use128 registers; M6-local reports8 stack bytes, global M4/M6/M8 report0. LOCAL0 alone is not a no-spill or performance claim.

| Variant | Registers | Stack bytes | Local bytes | Static shared bytes |
|---|---:|---:|---:|---:|
{rows}

All71 original worker files retain exact decompressed bytes. Their22 mounted and46 contract hashes match immutable git `{REV}`. Imported BF16 helper and binding runtime receipts validate. Both receipts record CUDA uninitialized and successful final runtime recheck. Worker inventories bracket collection and bind source HEAD/clean state and capsule manifest. Image identity is retained from the original submission command; collection did not invoke Docker.

No tests, compiler, GPU or service operation was performed during archival. CPU compilation/contracts and byte equality do not establish GPU numerics, graph correctness, throughput, quality or adoption acceptance.
''')
(ROOT/'collect.py').write_bytes(Path('/tmp/glm53_archive_ep76_cpu6.py').read_bytes())
(ROOT/'collector-reference.py').write_bytes(Path('/tmp/glm53_archive_prep_cpu10.py').read_bytes())
(ROOT/'finalize.py').write_bytes(Path(__file__).read_bytes())
(ROOT/'verify.py').write_bytes(Path('/tmp/glm53_ep76_archive_verify.py').read_bytes())
entries={}
for name,data in raw.items():
 p=ROOT/'originals'/name;compress=p.suffix in ('.ptx','.cubin','.log');dest=p.with_name(p.name+'.gz') if compress else p
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
print(json.dumps(dict(archive=str(ROOT),stored_files=len(sums)+1,manifest_sha256=sha((ROOT/'manifest.json').read_bytes()),result_sha256=summary['result_sha256'],contracts=summary['contracts']),sort_keys=True))
