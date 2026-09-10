#!/usr/bin/env python3
"""Verify and freeze CPU7 PASS originals with actual Q1 lowering evidence."""
import ast,gzip,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path('/tmp/glm53-ep76-cpu7-archive')
REPO=Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
REV='ca076d35e64a6a19e90dffe54054269d1a5e1887'
IMAGE='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
CAPSULE='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
assert ROOT.is_dir() and not (ROOT/'manifest.json').exists(), 'fresh collected originals required'
assert all(not (ROOT/name).exists() for name in ('verification.json','source-verification.json','q1-ptx-verification.json','README.md','SHA256SUMS')), 'refuse to overwrite finalized evidence'
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
load('probes/glm53_ep_tiled_compile.py',('static_specialization','global_static_specialization','opt_static_specialization','opt_shared_capacity','validate_q1_register_layout'),('STATIC_ROWS','DYNAMIC_ROWS','GLOBAL_STATIC_CASES','OPT_STATIC_CASES','CPU_TESTS','CPU_TEST_COUNTS','EXPECTED_CPU_TESTS','CONTRACT_PATHS'))
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
assert len(r['mounted_sources'])==22 and len(r['contract_sources'])==47
oracle=json.loads(git('measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json'))
assert r['scatter_helper']==dict(path=oracle['source_path'],sha256=oracle['source_sha256'],size=oracle['source_bytes'],helper='scatter_add_v4_bf16x2')
resources=[];files={'result.json','contracts.json'};ptx_proof=[]
for group in ('static','global_static','opt_static','dynamic'):
 for passed in r[group+'_passes']:
  row=passed['resources'][0];met={k:int(v) for k,v in re.findall(r'\b(REG|STACK|LOCAL|SHARED):(\d+)',row['resources'])}
  entry=dict(arm=passed['arm'],**met)
  if group=='opt_static':
   entry.update(capacity=passed['shared_capacity'],q1_register_layout=passed['q1_register_layout'])
   assert set(met)=={'REG','STACK','LOCAL','SHARED'}
   assert entry['q1_register_layout']==ns['validate_q1_register_layout'](
       entry['q1_register_layout'],fast_math=passed['cache_key'][6])
   assert passed['cache_key'][-1]=='glm53_ep_static_sf6_q1_register_max_v5'
   ptxrow=passed['artifacts'][0];data=raw[ptxrow['path']];text=data.decode();lines=text.splitlines()
   matches=[dict(line=i+1,text=line.strip()) for i,line in enumerate(lines) if 'shfl.sync' in line]
   ptx_proof.append(dict(arm=passed['arm'],path=ptxrow['path'],sha256=sha(data),
       total_shuffle_count=len(matches),shuffle_instructions=matches,
       ptx_local_loads=len(re.findall(r'\bld\.local\.',text)),
       ptx_local_stores=len(re.findall(r'\bst\.local\.',text)),
       scope='Observed complete emitted PTX only; no inherited pair-v4 opcode count or no-spill assertion.'))
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
 original_files=71,original_bytes=sum(map(len,raw.values())),lowerings=23,contracts=r['contracts'],mounted_sources=22,contract_sources=47,
 cuda_initialized=False,runtime_post_recheck_recorded=True,elapsed_seconds=ex['finished']-ex['started'],resources=resources,
 artifact_validator='frozen source pure validator replayed over original files; no tests/compiler/GPU',
 source_binding='original receipt compared against immutable git ca076d35; worker before/after HEAD and clean state independently retained',
 image=dict(id=IMAGE,evidence='original submission command; no Docker re-inspection during collection'),
 gpu_numerics_acceptance=False,performance_acceptance=False)
(ROOT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'source-verification.json').write_text(json.dumps(dict(source=REV,verified=source),indent=2)+'\n')
(ROOT/'q1-ptx-verification.json').write_text(json.dumps(dict(source=REV,variants=ptx_proof,
 scope='PTX opcode, operands, compiler resources and source-bound layout witness only; no GPU numerical or performance claim'),indent=2)+'\n')
rows='\n'.join('| {arm} | {REG} | {STACK} | {LOCAL} | {SHARED} |'.format(**x) for x in resources)
(ROOT/'README.md').write_text(f'''# EP76 CPU7 register-max PASS originals

Source `{REV}`, normal fleet CPU session `epdecode76cpu0910v7`, worker srv4. Terminal rc0 after {summary['elapsed_seconds']:.3f}s. **183 tests PASS, zero failures/errors/skips; 23 actual CuTe lowerings**: seven local, ten global, four optimized static, two dynamic. Frozen artifact and layout validators were replayed against all original files locally. This is file validation, not another test or lowering run.

The four optimized variants carry `glm53_ep_static_sf6_q1_register_max_v5`. Each has an actual source-bound `q1_register_layout` receipt generated during CuTe setup: identity `partition_D` and `partition_S` shape/dense register index checks cover 128 threads and 2,048 values; the full sC1 mapping covers 4,096 bytes. The R0..8 records preserve 512-byte maximum scratch ownership, matching peer loads, packed A2 and scale consumer addresses and their hashes. The receipt binds the actual fast/precise mode and selected low-row branch. Prior FC1, FC2 and Q1 pair receipts cannot substitute for this witness.

The resource table below is the actual cubin report, including any nonzero stack or local values. Dynamic shared storage is checked against 98,304 bytes; actual cubin static storage must stay within 1,024 bytes and total within 101,376 bytes. `q1-ptx-verification.json` preserves observed shuffle instructions and local load/store counts without assuming the previous pair-v4 instruction count. `LOCAL=0` alone is not a no-spill or performance claim. No byte-equality claim is made against CPU5/6; their original evidence remains separate.

| Variant | Registers | Stack bytes | Local bytes | Static shared bytes |
|---|---:|---:|---:|---:|
{rows}

All 71 original worker files retain exact decompressed bytes. Their 22 mounted and 47 contract hashes match immutable git `{REV}`, including the native source oracle from `ep76_onepass4`. Imported BF16 helper and binding runtime receipts validate. Both receipts record CUDA uninitialized and successful final runtime recheck. Worker inventories bracket collection and bind source HEAD/clean state and capsule manifest. Image identity comes from the original submission command; collection invokes no Docker command.

The original result, contracts, execution metadata, fleet stdout, PTX, cubins and resource logs are preserved with separate original/stored hashes. Collection utilities are separate from measured source. No tests, compiler, GPU or service operation is performed by these collection/finalization helpers. CPU compilation and static layout witnesses do not establish GPU numerics, graph correctness, throughput, quality or default acceptance.
''')
(ROOT/'collect.py').write_bytes(Path('/tmp/glm53_archive_ep76_cpu7.py').read_bytes())
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
