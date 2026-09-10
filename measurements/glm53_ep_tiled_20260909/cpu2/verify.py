#!/usr/bin/env python3
"""Recheck unchanged CPU2 originals with its actual frozen artifact validator.
No GPU imports, compilation, requests, or remote operations.
"""
import hashlib,importlib.util,json,pathlib,re,sys,tarfile
ROOT=pathlib.Path(__file__).resolve().parent
RESULT_SHA='e36845c14815061e7bc2833152886b3af7d53cc047aa05b3e2b4f9d42f314b83'
REV='f6b0934eb3d14b46cc58c29f6c9983f776eed250'
def sha(raw):return hashlib.sha256(raw).hexdigest()

def verify():
 raw=(ROOT/'result.json').read_bytes();assert sha(raw)==RESULT_SHA
 result=json.loads(raw);before=json.loads((ROOT/'worker-before.json').read_text());after=json.loads((ROOT/'worker-after.json').read_text())
 assert result['verdict']=='PASS' and result['phase']=='complete' and result['compile_only'] is True
 assert result['cuda_initialized'] is False and result['binding_runtime_rechecked'] is True
 assert result['gpu_numerics_acceptance'] is False and result['performance_acceptance'] is False
 assert not any(k in result for k in ('error','recheck_error','cleanup_error'))
 assert before['head']==after['head']==REV and before['status']==after['status']==''
 assert before['image']=='sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
 assert before['source']=='/home/choiceoh/stkernel-ep-tiled-0909-sf6'
 assert before['capsule_manifest_sha256']=='b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
 assert not before['capsule_mismatches'] and before['capsule_files']==116
 assert len(before['files'])==20
 head=json.loads((ROOT/'head-source.json').read_text())
 assert head['head']==REV and head['status']=='' and head['source']==before['source']
 submission=json.loads((ROOT/'submission.json').read_text())
 assert submission['source_revision']==REV and submission['source']==head['source'] and submission['returncode']==0
 for key in ('image','source','capsule','capsule_manifest_sha256','capsule_files','capsule_mismatches','files'):assert before[key]==after[key]
 for name,row in before['files'].items():
  p=ROOT/name;assert not p.is_symlink() and p.is_file()
  raw=p.read_bytes();assert sha(raw)==row['sha256'] and len(raw)==row['size']
 with tarfile.open(ROOT/'original-evidence.tar.gz','r:gz') as t:
  assert {x.name for x in t.getmembers()}==set(before['files'])
  for m in t.getmembers():
   assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
   assert t.extractfile(m).read()==(ROOT/m.name).read_bytes()
 for path,expected in result['contract_sources'].items():
  assert sha((ROOT/'contract-source'/path).read_bytes())==expected
 sys.dont_write_bytecode=True
 sys.path.insert(0,str(ROOT/'contract-source/probes'))
 import glm53_ep_capsule_runtime
 from run_glm53_ep_tiled_cpu import validate_artifacts
 from glm53_ep_tiled_compile import CPU_TEST_COUNTS, EXPECTED_CPU_TESTS
 glm53_ep_capsule_runtime.validate_runtime_receipt(result['binding_runtime'])
 validate_artifacts(ROOT,result)
 assert EXPECTED_CPU_TESTS==96
 assert result['contracts']==dict(tests_run=96,failures=0,errors=0,skips=0)
 assert result['selected_test_counts']==CPU_TEST_COUNTS
 assert result['contracts_process_isolated'] is True
 contracts=json.loads((ROOT/'contracts.json').read_text())
 assert contracts['verdict']=='PASS' and contracts['phase']=='complete'
 assert contracts['cuda_initialized'] is False and contracts['binding_runtime_rechecked'] is True
 assert not any(k in contracts for k in ('error','recheck_error','cleanup_error'))
 for k in ('contracts','selected_test_counts','binding_runtime','mounted_sources','contract_sources'):
  assert contracts[k]==result[k]
 for passed in result['static_passes']:
  m=int(passed['arm'].split('/M')[1]); expected=['glm53_ep_static_tiled_fp32_v1',m,256,48,'torch.int32',False,True,
    [16,128,256] if m<=8 else [32,64,512], [16,256,128] if m<=8 else [32,128,128],
    'nvfp4','sf6_v1','swigluoai_uninterleave',1.,0.,10.,'fp32_scatter']
  assert passed['cache_key']==expected
 for passed in result['dynamic_passes']:
  assert passed['cache_key']==['dynamic','fp4','nvfp4',72,4096,2048,8,48,[128,128],'torch.int32',False,True,
    'swigluoai_uninterleave',1.,0.,10.,False,True,'glm53_ep_prefill_local_fp32_v2','glm53_ep_tiled_sf6_v1']
 resources=[]
 for passed in result['static_passes']+result['dynamic_passes']:
  log=passed['resources'][0]['resources'];fields={k:int(v) for k,v in re.findall(r'\b(REG|STACK|LOCAL|SHARED):(\d+)',log)}
  expected={'REG':121 if passed['arm']=='static/M6' else 118,'STACK':0,'SHARED':1024,'LOCAL':0} if passed['arm'].startswith('static/') else {'REG':168,'STACK':112,'SHARED':1024,'LOCAL':0}
  assert fields==expected
  ptx=(ROOT/passed['artifacts'][0]['path']).read_text()
  resources.append(dict(arm=passed['arm'],**fields,ptx_bytes=len(ptx.encode()),
      cubin_bytes=(ROOT/passed['resources'][0]['path']).stat().st_size,
      ptx_sha256=passed['artifacts'][0]['sha256'],cubin_sha256=passed['resources'][0]['sha256'],
      ptx_local_load_sites=len(re.findall(r'(?m)^\s*ld\.local(?:\.|\s)',ptx)),
      ptx_local_store_sites=len(re.findall(r'(?m)^\s*st\.local(?:\.|\s)',ptx))))
 assert result['dynamic_passes'][0]['cache_key']==result['dynamic_passes'][1]['cache_key']
 assert result['dynamic_passes'][0]['artifacts'][0]['sha256']==result['dynamic_passes'][1]['artifacts'][0]['sha256']
 assert result['dynamic_passes'][0]['resources'][0]['sha256']==result['dynamic_passes'][1]['resources'][0]['sha256']
 return dict(verdict='ORIGINAL_CPU_COMPILE_ARTIFACTS_VERIFIED',source_revision=REV,
   result_sha256=RESULT_SHA,original_files=len(before['files']),original_bytes=sum(x['size'] for x in before['files'].values()),
   original_tar_sha256=sha((ROOT/'original-evidence.tar.gz').read_bytes()),
   actual_frozen_artifact_validator_rerun=True,runtime_receipt_validator_rerun=True,
   mounted_sources=len(result['mounted_sources']),contract_sources=len(result['contract_sources']),
   cuda_initialized=False,compile_only=True,gpu_numerics_acceptance=False,performance_acceptance=False,
   contracts=result['contracts'], contracts_process_isolated=True, scale_mode='sf6_v1',
   dynamic_two_fresh_passes_same_key_and_ptx_cubin=True,resources=resources)
if __name__=='__main__':print(json.dumps(verify(),indent=2))
