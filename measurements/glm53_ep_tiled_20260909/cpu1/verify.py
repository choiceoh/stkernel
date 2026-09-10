#!/usr/bin/env python3
"""Recheck unchanged CPU1 originals with its actual frozen artifact validator.
No GPU imports, compilation, requests, or remote operations.
"""
import hashlib,importlib.util,json,pathlib,re,sys,tarfile
ROOT=pathlib.Path(__file__).resolve().parent
RESULT_SHA='5441230b4a0630dda85e643e4f9cb1f5bb74e69b381a3f3704220af7d3d52f62'
REV='a62d492b90caf712d0522528c0baa02926c256d6'
def sha(raw):return hashlib.sha256(raw).hexdigest()

def verify():
 raw=(ROOT/'result.json').read_bytes();assert sha(raw)==RESULT_SHA
 result=json.loads(raw);before=json.loads((ROOT/'worker-before.json').read_text());after=json.loads((ROOT/'worker-after.json').read_text())
 assert result['verdict']=='PASS' and result['phase']=='complete' and result['compile_only'] is True
 assert result['cuda_initialized'] is False and result['binding_runtime_rechecked'] is True
 assert result['gpu_numerics_acceptance'] is False and result['performance_acceptance'] is False
 assert not any(k in result for k in ('error','recheck_error','cleanup_error'))
 assert before['head']==after['head']==REV and before['status']==after['status']==''
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
 glm53_ep_capsule_runtime.validate_runtime_receipt(result['binding_runtime'])
 validate_artifacts(ROOT,result)
 resources=[]
 for passed in result['static_passes']+result['dynamic_passes']:
  log=passed['resources'][0]['resources'];fields={k:int(v) for k,v in re.findall(r'\b(REG|STACK|LOCAL|SHARED):(\d+)',log)}
  expected={'REG':115,'STACK':0,'SHARED':1024,'LOCAL':0} if passed['arm'].startswith('static/') else {'REG':168,'STACK':112,'SHARED':1024,'LOCAL':0}
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
   dynamic_two_fresh_passes_same_key_and_ptx_cubin=True,resources=resources)
if __name__=='__main__':print(json.dumps(verify(),indent=2))
