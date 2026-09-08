#!/usr/bin/env python3
"""Compile the exact EP decode candidate without devices or CUDA initialization."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
import unittest
from glm53_ep_capsule_runtime import verify_runtime


def compile_candidate(output, result):
    os.environ.update(CUTE_DSL_ARCH='sm_121a', CUTE_DSL_KEEP='ptx,cubin',
                      CUTE_DSL_DUMP_DIR=str(output),
                      CUTE_DSL_CACHE_DIR=str(output/'cache'),
                      CUTE_DSL_DISABLE_FILE_CACHING='1',
                      CUTE_DSL_COMPILER_OPT='ptx-options=-v',
                      VLLM_B12X_EP_ZERO_WEIGHT_MICRO='1')
    import torch
    assert not torch.cuda.is_initialized()
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *a, **kw: (12, 1)
    # Import-only hardware query, as in the existing CPU compilation checks.
    import flashinfer.utils
    flashinfer.utils.get_num_sm = lambda *a, **kw: 48
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    md.get_num_sm = lambda *a: 48
    md.get_max_active_clusters = lambda *a: 48
    md.build_and_load_cute_dsl_kernel = lambda module, name, build, **kw: build()
    result['phase'] = 'micro-cute-compile'
    md._MICRO_KERNEL_CACHE.clear()
    for sentinel, tile in ((72, (32,128)), (None, (64,128))):
        md._get_micro_kernel(72,72,8,4096,2048,8,64,
            activation='swigluoai_uninterleave',swiglu_alpha=1.0,
            swiglu_beta=0.0,swiglu_limit=10.0,quant_mode='nvfp4',
            skip_zero_weight_expert_id=sentinel,mac_override=48)
        keys=[key for key in md._MICRO_KERNEL_CACHE if key[17]==sentinel]
        assert len(keys)==1 and keys[0][10]==tile,keys
    result['micro_keys']=list(md._MICRO_KERNEL_CACHE)
    assert len(result['micro_keys'])==2
    artifacts=[]
    for path in sorted(output.rglob('*.ptx')):
        artifacts.append({'path':str(path.relative_to(output)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    resources=[]
    for path in sorted(output.rglob('*.cubin')):
        p=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(path)],text=True,capture_output=True)
        p.check_returncode();path.with_suffix('.resources.log').write_text(p.stdout+p.stderr)
        resources.append({'path':str(path.relative_to(output)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'resources':p.stdout+p.stderr})
    assert len(artifacts)>=2 and len(resources)>=2,'Both fresh CuTe kernels must produce PTX/cubin'
    result.update(micro_artifacts=artifacts,micro_resources=resources)
    result['phase']='fused-prepare-triton-compile'
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import glm53_ep_route_remap as helper
    variants=[]
    for kind in ('mapped','empty','offset'):
        for ids,weight,mapping in itertools.product(('i32','i64'),('fp32','fp16','bf16'),('i32','i64') if kind=='mapped' else ('i32',)):
            label='-'.join((kind,ids,weight,mapping))
            signature=dict(X='*bf16',IDS='*'+ids,WEIGHTS='*'+weight,EXPERT_MAP='*'+mapping,
                           PAD_X='*bf16',PAD_IDS='*i32',PAD_WEIGHTS='*'+weight,LOCAL_OFFSET='i32')
            constants=dict(MAP_LEN=288 if kind=='mapped' else 0,HAS_MAP=kind!='offset')
            kernel=triton.compile(ASTSource(helper._prepare_ep_short_decode_kernel,signature,constexprs=constants),
                                  target=GPUTarget('cuda',121,32),options={'num_warps':4})
            folder=output/'prepare'/label;folder.mkdir(parents=True,exist_ok=False)
            ptx=kernel.asm['ptx'].encode();cubin=kernel.asm['cubin']
            (folder/'kernel.ptx').write_bytes(ptx);(folder/'kernel.cubin').write_bytes(cubin)
            variants.append(dict(label=label,signature=signature,constants=constants,hash=kernel.hash,
                shared_bytes=kernel.metadata.shared,ptx_sha256=hashlib.sha256(ptx).hexdigest(),cubin_sha256=hashlib.sha256(cubin).hexdigest()))
    assert len(variants)==24
    result['prepare_variants']=variants
    assert not torch.cuda.is_initialized()
    result['phase']='cpu-contracts'
    root=Path(__file__).resolve().parents[1]
    names=('test_glm53_ep_micro_tile.py','test_glm53_ep_short_decode.py','test_glm53_ep_route_remap.py')
    suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(root/'tests'),pattern=name) for name in names)
    checked=unittest.TextTestRunner(verbosity=2).run(suite)
    result['contracts']=dict(tests_run=checked.testsRun,failures=len(checked.failures),errors=len(checked.errors),skips=len(checked.skipped))
    assert checked.wasSuccessful() and not checked.skipped,result['contracts']
    assert not torch.cuda.is_initialized(),'CPU compile/tests created a CUDA context'
    result['cuda_initialized']=False
    mounted={}
    for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,target,*_=line.split('\t')
        if '/flashinfer/' in target or name=='flashinfer_b12x_moe.py':
            source=root/'build/glm53'/name
            assert source.read_bytes()==Path(target).read_bytes(),target
            mounted[target]=hashlib.sha256(source.read_bytes()).hexdigest()
    result['mounted_sources']=mounted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capsule-root',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=dict(verdict='FAIL',phase='no-device-guard',started=time.time(),scope='CPU lowering and contracts only; no GPU numerics or throughput')
    try:
        assert not list(Path('/dev').glob('nvidia*')),'CPU container exposes devices'
        runtime=verify_runtime(args.capsule_root,args.manifest_sha256)
        result['binding_runtime']=runtime
        compile_candidate(args.output,result)
        assert verify_runtime(args.capsule_root,args.manifest_sha256)==runtime
        result.update(verdict='PASS',phase='complete',binding_runtime_rechecked=True)
    except BaseException as exc:
        result['error']=repr(exc);raise
    finally:
        result['finished']=time.time()
        (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps({k:result.get(k) for k in ('verdict','phase','error','contracts','cuda_initialized')},default=str),flush=True)

if __name__=='__main__':main()
