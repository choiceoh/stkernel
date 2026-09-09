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


CPU_TEST_MODULES = ('test_glm53_ep_micro_tile.py', 'test_glm53_ep_short_decode.py',
                    'test_glm53_ep_route_remap.py', 'test_glm53_ep_local_selftest.py',
                    'test_glm53_ep_scatter_fp32.py', 'test_glm53_ep_prefill_local.py',
                    'test_glm53_ep_local_probe.py', 'test_glm53_ep_micro_scatter.py',
                    'test_glm53_ep_micro_scatter_fp32.py',
                    'test_glm53_ep_micro_direct_scatter.py',
                    'test_glm53_ep_micro_shared_fc1_a.py',
                    'test_glm53_ep_t6_direct_output.py',
                    'test_glm53_ep_micro_m16.py')
CONTRACT_PATHS = tuple('tests/'+name for name in CPU_TEST_MODULES) + (
    'probes/glm53_ep_short_decode_compile.py', 'probes/run_glm53_ep_short_decode_cpu.py',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/fp4_common.py.gz',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/moe_micro_kernel_cpu11.py.gz',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/micro-kernel-cpu11-identity.json',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/moe_micro_kernel_cpu17.py.gz',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/micro-kernel-cpu17-identity.json',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/moe_micro_kernel_cpu19.py.gz',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/micro-kernel-cpu19-identity.json',
    'measurements/glm53_ep_local_20260908/micro-scatter-ownership/verify.py',
    'measurements/glm53_ep_local_20260908/micro-scatter-ownership/identity.json',
    'measurements/glm53_ep_local_20260908/micro-scatter-ownership/m32-topk8-fp32.ptx.gz',
    'measurements/glm53_ep_local_20260908/micro-scatter-ownership/m64-topk1-fp32.ptx.gz')


def source_receipt(root, *, verify_mounted=False):
    mounted = {}
    names = set()
    for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, target, *_ = line.split('\t')
        if '/flashinfer/' in target or name == 'flashinfer_b12x_moe.py':
            assert target not in mounted, 'duplicate mounted source: '+target
            content = (root/'build/glm53'/name).read_bytes()
            if verify_mounted:
                assert content == Path(target).read_bytes(), target
            mounted[target] = hashlib.sha256(content).hexdigest()
            names.add(name)
    assert {'moe_dispatch.py', 'moe_micro_kernel.py', 'glm53_ep_route_remap.py', 'glm53_ep_local_selftest.py'} <= names
    return dict(mounted_sources=mounted, contract_sources={
        name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CONTRACT_PATHS})


def compile_candidate(output, result):
    os.environ.update(CUTE_DSL_ARCH='sm_121a', CUTE_DSL_KEEP='ptx,cubin',
                      CUTE_DSL_DUMP_DIR=str(output),
                      CUTE_DSL_CACHE_DIR=str(output/'cache'),
                      CUTE_DSL_DISABLE_FILE_CACHING='1',
                      CUTE_DSL_COMPILER_OPT='ptx-options=-v',
                      VLLM_B12X_EP_ZERO_WEIGHT_MICRO='1',
                      VLLM_GLM53_EP_PREFILL_LOCAL='1')
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
    result['micro_passes'] = []
    variants = ((72, 8, 64, (16,128), True, True, True, 'm16-topk8-shared-a-direct-fp32'),
                (None, 1, 8, (64,128), True, False, False, 'm64-topk1-fp32'),
                (None, 8, 64, (64,128), False, False, False, 'm64-topk8-bf16'))
    for sentinel, topk, max_rows, tile, fp32, direct, shared_a, arm in variants:
        assert not list(output.glob('*.ptx')) and not list(output.glob('*.cubin')), 'stale root CuTe artifacts'
        md._get_micro_kernel(72,72,8,4096,2048,topk,max_rows,
            activation='swigluoai_uninterleave',swiglu_alpha=1.0,
            swiglu_beta=0.0,swiglu_limit=10.0,quant_mode='nvfp4',
            skip_zero_weight_expert_id=sentinel,mac_override=48)
        keys=[key for key in md._MICRO_KERNEL_CACHE if key[17]==sentinel and key[7]==topk]
        assert len(keys)==1 and keys[0][10]==tile,keys
        assert ('glm53_ep_micro_scatter_fp32_v1' in keys[0][22:]) is fp32, keys
        assert ('glm53_ep_micro_direct_scatter_v1' in keys[0][22:]) is direct, keys
        assert ('glm53_ep_micro_shared_fc1_a_v1' in keys[0][22:]) is shared_a, keys
        assert (keys[0][-1]=='glm53_ep_micro_m16_v1') is (tile == (16,128)), keys
        if direct:
            assert keys[0][-4:] == ('glm53_ep_micro_scatter_fp32_v1',
                                  'glm53_ep_micro_direct_scatter_v1',
                                  'glm53_ep_micro_shared_fc1_a_v1',
                                  'glm53_ep_micro_m16_v1'), keys
        # Both specializations use the same DSL dump basename. Preserve this
        # pass before the next compile overwrites it; the initialized dump
        # directory remains unchanged throughout the process.
        fresh = {suffix: sorted(output.glob('*'+suffix)) for suffix in ('.ptx', '.cubin')}
        assert all(fresh.values()), 'each fresh micro pass must emit root PTX and cubin'
        folder = output/'micro'/arm
        folder.mkdir(parents=True, exist_ok=False)
        preserved = []
        for paths in fresh.values():
            for path in paths:
                content = path.read_bytes()
                assert content, 'empty CuTe artifact: '+path.name
                digest = hashlib.sha256(content).hexdigest()
                destination = folder/path.name
                assert not destination.exists(), str(destination)
                path.rename(destination)
                assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
                preserved.append(dict(original_name=path.name,
                    path=str(destination.relative_to(output)), sha256=digest))
        result['micro_passes'].append(dict(arm=arm, cache_key=keys[0],
            scatter_fp32=fp32, ep_direct_scatter=direct, shared_fc1_a=shared_a,
            ep_m16=(tile == (16,128)),
            artifacts=preserved))
    result['micro_keys']=list(md._MICRO_KERNEL_CACHE)
    assert len(result['micro_keys'])==3
    artifacts=[]
    for path in sorted(output.rglob('*.ptx')):
        artifacts.append({'path':str(path.relative_to(output)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    resources=[]
    for path in sorted(output.rglob('*.cubin')):
        p=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(path)],text=True,capture_output=True)
        p.check_returncode();path.with_suffix('.resources.log').write_text(p.stdout+p.stderr)
        resources.append({'path':str(path.relative_to(output)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'resources':p.stdout+p.stderr})
    assert len(artifacts)>=3 and len(resources)>=3,'All three fresh micro kernels must produce PTX/cubin'
    result.update(micro_artifacts=artifacts,micro_resources=resources)
    result['phase'] = 'fp32-prefill-cute-compile'
    md._DYNAMIC_KERNEL_CACHE.clear()
    assert not list(output.glob('*.ptx')) and not list(output.glob('*.cubin'))
    md._get_dynamic_kernel(72, 8192, 4096, 2048, 8, 8192,
        activation='swigluoai_uninterleave', swiglu_alpha=1.0,
        swiglu_beta=0.0, swiglu_limit=10.0, tiled=False)
    keys = list(md._DYNAMIC_KERNEL_CACHE)
    assert len(keys) == 1 and keys[0][-1] == 'glm53_ep_prefill_local_fp32_v2', keys
    folder = output/'prefill'/'fp32-v2'
    folder.mkdir(parents=True, exist_ok=False)
    artifacts, resources = [], []
    for suffix, rows in (('.ptx', artifacts), ('.cubin', resources)):
        for path in sorted(output.glob('*'+suffix)):
            destination = folder/path.name
            assert path.stat().st_size > 0
            path.rename(destination)
            row = dict(path=str(destination.relative_to(output)),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            if suffix == '.cubin':
                inspected = subprocess.run(['/usr/local/cuda/bin/cuobjdump',
                    '--dump-resource-usage', str(destination)], text=True, capture_output=True)
                inspected.check_returncode()
                row['resources'] = inspected.stdout+inspected.stderr
                destination.with_suffix('.resources.log').write_text(row['resources'])
            rows.append(row)
    assert artifacts and resources, 'fresh FP32 prefill compilation needs PTX and cubin'
    result['prefill_pass'] = dict(cache_key=keys[0], artifacts=artifacts, resources=resources)
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
    suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(root/'tests'),pattern=name) for name in CPU_TEST_MODULES)
    checked=unittest.TextTestRunner(verbosity=2).run(suite)
    result['contracts']=dict(tests_run=checked.testsRun,failures=len(checked.failures),errors=len(checked.errors),skips=len(checked.skipped))
    assert checked.wasSuccessful() and not checked.skipped,result['contracts']
    assert not torch.cuda.is_initialized(),'CPU compile/tests created a CUDA context'
    result['cuda_initialized']=False


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
        root=Path(__file__).resolve().parents[1]
        sources=source_receipt(root,verify_mounted=True)
        result.update(sources)
        compile_candidate(args.output,result)
        assert source_receipt(root,verify_mounted=True)==sources,'compile/test source changed'
        assert verify_runtime(args.capsule_root,args.manifest_sha256)==runtime
        result.update(verdict='PASS',phase='complete',binding_runtime_rechecked=True)
    except BaseException as exc:
        result['error']=repr(exc);raise
    finally:
        result['finished']=time.time()
        (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps({k:result.get(k) for k in ('verdict','phase','error','contracts','cuda_initialized')},default=str),flush=True)

if __name__=='__main__':main()
