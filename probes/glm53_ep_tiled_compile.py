#!/usr/bin/env python3
"""Fresh EP tiled lowering only, inside the normal no-device CPU fleet job.

This early compiler receipt does not replace startup numerics or onepass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from glm53_ep_capsule_runtime import verify_runtime

STATIC_ROWS = (6, 12, 24, 32)
DYNAMIC_ROWS = (33, 8192)
CONTRACT_PATHS = (
    'probes/glm53_ep_tiled_compile.py', 'probes/run_glm53_ep_tiled_cpu.py',
    'probes/glm53_ep_capsule_runtime.py', 'probes/glm53_ep_bindings_capsule.py',
    'probes/glm53_ep_bindings_pair_check.py',
)


def source_receipt(root, *, verify_mounted=False):
    mounted, names = {}, set()
    for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, target, *_ = line.split('\t')
        if '/flashinfer/' in target or name in ('flashinfer_b12x_moe.py', 'gpu_worker.py'):
            assert target not in mounted, 'duplicate mounted source: '+target
            content = (root/'build/glm53'/name).read_bytes()
            if verify_mounted:
                assert content == Path(target).read_bytes(), target
            mounted[target] = hashlib.sha256(content).hexdigest()
            names.add(name)
    assert {'moe_static_ep_tiled.py','moe_dynamic_ep_local.py','moe_dispatch.py',
            'moe_static_kernel_v4.py','moe_static_kernel_v5.py'} <= names
    return dict(mounted_sources=mounted, contract_sources={
        name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in CONTRACT_PATHS})


def preserve_pass(output, arm, cache_key):
    folder = output/arm
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(arm=arm,cache_key=cache_key,artifacts=[],resources=[])
    for suffix, name in (('.ptx','artifacts'),('.cubin','resources')):
        paths = sorted(output.glob('*'+suffix))
        assert len(paths)==1, 'each fresh pass needs exactly one '+suffix
        for path in paths:
            content = path.read_bytes()
            assert content, 'empty compiler output'
            destination = folder/path.name
            path.rename(destination)
            row = dict(path=str(destination.relative_to(output)),sha256=hashlib.sha256(content).hexdigest())
            if suffix == '.cubin':
                inspected = subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(destination)],text=True,capture_output=True)
                inspected.check_returncode()
                row['resources'] = inspected.stdout+inspected.stderr
                destination.with_suffix('.resources.log').write_text(row['resources'])
            result[name].append(row)
    return result


def compile_candidate(output, result):
    os.environ.update(CUTE_DSL_ARCH='sm_121a',CUTE_DSL_KEEP='ptx,cubin',
        CUTE_DSL_DUMP_DIR=str(output),CUTE_DSL_CACHE_DIR=str(output/'cache'),
        CUTE_DSL_DISABLE_FILE_CACHING='1',CUTE_DSL_COMPILER_OPT='ptx-options=-v',
        VLLM_GLM53_EP_TILED='1',VLLM_GLM53_EP_PREFILL_LOCAL='1')
    import torch
    assert not torch.cuda.is_initialized()
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *a,**kw: (12,1)
    import flashinfer.utils
    flashinfer.utils.get_num_sm = lambda *a,**kw: 48
    import cutlass.cute as cute
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_static_ep_tiled as ep
    result['static_passes'] = []
    for rows in STATIC_ROWS:
        result['phase'] = 'static-M'+str(rows)
        assert not list(output.glob('*.ptx')) and not list(output.glob('*.cubin'))
        kernel,args,key = ep.ep_tiled_compile_spec(num_tokens=rows,max_rows=256,
            max_active_clusters=48,topk_ids_dtype=torch.int32)
        assert key[0] == 'glm53_ep_static_tiled_fp32_v1'
        cute.compile(kernel,*args,options='--opt-level 2 --enable-tvm-ffi')
        result['static_passes'].append(preserve_pass(output,'static/M'+str(rows),key))
        assert not torch.cuda.is_initialized()
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    md.get_num_sm = lambda *a: 48
    md.get_max_active_clusters = lambda *a: 48
    md.build_and_load_cute_dsl_kernel = lambda module,name,build,**kw: build()
    result['dynamic_passes'] = []
    for rows in DYNAMIC_ROWS:
        result['phase'] = 'dynamic-M'+str(rows)
        # Each selected runtime shape receives a fresh lowering, even when
        # both happen to share a dynamic key. This is not startup cost evidence.
        md._DYNAMIC_KERNEL_CACHE.clear()
        assert not list(output.glob('*.ptx')) and not list(output.glob('*.cubin'))
        md._get_dynamic_kernel(72,rows,4096,2048,8,8192,
            activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,
            swiglu_limit=10.,tiled=True,reform_sf_pack=False)
        keys = set(md._DYNAMIC_KERNEL_CACHE)
        assert len(keys)==1, 'each fresh dynamic lowering must publish one key'
        key = next(iter(keys))
        assert key[3:7] == (72,4096,2048,8) and key[17] is True,key
        assert key[-1] == 'glm53_ep_prefill_local_fp32_v2',key
        result['dynamic_passes'].append(preserve_pass(output,'dynamic/M'+str(rows),key))
        assert not torch.cuda.is_initialized()
    result['cuda_initialized'] = torch.cuda.is_initialized()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capsule-root',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    args=p.parse_args()
    result=dict(verdict='FAIL',phase='no-device-guard',started=time.time(),compile_only=True,
                gpu_numerics_acceptance=False,performance_acceptance=False,
                scope='four static and two dynamic EP tiled compiler variants; no GPU or CPU test suite')
    try:
        assert not list(Path('/dev').glob('nvidia*')),'CPU container exposes CUDA devices'
        runtime=verify_runtime(args.capsule_root,args.manifest_sha256)
        result['binding_runtime']=runtime
        root=Path(__file__).resolve().parents[1]
        sources=source_receipt(root,verify_mounted=True)
        result.update(sources)
        compile_candidate(args.output,result)
        assert source_receipt(root,verify_mounted=True)==sources
        assert verify_runtime(args.capsule_root,args.manifest_sha256)==runtime
        result.update(verdict='PASS',phase='complete',binding_runtime_rechecked=True)
    except BaseException as exc:
        result['error']=repr(exc)
        raise
    finally:
        result['finished']=time.time()
        (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps({k:result.get(k) for k in ('verdict','phase','error','cuda_initialized','compile_only')}),flush=True)


if __name__=='__main__':
    main()
