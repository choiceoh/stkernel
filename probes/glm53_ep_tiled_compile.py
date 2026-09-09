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
import sys
import time
import unittest

from glm53_ep_capsule_runtime import verify_runtime

STATIC_ROWS = (6, 12, 24, 32)
DYNAMIC_ROWS = (33, 8192)
CPU_TESTS = ("test_glm53_ep_tiled_static.py", "test_glm53_ep_tiled_prefill.py",
             "test_glm53_ep_tiled_owner.py", "test_glm53_ep_tiled_selftest.py",
             "test_glm53_ep_tiled_proof.py", "test_moe_sf6_owner.py",
             "test_moe_static_sf6_direct.py", "test_moe_dynamic_sf6.py",
             "test_moe_sf6_dispatch.py", "test_glm53_ep_tiled_a_ring.py",
             "test_glm53_ep_tiled_sf6_word_unpack.py")
CPU_TEST_COUNTS = dict(zip(CPU_TESTS, (10, 12, 19, 12, 12, 12, 6, 7, 6, 5, 6)))
EXPECTED_CPU_TESTS = 107
CONTRACT_PATHS = (
    'probes/glm53_ep_tiled_compile.py', 'probes/run_glm53_ep_tiled_cpu.py',
    'probes/glm53_ep_capsule_runtime.py', 'probes/glm53_ep_bindings_capsule.py',
    'probes/glm53_ep_bindings_pair_check.py',
    'profiles/glm53.env', 'bench/proof.py', 'bench/proof-markers.tsv',
    'overlay/modules/glm53_model/glm5next_model.py',
    'measurements/glm53_ep_local_20260908/onepass20-completed/source/moe_dynamic_ep_local.py.gz',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/fp4_common.py.gz',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json',
) + tuple('tests/' + name for name in CPU_TESTS)


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


def static_specialization(rows, key, a_ring, word_unpack):
    """Bind actual constructor selection and its persisted cache namespace."""
    assert rows in STATIC_ROWS and type(a_ring) is bool and type(word_unpack) is bool
    assert tuple(key[:4]) == ('glm53_ep_static_tiled_fp32_v1', rows, 256, 48), key
    assert key[10] == 'sf6_v1', key
    expected_ring = rows <= 8
    assert a_ring is expected_ring, (rows, a_ring)
    assert word_unpack is expected_ring, (rows, word_unpack)
    if expected_ring:
        assert len(key) == 18 and tuple(key[-3:]) == (
            'fp32_scatter', 'glm53_ep_static_sf6_a_ring_v1',
            'glm53_ep_static_sf6_word_unpack_v1'), key
    else:
        assert len(key) == 16 and key[-1] == 'fp32_scatter', key
    return dict(a_ring=a_ring, word_unpack=word_unpack,
                scale_mode=key[10], cache_tag=key[-1])


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
            max_active_clusters=48,topk_ids_dtype=torch.int32,reform_sf_pack=True)
        specialization = static_specialization(rows, key, kernel.a_ring, kernel.word_unpack)
        cute.compile(kernel,*args,options='--opt-level 2 --enable-tvm-ffi')
        assert static_specialization(rows, key, kernel.a_ring, kernel.word_unpack) == specialization
        passed = preserve_pass(output,'static/M'+str(rows),key)
        passed['specialization'] = specialization
        result['static_passes'].append(passed)
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
            swiglu_limit=10.,tiled=True,reform_sf_pack=True)
        keys = set(md._DYNAMIC_KERNEL_CACHE)
        assert len(keys)==1, 'each fresh dynamic lowering must publish one key'
        key = next(iter(keys))
        assert key[3:7] == (72,4096,2048,8) and key[17] is True,key
        assert key[-2:] == ('glm53_ep_prefill_local_fp32_v2','glm53_ep_tiled_sf6_v1'),key
        result['dynamic_passes'].append(preserve_pass(output,'dynamic/M'+str(rows),key))
        assert not torch.cuda.is_initialized()
    result['cuda_initialized'] = torch.cuda.is_initialized()


def check_cpu_contracts(root, result):
    """Runs in a fresh process with the original pre-compiler environment."""
    import torch
    assert not torch.cuda.is_initialized()
    suite = unittest.TestSuite()
    counts = {}
    for name in CPU_TESTS:
        selected = unittest.defaultTestLoader.discover(str(root/'tests'), pattern=name)
        counts[name] = selected.countTestCases()
        suite.addTests(selected)
    result['selected_test_counts'] = counts
    assert counts == CPU_TEST_COUNTS and suite.countTestCases() == EXPECTED_CPU_TESTS
    checked = unittest.TextTestRunner(verbosity=2).run(suite)
    result['contracts'] = dict(tests_run=checked.testsRun, failures=len(checked.failures),
        errors=len(checked.errors), skips=len(checked.skipped))
    result['cuda_initialized'] = torch.cuda.is_initialized()
    assert checked.testsRun == EXPECTED_CPU_TESTS and checked.wasSuccessful() and not checked.skipped
    assert result['cuda_initialized'] is False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capsule-root',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--contracts-only',action='store_true')
    args=p.parse_args()
    result=dict(verdict='FAIL',phase='no-device-guard',started=time.time(),compile_only=True,
                gpu_numerics_acceptance=False,performance_acceptance=False,
                scope='four static and two dynamic EP tiled compiler variants plus CPU ownership contracts; no GPU')
    try:
        assert not list(Path('/dev').glob('nvidia*')),'CPU container exposes CUDA devices'
        runtime=verify_runtime(args.capsule_root,args.manifest_sha256)
        result['binding_runtime']=runtime
        root=Path(__file__).resolve().parents[1]
        sources=source_receipt(root,verify_mounted=True)
        result.update(sources)
        if args.contracts_only:
            result['phase'] = 'cpu-contracts'
            result['scope'] = 'fresh-process CPU ownership and SF6 regression contracts; no compilation or GPU'
            check_cpu_contracts(root,result)
        else:
            # torch.cuda/device/SM compiler shims must never leak into the
            # regression suites. The child also receives the original env.
            contracts_env = dict(os.environ)
            compile_candidate(args.output,result)
            result['phase'] = 'cpu-contracts'
            completed = subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),
                '--output',str(args.output),'--capsule-root',str(args.capsule_root),
                '--manifest-sha256',args.manifest_sha256,'--contracts-only'],env=contracts_env)
            report = json.loads((args.output/'contracts.json').read_text())
            result['contracts'] = report.get('contracts')
            result['selected_test_counts'] = report.get('selected_test_counts')
            completed.check_returncode()
            assert report['verdict'] == 'PASS' and report['phase'] == 'complete'
            assert report['cuda_initialized'] is False and report['binding_runtime_rechecked'] is True
            assert report['binding_runtime'] == runtime
            for key,value in sources.items():assert report[key] == value
            assert not any(k in report for k in ('error','cleanup_error','recheck_error'))
            assert result['selected_test_counts'] == CPU_TEST_COUNTS
            assert result['contracts'] == dict(tests_run=EXPECTED_CPU_TESTS,failures=0,errors=0,skips=0)
            result['contracts_process_isolated'] = True
        import torch
        assert not torch.cuda.is_initialized()
        assert source_receipt(root,verify_mounted=True)==sources
        assert verify_runtime(args.capsule_root,args.manifest_sha256)==runtime
        result.update(verdict='PASS',phase='complete',binding_runtime_rechecked=True)
    except BaseException as exc:
        result['error']=repr(exc)
        raise
    finally:
        result['finished']=time.time()
        filename = 'contracts.json' if args.contracts_only else 'result.json'
        (args.output/filename).write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps({k:result.get(k) for k in ('verdict','phase','error','cuda_initialized','compile_only')}),flush=True)


if __name__=='__main__':
    main()
