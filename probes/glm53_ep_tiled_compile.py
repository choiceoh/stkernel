#!/usr/bin/env python3
"""Fresh EP tiled lowering only, inside the normal no-device CPU fleet job.

This early compiler receipt does not replace startup numerics or onepass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest

from glm53_ep_capsule_runtime import verify_runtime

STATIC_ROWS = (4, 6, 8, 12, 16, 24, 32)
DYNAMIC_ROWS = (33, 8192)
# Ten global lowerings: the actual runtime map specialization at every native
# canary shape, plus independent 64-bit loads and both map-free control paths.
# These are compiler witnesses, not a parameter sweep or timing experiment.
GLOBAL_STATIC_CASES = tuple(
    ("M%d-map288-i32" % rows, rows, "int32", 288, "int32", 0)
    for rows in STATIC_ROWS
) + (
    ("M6-map288-i64", 6, "int64", 288, "int64", 0),
    ("M6-offset216-i64", 6, "int64", None, None, 216),
    ("M6-empty-i64", 6, "int64", 0, "int64", 0),
)
OPT_STATIC_CASES = (("M6-local", 6, "local"),) + tuple(
    ("M%d-map288-i32" % rows, rows, "global") for rows in (4, 6, 8))
CPU_TESTS = ("test_glm53_ep_tiled_static.py", "test_glm53_ep_tiled_prefill.py",
             "test_glm53_ep_tiled_owner.py", "test_glm53_ep_tiled_selftest.py",
             "test_glm53_ep_tiled_proof.py", "test_moe_sf6_owner.py",
             "test_moe_static_sf6_direct.py", "test_moe_dynamic_sf6.py",
             "test_moe_sf6_dispatch.py", "test_glm53_ep_tiled_a_ring.py",
             "test_glm53_ep_tiled_sf6_word_unpack.py",
             "test_onepass_speculation_proof.py",
             "test_glm53_ep_tiled_route_fusion.py",
             "test_glm53_prep_fused_kv_integration.py", "test_onepass_prep_proof.py",
             "test_glm53_prep_checkpoint_logging.py",
             "test_glm53_ep_tiled_decode_opt.py", "test_glm53_prep_decode_opt.py",
             "test_onepass_ep_default_proof.py", "test_glm53_ep_decode_opt_proof.py")
CPU_TEST_COUNTS = dict(zip(CPU_TESTS, (12, 12, 27, 14, 12, 12, 6, 7, 6, 5, 6, 10, 6, 10, 10, 4,
                                     6, 8, 4, 4)))
EXPECTED_CPU_TESTS = 181
CONTRACT_PATHS = (
    'probes/glm53_ep_tiled_compile.py', 'probes/run_glm53_ep_tiled_cpu.py',
    'probes/glm53_ep_capsule_runtime.py', 'probes/glm53_ep_bindings_capsule.py',
    'probes/glm53_ep_bindings_pair_check.py',
    'profiles/glm53.env', 'bench/proof.py', 'bench/proof-markers.tsv',
    'bench/onepass.py', 'bench/glm53_launch_metadata.py', 'bench/glm53_prep_proof.py',
    'bench/baseline.py', 'bench/judge.py',
    'overlay/modules/glm53_runtime/glm53_prep_fused.py',
    'overlay/modules/glm53_runtime/kv_zero_worker_utils.py',
    'overlay/modules/glm53_runtime/manifest.tsv',
    'tests/fixtures/glm53_prep_fused_runtime/identity.json',
    'tests/fixtures/glm53_prep_fused_runtime/model_runner.py.gz',
    'tests/fixtures/glm53_prep_fused_runtime/worker_utils.image.py.gz',
    'overlay/modules/glm53_model/glm5next_model.py',
    'measurements/glm53_ep_local_20260908/onepass20-completed/source/moe_dynamic_ep_local.py.gz',
    'measurements/glm53_ep_tiled_20260909/ep76_onepass1/source/moe_static_ep_tiled.py.gz',
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


def static_specialization(rows, key, a_ring, word_unpack, scatter_bf16, output_dtype):
    """Bind actual constructor selection and its persisted cache namespace."""
    assert rows in STATIC_ROWS and type(a_ring) is bool and type(word_unpack) is bool
    assert type(scatter_bf16) is bool and type(output_dtype) is str
    assert tuple(key[:4]) == ('glm53_ep_static_tiled_fp32_v1', rows, 256, 48), key
    assert key[10] == 'sf6_v1', key
    expected_ring = rows <= 8
    assert a_ring is expected_ring, (rows, a_ring)
    assert word_unpack is expected_ring, (rows, word_unpack)
    assert scatter_bf16 is expected_ring, (rows, scatter_bf16)
    assert output_dtype == ('bfloat16' if expected_ring else 'float32'), (rows, output_dtype)
    if expected_ring:
        assert len(key) == 19 and tuple(key[-4:]) == (
            'bf16_scatter', 'glm53_ep_static_sf6_a_ring_v1',
            'glm53_ep_static_sf6_word_unpack_v1', 'glm53_ep_static_bf16_scatter_v1'), key
    else:
        assert len(key) == 16 and key[-1] == 'fp32_scatter', key
    return dict(a_ring=a_ring, word_unpack=word_unpack, scatter_bf16=scatter_bf16,
                output_dtype=output_dtype,
                scale_mode=key[10], cache_tag=key[-1])


def global_static_specialization(case, key, a_ring, word_unpack, scatter_bf16,
                                 output_dtype, route):
    """Check actual fake/constructor ABI against one declared global lowering."""
    name, rows, ids_dtype, map_len, map_dtype, offset = case
    assert case in GLOBAL_STATIC_CASES, case
    selected = static_specialization(rows, key[:-4], a_ring, word_unpack,
                                     scatter_bf16, output_dtype)
    canonical_offset = 0 if map_len is not None else offset
    assert key[4] == 'torch.'+ids_dtype, key
    assert tuple(key[-4:]) == ('glm53_ep_static_fused_route_v1', map_len,
        'torch.'+map_dtype if map_len else None, canonical_offset), key
    expected_route = dict(route_mode='global',expert_map_len=map_len,
        local_expert_offset=canonical_offset,topk_ids_dtype=ids_dtype,
        expert_map_operand_dtype=map_dtype if map_len else 'int32',
        expert_map_operand_shape=[map_len or 1],compile_argument_count=31)
    assert isinstance(route,dict) and type(route.get('local_expert_offset')) is int
    assert type(route.get('compile_argument_count')) is int
    assert route.get('expert_map_len') is None or type(route['expert_map_len']) is int
    assert type(route.get('expert_map_operand_shape')) is list
    assert all(type(x) is int for x in route['expert_map_operand_shape'])
    assert route == expected_route, (name, route, expected_route)
    return dict(**selected, route=route)


def validate_scatter_helper_receipt(root, receipt):
    identity = json.loads((root/'measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json').read_text())
    assert receipt == dict(path=identity['source_path'], sha256=identity['source_sha256'],
                           size=identity['source_bytes'], helper='scatter_add_v4_bf16x2')


def opt_static_specialization(case, key, a_ring, word_unpack, scatter_bf16,
                              output_dtype, route, decode_opt, storage_bytes):
    """An optimized artifact preserves the old ABI and fits one resident CTA."""
    name, rows, mode = case
    assert case in OPT_STATIC_CASES
    assert key[-1] == 'glm53_ep_static_sf6_fc1_register_v2'
    assert decode_opt is True and type(storage_bytes) is int and storage_bytes == 98304
    if mode == 'global':
        original = next(item for item in GLOBAL_STATIC_CASES if item[0] == name)
        selected = global_static_specialization(original, key[:-1], a_ring, word_unpack,
                                                scatter_bf16, output_dtype, route)
    else:
        assert route is None
        selected = static_specialization(rows, key[:-1], a_ring, word_unpack,
                                         scatter_bf16, output_dtype)
    return dict(**selected, decode_opt=True, storage_bytes=storage_bytes)


def opt_shared_capacity(passed):
    """cuobjdump includes static allocation in addition to dynamic Storage."""
    assert len(passed['resources']) == 1
    resource = passed['resources'][0]['resources']
    assert len(re.findall(r'(?m)^ Function ', resource)) == 1
    values = re.findall(r'(?m)^  REG:\d+ STACK:\d+ SHARED:(\d+) LOCAL:', resource)
    assert len(values) == 1
    static_bytes = int(values[0])
    dynamic_bytes = passed['specialization']['storage_bytes']
    assert type(dynamic_bytes) is int and dynamic_bytes == 98304
    assert 0 <= static_bytes <= 1024 and static_bytes + dynamic_bytes <= 101376
    return dict(dynamic_bytes=dynamic_bytes, static_bytes=static_bytes,
                total_bytes=dynamic_bytes+static_bytes, block_limit_bytes=101376)


def opt_register_layout(kernel):
    """Preserve the actual CuTe copy-layout check performed during lowering."""
    assert kernel.ep_sf1_register_layout_proven is True
    return validate_register_layout(kernel.ep_sf1_register_layout_receipt)


def validate_register_layout(receipt):
    """Validate the layout witness both before and after artifact transfer."""
    fixed = dict(proven=True, threads=128, raw_stage_bytes=2048, num_k_blocks=4,
                 word_coverage_bytes=2048, stages=2, stage_stride_bytes=2048,
                 offset_engine='static_scalar_physical_layout',
                 slot_zero_relative_offsets=True)
    assert type(receipt) is dict and set(receipt) == set(fixed) | {'copy_shape', 'words_per_thread'}
    assert all(type(receipt[k]) is type(value) and receipt[k] == value
               for k, value in fixed.items())
    assert type(receipt['copy_shape']) is str and receipt['copy_shape']
    words = receipt['words_per_thread']
    assert type(words) is list and words and all(type(x) is int and x > 0 for x in words)
    assert words == sorted(set(words))
    return dict(receipt)


def scatter_helper_receipt(root, source_path):
    path = Path(source_path)
    content = path.read_bytes()
    receipt = dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),
                   size=len(content),helper='scatter_add_v4_bf16x2')
    validate_scatter_helper_receipt(root,receipt)
    return receipt


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
    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl import fp4_common
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_static_ep_tiled as ep
    assert ep.scatter_add_v4_bf16x2 is fp4_common.scatter_add_v4_bf16x2
    root = Path(__file__).resolve().parents[1]
    result['scatter_helper'] = scatter_helper_receipt(root, fp4_common.__file__)
    result['static_passes'] = []
    for rows in STATIC_ROWS:
        result['phase'] = 'static-M'+str(rows)
        assert not list(output.glob('*.ptx')) and not list(output.glob('*.cubin'))
        kernel,args,key = ep.ep_tiled_compile_spec(num_tokens=rows,max_rows=256,
            max_active_clusters=48,topk_ids_dtype=torch.int32,reform_sf_pack=True)
        output_types = {cutlass.BFloat16:'bfloat16', cutlass.Float32:'float32'}
        specialization = static_specialization(rows, key, kernel.a_ring, kernel.word_unpack,
                                                kernel.scatter_bf16, output_types[args[21].element_type])
        cute.compile(kernel,*args,options='--opt-level 2 --enable-tvm-ffi')
        assert static_specialization(rows, key, kernel.a_ring, kernel.word_unpack,
                                     kernel.scatter_bf16, output_types[args[21].element_type]) == specialization
        passed = preserve_pass(output,'static/M'+str(rows),key)
        passed['specialization'] = specialization
        result['static_passes'].append(passed)
        assert not torch.cuda.is_initialized()
    result['global_static_passes'] = []
    integer_types = {cutlass.Int32:'int32',cutlass.Int64:'int64'}
    for case in GLOBAL_STATIC_CASES:
        name,rows,ids_dtype,map_len,map_dtype,offset = case
        result['phase'] = 'global-static-'+name
        assert not list(output.glob('*.ptx')) and not list(output.glob('*.cubin'))
        kernel,args,key = ep.ep_tiled_compile_spec(num_tokens=rows,max_rows=256,
            max_active_clusters=48,topk_ids_dtype=getattr(torch,ids_dtype),
            reform_sf_pack=True,route_mode='global',expert_map_len=map_len,
            expert_map_dtype=getattr(torch,map_dtype) if map_dtype else None,
            local_expert_offset=offset)
        def actual_specialization():
            # Position21 remains output;28 is constexpr CTA count,29 the
            # environment stream,30 the only new real/dummy map operand.
            assert len(args)==31 and args[28]==48
            assert tuple(args[1].shape)==(rows*8,)
            assert args[2].element_type==cutlass.Float32
            route=dict(route_mode=kernel.ep_route_mode,
                expert_map_len=kernel.ep_route_map_len,
                local_expert_offset=kernel.ep_local_expert_offset,
                topk_ids_dtype=integer_types[args[1].element_type],
                expert_map_operand_dtype=integer_types[args[30].element_type],
                expert_map_operand_shape=list(args[30].shape),compile_argument_count=len(args))
            return global_static_specialization(case,key,kernel.a_ring,kernel.word_unpack,
                kernel.scatter_bf16,output_types[args[21].element_type],route)
        specialization = actual_specialization()
        cute.compile(kernel,*args,options='--opt-level 2 --enable-tvm-ffi')
        assert actual_specialization()==specialization
        passed=preserve_pass(output,'global-static/'+name,key)
        passed['specialization']=specialization
        result['global_static_passes'].append(passed)
        assert not torch.cuda.is_initialized()
    result['opt_static_passes'] = []
    for case in OPT_STATIC_CASES:
        name, rows, mode = case
        result['phase'] = 'opt-static-' + name
        assert not list(output.glob('*.ptx')) and not list(output.glob('*.cubin'))
        kernel, args, key = ep.ep_tiled_compile_spec(num_tokens=rows, max_rows=256,
            max_active_clusters=48, topk_ids_dtype=torch.int32, reform_sf_pack=True,
            route_mode=mode, expert_map_len=288 if mode == 'global' else None,
            expert_map_dtype=torch.int32 if mode == 'global' else None, decode_opt=True)
        assert len(args) == (31 if mode == 'global' else 30) and args[28] == 48
        assert tuple(args[1].shape) == (rows*8,) and args[2].element_type == cutlass.Float32
        route = None
        if mode == 'global':
            route = dict(route_mode=kernel.ep_route_mode, expert_map_len=kernel.ep_route_map_len,
                local_expert_offset=kernel.ep_local_expert_offset,
                topk_ids_dtype=integer_types[args[1].element_type],
                expert_map_operand_dtype=integer_types[args[30].element_type],
                expert_map_operand_shape=list(args[30].shape), compile_argument_count=len(args))
        cute.compile(kernel, *args, options='--opt-level 2 --enable-tvm-ffi')
        selected = opt_static_specialization(case, key, kernel.a_ring, kernel.word_unpack,
            kernel.scatter_bf16, output_types[args[21].element_type], route,
            kernel.ep_decode_opt, kernel.ep_storage_bytes)
        assert kernel.smem_bytes == kernel.ep_storage_bytes <= kernel.smem_capacity
        passed = preserve_pass(output, 'opt-static/' + name, key)
        passed['specialization'] = selected
        passed['shared_capacity'] = opt_shared_capacity(passed)
        passed['register_layout'] = opt_register_layout(kernel)
        result['opt_static_passes'].append(passed)
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
    assert scatter_helper_receipt(root, fp4_common.__file__) == result['scatter_helper']


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
    assert counts == CPU_TEST_COUNTS and suite.countTestCases() == EXPECTED_CPU_TESTS, (
        counts, unittest.defaultTestLoader.errors)
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
                scope='seven local static, ten global-route static, four decode-optimized static and two dynamic EP tiled compiler variants plus CPU contracts; no GPU')
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
            assert scatter_helper_receipt(root,result['scatter_helper']['path']) == result['scatter_helper']
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
