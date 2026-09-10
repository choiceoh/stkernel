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
             "test_onepass_ep_default_proof.py", "test_glm53_ep_decode_opt_proof.py",
             "test_glm53_ep_q1_cpu_receipt.py")
CPU_TEST_COUNTS = dict(zip(CPU_TESTS, (12, 12, 27, 14, 12, 12, 6, 7, 6, 5, 6, 10, 6, 10, 10, 4,
                                     6, 8, 4, 4, 2)))
# Loader owner reported eight focused methods PASS before this source freeze.
# Rebind this value if that suite changes; None is never executable admission.
HYBRID_LOADER_TEST_COUNT = 8
CPU_TESTS += ("test_glm53_ep_hybrid_geometry.py", "test_glm53_ep_hybrid_owner.py",
              "test_glm53_ep_hybrid_remap.py", "test_glm53_ep_hybrid_proof.py",
              "test_glm53_ep_hybrid_loader.py")
CPU_TEST_COUNTS.update({"test_glm53_ep_hybrid_geometry.py":10,
    "test_glm53_ep_hybrid_owner.py":9,"test_glm53_ep_hybrid_remap.py":3,
    "test_glm53_ep_hybrid_proof.py":8,
    "test_glm53_ep_hybrid_loader.py":HYBRID_LOADER_TEST_COUNT})
HYBRID_Q0_TEST_COUNTS = {
    "test_glm53_ep_hybrid_q0_cpu_receipt.py":4,
    "test_glm53_ep_hybrid_q0_dispatch.py":6,
    "test_glm53_ep_hybrid_q0_dual_warp.py":8,
}
CPU_TESTS += tuple(HYBRID_Q0_TEST_COUNTS)
CPU_TEST_COUNTS.update(HYBRID_Q0_TEST_COUNTS)
EXPECTED_CPU_TESTS = (213 + HYBRID_LOADER_TEST_COUNT + sum(HYBRID_Q0_TEST_COUNTS.values())
                      if type(HYBRID_LOADER_TEST_COUNT) is int else None)
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
    'measurements/glm53_ep_tiled_20260909/ep76_onepass3/source/moe_static_ep_tiled.py.gz',
    'measurements/glm53_ep_tiled_20260909/ep76_onepass4/source/moe_static_ep_tiled.py.gz',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/fp4_common.py.gz',
    'measurements/glm53_ep_local_20260908/micro-stock-oracle/identity.json',
    'overlay/modules/glm53_moe/glm53_ep_shard_geometry.py',
    'overlay/modules/glm53_moe/glm53_ep_route_remap.py',
    'overlay/modules/glm53_moe/manifest.tsv',
    'overlay/modules/glm53_model/glm53_ep_hybrid.py',
    'overlay/modules/glm53_model/manifest.tsv',
    'tests/fixtures/glm53_ep_hybrid/ep4_route_remap.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/source_pins.json',
    'measurements/glm53_ep_tiled_20260909/ep76_onepass5/source/moe_static_ep_tiled.py.gz',
    'measurements/glm53_ep_tiled_20260909/ep76_onepass5/source/moe_dynamic_ep_local.py.gz',
    'measurements/glm53_ep_tiled_20260909/ep76_onepass6/source/moe_dynamic_ep_local.py.gz',
    'tests/test_glm53_ep_route_scale_cache.py',
    'measurements/glm53_ep_tiled_20260909/onepass1/source/moe_dispatch.py.gz',
    'measurements/glm53_ep_local_20260908/cpu13/stock-gated.py.gz',
    'measurements/glm53_ep_tiled_20260909/ep76_cpu7/originals/result.json',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/config.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/layer.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/expert_map_manager.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/routed_experts.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/runner/moe_runner.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/oracle/nvfp4.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/all2all_utils.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/quantization/compressed_tensors/utils.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/quantization/utils/quant_utils.py.gz',
    'tests/fixtures/glm53_ep_hybrid/runtime/model_executor/layers/fused_moe/modular_kernel.py.gz',
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
            'moe_static_kernel_v4.py','moe_static_kernel_v5.py','glm53_ep_shard_geometry.py'} <= names
    for name in ('moe_static_ep_tiled.py','moe_dynamic_ep_local.py',
                 'moe_dispatch.py','glm53_ep_shard_geometry.py'):
        assert (root/'build/glm53'/name).read_bytes() == (
            root/'overlay/modules/glm53_moe'/name).read_bytes(), 'stale composed source: '+name
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
    assert key[-1] == 'glm53_ep_static_sf6_q1_register_max_v5'
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


def opt_q1_register_layout(kernel, key):
    """Bind the actual CuTe partition_D/flat-register and physical-address witness."""
    assert kernel.ep_decode_opt is True and kernel.ep_q1_register_layout_proven is True
    assert type(kernel.fast_math) is bool and kernel.fast_math is key[6]
    assert key[-1] == 'glm53_ep_static_sf6_q1_register_max_v5'
    return validate_q1_register_layout(kernel.ep_q1_register_layout_receipt,
                                       fast_math=kernel.fast_math)


def validate_q1_register_layout(receipt, *, fast_math):
    """Validate transported evidence produced by the source-bound CuTe setup guard.

    selected describes the compiled R<=8 branch, not a runtime branch counter.
    The source mapping is obtained from actual identity partition_D and dense
    register indices. Per-row digests preserve max scratch load/store ownership
    and scalar-equivalent packed/scale addresses. Format/count checks alone do
    not prove those mappings, GPU numerics, or execution in a timed request.
    """
    assert type(fast_math) is bool
    fixed = dict(proven=True, selected=True, math_mode='fast' if fast_math else 'precise',
                 threads=128, subgroup_threads=4, partner_warp_xor=2, max_rows=8,
                 scratch_bytes=512, sc1_capacity_bytes=4096, source_values=2048,
                 source_element_bits=16, max_shuffle_collectives=8,
                 quant_shuffle_collectives=8)
    assert type(receipt) is dict and set(receipt) == set(fixed) | {
        'source_mapping_sha256', 'copy_shapes', 'layout', 'rows'}
    assert all(type(receipt[k]) is type(value) and receipt[k] == value for k, value in fixed.items())
    digest = receipt['source_mapping_sha256']
    assert type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest)
    empty_digest = hashlib.sha256(b'[]').hexdigest()
    assert digest != empty_digest
    shapes = receipt['copy_shapes']
    assert type(shapes) is list and shapes
    assert all(type(value) is str and 0 < len(value) <= 4096 for value in shapes)
    assert shapes == sorted(set(shapes))
    layout = receipt['layout']
    assert type(layout) is dict and set(layout) == {
        'sc1_shape', 'sc1_stride', 'sc1_swizzle', 'a2_shape', 'a2_stride',
        'a2_swizzle', 'sfa2_shape', 'sfa2_stride'}
    assert all(type(value) is str and 0 < len(value) <= 4096 for value in layout.values())
    assert layout['a2_swizzle'] == '(2, 4, 3)'
    rows = receipt['rows']
    hashes = ('max_stores_sha256', 'max_loads_sha256', 'packed_sha256',
              'scales_sha256', 'ownership_sha256')
    assert type(rows) is list and len(rows) == 9
    for r, row in enumerate(rows):
        counts = dict(rows=r, max_store_bytes=r*64, max_load_bytes=r*64,
                      a2_bytes=r*64, sfa2_bytes=r*8, source_values=r*128)
        assert type(row) is dict and set(row) == set(counts) | set(hashes)
        assert all(type(row[k]) is int and row[k] == value for k,value in counts.items())
        assert all(type(row[k]) is str and re.fullmatch('[0-9a-f]{64}', row[k]) for k in hashes)
        if r == 0:
            assert all(row[k] == empty_digest for k in hashes)
    assert all(len({row[k] for row in rows}) == len(rows) for k in hashes)
    return dict(receipt)


def scatter_helper_receipt(root, source_path):
    path = Path(source_path)
    content = path.read_bytes()
    receipt = dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),
                   size=len(content),helper='scatter_add_v4_bf16x2')
    validate_scatter_helper_receipt(root,receipt)
    return receipt


# The actual hybrid gate preserves its eight variants and adds one fresh
# hybrid Q0 candidate lowering. Legacy specialization validators above remain
# used by the unchanged CPU contracts.
BASELINE_STATIC_CASES = (("M6-map288-i32",6,"global"),("M32-map288-i32",32,"global"))
HYBRID_STATIC_CASES = (("M6-local",6,"local"),("M6-map288-i32",6,"global"),
                       ("M32-local",32,"local"),("M32-map288-i32",32,"global"))
HYBRID_TAG = 'glm53_ep2tp2_tiled_e144_i1024_v1'
HYBRID_Q0_TAG = 'glm53_ep2tp2_q0_dual_warp_v1'
HISTORICAL_CPU7_PATH = 'measurements/glm53_ep_tiled_20260909/ep76_cpu7/originals/result.json'
HISTORICAL_CPU7_SHA256 = 'eaa69a1651341a21e66e5353b1de5a03beb4689198ae070e72a893673fa599d7'
COMPILE_GROUPS = (
    ('baseline_static',72,2048,BASELINE_STATIC_CASES),
    ('baseline_dynamic',72,2048,(("M8192",8192,"dynamic"),)),
    ('hybrid_static',144,1024,HYBRID_STATIC_CASES),
    ('hybrid_dynamic',144,1024,(("M8192",8192,"dynamic"),)),
    ('hybrid_q0_dynamic',144,1024,(("M8192",8192,"dynamic"),)),
)


def require_bound_hybrid_loader_contracts():
    # No admission at the provisional 209 count while loader tests are pending.
    assert type(HYBRID_LOADER_TEST_COUNT) is int and HYBRID_LOADER_TEST_COUNT > 0, (
        'bind the completed loader test count before freeze')
    assert CPU_TEST_COUNTS['test_glm53_ep_hybrid_loader.py'] == HYBRID_LOADER_TEST_COUNT
    assert all(type(n) is int and n > 0 for n in CPU_TEST_COUNTS.values())
    assert len(CPU_TESTS) == len(set(CPU_TESTS)) == len(CPU_TEST_COUNTS)
    assert sum(CPU_TEST_COUNTS.values()) == EXPECTED_CPU_TESTS
    assert EXPECTED_CPU_TESTS == 213 + HYBRID_LOADER_TEST_COUNT + sum(HYBRID_Q0_TEST_COUNTS.values())
    assert len(CONTRACT_PATHS) == len(set(CONTRACT_PATHS))


def historical_cpu7(root):
    raw = (root/HISTORICAL_CPU7_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HISTORICAL_CPU7_SHA256
    old = json.loads(raw)
    assert old['verdict'] == 'PASS' and old['phase'] == 'complete'
    assert old['contracts'] == dict(tests_run=183,failures=0,errors=0,skips=0)
    assert old['cuda_initialized'] is False and old['binding_runtime_rechecked'] is True
    assert [len(old[k]) for k in ('static_passes','global_static_passes',
                                  'opt_static_passes','dynamic_passes')] == [7,10,4,2]
    return old


def compiler_resource_summary(passed, estimated_shared=None, kernel_smem_capacity=None):
    raw = passed['resources'][0]['resources']
    assert len(re.findall(r'^\s*Function\s',raw,re.M)) == 1
    found = re.findall(r'\bREG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',raw)
    assert len(found) == 1,raw
    reg,stack,shared,local = map(int,found[0])
    # kernel.smem_bytes comes from V4._smem_bytes_estimate(), not the
    # allocator's Storage size or actual launch dynamic-shared argument.
    # Keep the cuobjdump SHARED value separate; their sum is not a bound
    # actual per-block allocation witness. No physical limit is increased.
    if estimated_shared is None:
        assert kernel_smem_capacity is None
    else:
        assert type(estimated_shared) is int and type(kernel_smem_capacity) is int
        assert 0 < estimated_shared <= kernel_smem_capacity <= 101376
    return dict(registers=reg,stack_bytes=stack,static_shared_bytes=shared,
        local_bytes=local,estimated_shared_bytes=estimated_shared,
        kernel_smem_capacity_bytes=kernel_smem_capacity,
        dynamic_shared_bytes=None,total_shared_bytes=None)


def expected_native_key(rows, mode, e, n):
    assert (e,n) in ((72,2048),(144,1024)) and rows in (6,32)
    assert mode in ('local','global')
    low = rows <= 8
    key = ['glm53_ep_static_tiled_fp32_v1',rows,256,48,'torch.int32',False,True,
           [16,128,256] if low else [32,64,512],
           [16,256,128] if low else [32,128,128],
           'nvfp4','sf6_v1','swigluoai_uninterleave',1.,0.,10.,
           'bf16_scatter' if low else 'fp32_scatter']
    if low:
        key += ['glm53_ep_static_sf6_a_ring_v1','glm53_ep_static_sf6_word_unpack_v1',
                'glm53_ep_static_bf16_scatter_v1']
    if mode == 'global':key += ['glm53_ep_static_fused_route_v1',288,'torch.int32',0]
    if e == 144:key += [HYBRID_TAG,144,1024]
    return key


def expected_native_specialization(rows, mode, e, n):
    low = rows <= 8
    shapes = {0:[rows,4096],1:[rows*8],2:[rows*8],3:[256,4096,e],
        5:[e*256*2048],6:[e*256*256],9:[2*n,512,8,e],11:[4096,128,n//128,e],
        13:[e],15:[e],16:[e],17:[e],18:[e],19:[e],20:[e],21:[rows,4096],
        22:[e,256],23:[e,256],26:[e,(2*n//128)*16,1552],27:[e,16*(n//128),1552]}
    if mode == 'global':shapes[30] = [288]
    return dict(E=e,H=4096,I=n,top_k=8,rows=rows,route_mode=mode,
        map_length=288 if mode=='global' else None,local_expert_offset=0,
        argument_count=31 if mode=='global' else 30,dtype_ids='int32',
        dtype_output='bfloat16' if low else 'float32',a_ring=low,word_unpack=low,
        scatter_bf16=low,decode_opt=False,native_slices=n//128,
        tensor_shapes={str(k):v for k,v in shapes.items()})


def native_actual_specialization(kernel,args,key,rows,mode,e,n,cutlass):
    expected = expected_native_specialization(rows,mode,e,n)
    assert kernel.ep_num_experts == e and kernel.ep_intermediate_size == n
    assert kernel.output_tile_count_n == n//128 and kernel.ep_decode_opt is False
    assert kernel.ep_route_mode == mode and kernel.ep_route_map_len == expected['map_length']
    assert kernel.ep_local_expert_offset == 0 and args[28] == 48
    assert len(args) == expected['argument_count']
    for index,shape in expected['tensor_shapes'].items():
        assert list(args[int(index)].shape) == shape,(index,args[int(index)].shape,shape)
    assert args[1].element_type == cutlass.Int32 and args[2].element_type == cutlass.Float32
    assert args[9].element_type == args[11].element_type == cutlass.Float4E2M1FN
    assert args[26].element_type == args[27].element_type == cutlass.Uint8
    if mode == 'global':assert args[30].element_type == cutlass.Int32
    low = rows <= 8
    assert kernel.a_ring is low and kernel.word_unpack is low and kernel.scatter_bf16 is low
    assert args[21].element_type == (cutlass.BFloat16 if low else cutlass.Float32)
    assert json.loads(json.dumps(key,default=str)) == expected_native_key(rows,mode,e,n)
    return expected


def expected_dynamic_specialization(e,n,q0_dual_warp=False):
    assert type(q0_dual_warp) is bool
    assert (e,n) in ((72,2048),(144,1024))
    assert not q0_dual_warp or (e,n) == (144,1024)
    return dict(kernel_class='MoEGatedEPLocalKernelSF6',argument_count=36,
        E=e,H=4096,I=n,top_k=8,requested_rows=8192,output_dtype='float32',
        intermediate_slices=n//128,dynamic_four_slice_groups=n//512,SF6=True,
        q0_dual_warp=q0_dual_warp,math_warps=8,threads_per_cta=288,
        q0_batch_tokens=4,q0_warps_per_token=2 if q0_dual_warp else 1,
        tensor_shapes={'14':[2*n,512,8,e],'16':[4096,128,n//128,e],
                       '18':[e],'20':[e+1],'28':[e,(2*n//128)*16,1552],
                       '29':[e,16*(n//128),1552]})


def expected_dynamic_key(e,n,q0_dual_warp=False):
    expected_dynamic_specialization(e,n,q0_dual_warp)
    key = ['dynamic','fp4','nvfp4',e,4096,n,8,48,[128,128],
        'torch.int32',False,True,'swigluoai_uninterleave',1.,0.,10.,False,True,
        'glm53_ep_prefill_local_fp32_v2','glm53_ep_tiled_sf6_v1']
    if e == 144:key.append(HYBRID_TAG)
    if q0_dual_warp:key.append(HYBRID_Q0_TAG)
    return key


def dynamic_actual_specialization(kernel,args,e,n,q0_dual_warp,cutlass,*,lowered):
    """Observe constructor and actual pointer ABI; layout fields follow lowering."""
    assert type(lowered) is bool
    expected = expected_dynamic_specialization(e,n,q0_dual_warp)
    assert type(kernel).__name__ == expected['kernel_class']
    assert type(kernel.q0_dual_warp) is bool and kernel.q0_dual_warp is q0_dual_warp
    assert len(args) == expected['argument_count'] and args[34] == 48
    assert kernel.reform_sf_pack is True
    for index,shape in expected['tensor_shapes'].items():
        assert list(args[int(index)].shape) == shape,(index,args[int(index)].shape,shape)
    # make_ptr exposes dtype, whereas make_fake_compact_tensor has element_type.
    assert args[25].dtype == cutlass.Float32
    if lowered:
        assert kernel.num_mma_warps == expected['math_warps']
        assert kernel.threads_per_cta == expected['threads_per_cta']
        assert tuple(kernel.tile_shape_mnk) == (128,128,128)
        assert min(kernel.tile_shape_mnk[0]*kernel.tile_shape_mnk[1]//4096,
                   kernel.num_mma_warps) == expected['q0_batch_tokens']
    return expected


def validate_compile_matrix(result):
    expected_groups = {kind+'_passes' for kind,_,_,_ in COMPILE_GROUPS}
    assert {k for k in result if k.endswith('_passes')} == expected_groups
    count = 0
    for kind,e,n,cases in COMPILE_GROUPS:
        q0_dual_warp = kind == 'hybrid_q0_dynamic'
        passes = result[kind+'_passes']
        assert [p['arm'] for p in passes] == [kind.replace('_','-')+'/'+c[0] for c in cases]
        for passed,(name,rows,mode) in zip(passes,cases):
            assert passed['compiled_in_this_run'] is True
            assert passed['candidate'] is (e==144)
            assert type(passed['q0_dual_warp']) is bool
            assert passed['q0_dual_warp'] is q0_dual_warp
            assert not {'q1_register_layout','q1_pair_layout','register_layout'} & set(passed)
            summary = passed['resource_summary']
            assert type(summary) is dict
            assert all(type(summary[k]) is int for k in (
                'registers','stack_bytes','static_shared_bytes','local_bytes'))
            assert summary['dynamic_shared_bytes'] is None and summary['total_shared_bytes'] is None
            if mode != 'dynamic':
                assert passed['cache_key'] == expected_native_key(rows,mode,e,n)
                assert passed['specialization'] == expected_native_specialization(rows,mode,e,n)
                assert type(passed['resource_summary']['estimated_shared_bytes']) is int
                assert type(passed['resource_summary']['kernel_smem_capacity_bytes']) is int
            else:
                assert passed['cache_key'] == expected_dynamic_key(e,n,q0_dual_warp)
                assert type(passed['specialization']['q0_dual_warp']) is bool
                assert passed['specialization'] == expected_dynamic_specialization(e,n,q0_dual_warp)
                assert passed['resource_summary']['estimated_shared_bytes'] is None
                assert passed['resource_summary']['kernel_smem_capacity_bytes'] is None
            assert passed['resource_summary'] == compiler_resource_summary(
                passed,passed['resource_summary']['estimated_shared_bytes'],
                passed['resource_summary']['kernel_smem_capacity_bytes'])
            count += 1
    assert count == 9 and result['fresh_lowerings'] == 9
    assert result['same_source_baseline_lowerings'] == 3 and result['hybrid_lowerings'] == 6
    assert result['hybrid_q0_control_lowerings'] == result['hybrid_q0_candidate_lowerings'] == 1


def compile_candidate(output, result):
    os.environ.update(CUTE_DSL_ARCH='sm_121a',CUTE_DSL_KEEP='ptx,cubin',
        CUTE_DSL_DUMP_DIR=str(output),CUTE_DSL_CACHE_DIR=str(output/'cache'),
        CUTE_DSL_DISABLE_FILE_CACHING='1',CUTE_DSL_COMPILER_OPT='ptx-options=-v',
        VLLM_GLM53_EP_TILED='1',VLLM_GLM53_EP_PREFILL_LOCAL='1',VLLM_GLM53_EP_DECODE_OPT='0',
        VLLM_GLM53_EP_HYBRID_Q0_DUAL_WARP='0')
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
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    assert ep.scatter_add_v4_bf16x2 is fp4_common.scatter_add_v4_bf16x2
    root = Path(__file__).resolve().parents[1]
    result['scatter_helper'] = scatter_helper_receipt(root,fp4_common.__file__)
    old = historical_cpu7(root)
    result['historical_cpu7_reference'] = dict(
        source_revision='ca076d35e64a6a19e90dffe54054269d1a5e1887',
        result_sha256=HISTORICAL_CPU7_SHA256,tests_run=183,lowerings=23,
        role='historical cache/ABI reference only; not this run or a performance baseline')
    md.get_num_sm = lambda *a: 48
    md.get_max_active_clusters = lambda *a: 48
    md.build_and_load_cute_dsl_kernel = lambda module,name,build,**kw: build()
    for kind,e,n,cases in COMPILE_GROUPS:
        q0_dual_warp = kind == 'hybrid_q0_dynamic'
        result[kind+'_passes'] = []
        for name,rows,mode in cases:
            result['phase'] = kind+'-'+name
            assert not list(output.glob('*.ptx')) and not list(output.glob('*.cubin'))
            if mode != 'dynamic':
                kwargs = dict(num_tokens=rows,num_local_experts=e,intermediate_size=n,
                    max_rows=256,max_active_clusters=48,topk_ids_dtype=torch.int32,
                    reform_sf_pack=True,decode_opt=False,route_mode=mode)
                if mode == 'global':kwargs.update(expert_map_len=288,expert_map_dtype=torch.int32)
                kernel,args,key = ep.ep_tiled_compile_spec(**kwargs)
                selected = native_actual_specialization(kernel,args,key,rows,mode,e,n,cutlass)
                cute.compile(kernel,*args,options='--opt-level 2 --enable-tvm-ffi')
                assert native_actual_specialization(kernel,args,key,rows,mode,e,n,cutlass) == selected
                assert not hasattr(kernel,'ep_q1_register_layout_receipt'), 'retired optimization executed'
                estimated_shared = int(kernel.smem_bytes)
                kernel_smem_capacity = int(kernel.smem_capacity)
                if e == 72:
                    prior = next(p for p in old['global_static_passes']
                                 if p['arm']=='global-static/'+name)
                    assert json.loads(json.dumps(key,default=str)) == prior['cache_key']
            else:
                seen=[]
                real_compile=cute.compile
                def observe_compile(launch,*args,**kw):
                    kernel=launch._kernel
                    expected=dynamic_actual_specialization(
                        kernel,args,e,n,q0_dual_warp,cutlass,lowered=False)
                    compiled=real_compile(launch,*args,**kw)
                    assert dynamic_actual_specialization(
                        kernel,args,e,n,q0_dual_warp,cutlass,lowered=True) == expected
                    seen.append(expected)
                    return compiled
                md._DYNAMIC_KERNEL_CACHE.clear()
                cute.compile=observe_compile
                try:
                    md._get_dynamic_kernel(e,rows,4096,n,8,8192,
                        activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,
                        swiglu_limit=10.,tile_m=128,tiled=True,reform_sf_pack=True,
                        _ep_hybrid_q0_dual_warp_override=q0_dual_warp)
                finally:
                    cute.compile=real_compile
                assert len(seen) == len(md._DYNAMIC_KERNEL_CACHE) == 1
                key=next(iter(md._DYNAMIC_KERNEL_CACHE))
                selected=seen[0]
                estimated_shared=kernel_smem_capacity=None
                if e == 72:
                    prior=next(p for p in old['dynamic_passes'] if p['arm']=='dynamic/M8192')
                    assert json.loads(json.dumps(key,default=str)) == prior['cache_key']
            passed=preserve_pass(output,kind.replace('_','-')+'/'+name,key)
            passed.update(compiled_in_this_run=True,candidate=e==144,q0_dual_warp=q0_dual_warp,
                          specialization=selected,
                          resource_summary=compiler_resource_summary(passed,estimated_shared,kernel_smem_capacity))
            result[kind+'_passes'].append(passed)
            assert not torch.cuda.is_initialized()
    result.update(fresh_lowerings=9,same_source_baseline_lowerings=3,hybrid_lowerings=6,
                  hybrid_q0_control_lowerings=1,hybrid_q0_candidate_lowerings=1,
                  cuda_initialized=torch.cuda.is_initialized())
    assert result['cuda_initialized'] is False
    validate_compile_matrix(json.loads(json.dumps(result,default=str)))
    assert scatter_helper_receipt(root,fp4_common.__file__) == result['scatter_helper']


def check_cpu_contracts(root, result):
    """Runs in a fresh process with the original pre-compiler environment."""
    require_bound_hybrid_loader_contracts()
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
                scope='same-source EP4 baseline three plus EP2xTP2 hybrid five fresh lowerings, and isolated full CPU contracts; no GPU')
    try:
        require_bound_hybrid_loader_contracts()
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
            validate_compile_matrix(json.loads(json.dumps(result,default=str)))
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
