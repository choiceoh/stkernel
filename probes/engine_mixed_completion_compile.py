"""Compile both M1 producers and ordinary/prepared M16, M32, M128 bodies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from probes.engine_mixed_experts_compile import fingerprint as hot_fingerprint

SOURCES = ('engine/modules/mixed_completion.py', 'engine/kernels/b12x/moe_mixed_completion.py',
           'engine/kernels/b12x/moe_cold_frontend.py', 'engine/kernels/b12x/moe_prepared_prefill.py',
           'engine/kernels/b12x/moe_dynamic_gated_sf6_prefill.py',
           'engine/kernels/b12x/moe_dynamic_gated_sf6_words.py',
           'engine/kernels/b12x/moe_dynamic_gated_sf6.py', 'engine/kernels/b12x/_moe_dynamic/gated.py',
           'engine/kernels/b12x/moe_dynamic_prefill.py', 'engine/kernels/b12x/moe_dynamic_prefill_n128.py',
           'engine/kernels/b12x/moe_dynamic_prefill_n128_tiled.py',
           'engine/kernels/b12x/moe_sf6_prefill_scales.py', 'engine/kernels/b12x/moe_sf6_prefill_scales_kernel.py',
           'probes/engine_mixed_completion_compile.py', 'tests/test_engine_mixed_completion.py')


def fingerprint():
    root = Path(__file__).resolve().parents[1]
    return dict(hot_fingerprint(), **{p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCES})


def compile_all(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('offline compile requires CUDA_VISIBLE_DEVICES=')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    import torch
    records, selected = [], {}
    report = dict(status='FAIL', gpu_used=False, source_sha256=fingerprint(),
        scope='actual CuTe, PTXAS and TVM-FFI compilation; no GPU execution',
        torch=torch.__version__, cuda=torch.version.cuda, kernels=records)
    try:
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
            from engine.kernels.b12x import moe_dispatch as md
            from engine.kernels.b12x.moe_mixed_frontend import compile_producer
            from engine.kernels.b12x.moe_cold_frontend import compile_cold_producer
            def builder(module, name, build, **kwargs):
                start = time.monotonic()
                result = build()
                records.append(dict(selected, name=name, status='PASS', seconds=time.monotonic()-start))
                print(json.dumps(records[-1]), flush=True)
                return result
            with patch.object(md, 'get_num_sm', return_value=48), \
                    patch.object(md, 'get_max_active_clusters', return_value=48), \
                    patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
                for kind, build in (('hot_producer', compile_producer),
                        ('cold_route_reference', lambda: compile_cold_producer(token_major=False)),
                        ('cold_token_producer', compile_cold_producer)):
                    selected.update(kind=kind, runtime_row_extents=True); build()
                selected.clear()
                for rows in (8, 32):
                    for prepared in (False, True):
                        selected.update(kind='static', rows=rows, prepared=prepared)
                        config = md._parse_glm53_static_v2('t,r,sf6')
                        if prepared:
                            config = dict(config, probe_prepared_routes=384)
                        md._get_static_kernel_v2(288, 288, rows, 4096, 512, 8, 32 if prepared else rows*8,
                            config=config, activation='swigluoai_uninterleave',
                            swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                selected.clear()
                for prepared in (False, True):
                    selected.update(kind='dynamic', tile_m=128, prepared=prepared)
                    handles = [md._get_dynamic_kernel(288, rows, 4096, 512, 8, (rows*8//128+287)*128,
                        activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
                        tile_m=128, tiled=True, reform_sf_pack=True, _prepared_prefill=prepared)
                        for rows in (9240, 32768)]
                    if handles[0] is not handles[1]:
                        raise RuntimeError('long-prefill runtime row extents recompiled the same body')
                selected.update(kind='dynamic_n128', tile_m=128, prepared=True)
                handles = [md._get_dynamic_kernel(288, rows, 4096, 512, 8, (rows*8//128+287)*128,
                    activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
                    tile_m=128, tiled=True, reform_sf_pack=False, _prepared_prefill=True,
                    _prefill_scale_expansion=True, _prefill_n128=True) for rows in (9240, 32768)]
                if handles[0] is not handles[1]:
                    raise RuntimeError('N128 runtime row extents recompiled the same body')
        if torch.cuda.is_initialized() or len(records) != 10:
            raise RuntimeError('compile must build all ten handles without initializing CUDA')
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        from engine.kernels.mixed_checks import _check
        start = time.monotonic()
        source = ASTSource(fn=_check, signature=dict(DX='*bf16', PX='*bf16', DR='*fp32', PR='*fp32',
            S0='*fp32', S1='*fp32', S2='*fp32', S3='*fp32', Status='*i32', D='i32', P='i32'), constexprs=dict(B=16384))
        kernel = triton.compile(source, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=8))
        report['value_check'] = dict(status='PASS', runtime_row_extents=True,
            seconds=time.monotonic()-start, shared_bytes=kernel.metadata.shared, triton=triton.__version__)
        from engine.kernels.b12x.moe_sf6_prefill_scales_kernel import expand
        report['scale_expansion'] = []
        for fc2, k_tiles in ((False, 16), (True, 4)):
            start = time.monotonic()
            source = ASTSource(fn=expand, signature=dict(Packed='*u8', Out='*u8'),
                               constexprs=dict(K_TILES=k_tiles, FC2=fc2))
            kernel = triton.compile(source, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
            report['scale_expansion'].append(dict(status='PASS', fc2=fc2,
                seconds=time.monotonic()-start, shared_bytes=kernel.metadata.shared))
        if torch.cuda.is_initialized():
            raise RuntimeError('value-check compile initialized CUDA')
        report.update(status='PASS', dynamic_runtime_shape_reuse=True)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    compile_all(parser.parse_args().output)
