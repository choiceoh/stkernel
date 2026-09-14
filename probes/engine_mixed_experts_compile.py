"""Compile the explicit producer and ordinary/prepared M16/M32 expert bodies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch


SOURCES = ('engine/modules/mixed_experts.py', 'engine/kernels/b12x/moe_mixed.py',
           'engine/modules/route_table.py', 'tests/test_engine_mixed_plan.py',
           'engine/modules/mixed_route_plan.py', 'engine/modules/mixed_metadata.py',
           'engine/modules/mixed_route_native.py', 'engine/modules/mixed_route_plan.cpp',
           'engine/kernels/common/native_cache.py',
           'engine/kernels/mixed_checks.py',
           'engine/kernels/b12x/moe_mixed_frontend.py', 'engine/kernels/b12x/moe_dispatch.py',
           'engine/kernels/b12x/moe_static_common.py',
           'engine/kernels/b12x/moe_static_kernel_v4.py', 'engine/kernels/b12x/moe_static_kernel_v5.py',
           'engine/kernels/b12x/moe_micro_kernel.py', 'probes/engine_mixed_experts_compile.py',
           'tests/test_engine_mixed_experts.py')


def fingerprint():
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCES}


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
            def builder(module, name, build, **kwargs):
                start = time.monotonic()
                compiled = build()
                records.append(dict(selected, name=name, status='PASS', seconds=time.monotonic()-start))
                print(json.dumps(records[-1]), flush=True)
                return compiled
            with patch.object(md, 'get_num_sm', return_value=48), \
                    patch.object(md, 'get_max_active_clusters', return_value=48), \
                    patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
                selected.update(kind='producer', runtime_row_extents=True)
                compile_producer()
                selected.clear()
                for rows in (8, 32):
                    for prepared in (False, True):
                        selected.update(kind='expert_body', rows=rows, prepared=prepared)
                        config = md._parse_glm53_static_v2('t,r,sf6')
                        if prepared:
                            config = dict(config, probe_prepared_routes=384)
                        md._get_static_kernel_v2(288, 288, rows, 4096, 512, 8, 32 if prepared else rows*8,
                            config=config, activation='swigluoai_uninterleave',
                            swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
        if torch.cuda.is_initialized() or len(records) != 5:
            raise RuntimeError('compile must build all five handles without initializing CUDA')
        report['status'] = 'PASS'
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
