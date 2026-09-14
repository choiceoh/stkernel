"""Compile the actual TP4 packet consumers for SM121 without initializing CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback
from unittest.mock import patch


SOURCES = (
    'engine/modules/prefill_packets.py', 'engine/modules/token_shards.py',
    'engine/kernels/prefill_collectives/__init__.py', 'engine/kernels/prefill_collectives/consumer.py',
    'engine/kernels/prefill_collectives/kernels.py', 'engine/kernels/prefill_router.py',
    'engine/kernels/prefill_router_packets.py',
    'engine/kernels/dense/__init__.py', 'engine/kernels/dense/fp8.py',
    'engine/kernels/b12x/moe_dispatch.py', 'engine/kernels/b12x/moe_packet_input.py',
    'engine/kernels/b12x/moe_dynamic_prefill_packets.py',
    'engine/kernels/b12x/moe_dynamic_gated_sf6_prefill.py', 'engine/kernels/b12x/moe_w4a16_fp4_helpers.py',
    'engine/profiles/glm53/net.py', 'engine/profiles/glm53/lanes.py',
    'engine/profiles/glm53/execution.py', 'engine/profiles/glm53/boot.py',
    'probes/engine_ffn_packets_compile.py',
    'probes/engine_ffn_packets_check.py', 'tests/test_engine_ffn_packets.py',
    'tests/test_engine_prefill_fp8_consumer.py',
)

# A bounded diagnostic sweep, never an engine autotuner or a serving knob.
ROUTER_VARIANTS = (
    ('current', 64, 64, 64, False, False),
    ('explicit-64x64x128', 64, 64, 128, True, False),
    ('explicit-64x64x256', 64, 64, 256, True, False),
    ('explicit-64x128x128', 64, 128, 128, True, False),
    ('native-64x64x64', 64, 64, 64, True, True),
    ('native-64x64x128', 64, 64, 128, True, True),
    ('native-32x64x128', 32, 64, 128, True, True),
    ('native-64x128x64', 64, 128, 64, True, True),
)


def fingerprint():
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCES}


def compile_consumers(output, router_only=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('compile requires CUDA_VISIBLE_DEVICES= and a container without GPUs')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.experimental.gluon._runtime import GluonASTSource
    from engine.kernels.prefill_router import _router_gemm
    from engine.kernels.prefill_router_packets import _router_packet_gemm
    from engine.kernels.prefill_collectives.consumer import _quantize_gather
    output.mkdir(parents=True, exist_ok=True)
    records = []
    report = dict(scope='SM121 compilation of both router/MoE ABIs and shared quantization; no GPU execution',
                  gpu_used=False, torch=torch.__version__, triton=triton.__version__,
                  source_sha256=fingerprint(), kernels=records)
    try:
        variants = [('bf16', 64, 64, 64, False, False)] + (list(ROUTER_VARIANTS) if router_only
                    else [('packets', 64, 64, 64, False, False)])
        for label, bm, bn, bk, explicit, native in variants:
            packets = label != 'bf16'
            signature = dict(X='*fp8e4nv' if packets else '*bf16', W='*bf16', Out='*fp32', M='i32')
            constants = dict(BM=bm, BN=bn, BK=bk, PACKETS=packets)
            if packets:
                signature.update(Scales='*fp32', LOCAL_ROWS='i32', PACKET_BYTES='i32')
            else:
                constants.update(Scales=None, LOCAL_ROWS=0, PACKET_BYTES=0)
            start = time.monotonic()
            function = _router_gemm
            if explicit:
                function = _router_packet_gemm
                signature['Packed'] = signature.pop('X')
                constants.pop('PACKETS')
                constants['NATIVE'] = native
            source = GluonASTSource if explicit else ASTSource
            kernel = triton.compile(source(function, signature, constexprs=constants),
                target=GPUTarget('cuda', 121, 32),
                options=dict(num_warps=4, num_stages=1 if packets else 3, enable_fp_fusion=False))
            name = 'router-'+label
            dot_ir = '\n'.join(line for line in kernel.asm['ttgir'].splitlines() if 'tt.dot ' in line)
            k_widths = [int(value) for value in re.findall(r'kWidth = (\d+)', dot_ir)]
            if k_widths != [2, 2]:
                raise RuntimeError(f'{name} changed ordinary BF16 dot operand packing: {k_widths}')
            (output/(name+'.ttgir')).write_text(kernel.asm['ttgir'])
            (output/(name+'.ptx')).write_text(kernel.asm['ptx'])
            (output/(name+'.cubin')).write_bytes(kernel.asm['cubin'])
            records.append(dict(name=name, status='PASS', seconds=time.monotonic()-start,
                                shared_bytes=kernel.metadata.shared,
                                dot_k_widths=k_widths,
                                cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
            print(json.dumps(records[-1]), flush=True)
        if router_only:
            if torch.cuda.is_initialized():
                raise RuntimeError('router compilation initialized CUDA')
            report.update(status='PASS', scope='bounded SM121 router compilation only; no GPU or FFN proof')
            return report
        kernel = triton.compile(ASTSource(_quantize_gather,
            dict(Packed='*fp8e4nv', Scales='*fp32', Q='*fp8e4nv', S='*fp32', LOCAL_N='i32', PAYLOAD_BYTES='i32'),
            constexprs=dict(K=4096, G=32, PACK_BLOCK=2048)),
            target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
        records.append(dict(name='shared-quantize', status='PASS', shared_bytes=kernel.metadata.shared,
                            cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
            from engine.kernels.b12x import moe_dispatch as md
            selected = {}

            def builder(module, name, build, **kwargs):
                start = time.monotonic()
                result = build()  # Actual CuTe/ptxas/TVM-FFI build, not a cached substitute.
                records.append(dict(selected, name=name, status='PASS', seconds=time.monotonic()-start))
                print(json.dumps(records[-1]), flush=True)
                return result

            with patch.object(md, 'get_num_sm', return_value=48), \
                    patch.object(md, 'get_max_active_clusters', return_value=48), \
                    patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
                for packets in (False, True):
                    selected.update(packets=packets)
                    md._get_dynamic_kernel(288, 9216, 4096, 512, 8, 9216*8,
                        activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
                        tiled=True, reform_sf_pack=True, tile_m=128, _prefill_packets=packets)
        if torch.cuda.is_initialized():
            raise RuntimeError('compile initialized CUDA')
        if len(records) != 5 or any(r['status'] != 'PASS' for r in records):
            raise RuntimeError('every requested consumer must actually compile')
        report['status'] = 'PASS'
    except BaseException:
        report.update(status='FAIL', error=traceback.format_exc())
        raise
    finally:
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--router-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(compile_consumers(args.output, args.router_only)), flush=True)
