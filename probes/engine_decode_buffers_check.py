"""CPU interpreter address checks or device-free SM121 compile for direct draft V reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def constants(n, b, pitch, *, cells=33, parts=2):
    return dict(SLOT_STRIDE=2*2*cells*2*128, LAYER_OFFSET=2*cells*2*128,
                V_ROW=(b+2)*pitch, V_TOKEN=pitch, B=b, H=8, HK=2, RHK=2, D=128,
                W=cells, RS=cells*2*128, SCALE=128**-.5, BN=32, SPAN=32, TILES=1, BQ=32)


def interpret():
    import torch
    from engine.kernels.draft_attention import _attend, _combine
    torch.set_num_threads(1)
    torch.manual_seed(926)
    records = []
    for n, b in ((1, 1), (1, 8), (2, 8), (3, 8), (4, 8)):
        q, k = torch.randn(n, b, 8, 128), torch.randn(n, b, 2, 128)
        packed = torch.full((n, b+2, 12*128), float('nan'))
        v = packed[:, 1:b+1, -256:].view(n, b, 2, 128)
        v.normal_()
        ring = torch.randn(6, 2, 2, 33, 2, 128)
        slots = torch.arange(n).flip(0) + 1
        saved = packed.clone()
        for start in (0, 32, 10000):
            ctx = torch.arange(n) + start
            def run(values):
                c = constants(n, b, values.stride(1))
                c['V_ROW'] = values.stride(0)
                acc = torch.empty(n*2*2*32, 128)
                maximum, denominator = torch.empty(n*2*2*32), torch.empty(n*2*2*32)
                _attend[(n, 2, 2)](q, k, values, ring, ctx, slots, acc, maximum, denominator,
                                    **c, num_warps=4, enable_fp_fusion=False)
                out = torch.empty_like(q)
                _combine[(n, b, 8)](acc, maximum, denominator, out, b, 8, 2, 128,
                                     2, 2, 1, 32, num_warps=4, enable_fp_fusion=False)
                return out
            actual, expected = run(v), run(v.contiguous())
            assert torch.isfinite(actual).all()
            assert torch.equal(actual, expected)
            assert torch.allclose(packed, saved, equal_nan=True, rtol=0, atol=0)
            records.append(dict(rows=n, tokens=b, context=start, value_strides=list(v.stride()), exact=True))
    return records


def compile_native():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.draft_attention import _attend
    signature = dict(Q='*bf16', K='*bf16', V='*bf16', R='*bf16', P='*i64', Slot='*i64',
                     ACC='*fp32', MAX='*fp32', DEN='*fp32')
    records = []
    for b in (1, 8, 32):
        for pitch in (2*128, 12*128):
            c = constants(4, b, pitch, cells=2056)
            c.update(V_ROW=b*pitch, SPAN=256, TILES=triton.cdiv(b*4, 32))
            kernel = triton.compile(ASTSource(_attend, signature, constexprs=c),
                                    target=GPUTarget('cuda', 121, 32),
                                    options=dict(num_warps=4, enable_fp_fusion=False))
            records.append(dict(tokens=b, value_token_stride=pitch, status='PASS',
                                shared_bytes=kernel.metadata.shared,
                                cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('interpreter', 'compile'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise RuntimeError('requires CUDA and NVIDIA devices hidden')
    if (os.environ.get('TRITON_INTERPRET') == '1') != (args.mode == 'interpreter'):
        raise RuntimeError('TRITON_INTERPRET=1 is required only for interpreter mode')
    records = interpret() if args.mode == 'interpreter' else compile_native()
    import torch
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', mode=args.mode, gpu_used=False, cases=records,
                  scope='CPU address/equality or native compile only; GPU numerics, replay and serving speed pending',
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                 ('engine/kernels/draft_attention.py', 'probes/engine_decode_buffers_check.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
