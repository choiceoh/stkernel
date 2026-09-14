"""Paired draft Q/K norms: CPU address/compile gates and native same-build replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inputs(rows, device, dtype, heads=8, kv=2, dim=128):
    import torch
    # Padding and the untouched V region expose accidental contiguous reads.
    packed = torch.randn(rows+2, (heads+2*kv+1)*dim, device=device, dtype=dtype)
    q = packed[1:rows+1, :heads*dim].view(rows, heads, dim)
    k = packed[1:rows+1, heads*dim:(heads+kv)*dim].view(rows, kv, dim)
    weights = [torch.randn(dim, device=device, dtype=dtype).abs()+.5 for _ in range(2)]
    positions = torch.arange(rows, device=device, dtype=torch.int64)+31997
    return packed, q, k, *weights, positions


def interpret():
    import torch
    from engine.kernels.common.norm_rope import _norm_rope, _norm_rope_pair, warm
    torch.manual_seed(933)
    results = []
    for rows, heads, kv in ((1, 8, 2), (8, 8, 2), (32, 8, 2), (8, 32, 8)):
        packed, q, k, qw, kw, pos = inputs(rows, 'cpu', torch.float32, heads, kv)
        before = packed.clone()
        inv = warm(q.device, 128, 10000.)
        oq, ok = torch.empty(q.shape), torch.empty(k.shape)
        _norm_rope_pair[(rows, heads+kv)](q, k, qw, kw, pos, inv, oq, ok,
            *q.stride()[:2], *k.stride()[:2], oq.stride(0), ok.stride(0), 1e-5,
            QH=heads, D=128, H=64, BH=64, num_warps=4)
        for x, w, out in ((q, qw, oq), (k, kw, ok)):
            want = torch.empty(x.shape)
            _norm_rope[(rows, x.shape[1])](x, w, pos, inv, want, *x.stride()[:2], want.stride(0), 1e-5,
                                          D=128, H=64, BH=64, num_warps=4)
            assert torch.equal(out, want)
        assert torch.equal(packed, before)
        results.append(dict(rows=rows, heads=heads, kv_heads=kv, exact=True, dtype='float32'))
    return results


def compile_native():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.common.norm_rope import _norm_rope_pair
    signature = {name: '*bf16' for name in ('Q', 'K', 'WQ', 'WK', 'OQ', 'OK')}
    signature.update(POS='*i64', INV='*fp32', EPS='fp32')
    signature.update({name: 'i32' for name in ('sQr', 'sQh', 'sKr', 'sKh', 'sOq', 'sOk')})
    results = []
    for heads in (8, 32):
        kernel = triton.compile(ASTSource(_norm_rope_pair, signature,
            constexprs=dict(QH=heads, D=128, H=64, BH=64)),
            target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
        results.append(dict(q_heads=heads, status='PASS', shared_bytes=kernel.metadata.shared,
                            cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    return results


def check(report, *, timing=True):
    import torch
    from engine.kernels.common.norm_rope import norm_rope, norm_rope_pair
    from probes.engine_decode_fusions import _capture, _time
    for rows in (1, 8, 16, 24, 32):
        packed, q, k, qw, kw, pos = inputs(rows, 'cuda', torch.bfloat16)
        graphs, outputs = [], []
        try:
            for paired in (False, True):
                def run():
                    if paired:
                        return norm_rope_pair(q, k, qw, kw, 1e-5, pos, 10000.)
                    return norm_rope(q, qw, 1e-5, pos, 10000.), norm_rope(k, kw, 1e-5, pos, 10000.)
                graph, out = _capture(run)
                graphs.append(graph); outputs.append(out)
            for start, scale in ((0, 1.), (31997, .001), (131064, 8.), (31990, 1.), (899900, .1)):
                packed.normal_().mul_(scale)
                qw.normal_(); kw.normal_()
                pos.copy_(torch.arange(rows, device='cuda').flip(0)+start)
                saved = packed.clone()
                for order in ((0, 1), (1, 0)):
                    for arm in order:
                        for out in outputs[arm]:
                            out.fill_(float('nan'))
                        graphs[arm].replay()
                    for got, want in zip(outputs[1], outputs[0]):
                        assert torch.isfinite(got).all() and torch.equal(got.view(torch.int16), want.view(torch.int16))
                    assert torch.equal(packed, saved)
            measurements = [dict(arm=label, ms=_time(graphs[i], iterations=128))
                            for label, i in (('B', 0), ('A', 1), ('A', 1), ('B', 0))] if timing else []
            report('draft_qk_pair', rows=rows, q_heads=8, kv_heads=2, exact_bf16=True, eps=1e-5, theta=10000.,
                   launch_calls=[2, 1], changed_inputs=True, replay_orders='BA/AB',
                   measurements=measurements, consumer_metrics_measured=False)
        finally:
            for graph in graphs:
                graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('interpreter', 'compile'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise RuntimeError('CPU gates require CUDA and NVIDIA devices hidden')
    if (os.environ.get('TRITON_INTERPRET') == '1') != (args.mode == 'interpreter'):
        raise RuntimeError('TRITON_INTERPRET=1 is required only for interpreter mode')
    cases = interpret() if args.mode == 'interpreter' else compile_native()
    import torch
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS', mode=args.mode, gpu_used=False, cases=cases,
                  scope='FP32 CPU address/equality or SM121 compilation only; BF16 GPU/replay/acceptance pending')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
