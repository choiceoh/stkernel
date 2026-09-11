"""Compare the b12x quantizer instruction variants on a small CUDA allocation."""
import argparse
import json
import statistics
from pathlib import Path

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from flashinfer.cute_dsl.fp4_common import (
    fabs_f32, fmax_f32, quantize_block_fp4_fast as original_quant,
)
from engine.kernels.b12x.fp4_quant import max_abs_16
from cutlass import Float32, Uint8, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from flashinfer.cute_dsl.fp4_common import (
    cvt_f32_to_e4m3, fmin_f32, fp8_e4m3_to_f32_and_rcp, rcp_approx_ftz,
)


def paired_timing(samples):
    """Pair adjacent AB/BA rounds to balance first/second launch order."""
    assert len(samples) == 2 and len(samples[0]) == len(samples[1]) and len(samples[0]) % 2 == 0
    cycles = [[statistics.mean(s[i:i+2]) for i in range(0, len(s), 2)] for s in samples]
    deltas = [old-new for old, new in zip(*cycles)]
    return dict(cycle_mean_us=cycles, median_cycle_us=[statistics.median(s) for s in cycles],
                cycle_delta_us=deltas, median_delta_us=statistics.median(deltas),
                median_percent_reduction=statistics.median(100*(a-b)/a for a, b in zip(*cycles)))


# Experimental paired-multiply alternative; it is not part of the served lane.
@dsl_user_op
def _scale_pack8(v0: Float32, v1: Float32, v2: Float32, v3: Float32,
                 v4: Float32, v5: Float32, v6: Float32, v7: Float32,
                 inv: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(), [Float32(v).ir_value(loc=loc, ip=ip)
                  for v in (v0, v1, v2, v3, v4, v5, v6, v7, inv)],
        """{
            .reg .b64 s, x, y;
            .reg .f32 a, b;
            .reg .b8 q0, q1, q2, q3;
            mov.b64 s, {$9, $9};
            mov.b64 x, {$1, $2};
            mul.rn.f32x2 y, x, s;
            mov.b64 {a, b}, y;
            cvt.rn.satfinite.e2m1x2.f32 q0, b, a;
            mov.b64 x, {$3, $4};
            mul.rn.f32x2 y, x, s;
            mov.b64 {a, b}, y;
            cvt.rn.satfinite.e2m1x2.f32 q1, b, a;
            mov.b64 x, {$5, $6};
            mul.rn.f32x2 y, x, s;
            mov.b64 {a, b}, y;
            cvt.rn.satfinite.e2m1x2.f32 q2, b, a;
            mov.b64 x, {$7, $8};
            mul.rn.f32x2 y, x, s;
            mov.b64 {a, b}, y;
            cvt.rn.satfinite.e2m1x2.f32 q3, b, a;
            mov.b32 $0, {q0, q1, q2, q3};
        }""", "=r,f,f,f,f,f,f,f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@cute.jit
def quantize_block_fp4_fast(values: cute.Tensor, max_abs: Float32,
                            global_scale_val: Float32):
    scale_byte = Uint8(0)
    packed = Uint64(0)
    if global_scale_val != Float32(0.0):
        gs_recip = rcp_approx_ftz(global_scale_val)
        scale_float = gs_recip * (max_abs * rcp_approx_ftz(Float32(6.0)))
        scale_u32 = cvt_f32_to_e4m3(fmin_f32(scale_float, Float32(448.0)))
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv = fp8_e4m3_to_f32_and_rcp(scale_u32)
        if inv != Float32(0.0):
            inv = inv * gs_recip
            lo = _scale_pack8(values[0], values[1], values[2], values[3],
                              values[4], values[5], values[6], values[7], inv)
            hi = _scale_pack8(values[8], values[9], values[10], values[11],
                              values[12], values[13], values[14], values[15], inv)
            packed = (Uint64(hi) << Uint64(32)) | Uint64(lo)
    return packed, scale_byte


class QuantProbe:
    def __init__(self, variant):
        self.variant = variant

    @cute.jit
    def __call__(self, x: cute.Tensor, gs: cute.Tensor, packed: cute.Tensor,
                 scales: cute.Tensor, maxima: cute.Tensor, stream: cuda.CUstream):
        self.kernel(x, gs, packed, scales, maxima).launch(
            grid=(cute.ceil_div(x.shape[0], 128), 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, gs: cute.Tensor, packed: cute.Tensor,
               scales: cute.Tensor, maxima: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        row = bid * 128 + tid
        if row < x.shape[0]:
            values = cute.make_rmem_tensor((16,), cutlass.Float32)
            mx = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(16):
                values[j] = x[row, j].to(cutlass.Float32)
                if cutlass.const_expr(self.variant == 0):
                    mx = fmax_f32(mx, fabs_f32(values[j]))
            if cutlass.const_expr(self.variant != 0):
                mx = max_abs_16(values)
            if cutlass.const_expr(self.variant == 2):
                p, s = quantize_block_fp4_fast(values, mx, gs[row])
            else:
                p, s = original_quant(values, mx, gs[row])
            packed[row] = p.to(cutlass.Int64)
            scales[row] = s
            maxima[row] = mx


def compile_probe(variant, x, gs, dump_dir=None):
    outputs = (torch.empty(x.shape[0], device=x.device, dtype=torch.int64),
               torch.empty(x.shape[0], device=x.device, dtype=torch.uint8),
               torch.empty(x.shape[0], device=x.device, dtype=torch.float32))
    args = tuple(from_dlpack(t, assumed_align=16) for t in (x, gs, *outputs))
    options = '--enable-tvm-ffi'
    if dump_dir is not None:
        options += ' --keep-ptx --keep-cubin'
    fn = cute.compile(QuantProbe(variant), *args,
                      cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                      options=options)
    if dump_dir is not None:
        dump_dir.mkdir(exist_ok=True)
        stem = dump_dir/f'quant-v{variant}-{x.shape[0]}-{x.dtype}'
        stem.with_suffix('.ptx').write_text(fn.__ptx__)
        stem.with_suffix('.cubin').write_bytes(fn.__cubin__)
    return lambda: fn(x, gs, *outputs), outputs


def adversarial_inputs():
    # Every BF16 encoding, including signed zero, subnormals, infinities,
    # NaNs; both uniform and mixed blocks expose all-NaN reduction behavior.
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    bf = bits.view(torch.bfloat16).float()
    uniform = bf[:, None].expand(-1, 16).contiguous()
    mixed = bf.reshape(-1, 16)
    torch.manual_seed(721)
    random = torch.randn(65536, 16, device="cuda")
    random *= torch.exp2(torch.randint(-135, 120, (65536, 1), device="cuda").float())
    # FP4 midpoints and immediate FP32 neighbors, under varied scale maxima.
    ties = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device="cuda")
    neighbors = torch.cat((ties, -ties))
    neighbors = torch.cat((neighbors, torch.nextafter(neighbors, torch.full_like(neighbors, float('inf'))),
                           torch.nextafter(neighbors, torch.full_like(neighbors, -float('inf')))))
    edges = neighbors[:, None].expand(-1, 16).clone()
    edges[:, -1] = 6.0
    x = torch.cat((uniform, mixed, random, edges))
    gs = torch.ones(x.shape[0], device="cuda")
    gs[-random.shape[0]-edges.shape[0]:-edges.shape[0]] = torch.exp2(
        torch.randint(-8, 8, (random.shape[0],), device="cuda").float())
    return x, gs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(1024**3 / torch.cuda.get_device_properties(0).total_memory)
    x, gs = adversarial_inputs()
    checked_blocks = x.shape[0]
    dump = Path(args.out).parent/'quant-compiled'
    compiled = [compile_probe(v, x, gs, dump) for v in range(3)]
    checks = []
    for scale in ('varied', 'zero', 'negative_zero'):
        if scale != 'varied':
            gs.fill_(-0. if scale == 'negative_zero' else 0.)
        for call, _ in compiled:
            call()
        torch.cuda.synchronize()
        reference = compiled[0][1]
        for variant in (1, 2):
            differences = [int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item())
                           for a, b in zip(reference, compiled[variant][1])]
            checks.append(dict(scale=scale, variant=variant, changed_bytes=differences))
            assert differences == [0, 0, 0], checks[-1]
    timings = []
    for rows in (256, 1536, 16384):
        x = torch.randn(rows, 16, device='cuda', dtype=torch.bfloat16)
        gs = torch.ones(rows, device='cuda')
        calls = [compile_probe(v, x, gs, dump) for v in range(3)]
        graphs = []
        for call, outputs in calls:
            call()
            expected = [o.clone() for o in outputs]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(16):
                    call()
            for output in outputs:
                output.fill_(42)
            graph.replay()
            for output, expect in zip(outputs, expected):
                assert torch.equal(output, expect), 'graph must execute and overwrite every output'
            graphs.append(graph)
        for variant in (1, 2):
            samples = [[], []]
            for round in range(12):
                for index in ((0, 1) if round % 2 == 0 else (1, 0)):
                    graph = graphs[variant if index else 0]
                    for _ in range(4):
                        graph.replay()
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    for _ in range(16):
                        graph.replay()
                    end.record(); end.synchronize()
                    samples[index].append(start.elapsed_time(end)*1000/256)
            timings.append(dict(rows=rows, variant=variant, samples_us=samples,
                                paired=paired_timing(samples)))
    result = dict(checks=checks, checked_blocks=checked_blocks, timings=timings)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
