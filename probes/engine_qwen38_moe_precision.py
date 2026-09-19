"""Qwen's bound EP precision: isolate FC2 accumulation from activation search.

Single GB10, synthetic weights at E128/H2560/I640. No consumer quality or speed claim.
Run through engine_kernel_check --lanes qwen38_moe_precision (8 GiB fleet lane).
"""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import torch


@contextmanager
def arm(md, radius, *, fp32=True):
    with patch.object(md, '_STATIC_V2_OVERRIDE', dict(activation_scale_search=radius)):
        if fp32:
            yield
        else:
            with patch.object(md, '_bound_ep_prefill_fp32', return_value=False):
                yield


class Quant:
    """Native packing with independent torch GEMMs/decoding in the projection oracle.

The quantizer itself is checked independently in FP64 by quant_check below.
Power-of-two buckets keep this probe's compilation bounded.
"""
    def __init__(self, radius):
        self.radius, self.cache = radius, {}

    def __call__(self, values, gs):
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from probes.engine_fp4_scale_search import Pack
        rows, cols = values.shape
        blocks = values.numel() // 16
        capacity = 1 << (blocks - 1).bit_length()
        if capacity not in self.cache:
            x = torch.zeros(capacity, 16, device=values.device)
            scales = torch.ones(capacity, device=values.device)
            packed = torch.empty(capacity, device=values.device, dtype=torch.uint64)
            sf = torch.empty(capacity, device=values.device, dtype=torch.uint8)
            tensors = (x, scales, packed, sf)
            fn = cute.compile(Pack(self.radius), *(from_dlpack(t, assumed_align=16) for t in tensors),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options='--enable-tvm-ffi')
            self.cache[capacity] = fn, tensors
        fn, (x, scales, packed, sf) = self.cache[capacity]
        x[:blocks].copy_(values.reshape(-1, 16)); scales.fill_(float(gs))
        fn(x, scales, packed, sf)
        return (packed[:blocks].view(torch.uint8).reshape(rows, cols // 2).clone(),
                sf[:blocks].view(torch.float8_e4m3fn).reshape(rows, cols // 16).clone())


def run(output=None):
    from engine.base import kernel_shape as ks
    from probes import engine_qwen38_moe as fixture
    from engine.profiles.qwen38 import lanes
    from engine.kernels.b12x import moe_dispatch as md
    from probes.engine_fp4_scale_search import quant_check

    events = []
    def report(event, **fields):
        record = dict(event=event, **fields); events.append(record)
        print(json.dumps(record), flush=True)
        if output:
            Path(output).write_text(''.join(json.dumps(e) + '\n' for e in events))

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)
    ks.bind(fixture.kernel_shape())
    c = fixture.cell_of(ks.bound())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device='cuda').manual_seed(919)
    lane = lanes.served()
    experts = fixture.build_experts(c, generator, 'cuda')
    prepared = lane.moe_prepare(experts.w13, experts.w13_sf, experts.w2, experts.w2_sf, c.topk, scales=experts.scales)
    report('device', name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
           shape=[c.local, c.hidden, c.inter, c.topk], default_search=md._ACTIVATION_SCALE_SEARCH_RADIUS)
    quant_check(report)

    def inputs(rows, topk=1):
        x = torch.randn(rows, c.hidden, device='cuda', generator=generator).bfloat16() * .5
        ids = (torch.arange(rows * topk, device='cuda') % 4 * 31).int().reshape(rows, topk)
        w = torch.rand(rows, topk, device='cuda', generator=generator) / topk
        return x, ids, w

    def call(x, ids, w, layer=experts, views=prepared):
        return fixture.dispatch(x, ids, w, layer, views)

    # Independent accumulation oracle: run each of the five native FC2
    # slices alone with identical packing/scales, then sum their BF16 outputs.
    # This keeps quantization and GEMM rounding identical in both sum arms.
    x, ids, w = inputs(64)
    partials, sliced_layers = [], []
    with arm(md, 0):
        for start in range(0, c.inter, 128):
            down = torch.zeros_like(experts.w2)
            down[:, :, start // 2:(start + 128) // 2].copy_(experts.w2[:, :, start // 2:(start + 128) // 2])
            layer = fixture.Experts(experts.w13, experts.w13_sf, down, experts.w2_sf, experts.scales)
            sliced_layers.append(layer)  # keep source pointers live in the lane's prepared-view cache
            views = lane.moe_prepare(layer.w13, layer.w13_sf, layer.w2, layer.w2_sf, 1, scales=layer.scales)
            partials.append(call(x, ids, w, layer, views).clone())
        expected = torch.stack([p.double() for p in partials]).sum(0)
        precise = call(x, ids, w).clone()
    with arm(md, 0, fp32=False):
        before = call(x, ids, w).clone()
    before_sse = float((before.double() - expected).square().sum())
    after_sse = float((precise.double() - expected).square().sum())
    assert torch.equal(precise, expected.bfloat16()), 'FP32 accumulation must match independently summed native slices'
    assert after_sse < before_sse, (before_sse, after_sse)
    report('accumulation', rows=64, slices=len(partials), baseline_sse=before_sse,
           fp32_sse=after_sse, reduction_pct=100 * (1 - after_sse / before_sse), exact_rounded_oracle=True)

    quant = {radius: Quant(radius) for radius in (0, 2)}
    for topk, rows in ((1, 1), (1, 8), (1, 64), (10, 4), (10, 16)):
        x, ids, w = inputs(rows, topk)
        for radius in (0, 2):
            with arm(md, radius):
                got = call(x, ids, w).clone()
            expected = fixture.oracle(x, ids, w, experts, c, quant[radius])
            error = fixture.relative(got, expected)
            assert error < fixture.ORACLE_RELATIVE, (rows, topk, radius, error)
            report('projection', rows=rows, topk=topk, radius=radius, relative_error=error)

    # Exercise each generic tile, every byte of its FP32 zero-fill, and graph
    # replay on changed inputs with the old accumulator deliberately poisoned.
    for tile in (32, 64, 128):
        x, ids, w = inputs(129)
        with patch.object(md, '_DYNAMIC_TILE_M_OVERRIDE', tile), arm(md, 2):
            eager = call(x, ids, w).clone()
            again = call(x, ids, w).clone()
            assert torch.equal(eager, again)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = call(x, ids, w)
            x.mul_(.75)
            expected = call(x, ids, w).clone()
            for ws in md._WORKSPACE_CACHE.values():
                if getattr(ws, 'ep_scatter_fp32', None) is not None:
                    ws.ep_scatter_fp32.fill_(float('nan'))
            graph.replay(); torch.cuda.synchronize()
            assert torch.equal(captured, expected), ('graph', tile)
            w.zero_()
            assert torch.count_nonzero(call(x, ids, w)) == 0
            report('graph', tile=tile, rows=129, changed_inputs=True, poisoned_accumulator=True, zero_weights=True)

    x, ids, w = inputs(4096)
    got = lane.moe(x, ids, w, experts.w13, experts.w13_sf, experts.w2, experts.w2_sf,
                   scales=experts.scales, first_expert=0, compact=True)
    assert torch.isfinite(got).all()
    # All routes foreign: compact lane must return zero without allocating pairs.
    foreign = ids + 384
    zero = lane.moe(x, foreign, w, experts.w13, experts.w13_sf, experts.w2, experts.w2_sf,
                    scales=experts.scales, first_expert=0, compact=True)
    assert torch.count_nonzero(zero) == 0
    report('served_prefill', rows=4096, finite=True, foreign_routes_zero=True)
    report('passed', passed=True, peak_bytes=torch.cuda.max_memory_allocated(),
           not_measured='real checkpoint, TP4 consumer output quality, acceptance and throughput')
    return events


if __name__ == '__main__':
    run()
