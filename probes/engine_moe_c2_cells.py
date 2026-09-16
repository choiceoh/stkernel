"""C=2 routed-expert cells on real rank weights: the tile-major w13 chunk, FC2 ring depth, the per-item timeline.

Same build, same process, one set of rank bytes. An arm is the served b12x MoE (t,r,sf6,batch,q0) over one w13
chunk (moe_static_kernel_v5.TILED_W13_CHUNKS) and, for the depth arm, one static-lane config; w2, the SF6 scale
planes and the alphas are the served layer's. Chunk 512 (moe_static_kernel_v5.TILED_W13_K_IN) with the served
config is the same-build control.

Numerics precede timing. The kernel adds its FP32 (decode, Q0 prefill) or BF16 (long prefill) route partials
atomically in an order it does not specify, so the same handle replayed twice can differ in the last ulp of a
sum. Every comparison therefore also replays a second copy of the control and reports both pairs: how many FP32
elements differ and by how many ulps, how many BF16 model-boundary values (BF16(FP32 scatter), what the
finalizer consumes) differ. The gate is the ulp bound: an arm reading wrong bytes or scales moves values by
orders of magnitude, an add order by a few ulps; the BF16 counts are reported against the control's own.

Occupancy is each layer's real router over synthetic rows. A request is a group of related verify rows whose
spread is calibrated so one 8-row request reads the fleet's 41.9 distinct experts per layer
(measurements/st_draft_rank_overlap_20260915); C=2 is two independent requests (16 rows), C=1 one request, and
16 independent rows bound the top. Scope `single` is the first listed layer; `chain` is every listed layer in
one graph, each with its own weights and routes, which no L2 holds at once. Timing is B/A/A/B brackets, warm
(64 replays per sample) and evicted (a 128 MiB flush before every replay, outside the events).

Sections (engine_kernel_check.py --lanes moe_c2_cells[:section...][:layers=3,4,5][:chunks=512,256]):
  chunk    noise-controlled exactness and timing of every chunk against 512: C=2 two requests, C=1 one request,
           16 independent rows
  depth    the C=2 tile's FC2 prefetch ring (three slots, #962) against two slots: does a deeper ring stream
           faster per CTA (exactness, timing, stamps)
  stamps   the stamped 16-row tile per chunk (eager calls): per-item FC1+quant / publication / FC2 and the CTA
           timeline, so FC1's and FC2's rates can be read apart
  prefill  noise-controlled exactness and eager timing of the served prefill kernels per chunk (m=2304 Q0 words,
           m=9216 SF6 words)
  shapes   short-prefill static row counts (12, 32): the t tile over a 256 chunk, and the M16 reform tile for
           every static row count (probe config reform_every_static, not served) over 512 and 256
  price    what the FC1 input (A + SFA) and scale (SF6) boxes cost the served 16-row tile: the probe-only timing
           cells xa / xs skip those TMA issues (their numerics are garbage and are not compared)
  bulk     cell z (2026-09-17): the tile-major boxes pre-swizzled into the reform stages' byte order and every B
           stage landed by one cp.async.bulk -- exactness (the same bytes must reach the MMA) and timing at C=2 and
           C=1, single and chain, warm and evicted, plus the stamped per-item rates of z against the served tile
  prefetch the l<n> cells (2026-09-16): the DMA warp bulk-prefetches the B stage n stages ahead into L2 over the
           served chunk -- exactness (a hint must not move a bit beyond the add-order floor) and timing at C=2
           (two requests) and C=1 (one request), single and chain, warm and evicted, plus the stamped per-item
           rates of l4 against the served tile
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from probes.engine_decode_fusions import _capture, _time

TARGET_U8 = 41.9      # distinct experts per layer an 8-row C=1 verify reads on the fleet
LAYERS = (3, 4, 5)
CHUNKS = (512, 256)
SECTIONS = ('chunk', 'depth', 'stamps', 'prefill', 'shapes', 'price', 'prefetch', 'bulk')
PREFETCH_CELLS = ('l2', 'lf2', 'lf4', 'lf8')   # l2 repeats the first ticket's control arm beside the FC2-only cells
RANKS = '/home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors'
# bytes a unique expert streams per layer: w13 + w2 + SF6 FC1 (128 x 1552) + SF6 FC2 (64 x 1552)
EXPERT_BYTES = 1024 * 2048 + 4096 * 256 + (128 + 64) * 1552
# per-item bytes of the M16 reform tile (4 items per expert): FC1 = 32 B stages x 16 KB + 16 A x 2 KB
# + 16 SFA x 256 B + 32 SF6 x 1552 B; FC2 = 16 B stages x 16 KB + 16 SF6 x 1552 B
FC1_ITEM_BYTES = 32 * 16384 + 16 * 2048 + 16 * 256 + 32 * 1552
FC2_ITEM_BYTES = 16 * 16384 + 16 * 1552
# the numerical gate: an add order moves a sum by a few ulps of its largest partial, which is the tensor's
# scale for a sum that cancels to ~0; wrong bytes or scales move it by that scale itself (>= 1e4 ulps)
MAX_ULPS = 64
MAX_BF16_REL = 0.25


def _sample_sha(t):
    flat = t.reshape(-1).view(torch.uint8)
    n = flat.numel()
    parts = torch.cat([flat[:4096], flat[n // 2:n // 2 + 4096], flat[-4096:]]).cpu().numpy().tobytes()
    return hashlib.sha256(parts).hexdigest()


class Layer:
    """One MoE layer's rank bytes, served views (in place) and the other chunks' w13 copies."""

    def __init__(self, loader, keys, L, lane, chunks):
        from engine.kernels.b12x import moe_dispatch as md
        from engine.modules.nvfp4_sf import mma_sf_view
        from engine.profiles.glm53.modelopt_scales import ModelOptScales
        p = f'L{L}.moe.'
        names = [p + s for s in ('w13', 'w13_sf', 'w2', 'w2_sf', 'gate', 'bias')]
        extra = [p + s for s in ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')]
        modelopt = all(k in keys for k in extra)
        got = loader.load(names + (extra if modelopt else []), device='cuda')
        self.L = L
        self.w13, self.w13_sf, self.w2, self.w2_sf, self.gate, self.bias = (got[k] for k in names)
        # the router is IEEE FP32 with resident FP32 weights since #1014 (engine/kernels/router_fp32.py):
        # the rank file's BF16 gate is upcast once, as net._router_weights holds it
        self.gate = self.gate.float()
        self.identity = dict(layer=L, scales='ModelOpt' if modelopt else 'folded',
                             sample_sha256={k.split('.')[-1]: _sample_sha(got[k]) for k in names},
                             gate_sha256=hashlib.sha256(self.gate.float().cpu().numpy().tobytes()).hexdigest())
        self.scales = (ModelOptScales.bind(*(got[k] for k in extra), experts=288, device=self.w13.device)
                       if modelopt else None)
        served = md._w13_tile_chunk()
        copies = {}
        for chunk in chunks:
            if chunk != served:
                w13_t, w2_t = md._tile_expert_weights(self.w13, self.w2, w13_chunk=chunk)
                del w2_t
                copies[chunk] = w13_t
        views = lane.moe_prepare(self.w13, self.w13_sf, self.w2, self.w2_sf, 8, 10., scales=self.scales)
        if views.w13_chunk != served or not views.tiled:
            raise RuntimeError('the served bind did not lay w13 out at the served chunk')
        self.views = {served: views}
        for chunk, w13_t in copies.items():
            self.views[chunk] = dataclasses.replace(
                views, w13_fp4=w13_t.view(torch.float4_e2m1fn_x2).permute(2, 3, 1, 0),
                w13_tiled_storage=w13_t, w13_chunk=chunk)
        if served == 256:
            # cell z: a pre-swizzled copy of BOTH tile-major storages (w13 over the served 256 chunk, w2), so
            # the served view's bytes are untouched and every other arm keeps reading them
            w13_t, w2_t = md._tile_expert_weights(self.w13, self.w2, w13_chunk=served)
            w13_z, w2_z = md._swizzle_tile_boxes(w13_t, w2_t)
            del w13_t, w2_t
            self.views['z'] = dataclasses.replace(
                views, w13_fp4=w13_z.view(torch.float4_e2m1fn_x2).permute(2, 3, 1, 0),
                down_fp4=w2_z.view(torch.float4_e2m1fn_x2).permute(2, 3, 1, 0),
                w13_tiled_storage=w13_z, w2_tiled_storage=w2_z, w13_chunk=served, swizzled=True)
        self.sf13 = mma_sf_view(self.w13_sf, self.w13.shape[1], 4096)
        self.sf2 = mma_sf_view(self.w2_sf, self.w2.shape[1], self.w2.shape[2] * 2)
        ones = torch.ones(288, device=self.w13.device, dtype=torch.float32)
        self.a13, self.a2, self.q13, self.q2 = ((ones, ones, None, ones) if self.scales is None else
                                                (self.scales.alpha13, self.scales.alpha2,
                                                 self.scales.input13, self.scales.input2))

    def route(self, x):
        from engine.kernels.glm_pointwise import router_logits, route_weights
        return route_weights(router_logits(x, self.gate), self.bias, 8, 2.5)

    def moe(self, chunk, x, ids, routes, *, finalize=None, output=None):
        from engine.kernels.b12x import b12x_fused_moe
        return b12x_fused_moe(
            x=x, output=x if finalize is not None else output,
            w1_weight=self.w13, w1_weight_sf=self.sf13, w2_weight=self.w2, w2_weight_sf=self.sf2,
            token_selected_experts=ids, token_final_scales=routes, num_experts=288, num_local_experts=288, top_k=8,
            w1_alpha=self.a13, w2_alpha=self.a2, fc2_input_scale=self.q2, input_global_scale=self.q13,
            activation='swigluoai_uninterleave', swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.,
            activation_precision='fp4', quant_mode='nvfp4', _weight_views=self.views[chunk], _output_finalize=finalize)


def grouped(rows, groups, spread, seed):
    """`groups` requests of rows/groups related rows each: a request base plus `spread` x independent noise."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    base = torch.randn(groups, 4096, device='cuda', generator=g).repeat_interleave(rows // groups, 0)
    return (base + spread * torch.randn(rows, 4096, device='cuda', generator=g)).bfloat16()


def unique_counts(layers, x):
    return [int(layer.route(x)[0].unique().numel()) for layer in layers]


def calibrate(layers, report):
    """The spread whose 8-row request reads TARGET_U8 distinct experts per layer, averaged over layers and seeds."""
    def mean_u(spread):
        return statistics.mean(u for seed in range(4) for u in unique_counts(layers, grouped(8, 1, spread, 700 + seed)))
    lo, hi = 0.0, 4.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if mean_u(mid) < TARGET_U8:
            lo = mid
        else:
            hi = mid
    spread = (lo + hi) / 2
    report('calibration', target_u8=TARGET_U8, spread=spread, mean_u8=mean_u(spread),
           mean_u16_two_requests=statistics.mean(
               u for seed in range(4) for u in unique_counts(layers, grouped(16, 2, spread, 800 + seed))),
           mean_u16_independent=statistics.mean(
               u for seed in range(4) for u in unique_counts(layers, grouped(16, 16, 1.0, 900 + seed))))
    return spread


def fp32_noise(got, want):
    """FP32 accumulator difference in ulps, and the BF16 model-boundary values it changes."""
    d = (got - want).abs()
    rms = want.float().pow(2).mean().sqrt().clamp_min(2.0 ** -20)
    scale = torch.maximum(torch.maximum(got.abs(), want.abs()), rms.expand_as(want))
    ulp = torch.nextafter(scale, torch.full_like(scale, float('inf'))) - scale
    return dict(fp32_diff=int((d != 0).sum()), fp32_max_abs=float(d.max()), fp32_max_ulps=float((d / ulp).max()),
                bf16_diff=int((got.bfloat16() != want.bfloat16()).sum()))


def bf16_noise(got, want):
    """BF16 output difference, relative to the larger magnitude."""
    g, w = got.float(), want.float()
    d = (g - w).abs()
    rms = w.pow(2).mean().sqrt().clamp_min(2.0 ** -10)
    scale = torch.maximum(torch.maximum(g.abs(), w.abs()), rms.expand_as(w))
    return dict(bf16_diff=int((d != 0).sum()), bf16_max_abs=float(d.max()), bf16_max_rel=float((d / scale).max()))


def _merge(total, cell):
    for k, v in cell.items():
        total[k] = max(total.get(k, 0), v) if k.startswith(('fp32_max', 'bf16_max')) else total.get(k, 0) + v
    return total


def bracket(report, graphs, control, candidate, *, brackets, **meta):
    flush = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)
    out = {}
    for cache in ('warm', 'evicted'):
        samples = []
        for _ in range(brackets):
            for arm in (control, candidate, candidate, control):
                if cache == 'warm':
                    us = _time(graphs[arm], iterations=64) * 1000
                else:
                    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                              for _ in range(32)]
                    for start, end in events:
                        flush.zero_()
                        start.record()
                        graphs[arm].replay()
                        end.record()
                    events[-1][1].synchronize()
                    us = sum(s.elapsed_time(e) for s, e in events) / len(events) * 1000
                samples.append(dict(arm=arm, us=us))
        b = [s['us'] for s in samples if s['arm'] == control]
        a = [s['us'] for s in samples if s['arm'] == candidate]
        row = dict(cache=cache, control=control, candidate=candidate, samples=samples,
                   control_us=dict(mean=statistics.mean(b), min=min(b)),
                   candidate_us=dict(mean=statistics.mean(a), min=min(a)),
                   mean_change_pct=100 * (statistics.mean(a) / statistics.mean(b) - 1),
                   min_change_pct=100 * (min(a) / min(b) - 1), **meta)
        report('timing', **row)
        out[cache] = row
    return out


class Fixtures:
    """Static x / per-layer ids and routes a row count's graphs read; `load` refills them for a fixture."""

    def __init__(self, layers, rows):
        self.layers, self.rows = layers, rows
        self.x = torch.empty(rows, 4096, device='cuda', dtype=torch.bfloat16)
        self.ids = [torch.zeros(rows, 8, device='cuda', dtype=torch.int32) for _ in layers]
        self.routes = [torch.full((rows, 8), 1. / 8, device='cuda') for _ in layers]

    def load(self, fixture, seed, *, zero_route=False):
        _, _, groups, spread = fixture
        self.x.copy_(grouped(self.rows, groups, spread, seed))
        for layer, i, r in zip(self.layers, self.ids, self.routes):
            picked, weight = layer.route(self.x)
            i.copy_(picked)
            r.copy_(weight)
            if zero_route:
                r[0, 0] = 0.
        return [int(i.unique().numel()) for i in self.ids]


def capture_arms(group, fx, arms):
    """arms: [(label, chunk, static-lane config or None for the served one)] -> (graphs, FP32 accumulators)."""
    from engine.kernels.b12x import moe_dispatch as md
    accs = {label: [torch.empty(fx.rows, 4096, device='cuda', dtype=torch.float32) for _ in group]
            for label, _, _ in arms}
    graphs = {}
    previous = md._STATIC_V2_OVERRIDE
    try:
        for label, chunk, config in arms:
            md._STATIC_V2_OVERRIDE = config if config is not None else previous

            def run(label=label, chunk=chunk):
                outs = []
                for layer, i, r, acc in zip(group, fx.ids, fx.routes, accs[label]):
                    def finalize(accumulator, acc=acc):
                        acc.copy_(accumulator)
                        return acc
                    outs.append(layer.moe(chunk, fx.x, i, r, finalize=finalize))
                return outs
            graphs[label], _ = _capture(run)
    except BaseException:
        for graph in graphs.values():
            graph.reset()
        raise
    finally:
        md._STATIC_V2_OVERRIDE = previous
    return graphs, accs


def exact_arms(report, group, fx, fixtures, graphs, accs, control, second, candidates, *, scope, seeds=3):
    """Replay every arm in both orders over changed inputs/routes, a zero route and poisoned accumulators; gate
    each candidate (and the control's second copy) on the ulp bound against the control. Returns the labels that
    failed (their timing is meaningless; the caller times the rest and fails the section)."""
    labels = [control, *candidates, second]
    failed = []
    for fixture in fixtures:
        totals = {label: {} for label in (second, *candidates)}
        cells, uniques = 0, []
        for seed in range(seeds):
            uniques = fx.load(fixture, 1000 * seed + fx.rows, zero_route=seed == seeds - 1)[:len(group)]
            for order in (labels, labels[::-1]):
                for label in labels:
                    for acc in accs[label]:
                        acc.fill_(float('nan'))
                for label in order:
                    graphs[label].replay()
                torch.cuda.synchronize()
                for label in (second, *candidates):
                    for n, (got, want) in enumerate(zip(accs[label], accs[control])):
                        if not want.isfinite().all().item() or not got.isfinite().all().item():
                            raise RuntimeError(f'{fixture[0]} {scope} layer {group[n].L} {label}: unwritten accumulator')
                        _merge(totals[label], fp32_noise(got, want))
                cells += 1
        for label, total in totals.items():
            ok = total['fp32_max_ulps'] <= MAX_ULPS
            report('exact', fixture=fixture[0], rows=fx.rows, scope=scope, layers=[l.L for l in group],
                   reference=control, arm=label, noise_control=label == second, cells=cells, unique_experts=uniques,
                   replay_orders='forward/reverse', poisoned=True, zero_route=True, elements=cells * len(group) * fx.rows * 4096,
                   max_ulps_gate=MAX_ULPS, passed=ok, **total)
            if not ok and label not in failed:
                failed.append(label)
    if second in failed:
        raise RuntimeError(f'{scope}: the control differs from itself beyond {MAX_ULPS} ulps -- no candidate can be judged')
    return failed


def chunk_cells(report, layers, chunks, spread, brackets):
    control = 512
    fixtures = (('c2_two_requests', 16, 2, spread), ('c1_one_request', 8, 1, spread),
                ('c2_independent', 16, 16, 1.0))
    arms = [(str(c), c, None) for c in chunks] + [('512b', control, None)]
    candidates = [str(c) for c in chunks if c != control]
    failures = []
    for rows in (16, 8):
        fx = Fixtures(layers, rows)
        mine = [f for f in fixtures if f[1] == rows]
        fx.load(mine[0], 1)
        for scope, group in (('single', layers[:1]), ('chain', layers)):
            graphs, accs = capture_arms(group, fx, arms)
            try:
                failed = exact_arms(report, group, fx, mine, graphs, accs, str(control), '512b', candidates, scope=scope)
                failures += [f'{label}@{rows}/{scope}' for label in failed]
                for fixture in mine:
                    uniques = fx.load(fixture, 7)[:len(group)]
                    for label in (c for c in candidates if c not in failed):
                        res = bracket(report, graphs, str(control), label, brackets=brackets, fixture=fixture[0],
                                      rows=rows, scope=scope, layers=len(group), unique_experts=uniques)
                        stream_bytes = sum(uniques) * EXPERT_BYTES
                        report('rate', fixture=fixture[0], rows=rows, scope=scope, unique_experts=uniques,
                               expert_bytes=stream_bytes, candidate=label,
                               control_gbps_evicted=stream_bytes / res['evicted']['control_us']['mean'] * 1e6 / 1e9,
                               candidate_gbps_evicted=stream_bytes / res['evicted']['candidate_us']['mean'] * 1e6 / 1e9)
            finally:
                for graph in graphs.values():
                    graph.reset()
    if failures:
        raise RuntimeError(f'chunk cells beyond the ulp bound: {failures}')


def shape_cells(report, layers, spread, brackets):
    """The static row counts outside C=1 (1..8) and C=2 (16): a short prefill of 9..15 or 17..80 tokens takes the
    t tile, whose K512 FC1 box spans two 256 w13 chunks. Arms against t over 512: t over 256 (what a 256 chunk
    serves), the M16 reform tile for every static row count over 512 (the geometry alone) and over 256."""
    from engine.kernels.b12x import moe_dispatch as md
    every = dict(md._parse_glm53_static_v2('t,r,sf6,batch'), reform_every_static=True)
    arms = [('t512', 512, None), ('t256', 256, None), ('reform512', 512, every), ('reform256', 256, every),
            ('t512b', 512, None)]
    candidates = ['t256', 'reform512', 'reform256']
    failures = []
    for rows in (32, 12):
        fixtures = ((f'prefill{rows}_independent', rows, rows, 1.0), (f'prefill{rows}_one_request', rows, 1, 0.3))
        fx = Fixtures(layers, rows)
        fx.load(fixtures[0], 1)
        for scope, group in (('single', layers[:1]), ('chain', layers)):
            graphs, accs = capture_arms(group, fx, arms)
            try:
                failed = exact_arms(report, group, fx, fixtures, graphs, accs, 't512', 't512b', candidates, scope=scope)
                failures += [f'{label}@{rows}/{scope}' for label in failed]
                for fixture in fixtures:
                    uniques = fx.load(fixture, 7)[:len(group)]
                    for label in (c for c in ('t256', 'reform512', 'reform256') if c not in failed):
                        bracket(report, graphs, 't512', label, brackets=brackets, fixture=fixture[0], rows=rows,
                                scope=scope, layers=len(group), unique_experts=uniques)
            finally:
                for graph in graphs.values():
                    graph.reset()
    report('shape_verdict', failed=failures)
    if failures:
        raise RuntimeError(f'shape cells beyond the ulp bound: {failures}')


def depth_cells(report, layers, spread, brackets):
    from engine.kernels.b12x import moe_dispatch as md
    fixture = ('c2_two_requests', 16, 2, spread)
    two_slots = dict(md._parse_glm53_static_v2('t,r,sf6,batch'), c2_fc2_prefetch=False)
    arms = [('served', 512, None), ('fc2_two_slots', 512, two_slots), ('served_b', 512, None)]
    fx = Fixtures(layers, 16)
    fx.load(fixture, 1)
    for scope, group in (('single', layers[:1]), ('chain', layers)):
        graphs, accs = capture_arms(group, fx, arms)
        try:
            if exact_arms(report, group, fx, [fixture], graphs, accs, 'served', 'served_b', ['fc2_two_slots'], scope=scope):
                raise RuntimeError(f'depth {scope}: two FC2 slots beyond the ulp bound')
            uniques = fx.load(fixture, 7)[:len(group)]
            bracket(report, graphs, 'served', 'fc2_two_slots', brackets=brackets, fixture='c2_two_requests_depth',
                    rows=16, scope=scope, layers=len(group), unique_experts=uniques)
        finally:
            for graph in graphs.values():
                graph.reset()
    stamp_cells(report, layers, [('served', 512, None), ('fc2_two_slots', 512, two_slots)], spread)


def prefetch_cells(report, layers, spread, brackets):
    """l<n> against the served tile over the served chunk: C=2 two requests and C=1 one request."""
    from engine.kernels.b12x import moe_dispatch as md
    served = md._w13_tile_chunk()
    labels = list(PREFETCH_CELLS)
    arms = ([('served', served, None)]
            + [(cell, served, md._parse_glm53_static_v2(f't,r,sf6,batch,{cell}')) for cell in PREFETCH_CELLS]
            + [('served_b', served, None)])
    failures = []
    for fixture in (('c2_two_requests', 16, 2, spread), ('c1_one_request', 8, 1, spread)):
        fx = Fixtures(layers, fixture[1])
        fx.load(fixture, 1)
        for scope, group in (('single', layers[:1]), ('chain', layers)):
            graphs, accs = capture_arms(group, fx, arms)
            try:
                failed = exact_arms(report, group, fx, [fixture], graphs, accs, 'served', 'served_b', labels, scope=scope)
                failures += [f'{label}@{fixture[1]}/{scope}' for label in failed]
                uniques = fx.load(fixture, 7)[:len(group)]
                for label in (l for l in labels if l not in failed):
                    res = bracket(report, graphs, 'served', label, brackets=brackets, fixture=fixture[0] + '_prefetch',
                                  rows=fixture[1], scope=scope, layers=len(group), unique_experts=uniques)
                    stream_bytes = sum(uniques) * EXPERT_BYTES
                    report('rate', fixture=fixture[0] + '_prefetch', rows=fixture[1], scope=scope, unique_experts=uniques,
                           expert_bytes=stream_bytes, candidate=label,
                           control_gbps_evicted=stream_bytes / res['evicted']['control_us']['mean'] * 1e6 / 1e9,
                           candidate_gbps_evicted=stream_bytes / res['evicted']['candidate_us']['mean'] * 1e6 / 1e9)
            finally:
                for graph in graphs.values():
                    graph.reset()
    if failures:
        raise RuntimeError(f'prefetch cells beyond the ulp bound: {failures}')
    stamp_cells(report, layers, [('served', served, None),
                                 ('lf4', served, md._parse_glm53_static_v2('t,r,sf6,batch,lf4'))], spread)


def bulk_cells(report, layers, spread, brackets):
    """cell z against the served tile: the same bytes through one bulk copy per B stage."""
    from engine.kernels.b12x import moe_dispatch as md
    served = md._w13_tile_chunk()
    if any('z' not in layer.views for layer in layers):
        raise RuntimeError('cell z needs the served 256 chunk (its boxes are the reform stages)')
    z = md._parse_glm53_static_v2('t,r,sf6,batch,z')
    arms = [('served', served, None), ('z', 'z', z), ('served_b', served, None)]
    failures = []
    for fixture in (('c2_two_requests', 16, 2, spread), ('c1_one_request', 8, 1, spread)):
        fx = Fixtures(layers, fixture[1])
        fx.load(fixture, 1)
        for scope, group in (('single', layers[:1]), ('chain', layers)):
            graphs, accs = capture_arms(group, fx, arms)
            try:
                failed = exact_arms(report, group, fx, [fixture], graphs, accs, 'served', 'served_b', ['z'], scope=scope)
                failures += [f'{label}@{fixture[1]}/{scope}' for label in failed]
                if failed:
                    continue
                uniques = fx.load(fixture, 7)[:len(group)]
                res = bracket(report, graphs, 'served', 'z', brackets=brackets, fixture=fixture[0] + '_bulk',
                              rows=fixture[1], scope=scope, layers=len(group), unique_experts=uniques)
                stream_bytes = sum(uniques) * EXPERT_BYTES
                report('rate', fixture=fixture[0] + '_bulk', rows=fixture[1], scope=scope, unique_experts=uniques,
                       expert_bytes=stream_bytes, candidate='z',
                       control_gbps_evicted=stream_bytes / res['evicted']['control_us']['mean'] * 1e6 / 1e9,
                       candidate_gbps_evicted=stream_bytes / res['evicted']['candidate_us']['mean'] * 1e6 / 1e9)
            finally:
                for graph in graphs.values():
                    graph.reset()
    if failures:
        raise RuntimeError(f'bulk cells beyond the ulp bound: {failures}')
    stamp_cells(report, layers, [('served', served, None), ('z', 'z', z)], spread)


def price_cells(report, layers, spread, brackets):
    from engine.kernels.b12x import moe_dispatch as md
    fixture = ('c2_two_requests', 16, 2, spread)
    arms = [('served', 512, None), ('xa', 512, md._parse_glm53_static_v2('t,r,sf6,batch,xa', probe=True)),
            ('xs', 512, md._parse_glm53_static_v2('t,r,sf6,batch,xs', probe=True))]
    fx = Fixtures(layers, 16)
    for scope, group in (('single', layers[:1]), ('chain', layers)):
        uniques = fx.load(fixture, 7)[:len(group)]
        graphs, _ = capture_arms(group, fx, arms)
        try:
            for name in ('xa', 'xs'):
                bracket(report, graphs, 'served', name, brackets=brackets, fixture='c2_two_requests_price', rows=16,
                        scope=scope, layers=len(group), unique_experts=uniques, numerics='garbage (timing cell)')
        finally:
            for graph in graphs.values():
                graph.reset()


def stamp_summary(st):
    from engine.kernels.b12x import moe_static_common as k2
    s = st.cpu()
    t0, t1 = s[:, 0], s[:, 1]
    t_end, n_items = s[:, k2.STAMP_MMA_END], s[:, k2.STAMP_MMA_END + 1]
    live = [b for b in range(s.shape[0]) if int(t0[b]) > 0]
    base, kernel_end = min(int(t0[b]) for b in live), max(int(t_end[b]) for b in live)
    fc1, publish, fc2, dma_fc1, dma_fc2 = [], [], [], [], []
    for b in live:
        for i in range(min(int(n_items[b]), k2.STAMP_ITEMS)):
            a, f1, q, f2 = (int(s[b, 2 + 5 * i + j]) for j in range(4))
            if 0 in (a, f1, q, f2):
                continue
            fc1.append((f1 - a) / 1e3)
            publish.append((q - f1) / 1e3)
            fc2.append((f2 - q) / 1e3)
            d0, d1, d2 = (int(s[b, k2.STAMP_DMA_BASE + 3 * i + j]) for j in range(3))
            if 0 not in (d0, d1, d2):
                dma_fc1.append((d1 - d0) / 1e3)
                dma_fc2.append((d2 - d1) / 1e3)
    med = statistics.median
    tb1 = s[:, k2.STAMP_BARRIER1]
    phase0 = [(int(tb1[b]) - int(t0[b])) / 1e3 for b in live if int(tb1[b]) > 0]
    phase1 = [(int(t1[b]) - int(tb1[b])) / 1e3 for b in live if int(tb1[b]) > 0]
    return dict(span_us=(kernel_end - base) / 1e3, frontend_us=med((int(t1[b]) - int(t0[b])) / 1e3 for b in live),
                phase0_barrier1_us=med(phase0) if phase0 else None, route_quant_barrier2_us=med(phase1) if phase1 else None,
                start_skew_us=max((int(t0[b]) - base) / 1e3 for b in live),
                items_per_cta=sorted(int(n_items[b]) for b in live),
                fc1_quant_us=med(fc1), publish_us=med(publish), fc2_us=med(fc2),
                dma_fc1_us=med(dma_fc1) if dma_fc1 else None, dma_fc2_us=med(dma_fc2) if dma_fc2 else None,
                idle_tail_us=med((kernel_end - int(t_end[b])) / 1e3 for b in live), items_timed=len(fc1))


def stamp_cells(report, layers, arms, spread, calls=12, rows=16):
    """arms: [(label, chunk, static-lane config or None)]; each gets the stamps cell on top of its config.
    16 rows are C=2's two requests, 8 rows C=1's one."""
    from engine.kernels.b12x import moe_dispatch as md
    layer = layers[0]
    x = grouped(rows, rows // 8, spread, 31)
    ids, routes = layer.route(x)
    acc = torch.empty(rows, 4096, device='cuda', dtype=torch.float32)
    flush = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)

    def finalize(accumulator):
        acc.copy_(accumulator)
        return acc

    previous = md._STATIC_V2_OVERRIDE
    try:
        for label, chunk, config in arms:
            md._STATIC_V2_OVERRIDE = dict(config or md._parse_glm53_static_v2('t,r,sf6,batch'), stamps=True)
            summaries = []
            for call in range(calls + 2):
                flush.zero_()
                for st in md._STATIC_V2_STAMPS.values():
                    st.zero_()
                layer.moe(chunk, x, ids, routes, finalize=finalize)
                torch.cuda.synchronize()
                if call >= 2:   # the first calls compile and fault the handle in
                    summaries.append(stamp_summary(list(md._STATIC_V2_STAMPS.values())[-1]))
            keys = ('span_us', 'frontend_us', 'phase0_barrier1_us', 'route_quant_barrier2_us', 'start_skew_us',
                    'fc1_quant_us', 'publish_us', 'fc2_us', 'idle_tail_us')
            medians = {k: statistics.median(s[k] for s in summaries) for k in keys}
            medians['dma_fc1_us'] = statistics.median(s['dma_fc1_us'] for s in summaries if s['dma_fc1_us'])
            medians['dma_fc2_us'] = statistics.median(s['dma_fc2_us'] for s in summaries if s['dma_fc2_us'])
            report('stamps', arm=label, chunk=chunk, layer=layer.L, rows=rows, unique_experts=int(ids.unique().numel()),
                   calls=calls, evicted=True, medians=medians, items_per_cta=summaries[-1]['items_per_cta'],
                   fc1_item_bytes=FC1_ITEM_BYTES, fc2_item_bytes=FC2_ITEM_BYTES,
                   fc1_gbps_per_cta=FC1_ITEM_BYTES / medians['fc1_quant_us'] * 1e6 / 1e9,
                   fc2_gbps_per_cta=FC2_ITEM_BYTES / medians['fc2_us'] * 1e6 / 1e9,
                   scope='stamped handle (keeps one extra barrier); per-item medians over CTA lane 0 of every CTA')
    finally:
        md._STATIC_V2_OVERRIDE = previous


def prefill_cells(report, layers, chunks, brackets, reps=4):
    layer = layers[0]
    control = 512
    flush = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)
    labels = [(str(c), c) for c in chunks] + [('512b', control)]
    errors = []
    for m in (2304, 9216):
        g = torch.Generator(device='cuda').manual_seed(m)
        x = torch.randn(m, 4096, device='cuda', generator=g).bfloat16()
        ids, routes = layer.route(x)
        routes[0, 0] = 0.
        outs = {label: torch.empty(m, 4096, device='cuda', dtype=torch.bfloat16) for label, _ in labels}
        try:
            for label, chunk in labels:           # compile and fault in
                layer.moe(chunk, x, ids, routes, output=outs[label])
            totals = {label: {} for label, _ in labels if label != str(control)}
            for order in (labels, labels[::-1]):
                for label, _ in labels:
                    outs[label].fill_(float('nan'))
                for label, chunk in order:
                    layer.moe(chunk, x, ids, routes, output=outs[label])
                torch.cuda.synchronize()
                for label in totals:
                    if not outs[label].isfinite().all().item():
                        raise RuntimeError(f'prefill m={m} {label}: non-finite output')
                    _merge(totals[label], bf16_noise(outs[label], outs[str(control)]))
            for label, total in totals.items():
                ok = total['bf16_max_rel'] <= MAX_BF16_REL
                report('exact', fixture='prefill', rows=m, scope='single', layers=[layer.L], reference=str(control),
                       arm=label, noise_control=label == '512b', cells=2, elements=2 * m * 4096,
                       unique_experts=[int(ids.unique().numel())], replay_orders='forward/reverse', poisoned=True,
                       zero_route=True, max_rel_gate=MAX_BF16_REL, passed=ok, **total)
                if not ok:
                    raise RuntimeError(f'prefill m={m} {label}: BF16 output {total["bf16_max_rel"]:.3g} relative '
                                       f'from {control} (bound {MAX_BF16_REL})')
            for chunk in chunks:
                if chunk == control:
                    continue
                samples = []
                for _ in range(brackets):
                    for arm in (control, chunk, chunk, control):
                        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                                  for _ in range(reps)]
                        for start, end in events:
                            flush.zero_()
                            torch.cuda.synchronize()
                            start.record()
                            layer.moe(arm, x, ids, routes, output=outs[str(arm)])
                            end.record()
                            end.synchronize()
                        samples.append(dict(arm=arm, us=statistics.mean(s.elapsed_time(e) for s, e in events) * 1000))
                b = [s['us'] for s in samples if s['arm'] == control]
                a = [s['us'] for s in samples if s['arm'] == chunk]
                report('timing', cache='evicted', control=control, candidate=chunk, samples=samples, fixture='prefill',
                       rows=m, scope='single', layers=1, host_in_timing=True,
                       control_us=dict(mean=statistics.mean(b), min=min(b)),
                       candidate_us=dict(mean=statistics.mean(a), min=min(a)),
                       mean_change_pct=100 * (statistics.mean(a) / statistics.mean(b) - 1),
                       min_change_pct=100 * (min(a) / min(b) - 1))
        except Exception as exc:
            report('component_failed', section='prefill', rows=m, error=f'{type(exc).__name__}: {exc}'[:1500])
            errors.append(f'm={m}')
    if errors:
        raise RuntimeError(f'prefill cells failed: {errors}')


def main(ranks=None, *, sections=(), samples=None, output=None):
    options = dict(token.split('=', 1) for token in sections if '=' in token)
    wanted = {token for token in sections if '=' not in token} or set(SECTIONS)
    if wanted - set(SECTIONS):
        raise ValueError(f'unknown moe_c2_cells sections: {sorted(wanted - set(SECTIONS))}')
    layer_ids = tuple(int(v) for v in options.get('layers', ','.join(map(str, LAYERS))).split(','))
    chunks = tuple(int(v) for v in options.get('chunks', ','.join(map(str, CHUNKS))).split(','))
    if 512 not in chunks or any(c not in CHUNKS for c in chunks):
        raise ValueError('the chunks must include the 512 control and come from TILED_W13_CHUNKS')
    brackets = int(samples) if samples else 2
    sink = open(output, 'w') if output else None

    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink is not None:
            sink.write(line + '\n')
            sink.flush()

    root = Path(__file__).resolve().parents[1]
    files = ('engine/kernels/b12x/moe_dispatch.py', 'engine/kernels/b12x/moe_static_kernel_v4.py',
             'engine/kernels/b12x/moe_static_kernel_v5.py', 'engine/kernels/b12x/moe_static_common.py',
             'engine/profiles/glm53/lanes.py', 'engine/modules/expert_layout.py', 'probes/engine_moe_c2_cells.py')
    failures = []
    try:
        from engine.kernels.b12x import moe_dispatch as md
        from engine.profiles.glm53.lanes import served
        from engine.profiles.glm53.weights import rank_loader
        path = Path(ranks or RANKS)
        if path.suffix != '.safetensors':
            path = path / 'rank3of4.safetensors'
        report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               rank_file=str(path), layers=layer_ids, chunks=chunks, sections=sorted(wanted), brackets=brackets,
               served_chunk=md._w13_tile_chunk(),
               source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
               scope='captured same-build routed experts beside production; no answer, acceptance or step/s verdict')
        lane = served(moe_static='t,r,sf6,batch,q0')
        loader = rank_loader(path)
        keys = set(loader.keys())
        layers = [Layer(loader, keys, L, lane, chunks) for L in layer_ids]
        report('weights', layers=[layer.identity for layer in layers],
               allocated_bytes=torch.cuda.memory_allocated())
        spread = calibrate(layers, report)
        for section, fn in (('chunk', lambda: chunk_cells(report, layers, chunks, spread, brackets)),
                            ('depth', lambda: depth_cells(report, layers, spread, brackets)),
                            ('stamps', lambda: (stamp_cells(report, layers, [(str(c), c, None) for c in chunks], spread),
                                                stamp_cells(report, layers, [('c1_served', 512, None)], spread, rows=8))),
                            ('prefill', lambda: prefill_cells(report, layers, chunks, brackets)),
                            # after the served shapes: an arm reading a mis-described box could fault the context
                            ('shapes', lambda: shape_cells(report, layers, spread, brackets)),
                            ('prefetch', lambda: prefetch_cells(report, layers, spread, brackets)),
                            ('bulk', lambda: bulk_cells(report, layers, spread, brackets)),
                            # last: the timing cells read garbage scales/inputs, a fault would poison the context
                            ('price', lambda: price_cells(report, layers, spread, brackets))):
            if section not in wanted:
                continue
            try:
                fn()
            except Exception as exc:  # the other sections' evidence is kept; the run still fails
                failures.append(section)
                report('component_failed', section=section, error=f'{type(exc).__name__}: {exc}'[:2000])
        report('complete', status='FAIL' if failures else 'PASS', failed=failures,
               max_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        if sink is not None:
            sink.close()
    if failures:
        raise RuntimeError(f'moe_c2_cells failed: {failures}')
