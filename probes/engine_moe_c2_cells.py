"""C=2 routed-expert cells on real rank weights: the tile-major w13 chunk, the per-item timeline, and prefill.

Same build, same process, one set of rank bytes. Every arm is the served b12x MoE (t,r,sf6,batch,q0) over its
own tile-major copy of w13 at one chunk (moe_static_kernel_v5.TILED_W13_CHUNKS); w2, the SF6 scale planes and
the alphas are the served layer's. Chunk 512 (moe_static_kernel_v5.TILED_W13_K_IN) is the same-build control.
Numerics precede timing and are zero-tolerance: the FP32 scatter accumulator a decode finalizer consumes, and
the BF16 output of a prefill call.

Occupancy is each layer's real router over synthetic rows. A request is a group of related verify rows whose
spread is calibrated so one 8-row request reads the fleet's 41.9 distinct experts per layer
(measurements/st_draft_rank_overlap_20260915); C=2 is two independent requests (16 rows), C=1 one request, and
16 independent rows bound the top. Scope `single` is the first listed layer; `chain` is every listed layer in
one graph, each with its own weights and routes, which no L2 holds at once. Timing is B/A/A/B brackets, warm
(64 replays per sample) and evicted (a 128 MiB flush before every replay, outside the events).

Sections (engine_kernel_check.py --lanes moe_c2_cells[:section...][:layers=3,4,5][:chunks=512,256,128]):
  chunk    exactness and timing of every chunk against 512: C=2 two requests, C=1 one request, 16 independent rows
  stamps   the stamped 16-row tile per chunk (eager calls): per-item FC1+quant / publication / FC2 and the CTA
           timeline, so FC1's and FC2's rates can be read apart
  prefill  exactness and eager timing of the served prefill kernels per chunk (m=2304 Q0 words, m=9216 SF6 words)
  price    what the FC1 input (A + SFA) and scale (SF6) boxes cost the served 16-row tile: the probe-only timing
           cells xa / xs skip those TMA issues (their numerics are garbage and are not compared)
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
CHUNKS = (512, 256, 128)
RANKS = '/home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors'
# bytes a unique expert streams per layer: w13 + w2 + SF6 FC1 (128 x 1552) + SF6 FC2 (64 x 1552)
EXPERT_BYTES = 1024 * 2048 + 4096 * 256 + (128 + 64) * 1552
# per-item bytes of the M16 reform tile (4 items per expert): FC1 = 32 B stages x 16 KB + 16 A x 2 KB
# + 16 SFA x 256 B + 32 SF6 x 1552 B; FC2 = 16 B stages x 16 KB + 16 SF6 x 1552 B
FC1_ITEM_BYTES = 32 * 16384 + 16 * 2048 + 16 * 256 + 32 * 1552
FC2_ITEM_BYTES = 16 * 16384 + 16 * 1552


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


def chunk_cells(report, layers, chunks, spread, brackets):
    control = 512
    fixtures = (('c2_two_requests', 16, 2, spread), ('c1_one_request', 8, 1, spread),
                ('c2_independent', 16, 16, 1.0))
    for rows in (16, 8):
        x = torch.empty(rows, 4096, device='cuda', dtype=torch.bfloat16)
        ids = [torch.zeros(rows, 8, device='cuda', dtype=torch.int32) for _ in layers]
        routes = [torch.full((rows, 8), 1. / 8, device='cuda') for _ in layers]
        mine = [f for f in fixtures if f[1] == rows]

        def load(fixture, seed):
            name, _, groups, s = fixture
            x.copy_(grouped(rows, groups, s, seed))
            for layer, i, r in zip(layers, ids, routes):
                picked, weight = layer.route(x)
                i.copy_(picked)
                r.copy_(weight)
            return [int(i.unique().numel()) for i in ids]

        load(mine[0], 1)
        for scope, group in (('single', layers[:1]), ('chain', layers)):
            accs = {c: [torch.empty(rows, 4096, device='cuda', dtype=torch.float32) for _ in group] for c in chunks}
            graphs = {}
            try:
                for chunk in chunks:
                    def run(chunk=chunk):
                        outs = []
                        for layer, i, r, acc in zip(group, ids, routes, accs[chunk]):
                            def finalize(accumulator, acc=acc):
                                acc.copy_(accumulator)
                                return acc
                            outs.append(layer.moe(chunk, x, i, r, finalize=finalize))
                        return outs
                    graphs[chunk], _ = _capture(run)
                # zero tolerance against the served control, before any timing
                cells = 0
                for fixture in mine:
                    for seed in range(3):
                        uniques = load(fixture, 1000 * seed + rows)
                        if seed == 2:
                            for r in routes:
                                r[0, 0] = 0.
                        for order in (chunks, chunks[::-1]):
                            for chunk in chunks:
                                for acc in accs[chunk]:
                                    acc.fill_(float('nan'))
                            for chunk in order:
                                graphs[chunk].replay()
                            torch.cuda.synchronize()
                            for chunk in chunks:
                                for n, (got, want) in enumerate(zip(accs[chunk], accs[control])):
                                    if not want.isfinite().all().item() or not got.isfinite().all().item():
                                        raise RuntimeError(f'{fixture[0]} {scope} chunk {chunk}: unwritten accumulator')
                                    if not torch.equal(got, want):
                                        diff = (got - want).abs().max().item()
                                        raise RuntimeError(f'{fixture[0]} {scope} layer {group[n].L} chunk {chunk}: '
                                                           f'differs from 512 by {diff}')
                            cells += 1
                    report('exact', fixture=fixture[0], rows=rows, scope=scope, layers=[l.L for l in group],
                           chunks=list(chunks), reference=control, cells=cells, unique_experts=uniques[:len(group)],
                           replay_orders='forward/reverse', poisoned=True, zero_route=True)
                for fixture in mine:
                    uniques = load(fixture, 7)[:len(group)]
                    for chunk in chunks:
                        if chunk == control:
                            continue
                        res = bracket(report, graphs, control, chunk, brackets=brackets, fixture=fixture[0], rows=rows,
                                      scope=scope, layers=len(group), unique_experts=uniques)
                        stream_bytes = sum(uniques) * EXPERT_BYTES
                        report('rate', fixture=fixture[0], rows=rows, scope=scope, unique_experts=uniques,
                               expert_bytes=stream_bytes,
                               control_gbps_evicted=stream_bytes / res['evicted']['control_us']['mean'] * 1e6 / 1e9,
                               candidate_gbps_evicted=stream_bytes / res['evicted']['candidate_us']['mean'] * 1e6 / 1e9,
                               candidate=chunk)
            finally:
                for graph in graphs.values():
                    graph.reset()


def price_cells(report, layers, spread, brackets):
    from engine.kernels.b12x import moe_dispatch as md
    rows, control = 16, 512
    x = grouped(rows, 2, spread, 7)
    ids, routes = zip(*(layer.route(x) for layer in layers))
    served = md._parse_glm53_static_v2('t,r,sf6,batch')
    arms = {'served': served, 'xa': md._parse_glm53_static_v2('t,r,sf6,batch,xa', probe=True),
            'xs': md._parse_glm53_static_v2('t,r,sf6,batch,xs', probe=True)}
    for scope, group in (('single', layers[:1]), ('chain', layers)):
        accs = {name: [torch.empty(rows, 4096, device='cuda', dtype=torch.float32) for _ in group] for name in arms}
        graphs = {}
        previous = md._STATIC_V2_OVERRIDE
        try:
            for name, config in arms.items():
                md._STATIC_V2_OVERRIDE = config
                def run(name=name):
                    outs = []
                    for layer, i, r, acc in zip(group, ids, routes, accs[name]):
                        def finalize(accumulator, acc=acc):
                            acc.copy_(accumulator)
                            return acc
                        outs.append(layer.moe(control, x, i, r, finalize=finalize))
                    return outs
                graphs[name], _ = _capture(run)
            md._STATIC_V2_OVERRIDE = previous
            uniques = [int(i.unique().numel()) for i in ids[:len(group)]]
            for name in ('xa', 'xs'):
                bracket(report, graphs, 'served', name, brackets=brackets, fixture='c2_two_requests_price', rows=rows,
                        scope=scope, layers=len(group), unique_experts=uniques, numerics='garbage (timing cell)')
        finally:
            md._STATIC_V2_OVERRIDE = previous
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
    return dict(span_us=(kernel_end - base) / 1e3, frontend_us=med((int(t1[b]) - int(t0[b])) / 1e3 for b in live),
                items_per_cta=sorted(int(n_items[b]) for b in live),
                fc1_quant_us=med(fc1), publish_us=med(publish), fc2_us=med(fc2),
                dma_fc1_us=med(dma_fc1) if dma_fc1 else None, dma_fc2_us=med(dma_fc2) if dma_fc2 else None,
                idle_tail_us=med((kernel_end - int(t_end[b])) / 1e3 for b in live), items_timed=len(fc1))


def stamp_cells(report, layers, chunks, spread, calls=12):
    from engine.kernels.b12x import moe_dispatch as md
    layer = layers[0]
    rows = 16
    x = grouped(rows, 2, spread, 31)
    ids, routes = layer.route(x)
    acc = torch.empty(rows, 4096, device='cuda', dtype=torch.float32)
    flush = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)

    def finalize(accumulator):
        acc.copy_(accumulator)
        return acc

    config = dict(md._parse_glm53_static_v2('t,r,sf6,batch'), stamps=True)
    previous = md._STATIC_V2_OVERRIDE
    md._STATIC_V2_OVERRIDE = config
    try:
        for chunk in chunks:
            summaries = []
            for call in range(calls + 2):
                flush.zero_()
                for st in md._STATIC_V2_STAMPS.values():
                    st.zero_()
                layer.moe(chunk, x, ids, routes, finalize=finalize)
                torch.cuda.synchronize()
                if call >= 2:   # the first calls compile and fault the handle in
                    stamps = [v for (mac, dev), v in md._STATIC_V2_STAMPS.items()]
                    summaries.append(stamp_summary(stamps[-1]))
            keys = ('span_us', 'frontend_us', 'fc1_quant_us', 'publish_us', 'fc2_us', 'idle_tail_us')
            medians = {k: statistics.median(s[k] for s in summaries) for k in keys}
            medians['dma_fc1_us'] = statistics.median(s['dma_fc1_us'] for s in summaries if s['dma_fc1_us'])
            medians['dma_fc2_us'] = statistics.median(s['dma_fc2_us'] for s in summaries if s['dma_fc2_us'])
            report('stamps', chunk=chunk, layer=layer.L, rows=rows, unique_experts=int(ids.unique().numel()),
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
    for m in (2304, 9216):
        g = torch.Generator(device='cuda').manual_seed(m)
        x = torch.randn(m, 4096, device='cuda', generator=g).bfloat16()
        ids, routes = layer.route(x)
        routes[0, 0] = 0.
        outs = {c: torch.empty(m, 4096, device='cuda', dtype=torch.bfloat16) for c in chunks}
        try:
            for chunk in chunks:           # compile and fault in
                layer.moe(chunk, x, ids, routes, output=outs[chunk])
            for order in (chunks, chunks[::-1]):
                for chunk in chunks:
                    outs[chunk].fill_(float('nan'))
                for chunk in order:
                    layer.moe(chunk, x, ids, routes, output=outs[chunk])
                torch.cuda.synchronize()
                for chunk in chunks:
                    if not outs[chunk].isfinite().all().item():
                        raise RuntimeError(f'prefill m={m} chunk {chunk}: non-finite output')
                    if not torch.equal(outs[chunk], outs[control]):
                        raise RuntimeError(f'prefill m={m} chunk {chunk}: differs from 512 by '
                                           f'{(outs[chunk].float() - outs[control].float()).abs().max().item()}')
            report('exact', fixture='prefill', rows=m, scope='single', layers=[layer.L], chunks=list(chunks),
                   reference=control, unique_experts=[int(ids.unique().numel())], replay_orders='forward/reverse',
                   poisoned=True, zero_route=True)
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
                            layer.moe(arm, x, ids, routes, output=outs[arm])
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


def main(ranks=None, *, sections=(), samples=None, output=None):
    options = dict(token.split('=', 1) for token in sections if '=' in token)
    wanted = {token for token in sections if '=' not in token} or {'chunk', 'stamps', 'prefill', 'price'}
    if wanted - {'chunk', 'stamps', 'prefill', 'price'}:
        raise ValueError(f'unknown moe_c2_cells sections: {sorted(wanted)}')
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
                            ('stamps', lambda: stamp_cells(report, layers, chunks, spread)),
                            ('prefill', lambda: prefill_cells(report, layers, chunks, brackets)),
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
