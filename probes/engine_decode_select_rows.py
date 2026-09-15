"""Captured same-build decode selection for joined rows (C=2) against the per-row launches; never consumer speed.

`Glm53Net._select_rows` joins two rows' indexer logits, horizon mask and top-k into one launch each over their
candidates laid end to end (DeepGEMM's compressed logits: each query's columns from its own row's window). The
control is the same function's `joined=False`: a logits kernel, a mask and a top-k per row. Two probe-local
alternatives are judged beside it: `rows_cat` (per-row logits, one cat, then one mask and one top-k) and `wide`
(one uncompressed launch over [rows*t, rows*n_cand] cleaned outside each window, one top-k over the wide rows,
ids less the row's offset). Every arm is an 11-layer captured graph over the served lanes; slots and counts must
equal the control's bit for bit in both replay orders, with rebound block tables, rolled-back contexts, rows short
of the top-k width, rows at the capacity's end and tied scores at the selection edge. Only exact arms are timed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from probes.engine_decode_fusions import _capture, _time

LAYERS, BLOCK, KP, D, HEADS, TOPK = 11, 768, 4, 128, 32, 2048
PER, RECORD = BLOCK // KP, D + 4
K = TOPK // KP
DEVICE = 'cuda'
NUMERIC_CAPACITIES = (4096, 8192, 16384, 32768, 65536, 131072, 202752)
TIMED = ((4096, 2000), (32768, 32000), (131072, 128000))    # capacity bucket, both rows' decode context


def rows_cat(net, L, q8, w_eff, keys, scales, n_cand, contexts, t, caches, slots_out, valid_out):
    """(b) per-row logits launches, their rows joined by one cat, then one horizon mask and one top-k."""
    from engine.base.constants import zeros
    rows, glue = contexts.shape[0], net.lanes.decode_rows
    seq_lens, ke = caches.row_lengths(contexts, t, KP, glue.lengths)
    keys_all, scales_all = glue.candidates(keys, scales, *caches.pool_maps(L), n_cand)
    ks = zeros(t, q8.device)
    logits = torch.cat([net.lanes.indexer_logits(q8[r * t:(r + 1) * t], keys_all[r], scales_all[r], w_eff[r * t:(r + 1) * t],
                                                 ke[r * t:(r + 1) * t], ks=ks) for r in range(rows)])
    glue.horizon(logits, ke)
    values = torch.empty((rows * t, K), dtype=torch.float32, device=q8.device)
    winners = torch.empty((rows * t, K), dtype=torch.int64, device=q8.device)
    torch.topk(logits, K, dim=-1, sorted=False, out=(values, winners))
    net.lanes.pool_slots(winners, seq_lens, KP, *caches.token_maps(L), slots_out, valid_out, tokens=t)


def wide(net, L, q8, w_eff, keys, scales, n_cand, contexts, t, caches, slots_out, valid_out):
    """(a) as first posed: one uncompressed logits launch cleaned outside each query's window, one top-k over the
    [rows*t, rows*n_cand] rows, then the row's offset taken off the ids (a -inf pick lands outside the pool range)."""
    from engine.kernels.deep_gemm import fp8_fp4_mqa_logits
    rows, glue = contexts.shape[0], net.lanes.decode_rows
    seq_lens, ke, starts, ends = caches.row_lengths(contexts, t, KP, glue.lengths, width=n_cand)
    keys_all, scales_all = glue.candidates(keys, scales, *caches.pool_maps(L), n_cand)
    logits = fp8_fp4_mqa_logits((q8, None), (keys_all.view(rows * n_cand, -1), scales_all.view(-1)), w_eff.contiguous(),
                                starts, ends, clean_logits=True)
    values = torch.empty((rows * t, K), dtype=torch.float32, device=q8.device)
    winners = torch.empty((rows * t, K), dtype=torch.int64, device=q8.device)
    torch.topk(logits, K, dim=-1, sorted=False, out=(values, winners))
    winners.sub_(starts[:, None])
    net.lanes.pool_slots(winners, seq_lens, KP, *caches.token_maps(L), slots_out, valid_out, tokens=t)


class Fixture:
    """One capacity bucket's paged pool records (record-strided keys, fp32 scale in each record's tail) and a
    gathered block table per row, the layout `GraphCaches.pool_maps`/`token_maps` read."""

    def __init__(self, rows, t, capacity, gen):
        from engine.profiles.glm53.decode_graphs import GraphCaches
        self.rows, self.t, self.capacity, self.gen = rows, t, capacity, gen
        self.n_cand = capacity // KP
        self.blocks = -(-capacity // BLOCK)
        self.pages = (rows + 1) * self.blocks + 3
        F = NS(kpool=KP, topk=TOPK, block=BLOCK, idx_dim=D, kv_lora=512)
        layout = NS(block_bytes=LAYERS * PER * RECORD,
                    token_offsets={L: L * BLOCK * 512 for L in range(LAYERS)}, pool_offsets={L: L * PER * RECORD for L in range(LAYERS)})
        self.paged = torch.empty(self.pages * layout.block_bytes, dtype=torch.uint8, device=DEVICE)
        records = self.paged.numel() // RECORD
        self.keys = self.paged.as_strided((records, D), (RECORD, 1)).view(torch.float8_e4m3fn)
        self.scales = self.paged.view(torch.float32).as_strided((records,), (RECORD // 4,), D // 4)
        self.real = NS(F=F, layout=layout, block_table=torch.empty(rows + 1, self.blocks, dtype=torch.int32, device=DEVICE))
        ids = torch.arange(1, rows + 1, device=DEVICE)                   # sequence ids gathered from a wider table
        self.caches = GraphCaches(self.real, ids, ids + 1, capacity)
        self.contexts = torch.zeros(rows, dtype=torch.int64, device=DEVICE)
        self.q8 = torch.empty(rows * t, HEADS, D, dtype=torch.float8_e4m3fn, device=DEVICE)
        self.w = torch.empty(rows * t, HEADS, dtype=torch.float32, device=DEVICE)
        self.net = NS(F=F, lanes=None)
        self.tables()
        self.data('random')

    def tables(self):
        order = torch.randperm(self.pages, device=DEVICE, generator=self.gen)[:(self.rows + 1) * self.blocks]
        self.real.block_table.copy_(order.view(self.rows + 1, self.blocks).int())

    def data(self, regime):
        records = self.keys.shape[0]
        for a in range(0, records, 1 << 18):
            b = min(records, a + (1 << 18))
            self.keys[a:b].copy_((torch.randn(b - a, D, device=DEVICE, generator=self.gen) * 96).clamp_(-448, 448).to(torch.float8_e4m3fn))
        self.scales.copy_(2. ** torch.randint(-9, 1, (records,), device=DEVICE, generator=self.gen).float())
        self.q8.copy_((torch.randn(self.q8.shape, device=DEVICE, generator=self.gen) * 96).clamp_(-448, 448).to(torch.float8_e4m3fn))
        self.w.copy_(torch.randn(self.w.shape, device=DEVICE, generator=self.gen) * 1e-3)
        if regime == 'tied':
            # a quarter of the records share the key aimed at query 0 (and the largest scale), an eighth another key:
            # identical logits per query, filling or straddling the top-k edge
            aimed = torch.where((self.w[0, :, None] * self.q8[0].float()).sum(0) >= 0, 448., -448.).to(torch.float8_e4m3fn)
            self.keys[::4] = aimed
            self.scales[::4] = 1.
            self.keys[1::8] = self.keys[3]
            self.scales[1::8] = self.scales[3]
        elif regime == 'zeros':
            # half the records are zero keys (logit exactly 0) under mostly negative gates: the top-k edge sits on ties at 0
            self.keys.view(torch.uint8)[::2] = 0
            self.w.copy_(-self.w.abs())
            self.w[:, ::5] = self.w[:, ::5].abs() * 1e-3


def arms(joined_only=False):
    from engine.profiles.glm53.net import Glm53Net
    table = {'control': lambda *a: Glm53Net._select_rows(*a, joined=False, native=False),
             'joined': lambda *a: Glm53Net._select_rows(*a, native=False),
             # the shipped path: joined rows, and the horizon mask + top-k as one CUDA launch
             # (engine/kernels/decode_topk). `native=False` above is its same-build control.
             'fused': lambda *a: Glm53Net._select_rows(*a),
             'fused_rows': lambda *a: Glm53Net._select_rows(*a, joined=False)}
    if not joined_only:
        table.update(rows_cat=rows_cat, wide=wide)
    return table


def capture(fx, arm):
    rows, t = fx.rows, fx.t
    outputs = [(torch.empty(rows * t, TOPK + KP - 1, dtype=torch.int32, device=DEVICE),
                torch.empty(rows * t, dtype=torch.int32, device=DEVICE)) for _ in range(LAYERS)]
    keys, scales, caches = fx.keys, fx.scales, fx.caches

    def run():
        caches.gather()
        try:
            for L, (slots, valid) in enumerate(outputs):
                arm(fx.net, L, fx.q8, fx.w, keys, scales, fx.n_cand, fx.contexts, t, caches, slots, valid)
        finally:
            caches._decode_lengths = None
            del caches.block_table
        return outputs

    graph, _ = _capture(run)
    return graph, outputs


def edge_ties(fx):
    """Queries of layer 0 whose k-th score is shared past the top-k (eager, served lanes): ties the selection must break."""
    from engine.base.constants import zeros
    lanes = fx.net.lanes
    fx.caches.gather()
    try:
        seq, ke = fx.caches.row_lengths(fx.contexts, fx.t, KP, lanes.decode_rows.lengths)
        keys_all, scales_all = lanes.decode_rows.candidates(fx.keys, fx.scales, *fx.caches.pool_maps(0), fx.n_cand)
        count = 0
        for r in range(fx.rows):
            sl = slice(r * fx.t, (r + 1) * fx.t)
            logits = lanes.indexer_logits(fx.q8[sl], keys_all[r], scales_all[r], fx.w[sl], ke[sl], ks=zeros(fx.t, DEVICE)).float()
            lanes.decode_rows.horizon(logits, ke[sl])
            kth = logits.topk(K, dim=-1, sorted=True).values[:, -1:]
            above, equal = (logits > kth).sum(-1), (logits == kth).sum(-1)
            count += int(((above + equal > K) & torch.isfinite(kth[:, 0])).sum())
        return count
    finally:
        fx.caches._decode_lengths = None
        del fx.caches.block_table


def numerics(report, fx, graphs, outputs, names):
    ends = fx.capacity - fx.t
    phases = (('random', 'tables', (ends, fx.capacity // 2 + 3)),
              ('tied', 'same', (ends - 5, 37)),
              ('zeros', 'tables', (3, ends)),
              ('tied', 'tables', (fx.capacity // 3, ends - 1)),        # rollback-like: contexts move back, tables rebound
              ('random', 'same', (1, 700)))
    mismatches = {name: 0 for name in names}
    ties = []
    for phase, (regime, tables, contexts) in enumerate(phases):
        fx.data(regime)
        if tables == 'tables':
            fx.tables()
        fx.contexts.copy_(torch.tensor(contexts[:fx.rows], device=DEVICE))
        ties.append(edge_ties(fx))
        for order in (list(range(len(names))), list(reversed(range(len(names))))):
            for i in order:
                for slots, valid in outputs[i]:
                    slots.fill_(-777); valid.fill_(-777)
                graphs[i].replay()
            torch.cuda.synchronize()
            for i, name in enumerate(names):
                for (slots, valid), (want_slots, want_valid) in zip(outputs[i], outputs[0]):
                    mismatches[name] += int((slots != want_slots).sum()) + int((valid != want_valid).sum())
        for slots, valid in outputs[0]:
            assert int(valid.min()) >= 0 and int((slots == -777).sum()) == 0, 'control left poisoned output'
    report('select_rows_numerics', rows=fx.rows, tokens=fx.t, capacity=fx.capacity, n_cand=fx.n_cand, layers=LAYERS,
           phases=[dict(regime=r, tables=tb, contexts=list(c[:fx.rows])) for r, tb, c in phases],
           edge_tie_queries=ties, mismatched_elements=mismatches,
           exact={name: count == 0 for name, count in mismatches.items()}, replay_orders='forward/reverse')
    return mismatches


def launches(graph, replays=4):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    kernels = Counter()
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name] += 1
    return {name: count / replays for name, count in kernels.items()}


def timings(report, name, fx, graphs, **metadata):
    # The same captured graphs in B/A/A/B twice; eviction (a 128 MiB write) sits outside each timed replay.
    flush = torch.empty(128 << 20, device=DEVICE, dtype=torch.uint8)
    for evicted in (False, True):
        samples = []
        for arm, i in (('B', 0), ('A', 1), ('A', 1), ('B', 0)) * 2:
            if not evicted:
                ms = _time(graphs[i], iterations=48)
            else:
                events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(24)]
                for start, end in events:
                    flush.zero_()
                    start.record()
                    graphs[i].replay()
                    end.record()
                events[-1][1].synchronize()
                ms = sum(start.elapsed_time(end) for start, end in events) / len(events)
            samples.append(dict(arm=arm, ms=ms))
        report('timing', candidate=name, rows=fx.rows, tokens=fx.t, capacity=fx.capacity, contexts=fx.contexts.tolist(),
               cache='evicted' if evicted else 'warm', samples=samples, **metadata)


def check(report):
    from engine.profiles.glm53 import lanes
    served = lanes.served(moe_static='stock')
    gen = torch.Generator(device=DEVICE).manual_seed(915)
    failures = []
    # numerics: every capacity bucket, one row (the loop both ways) and two rows (every arm), K=7 verify rows; one-token
    # rows keep the loop in `_select_rows` (a DeepGEMM query block would span rows), the alternatives still join them
    for rows, t, capacities in ((2, 8, NUMERIC_CAPACITIES), (1, 8, (4096, 32768, 131072)), (2, 1, (4096, 32768))):
        for capacity in capacities:
            fx = Fixture(rows, t, capacity, gen)
            fx.net.lanes = served
            table = arms(joined_only=rows == 1)
            names = list(table)
            graphs, outputs = [], []
            try:
                for name in names:
                    graph, out = capture(fx, table[name])
                    graphs.append(graph); outputs.append(out)
                mismatches = numerics(report, fx, graphs, outputs, names)
                if mismatches['joined']:
                    failures.append((rows, t, capacity))
            finally:
                for graph in graphs:
                    graph.reset()
                del fx, graphs, outputs
                torch.cuda.empty_cache()
    if failures:
        report('complete', status='FAIL', failed=failures, consumer_metrics_measured=False)
        raise RuntimeError(f'joined selection differs from the per-row control: {failures}')
    # timing: exact arms only, same graphs in B/A/A/B, warm and evicted; launch counts last (profiler)
    counts = []
    for rows, t, timed in ((2, 8, TIMED), (1, 8, TIMED), (2, 1, ((32768, 32000),))):
        for capacity, context in timed:
            fx = Fixture(rows, t, capacity, gen)
            fx.net.lanes = served
            table = arms(joined_only=rows == 1)
            names = list(table)
            graphs, outputs = [], []
            try:
                for name in names:
                    graph, out = capture(fx, table[name])
                    graphs.append(graph); outputs.append(out)
                mismatches = numerics(report, fx, graphs, outputs, names)
                if mismatches['joined']:
                    raise RuntimeError(f'joined selection differs from the per-row control: {(rows, t, capacity)}')
                fx.data('random')
                fx.contexts.copy_(torch.full((rows,), context, device=DEVICE) - torch.arange(rows, device=DEVICE))
                for i, name in enumerate(names[1:], 1):
                    if mismatches[name]:
                        report('timing_skipped', candidate=name, rows=rows, tokens=t, capacity=capacity, reason='not exact')
                        continue
                    timings(report, name, fx, [graphs[0], graphs[i]], layers=LAYERS)
                if rows == 2 and t == 8:
                    counts.append((capacity, fx, graphs, names, outputs))
                    graphs = []                                   # kept for the launch counts
            finally:
                for graph in graphs:
                    graph.reset()
    for capacity, fx, graphs, names, _ in counts:
        try:
            for name, graph in zip(names, graphs):
                kernels = launches(graph)
                report('launches', candidate=name, rows=fx.rows, tokens=fx.t, capacity=capacity, layers=LAYERS,
                       per_replay=sum(kernels.values()), kernels=kernels)
        except Exception as exc:                                  # the profiler is evidence only; timing already stands
            report('launches_failed', capacity=capacity, error=f'{type(exc).__name__}: {exc}')
        finally:
            for graph in graphs:
                graph.reset()
    report('complete', status='PASS', consumer_metrics_measured=False)


def run(output=None):
    events = []

    def report(event, **values):
        row = dict(event=event, **values)
        events.append(row)
        print(json.dumps(row), flush=True)
        if output:
            Path(output).write_text(''.join(json.dumps(e) + '\n' for e in events))

    root = Path(__file__).resolve().parents[1]
    files = ('engine/profiles/glm53/net.py', 'engine/profiles/glm53/lanes.py', 'engine/profiles/glm53/decode_graphs.py',
             'engine/kernels/indexer.py', 'engine/kernels/deep_gemm.py', 'engine/modules/sparse_indexer.py',
             'probes/engine_decode_select_rows.py')
    report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
           source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
           scope='captured same-build selection components on synthetic pool records; no model, answer or consumer verdict')
    torch.manual_seed(915)
    check(report)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, help='also write the events here (JSON lines)')
    run(ap.parse_args().output)
