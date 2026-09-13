"""What a prefill chunk costs by its size, and what a decode step costs by its rows -- one rank, real weights.

Why (45차, C=4/C=1): the 09-12 onepass record says a decode step at four rows is 2.3-2.5x a one-row step, and
that a 2,304-token prefill chunk beside live decoders costs ~1.19 s where a 9,216-token chunk costs ~3.37 s --
per token 1.4x more (measurements/c4_scaling_20260913). Both numbers come from the fleet with everything
in them: collectives, the drafter, the host. This takes the two questions to one GPU where CUPTI can see
the kernels:

  1. prefill: the same prompt prefilled in chunks of 2,304 / 4,608 / 6,912 / 9,216 tokens. The totals fit
     `total = tokens * v + chunks * F` -- F is what one more chunk boundary costs (CHARTER D9 measured 214.7 ms
     on the vLLM stack; the ST engine has its own number), and the kernel table of the first chunk of each
     size says which kernel carries it.
  2. decode: the captured graph replayed at 1..4 rows of K+1 tokens, and one eager step per width with the
     router counted, so the step time can be read against the unique experts it streams.

    bash probes/run_engine_probe.sh probes/engine_prefill_chunk_profile.py
    bash probes/run_engine_probe.sh probes/engine_prefill_chunk_profile.py --chunk 2304,9216 --tokens 18432

One rank: the collectives are the identity, the sequence-parallel prefill transport is off (its shards need
peers), there is no drafter and no prefix snapshot, and the token ids are random inside this rank's
vocabulary shard -- so expert routing is synthetic (a uniform-routing bound on unique experts) and the
mHC/norm lanes run on every row instead of a quarter of them. Shapes and bytes match production; this
attributes kernel cost, it is not consumer throughput (rule 6 of the ledger).
"""
import argparse
import json
from pathlib import Path
import statistics

import torch

from engine.base.instruments import Recorder
from engine.profiles.glm53 import facts
from engine.profiles.glm53.boot import build
from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs
from engine.profiles.glm53.lanes import MOE_STATIC_PRODUCTION, served
from engine.profiles.glm53.net import Step
from probes.engine_graph_profile import lane_of


class IsolatedRank:
    """One rank alone: every collective is the identity, so what is measured is this rank's arithmetic."""
    world_size = 4

    def __init__(self, rank: int):
        self.rank = rank

    def all_reduce(self, x):
        return x

    def all_reduce_max(self, x):
        return x

    def all_gather(self, x, dim=-1):
        return x

    def wait_prepared(self, phase, **kwargs):
        pass  # no peers: this is a diagnostic, not a fleet boot


def rank_on_this_node(ranks: str) -> int:
    """The rank whose file this node holds (srv4 holds rank3of4 only): the isolated rank is that one."""
    files = sorted(Path(ranks).glob(f"rank*of{facts.TP}.safetensors"))
    if not files:
        raise FileNotFoundError(f"no rank file under {ranks}")
    return int(files[0].name[len("rank"):].split("of")[0])


def kernel_table(prof, steps: int):
    """Per-step device time by lane and by kernel from one profile of `steps` repetitions."""
    rows = []
    for event in prof.key_averages():
        total = getattr(event, "self_device_time_total", 0.0) or 0.0
        if total > 0:
            rows.append((event.key, event.count, total))
    rows.sort(key=lambda r: -r[2])
    lanes = {}
    for name, calls, total in rows:
        lane = lanes.setdefault(lane_of(name), [0.0, 0])
        lane[0] += total
        lane[1] += calls
    return dict(total_us=sum(r[2] for r in rows) / steps,
                lanes={lane: dict(us=v[0] / steps, launches=v[1] / steps) for lane, v in sorted(lanes.items(), key=lambda kv: -kv[1][0])},
                kernels=[dict(kernel=name[:100], us=total / steps, launches=calls / steps) for name, calls, total in rows[:30]])


def print_table(title, table):
    print(f"\n  {title}: {table['total_us'] / 1000:.2f} ms of kernels")
    print(f"  {'lane':22s}{'ms':>10s}{'share':>8s}{'launches':>10s}")
    for lane, v in table["lanes"].items():
        print(f"  {lane:22s}{v['us'] / 1000:10.2f}{v['us'] / table['total_us'] * 100:7.1f}%{v['launches']:10.1f}")
    for k in table["kernels"][:12]:
        print(f"      {k['us'] / 1000:9.3f} ms x{k['launches']:6.1f}  {k['kernel'][:80]}")


def fit_fixed_cost(points):
    """points: (tokens, chunks, total_ms) -> (F ms per chunk, v ms per token) by least squares."""
    import numpy as np
    A = np.array([[c, t] for t, c, _ in points], dtype=float)
    y = np.array([ms for _, _, ms in points], dtype=float)
    (F, v), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(F), float(v)


def overlap_summary(windows, chunk, solo_decode_ms, solo_chunk_ms):
    """Pure arithmetic over the two streams' event times (ms from one origin).

    `windows`: the decode replays' (start, end); `chunk`: the prefill chunk's (start, end). A replay is
    'inside' when it ran wholly while the chunk ran, 'outside' when wholly before or after. The sequential
    time of the inside replays plus the chunk is what one stream would have taken for the same work."""
    inside = [e - s for s, e in windows if s >= chunk[0] and e <= chunk[1]]
    outside = [e - s for s, e in windows if e <= chunk[0] or s >= chunk[1]]
    chunk_ms = chunk[1] - chunk[0]
    together = dict(chunk_ms=chunk_ms, chunk_stretch=chunk_ms / solo_chunk_ms if solo_chunk_ms else None,
                    replays_inside=len(inside), replays_outside=len(outside),
                    decode_inside_ms=statistics.median(inside) if inside else None,
                    decode_outside_ms=statistics.median(outside) if outside else None,
                    decode_stretch=(statistics.median(inside) / solo_decode_ms) if inside and solo_decode_ms else None)
    if inside:
        sequential = len(inside) * solo_decode_ms + solo_chunk_ms
        together["sequential_ms_for_the_same_work"] = sequential
        together["overlap_gain"] = sequential / chunk_ms       # >1: the two streams did more than one would have
    return together


def coexist(net, caches, graphs, ids, ctx0, slots, chunk, lo, hi, samples, dev):
    """A prefill chunk on one stream beside four decoding rows replaying on another -- D9's unmeasured shape
    (45차 §80 asked it with a matmul stand-in), on the real kernels of one rank.

    The decode stream is queued deep first (its replays are one graph launch each), then the chunk's ~2,000
    eager launches stream in beside it. Rows 0..3 decode; the last row holds the chunk. Module-level device
    workspaces that a captured graph and an eager call would otherwise share are renewed for the eager side
    before each chunk (the graph keeps the addresses it captured): the MLA barrier/split workspaces and the
    b12x static kernel's claim counter -- two streams must never share a grid barrier."""
    from engine.kernels import mla as mk
    from engine.kernels.b12x import moe_dispatch as md
    dec = list(range(4))
    row_p, slot_p = len(slots) - 1, slots[-1]
    caches.pool.release(row_p)
    caches.reset_slot(slot_p)
    caches.pool.reserve(row_p, chunk)
    prompt = torch.randint(lo, hi, (chunk,), device=dev)
    step = Step.decode([(ids[i], ctx0, i, slots[i]) for i in dec])
    pf = Step.prefill(prompt, 0, row_p, slot_p)
    caches.prepare(step)
    caches.prepare(pf)
    torch.cuda.synchronize()
    A, B = torch.cuda.Stream(), torch.cuda.Stream()

    def prefill_once():
        mk._WS = None
        mk._MLA_WS = None
        md._STATIC_V2_COUNTERS.clear()
        md._STATIC_V2_STAMPS.clear()
        net.forward(pf, caches)

    def event():
        return torch.cuda.Event(enable_timing=True)

    for _ in range(2):                                   # warm: this chunk width's kernels, both streams
        with torch.cuda.stream(A):
            graphs.run(step)
        with torch.cuda.stream(B):
            prefill_once()
        torch.cuda.synchronize()
    solo_dec, solo_pf = [], []
    with torch.cuda.stream(A):
        for _ in range(samples):
            s, e = event(), event()
            s.record(A); graphs.run(step); e.record(A); e.synchronize()
            solo_dec.append(s.elapsed_time(e))
    with torch.cuda.stream(B):
        for _ in range(3):
            s, e = event(), event()
            s.record(B); prefill_once(); e.record(B); e.synchronize()
            solo_pf.append(s.elapsed_time(e))
    torch.cuda.synchronize()
    solo_decode_ms, solo_chunk_ms = statistics.median(solo_dec), min(solo_pf)
    n_rep = max(samples, int(3 * solo_chunk_ms / solo_decode_ms) + 4)
    origin, b0, b1 = event(), event(), event()
    marks = [(event(), event()) for _ in range(n_rep)]
    origin.record(A)
    with torch.cuda.stream(A):
        for s, e in marks:
            s.record(A); graphs.run(step); e.record(A)
    with torch.cuda.stream(B):
        b0.record(B); prefill_once(); b1.record(B)
    torch.cuda.synchronize()
    windows = [(origin.elapsed_time(s), origin.elapsed_time(e)) for s, e in marks]
    chunk_w = (origin.elapsed_time(b0), origin.elapsed_time(b1))
    together = overlap_summary(windows, chunk_w, solo_decode_ms, solo_chunk_ms)
    out = dict(decode_rows=len(dec), chunk_tokens=chunk, solo_decode_ms=solo_decode_ms, solo_chunk_ms=solo_chunk_ms,
               replays=n_rep, together=together, windows=windows, chunk_window=chunk_w)
    print(f"coexist: solo decode rows=4 {solo_decode_ms:.1f} ms/step, solo chunk {chunk} tokens {solo_chunk_ms:.0f} ms; together: "
          f"chunk {together['chunk_ms']:.0f} ms ({together['chunk_stretch']:.2f}x), decode inside the chunk "
          f"{together['decode_inside_ms'] if together['decode_inside_ms'] is None else round(together['decode_inside_ms'], 1)} ms "
          f"({together['decode_stretch'] if together['decode_stretch'] is None else round(together['decode_stretch'], 2)}x), "
          f"{together['replays_inside']} replays inside, overlap gain {together.get('overlap_gain')}", flush=True)
    caches.pool.release(row_p)
    caches.reset_slot(slot_p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # The queue invokes this with no arguments (bench/fleet_onepass.ST_FLAGS admits --ranks, --ckpt-meta,
    # --layers, --tokens, --chunk, --seed, --samples and --output): every default has to work on a node.
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--ckpt-meta", default=str(facts.CKPT))
    ap.add_argument("--layers", default="", help="a slice like 0-4; the whole model by default")
    ap.add_argument("--tokens", type=int, default=27648, help="the prompt prefilled once per chunk size (12 x 2304)")
    ap.add_argument("--chunk", default="2304,4608,6912,9216", help="chunk sizes, tokens, comma separated")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--samples", type=int, default=16, help="decode replays timed per row count")
    ap.add_argument("--moe-static", default=MOE_STATIC_PRODUCTION,
                    help="the b12x lane cell (lanes.parse_moe_static): production 't,r,sf6,q0'. A tiled cell without sf6 "
                         "('t,r') has no prefill kernel (the dispatcher refuses it); the row-major 'u' cell is the stock pair")
    ap.add_argument("--lanes", default="decode,prefill",
                    help="sections: decode (rows 1..4), prefill (the chunk sweep), coexist (a chunk beside four decoding rows, two streams)")
    ap.add_argument("--output", default="/cache/prefill-chunk-profile.json")
    a = ap.parse_args()
    lanes = {x.strip() for x in a.lanes.split(",") if x.strip()}
    if not lanes or lanes - {"decode", "prefill", "coexist"}:
        raise SystemExit(f"--lanes takes decode, prefill and/or coexist: {a.lanes!r}")

    F = facts.load(a.ckpt_meta)
    layers = list(range(F.layers))
    if a.layers:
        first, last = (int(v) for v in a.layers.split("-"))
        layers = list(range(first, last + 1))
    chunks = sorted({int(c) for c in a.chunk.split(",") if c})
    if any(c <= 0 or c % F.chunk_align for c in chunks):
        raise SystemExit(f"chunk sizes must be positive multiples of {F.chunk_align}: {chunks}")
    if a.tokens <= 0 or a.tokens % F.block:
        raise SystemExit(f"--tokens must be a positive multiple of the block {F.block}")
    rank = rank_on_this_node(a.ranks)
    comm = IsolatedRank(rank)
    torch.manual_seed(a.seed)

    rows_max, t = (5 if "coexist" in lanes else 4), F.spec_k + 1        # coexist: four decoding rows and one for the chunk
    from engine.profiles.glm53.caches import layout
    shape = layout(F, layers)
    long_blocks = -(-a.tokens // F.block)
    blocks = max(long_blocks, rows_max * (-(-(F.chunk_align + t) // F.block)) + -(-chunks[0] // F.block)) + 8
    kv_gib = ((rows_max + 1) * shape.slot_bytes + blocks * (shape.block_bytes + rows_max * 4) + (64 << 20)) / 2**30
    _, net, caches, _, _ = build(comm, layers, served(moe_static=a.moe_static), a.ranks, kv_gib, rows_max, False,
                                 Recorder("profile"), ckpt_meta=a.ckpt_meta, execution="native")
    for layer in net.dense.values():                 # calibration observers off: this asks about the target kernels
        layer.observer = None
    net.prefill_transport = None                     # sequence parallelism needs peers; one rank runs the plain path
    caches.paged.zero_()
    caches.state.zero_()
    dev = caches.device
    vp = F.vocab_local
    lo, hi = rank * vp, rank * vp + min(vp, 30000)   # ids this rank embeds (the others are the peers' shards)
    print(f"weights loaded: rank {rank}, {len(layers)} layers, prompt {a.tokens} tokens, chunks {chunks}, "
          f"{kv_gib:.2f} GiB of KV ({blocks} blocks)", flush=True)
    result = dict(rank=rank, layers=len(layers), tokens=a.tokens, chunks=chunks, spec_k=F.spec_k, seed=a.seed,
                  moe_static=a.moe_static, lanes=sorted(lanes),
                  scope="one rank, identity collectives, no SP transport, no drafter, no prefix marks, synthetic routing")

    # -- 2. decode by rows: capture first (capture needs no live slots), then real contexts --------------------
    aux_layers = tuple(L for L in (5, 14, 24, 33, 42) if L in layers)
    graphs = Glm53DecodeGraphs(net, caches, rows_max, t, aux_layers=aux_layers, ceiling=4096)
    print("decode graphs captured", flush=True)
    slots = [caches.slots.take(i) for i in range(rows_max)]
    ctx0 = F.chunk_align
    for i in range(rows_max):
        caches.pool.reserve(i, ctx0 + 2 * t)
        prompt = torch.randint(lo, hi, (ctx0,), device=dev)
        step = Step.prefill(prompt, 0, i, slots[i])
        caches.prepare(step)
        net.forward(step, caches)
    torch.cuda.synchronize()
    ids = [torch.randint(lo, hi, (t,), device=dev) for _ in range(rows_max)]
    decode = []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for n in range(1, rows_max + 1):
        step = Step.decode([(ids[i], ctx0, i, slots[i]) for i in range(n)])
        caches.prepare(step)
        for _ in range(3):
            graphs.run(step)
        torch.cuda.synchronize()
        times = []
        for _ in range(a.samples):
            start.record()
            graphs.run(step)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        # the eager step with the router counted: how many experts this width streams (uniform-routing bound)
        unique = {}
        original = net.route

        def counted(L, x, _route=original):
            sel, w = _route(L, x)
            unique[L] = int(torch.unique(sel).numel())
            return sel, w
        net.route = counted
        try:
            caches.prepare(step)
            net.forward(step, caches, aux_layers=aux_layers)
            torch.cuda.synchronize()
        finally:
            net.route = original
        table = None
        if n in (1, 4):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(a.samples):
                    graphs.run(step)
                torch.cuda.synchronize()
            table = kernel_table(prof, a.samples)
        row = dict(rows=n, tokens=n * t, replay_ms_median=statistics.median(times), replay_ms_min=min(times),
                   unique_experts_mean=statistics.mean(unique.values()) if unique else None,
                   unique_experts=unique, kernels=table)
        decode.append(row)
        print(f"decode rows={n}: {row['replay_ms_median']:.2f} ms/step (min {row['replay_ms_min']:.2f}), "
              f"unique experts/layer {row['unique_experts_mean']:.1f}", flush=True)
        if table:
            print_table(f"decode rows={n} kernels", table)
    result["decode"] = decode
    if "coexist" in lanes:
        result["coexist"] = coexist(net, caches, graphs, ids, ctx0, slots, chunks[0], lo, hi, a.samples, dev)
    graphs.graphs.close()                            # its pools go back to the allocator before the long prefills
    for i in range(rows_max):
        caches.pool.release(i)
        caches.slots.give(slots[i])
    caches.reset()

    # -- 1. prefill by chunk size: the same prompt, prefilled from context 0 ------------------------------
    slot = caches.slots.take(0)
    prompt = torch.randint(lo, hi, (a.tokens,), device=dev)
    prefill = []
    tables = {}
    for pass_ in (("warm", "measure") if "prefill" in lanes else ()):
        for chunk in chunks:
            caches.reset_slot(slot)
            caches.pool.release(0)
            caches.pool.reserve(0, a.tokens)
            times = []
            ctx = 0
            while ctx < a.tokens:
                n = min(chunk, a.tokens - ctx)
                step = Step.prefill(prompt[ctx:ctx + n], ctx, 0, slot)
                caches.prepare(step)
                profile = pass_ == "measure" and ctx == 0 and chunk in (chunks[0], chunks[-1])
                if profile:
                    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                    prof.__enter__()
                start.record()
                net.forward(step, caches)
                end.record()
                end.synchronize()
                if profile:
                    prof.__exit__(None, None, None)
                    tables[chunk] = kernel_table(prof, 1)
                times.append(dict(ctx=ctx, tokens=n, ms=start.elapsed_time(end)))
                ctx += n
            if pass_ == "measure":
                total = sum(x["ms"] for x in times)
                prefill.append(dict(chunk=chunk, chunks=len(times), total_ms=total, per_token_us=total / a.tokens * 1000,
                                    steps=times, first_chunk_kernels=tables.get(chunk)))
                print(f"prefill chunk={chunk}: {len(times)} chunks, {total:.0f} ms for {a.tokens} tokens "
                      f"({a.tokens / total * 1000:.0f} tok/s); first chunk {times[0]['ms']:.0f} ms, last {times[-1]['ms']:.0f} ms", flush=True)
    for chunk, table in tables.items():
        print_table(f"prefill first chunk of {chunk} tokens", table)
    fit = None
    if len(prefill) >= 2:
        F_ms, v_ms = fit_fixed_cost([(a.tokens, p["chunks"], p["total_ms"]) for p in prefill])
        fit = dict(fixed_ms_per_chunk=F_ms, ms_per_token=v_ms)
        print(f"\n  fit over {len(prefill)} chunk sizes: total = {v_ms * 1000:.1f} us/token + {F_ms:.0f} ms/chunk", flush=True)
    result["prefill"] = prefill
    result["prefill_fit"] = fit
    if len(tables) == 2:
        import re
        small, big = (tables[c] for c in (chunks[0], chunks[-1]))
        # the same lane can pick a differently named specialization per width (the dynamic MoE's Q0 cell at
        # 2,304 rows, plain at 9,216): pair by the name with the cell suffix dropped
        family = lambda name: re.sub(r"_q0(?=MoE)", "", name)
        by = {family(k["kernel"]): k["us"] for k in small["kernels"]}
        pairs = [(family(k["kernel"]), by[family(k["kernel"])], k["us"]) for k in big["kernels"] if family(k["kernel"]) in by]
        ratio = chunks[-1] / chunks[0]
        print(f"\n  first-chunk kernels, {chunks[0]} vs {chunks[-1]} tokens (a kernel proportional to tokens grows {ratio:.1f}x):")
        for name, s, b in sorted(pairs, key=lambda p: -p[2])[:12]:
            print(f"      {s / 1000:8.2f} -> {b / 1000:8.2f} ms  ({b / max(s, 1e-9):4.2f}x)  {name[:70]}")
        result["first_chunk_pairs"] = [dict(kernel=n, small_us=s, big_us=b) for n, s, b in pairs]

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {out}", flush=True)
    print("RESULT " + json.dumps(dict(decode=[{k: v for k, v in r.items() if k not in ("kernels", "unique_experts")} for r in decode],
                                      prefill=[{k: v for k, v in p.items() if k not in ("steps", "first_chunk_kernels")} for p in prefill],
                                      fit=fit,
                                      coexist={k: v for k, v in result["coexist"].items() if k not in ("windows",)}
                                      if "coexist" in result else None)), flush=True)


if __name__ == "__main__":
    main()
