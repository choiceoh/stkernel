"""Qwen3.8's captured decode step on one GB10, kernel by kernel (probe, single-GPU lane).

The fleet measures a C=1 step as a whole -- 35.3 / 34.7 ms with one-shot, 42.2 ms on NCCL (2026-09-18/19 windows,
measurements/qwen38_boot_window_20260918 and the speed sample) -- and nothing inside it, because the step is a replayed
graph and CUPTI is the only thing that sees through one. The fleet's `/v1/engine/profile` does that, but it needs a
window, which is production's downtime. This builds the served net for ONE rank of TP=4 on one GPU from the rank file's
own weights, captures the target's verify graphs and the MTP head's draft graphs exactly as the fleet boot does
(decode_graphs.py), replays a shape many times under the CUDA profiler and says what the replay is made of: launches
and device microseconds, per kernel and per family (MoE, GEMM, hyper-connection, GDN, QSA, norms, ...).

The collectives are this rank's own contribution (OneRankComm: all_reduce returns its input, all_gather repeats it), so
what is timed is the step's compute and launches; the communication is the fleet's (one-shot vs NCCL: about 7 ms a step).

Beside production the single-GPU lane leaves a probe about 4 GiB (srv4's MemAvailable less the budget must clear 16
GiB), and 8 layers of weights are 4.2. So the step is solved from small nets built one after another, each 1-4 layers
of the rank file's own weights: the fixed part (embedding, head, closing mixer, addressing), a GDN layer, a QSA layer
and the PLE injection are four unknowns from four layer sets, per graph and per kernel family --

    [4, 5, 6, 7]   fixed + 3 GDN + 1 QSA        [4, 5]   fixed + 2 GDN
    [7]            fixed + 1 QSA                [1]      fixed + 1 GDN + PLE (facts.ple_layers: before layer 1)

    step(48 layers) = fixed + 36 GDN + 12 QSA + PLE        (the MTP head's draft graph is its own, whatever the layers)

    python3 probes/engine_kernel_check.py --lanes qwen38_step --ranks /home/choiceoh/models/st-qwen38-tep4 \\
        --output /cache/qwen38-step.json                                          (the queue's single-GPU lane)

`--lanes qwen38_step_overlap` (carry M5) builds the first layer set three times: served, and the shared expert forked
onto a second stream beside the routed experts at one request's rows (`overlap-one`) and at every captured step
(`overlap-all`). The fork launches what the served arm launches, so the arm shows in a replay's wall time alone.

`--lanes qwen38_step_ab` builds every layer set twice, one after the other: the served lanes (`served`) and the same
lanes with the skinny GEMV's table emptied (`mm`: the router back on torch.mm, the mixer sites back to five launches on
cuBLAS -- the lanes before engine/kernels/common/skinny_gemv) -- the step and its families per arm, and the served arm
less the other.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAYER_SETS = ((4, 5, 6, 7), (4, 5), (7,), (1,))
SHAPES = ((1, 6), (1, 43), (4, 43))       # (rows, bucket blocks): C=1 near 4K and 32K context, C=4 near 32K
REPLAYS = 50
KV_GIB = 0.25
SPEC_K = 3                                # the operator's K (#1182: fleet --spec-k 3): verify 4 tokens, draft a chain of 3
MAX_GIB = 4.0                             # this process's own device-memory ceiling: the lane's budget beside production
FULL = {"fixed": 1, "gdn": 36, "qsa": 12, "ple": 1}
ARMS = ("served", "mm")                   # qwen38_step_ab: the served lanes, and the lanes before the skinny GEMV
MTP_ARMS = ("served", "mtp-w4", "mtp-fp8", "experts-fp8")   # qwen38_step_mtp: MTP dense BF16 (served), W4A8, FP8; experts FP8
# qwen38_step_overlap (carry M5): the shared expert forked beside the routed ones -- at one request's rows, at every
# captured step. The launches and their bytes are the served arm's; what the arm moves is a replay's wall time
OVERLAP_ARMS = ("served", "overlap-one", "overlap-all")
MTP_FP8_DIR = Path("/home/choiceoh/models/st-qwen38-mtp-fp8")  # mtp_fp8.py's side files (srv4)

FAMILIES = (
    ("moe b12x", r"[Mm]oe|[Mm]icro|[Ss]tatic|[Dd]ynamic|b12x|kernel_cutlass"),
    ("hc gated residual", r"gated_residual|_enter|_leave|_mix|_inject"),
    ("gdn / kda", r"gdn|kda|recurrent|chunk_|_ring|gated_delta"),
    ("qsa", r"qsa|expand_|select_blocks"),
    ("conv", r"conv"),
    ("gemm (cublas/cutlass)", r"gemm|gemv|nvjet|cutlass|cublas|Gemm|matmul|sm90_|sm100_|sm120_"),
    ("dense w4/fp8", r"w4a|fp8|quant|dense"),
    ("norm / rope", r"norm|rope|rms"),
    ("route / moe glue", r"route|softmax|topk|moe_finish|gated_sum|swiglu|silu"),
    ("addressing", r"step_addr|address"),
    ("elementwise (torch)", r"elementwise|vectorized|unrolled|reduce_kernel|fill|copy|CatArray|index"),
)


class OneRankComm:
    """base/comm.Comm's surface the served net reads, for one rank of TP=4 on one GPU: every collective is this rank's
    own contribution -- the probe times compute and launches, the fleet times communication."""

    graph_capture_safe = True

    def __init__(self, rank: int, world: int = 4):
        self.rank, self.world_size = rank, world

    def all_reduce(self, t):
        return t

    def all_reduce_max(self, t):
        return t

    def all_gather(self, t, dim=-1):
        import torch
        return torch.cat([t] * self.world_size, dim=dim)


class ZeroPLETable:
    """ple_table.PLETable's surface, answering zeros: the probe reads no SSD table (a decode step's read of its rows
    is host work before the replay, which the fleet measures)."""

    def __init__(self, rows: int, width: int, scale: float):
        self.rows, self.width, self.scale = rows, width, scale
        self.reads = self.rows_read = 0

    def gather(self, rows):
        import numpy as np
        return np.zeros((len(rows), self.width), dtype=np.uint8)

    def close(self):
        pass


def family(name: str) -> str:
    for label, pattern in FAMILIES:
        if re.search(pattern, name):
            return label
    return "other"


def counts(F, layers) -> dict:
    """The unknowns a layer set holds: fixed once, GDN and QSA layers by type, PLE where its layer is in the set."""
    kinds = [F.config["layer_types"][L] for L in layers]
    return {"fixed": 1, "gdn": kinds.count("linear_attention"), "qsa": kinds.count("full_attention"),
            "ple": sum(L in F.ple_layers for L in layers)}


def solve(rows: "list[tuple[dict, float]]", unknowns=("fixed", "gdn", "qsa", "ple")) -> dict:
    """Least squares over (counts, value) rows -> {unknown: value}. Four sets for four unknowns is exact."""
    import numpy as np
    a = np.array([[c[u] for u in unknowns] for c, _ in rows], dtype=float)
    b = np.array([v for _, v in rows], dtype=float)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    return {u: float(v) for u, v in zip(unknowns, x)}


def extrapolate(parts: dict, full=FULL) -> float:
    return sum(parts[u] * n for u, n in full.items())


def build(meta: Path, ranks: Path, rank: int, layers, *, max_seqs: int, kv_gib: float, spec_k: int = SPEC_K,
          mtp_precision: str = "bf16", mtp_experts_dir: "Path | None" = None, shared_overlap: "bool | str" = False):
    """The served net, caches and captured graphs for one rank over `layers` -> (F, net, caches, target, draft), at
    `spec_k` drafts a step as fleet.build takes it (the facts replaced before anything sizes from them)."""
    import dataclasses
    import torch
    from engine.base.arena import Arena
    from engine.base.params import total_bytes
    from engine.profiles.qwen38 import facts, lanes as lane_tables
    from engine.profiles.qwen38.caches import Qwen38Caches, cache_capacity, layout, snapshot_layout
    from engine.profiles.qwen38.decode_graphs import DraftGraphs, TargetGraphs
    from engine.profiles.qwen38.fleet import rank_loader
    from engine.profiles.qwen38.net import Qwen38Net

    F = facts.load(meta)
    if spec_k != F.spec_k:
        F = dataclasses.replace(F, spec_k=spec_k)
    net = Qwen38Net(F, OneRankComm(rank), lane_tables.served(), layers=list(layers), mtp=True,
                    mtp_precision=mtp_precision, mtp_experts="fp8" if mtp_experts_dir else "nvfp4",
                    shared_overlap=shared_overlap)
    specs = net.specs()
    nb, snapshots = cache_capacity(F, net.layers, kv_gib, max_seqs, 0.05, mtp=True)
    snapshots = min(snapshots, 9)           # a net with no GDN layer has empty snapshots, and the count would run away
    arena = Arena(total_bytes(specs) + 256 * (len(specs) + 64) + layout(F, net.layers, mtp=True).nbytes(nb, max_seqs)
                  + snapshots * snapshot_layout(F, net.layers)[0])
    loader = rank_loader(ranks / f"rank{rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout)
    side = {s.name for s in net.side_specs()}
    views = loader.load([s.name for s in specs if s.name not in side], arena=arena)
    if side:
        from engine.profiles.qwen38 import mtp_fp8
        views.update(rank_loader(mtp_fp8.path(mtp_experts_dir, rank), expected_layout=mtp_fp8.LAYOUT)
                     .load(sorted(side), arena=arena))
    net.bind(views)
    if net._ple is not None:
        net.attach_ple(ZeroPLETable(F.ple_rows_per_rank, F.ple_head_dim, float(net._ple_scale)),
                       max_rows=max_seqs * (F.spec_k + 1))
    net.prepare_dense(None)
    caches = Qwen38Caches(arena, F, net.layers, nb, max_seqs, snapshots, mtp=True)
    tokens = F.spec_k + 1
    target = TargetGraphs(net, caches, max_seqs, tokens, ceiling=F.max_position)
    draft = DraftGraphs(net, caches, max_seqs, tokens, k=F.spec_k, ceiling=F.max_position)
    torch.cuda.synchronize()
    return F, net, caches, target, draft


def seat(graphs, caches, F, shape, seed: int = 0) -> None:
    """The shape's static inputs as a served step leaves them: each row's context at the end of its bucket, its block
    table covering it (pages spread over the pool), random token ids (the router sees real embeddings)."""
    import torch
    n, t, blocks = shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pool = caches.block_table.shape[1]
    for r in range(n):
        pages = (torch.arange(blocks) * max(1, pool // (blocks * n)) + r) % pool
        caches.block_table[r, :blocks].copy_(pages.to(torch.int32))
    graphs.metadata[shape][0].fill_(blocks * F.block - t - F.spec_k)      # room for the draft chain's reach too
    inputs = graphs.graphs.inputs[shape]
    ids = inputs[0].ids if isinstance(inputs, tuple) else inputs.ids
    ids.copy_(torch.randint(0, F.vocab, (ids.numel(),), generator=gen))


def replay_profile(graphs, shape, replays: int) -> dict:
    """{'wall_us', 'device_us', 'launches', 'families', 'kernels'} per replay of the shape's graph."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    g = graphs.graphs.graphs[shape]
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    wall = start.elapsed_time(end) * 1000 / replays
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    kernels, families = {}, {}
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", None)
        if us is None:
            us = getattr(e, "self_cuda_time_total", 0)
        if not us or e.count <= 0:
            continue
        k = kernels.setdefault(e.key, [0.0, 0.0])
        k[0] += e.count / replays
        k[1] += us / replays
        f = families.setdefault(family(e.key), [0.0, 0.0])
        f[0] += e.count / replays
        f[1] += us / replays
    return {"wall_us": round(wall, 1), "device_us": round(sum(v[1] for v in kernels.values()), 1),
            "launches": round(sum(v[0] for v in kernels.values()), 1),
            "families": {k: {"launches": round(v[0], 1), "us": round(v[1], 1)} for k, v in families.items()},
            "kernels": {k[:200]: {"launches": round(v[0], 2), "us": round(v[1], 2)}
                        for k, v in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:40]}}


def served_loop(F, net, caches, target, draft, *, prompt: int = 512, steps: int = 64) -> dict:
    """One request decoded at C=1 through the served model (adapter.build_model: base/composed.ComposedModel's verify
    over these graphs), the way the runner drives it -> wall a step, and what the host spent in its parts. A step's
    replays are serialized by the host's reads (the draft's `.tolist()`, the picks' `.tolist()`), so the host's own
    time is the step's wall less the two replays -- and it does not grow with the layers, so this net's is the
    model's. The PLE table here is ZeroPLETable: the fleet's step also reads its rows off the SSD."""
    import torch
    from engine.profiles.qwen38.adapter import build_model
    caches.reset()
    model, _ = build_model(net, caches, F, eos_ids=[F.vocab + 7], max_new=(steps + 8) * (F.spec_k + 1) + 16, temperature=0.0,
                           top_p=1.0, seed=0, drafter=True)
    model.composition.graphs, model.drafter.graphs = target, draft
    spent = {"stage_ple": 0.0, "target_run": 0.0, "draft_run": 0.0, "picks": 0.0}

    def timed(name, fn):
        def call(*args, **kwargs):
            began = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                spent[name] += time.perf_counter() - began
        return call

    net.stage_ple = timed("stage_ple", net.stage_ple)
    target.run = timed("target_run", target.run)
    draft.run = timed("draft_run", draft.run)
    model._draw_ahead = timed("picks", model._draw_ahead)
    seq, gen = 0, torch.Generator(device="cpu").manual_seed(1)
    slot = caches.slots.take(seq)
    try:
        caches.pool.reserve(seq, prompt + (steps + 2) * (F.spec_k + 1) + 8)
        model.add(seq, torch.randint(0, F.vocab, (prompt,), generator=gen).tolist(), temperature=0.0)
        model.open(seq, slot)
        model.prefill(seq, 0, prompt, None, slot)
        for _ in range(4):                                   # the first steps settle the host's own caches
            model.decode([seq], [None], [slot])
        torch.cuda.synchronize()
        for key in spent:
            spent[key] = 0.0
        made0, began = model.generated_count(seq), time.perf_counter()
        for _ in range(steps):
            model.decode([seq], [None], [slot])
        torch.cuda.synchronize()
        wall = (time.perf_counter() - began) / steps
        made = model.generated_count(seq) - made0
    finally:
        model.close(seq)
        caches.pool.release(seq)
        caches.slots.give(slot)
        caches.reset()
    return {"steps": steps, "wall_us": round(wall * 1e6, 1), "tokens_a_step": round(made / steps, 3),
            "spent_us_a_step": {k: round(v / steps * 1e6, 1) for k, v in spent.items()}}


def assemble(builds: dict, F) -> dict:
    """Per graph: the four unknowns solved from the layer sets, for wall, device time, launches and each family, and
    the 48-layer step they add up to."""
    out = {}
    keys = sorted({g for b in builds.values() for g in b["graphs"]})
    for key in keys:
        rows = [(b["counts"], b["graphs"][key]) for b in builds.values() if key in b["graphs"]]
        if len(rows) < 4:
            continue
        entry = {}
        for metric in ("wall_us", "device_us", "launches"):
            parts = solve([(c, g[metric]) for c, g in rows])
            entry[metric] = {"parts": {u: round(v, 1) for u, v in parts.items()},
                             "step_48": round(extrapolate(parts) if key.startswith("target") else rows[0][1][metric], 1)}
        fams = {}
        for fam in sorted({f for _, g in rows for f in g["families"]}):
            parts = solve([(c, g["families"].get(fam, {}).get("us", 0.0)) for c, g in rows])
            fams[fam] = {"parts_us": {u: round(v, 1) for u, v in parts.items()},
                         "step_48_us": round(extrapolate(parts) if key.startswith("target")
                                             else rows[0][1]["families"].get(fam, {}).get("us", 0.0), 1)}
        entry["families"] = dict(sorted(fams.items(), key=lambda kv: -kv[1]["step_48_us"]))
        out[key] = entry
    return out


def stacks(fn, *, top: int = 30) -> list:
    """The kernels one eager call of `fn` launches, grouped by the torch op that launched them and the innermost engine
    frame on its Python stack -> [{'op', 'where', 'launches', 'us', 'kernels'}], the largest first. A replay's kernels
    have no stack (CUPTI sees the graph, not the Python that recorded it); the same forward run eagerly does."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        fn()
        torch.cuda.synchronize()
    groups = {}
    for e in prof.events():
        kernels = [k for k in getattr(e, "kernels", []) or []]
        if not kernels:
            continue
        frames = [f for f in (e.stack or []) if "/engine/" in f or "/probes/" in f]
        where = frames[0] if frames else "?"
        key = (e.name, where)
        g = groups.setdefault(key, {"op": e.name, "where": where, "launches": 0, "us": 0.0, "kernels": set()})
        for k in kernels:
            g["launches"] += 1
            g["us"] += k.duration
            g["kernels"].add(k.name[:80])
    rows = sorted(groups.values(), key=lambda g: -g["us"])[:top]
    return [{**g, "us": round(g["us"], 1), "kernels": sorted(g["kernels"])[:3]} for g in rows]


def calls(fn, *, top: int = 40) -> list:
    """The torch functions one eager call of `fn` makes, by name and the innermost engine/probe frame that made them
    -> [{'func', 'where', 'count'}]: where `stacks`' torch ops come from (the profiler's stacks came back empty)."""
    import traceback
    import torch
    seen = {}

    class Where(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", None) or str(func)
            if not name.startswith("__get") and name not in ("size", "dim", "stride", "is_contiguous", "data_ptr",
                                                             "numel", "element_size", "device", "dtype", "shape"):
                frames = [f for f in traceback.extract_stack()[:-1] if "/engine/" in f.filename or "/probes/" in f.filename]
                where = f"{frames[-1].filename.split('/repo/')[-1]}:{frames[-1].lineno}" if frames else "?"
                seen[name, where] = seen.get((name, where), 0) + 1
            return func(*args, **(kwargs or {}))

    with Where():
        fn()
    torch.cuda.synchronize()
    rows = sorted(({"func": k[0], "where": k[1], "count": v} for k, v in seen.items()), key=lambda r: -r["count"])
    return rows[:top]


def measure(ranks: Path, rank: int, layers, *, shapes=SHAPES, replays: int = REPLAYS, kv_gib: float = KV_GIB,
            max_gib: float = MAX_GIB, max_seqs: int = 4, loop: bool = False, arm: str = "served") -> dict:
    """One layer set, in this process: the kernel shape bound, the net built, every shape replayed -> the build's row.
    `arm` "mm": the skinny GEMV's table emptied first -- the router on torch.mm, the mixers in five launches on cuBLAS;
    "mtp-w4" / "mtp-fp8": the MTP head's dense projections at that precision instead of the served BF16."""
    import torch
    if arm not in ARMS + MTP_ARMS + OVERLAP_ARMS:
        raise ValueError(f"arm {arm!r}: one of {ARMS + MTP_ARMS + OVERLAP_ARMS}")
    if arm == "mm":
        from engine.kernels.common import skinny_gemv
        skinny_gemv.CONFIGS.clear()
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_gib * (1 << 30) / total))
    # the fleet boot's first act (fleet.main): the record the preshard wrote, bound before any lane reads it -- the
    # lanes admit their cells against it (b12x's EP zero-weight skip refuses an unbound, GLM-shaped process)
    from engine.base import kernel_shape
    from engine.profiles.qwen38 import facts
    _, shape_source = kernel_shape.bind_recorded(ranks, ranks / "config.json", lambda: facts.load(ranks).kernel_shape())
    began = time.perf_counter()
    F, net, caches, target, draft = build(ranks, ranks, rank, layers, max_seqs=max_seqs, kv_gib=kv_gib,
                                          mtp_precision=arm[4:] if arm.startswith("mtp-") else "bf16",
                                          mtp_experts_dir=MTP_FP8_DIR if arm == "experts-fp8" else None,
                                          shared_overlap={"overlap-one": True, "overlap-all": "all"}.get(arm, False))
    built = time.perf_counter() - began
    graphs = {}
    for n, blocks in shapes:
        shape = (n, F.spec_k + 1, blocks)
        for label, g in (("target", target), ("draft", draft)):
            key = f"{label} rows {n} blocks {blocks}"
            if shape not in g.graphs.graphs:
                graphs[key] = {"error": f"no captured graph {shape} (buckets {g.buckets})"}
                continue
            seat(g, caches, F, shape)
            graphs[key] = replay_profile(g, shape, replays)
            print(json.dumps({"arm": arm, "layers": list(layers), "graph": key, "wall_us": graphs[key]["wall_us"],
                              "device_us": graphs[key]["device_us"], "launches": graphs[key]["launches"]}), flush=True)
    if loop:                                      # the first layer set: where the eager forward's kernels come from
        from engine.profiles.qwen38.decode_graphs import draft_chain
        n, blocks = shapes[0]
        shape = (n, F.spec_k + 1, blocks)
        where = {}
        for label, g, fn in (("target", target, lambda i: net.forward(i, caches, streams=True)),
                             ("draft", draft, lambda i: draft_chain(net, caches, *i, F.spec_k))):
            if shape in g.graphs.inputs:
                seat(g, caches, F, shape)
                inputs = g.graphs.inputs[shape]
                where[label] = stacks(lambda: fn(inputs))
                print(json.dumps({"arm": arm, "stacks": label, "top": where[label][:12]}), flush=True)
                watched = {"gather", "floor_divide", "__floordiv__", "__rfloordiv__", "mm", "linear", "matmul", "cat",
                           "index_select", "arange", "sigmoid", "bitwise_and", "__and__", "sub", "__rsub__", "__sub__"}
                where[label + " calls"] = [c for c in calls(lambda: fn(inputs), top=200) if c["func"] in watched]
                print(json.dumps({"arm": arm, "calls": label, "top": where[label + " calls"][:30]}), flush=True)
        if shape in draft.graphs.inputs:
            # net.mtp_forward `rows`: the rows past the attention alone against every row and then the same rows
            step, given, last, _ = draft.graphs.inputs[shape]
            seat(draft, caches, F, shape)
            full, full_streams = net.mtp_forward(step, given, caches, last_hidden_only=False)
            part, part_streams = net.mtp_forward(step, given, caches, last_hidden_only=False, rows=last)
            a, b = full.index_select(0, last).float(), part.float()
            where["mtp_rows"] = {"hidden_max_err": float((a - b).abs().max() / a.abs().max().clamp_min(1e-30)),
                                 "hidden_equal": bool(torch.equal(a, b)),
                                 "streams_equal": bool(torch.equal(full_streams.index_select(0, last), part_streams))}
            print(json.dumps({"arm": arm, "mtp_rows": where["mtp_rows"]}), flush=True)
        caches.reset()
        served = served_loop(F, net, caches, target, draft)
        one = shapes[0]
        replays = sum(graphs.get(f"{label} rows {one[0]} blocks {one[1]}", {}).get("wall_us", 0.0)
                      for label in ("target", "draft"))
        served["replays_us"] = round(replays, 1)
        served["host_us"] = round(served["wall_us"] - replays, 1)
        print(json.dumps({"arm": arm, "layers": list(layers), "served_loop": served}), flush=True)
    row = {"arm": arm, "counts": counts(F, layers), "kernel_shape": shape_source, "free_GiB_at_start": round(free / 2**30, 1),
           "build_s": round(built, 1), "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "graphs": {k: v for k, v in graphs.items() if "error" not in v},
           "errors": {k: v for k, v in graphs.items() if "error" in v}}
    if loop:
        row["served_loop"] = served
        row["stacks"] = where
    target.close()
    draft.close()
    return row


def run(output=None, ranks=None, *, layer_sets=LAYER_SETS, rank: "int | None" = None, arms=("served",)) -> dict:
    """The lane (probes/engine_kernel_check.py --lanes qwen38_step): each layer set measured in a process of its own --
    a built net's weights stay referenced by the lanes' prepared views, so one process cannot hold two in 4 GiB --
    then assembled. `ranks`: the rank files' directory; `rank`: which one (default: the highest this host has);
    `arms`: ARMS to build each layer set under, in turn (qwen38_step_ab: both, and the first less the second)."""
    import subprocess
    import tempfile
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        present = sorted(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
        if not present:
            raise SystemExit(f"no rank file under {ranks}")
        rank = present[-1]
    report = {"rank": rank, "layer_sets": [list(s) for s in layer_sets], "shapes": [list(s) for s in SHAPES],
              "replays": REPLAYS, "arms": list(arms), "builds": {arm: {} for arm in arms}, "failed": {}}
    with tempfile.TemporaryDirectory() as scratch:
        for layers in layer_sets:
            name = ",".join(map(str, layers))
            for arm in arms:                              # one after the other: production's load lands on both
                out = Path(scratch) / f"{arm}-{name}.json"
                args = ["--loop"] if layers == layer_sets[0] else []
                done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", name, "--arm", arm,
                                       "--ranks", str(ranks), "--rank", str(rank), "--output", str(out), *args],
                                      cwd=str(ROOT))
                if done.returncode or not out.exists():
                    report["failed"][f"{arm} {name}"] = f"rc={done.returncode}"
                    continue
                report["builds"][arm][name] = json.loads(out.read_text())
    from engine.profiles.qwen38 import facts
    F = facts.load(ranks)
    steps = {arm: assemble(report["builds"][arm], F) for arm in arms}
    report["step"] = steps[arms[0]]
    report["served_loop"] = next((b["served_loop"] for b in report["builds"][arms[0]].values() if "served_loop" in b), None)
    if len(arms) == 2:
        report["step_by_arm"] = steps
        report["served_loop_by_arm"] = {arm: next((b["served_loop"] for b in report["builds"][arm].values()
                                                   if "served_loop" in b), None) for arm in arms}
        report["delta"] = difference(*(steps[arm] for arm in arms))
    elif len(arms) > 2:                                   # every later arm less the first (qwen38_step_overlap)
        report["step_by_arm"] = steps
        report["delta_from_first"] = {arm: difference(steps[arm], steps[arms[0]]) for arm in arms[1:]}
    text = json.dumps(report, indent=1)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
    print(json.dumps({arm: {k: {m: v[m]["step_48"] for m in ("wall_us", "device_us", "launches")}
                            for k, v in steps[arm].items()} for arm in arms}, indent=1), flush=True)
    if "delta" in report:
        print(json.dumps({"delta": report["delta"]}, indent=1), flush=True)
    return report


def difference(a: dict, b: dict) -> dict:
    """Two assembled steps -> per graph, a less b: the 48-layer wall, device time and launches, and each family's
    device time where either arm has it."""
    out = {}
    for key in sorted(set(a) & set(b)):
        entry = {m: round(a[key][m]["step_48"] - b[key][m]["step_48"], 1) for m in ("wall_us", "device_us", "launches")}
        fams = set(a[key]["families"]) | set(b[key]["families"])
        entry["families_us"] = {f: round(a[key]["families"].get(f, {}).get("step_48_us", 0.0)
                                         - b[key]["families"].get(f, {}).get("step_48_us", 0.0), 1) for f in sorted(fams)}
        entry["families_us"] = {f: v for f, v in entry["families_us"].items() if abs(v) >= 1.0}
        out[key] = entry
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--one", default=None, help="one layer set (comma separated), measured in this process")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--loop", action="store_true", help="with --one: also decode one request through the served model")
    ap.add_argument("--arm", default="served", choices=ARMS + MTP_ARMS[1:] + OVERLAP_ARMS[1:],
                    help="with --one: the lanes it builds under")
    a = ap.parse_args()
    if a.one is not None:
        row = measure(Path(a.ranks), a.rank, tuple(int(x) for x in a.one.split(",")), loop=a.loop, arm=a.arm)
        Path(a.output).write_text(json.dumps(row) + "\n")
    else:
        run(a.output, a.ranks, rank=a.rank)
