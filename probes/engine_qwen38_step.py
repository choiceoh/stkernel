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
    [7]            fixed + 1 QSA                [2]      fixed + 1 GDN + PLE

    step(48 layers) = fixed + 36 GDN + 12 QSA + PLE        (the MTP head's draft graph is its own, whatever the layers)

    python3 probes/engine_kernel_check.py --lanes qwen38_step --ranks /home/choiceoh/models/st-qwen38-tep4 \\
        --output /cache/qwen38-step.json                                          (the queue's single-GPU lane)
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

LAYER_SETS = ((4, 5, 6, 7), (4, 5), (7,), (2,))
SHAPES = ((1, 6), (1, 43), (4, 43))       # (rows, bucket blocks): C=1 near 4K and 32K context, C=4 near 32K
REPLAYS = 50
KV_GIB = 0.25
MAX_GIB = 4.0                             # this process's own device-memory ceiling: the lane's budget beside production
FULL = {"fixed": 1, "gdn": 36, "qsa": 12, "ple": 1}

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


def build(meta: Path, ranks: Path, rank: int, layers, *, max_seqs: int, kv_gib: float):
    """The served net, caches and captured graphs for one rank over `layers` -> (F, net, caches, target, draft)."""
    import torch
    from engine.base.arena import Arena
    from engine.base.params import total_bytes
    from engine.profiles.qwen38 import facts, lanes as lane_tables
    from engine.profiles.qwen38.caches import Qwen38Caches, cache_capacity, layout, snapshot_layout
    from engine.profiles.qwen38.decode_graphs import DraftGraphs, TargetGraphs
    from engine.profiles.qwen38.fleet import rank_loader
    from engine.profiles.qwen38.net import Qwen38Net

    F = facts.load(meta)
    net = Qwen38Net(F, OneRankComm(rank), lane_tables.served(), layers=list(layers), mtp=True)
    specs = net.specs()
    nb, snapshots = cache_capacity(F, net.layers, kv_gib, max_seqs, 0.05, mtp=True)
    arena = Arena(total_bytes(specs) + 256 * (len(specs) + 64) + layout(F, net.layers, mtp=True).nbytes(nb, max_seqs)
                  + snapshots * snapshot_layout(F, net.layers)[0])
    loader = rank_loader(ranks / f"rank{rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout)
    net.bind(loader.load([s.name for s in specs], arena=arena))
    if net._ple is not None:
        net.attach_ple(ZeroPLETable(F.ple_rows_per_rank, F.ple_head_dim, float(net._ple_scale)),
                       max_rows=max_seqs * (F.spec_k + 1))
    net.prepare_dense(None)
    caches = Qwen38Caches(arena, F, net.layers, nb, max_seqs, snapshots, mtp=True)
    tokens = F.spec_k + 1
    target = TargetGraphs(net, caches, max_seqs, tokens, ceiling=F.max_position)
    draft = DraftGraphs(net, caches, max_seqs, tokens, ceiling=F.max_position)
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
    graphs.metadata[shape][0].fill_(blocks * F.block - t)
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


def run(output=None, ranks=None, *, layer_sets=LAYER_SETS, shapes=SHAPES, replays: int = REPLAYS,
        kv_gib: float = KV_GIB, max_gib: float = MAX_GIB, max_seqs: int = 4, rank: "int | None" = None) -> dict:
    """The lane (probes/engine_kernel_check.py --lanes qwen38_step): every layer set built, captured and replayed in
    turn, then assembled. `ranks`: the rank files' directory; `rank`: which one (default: the highest this host has)."""
    import torch
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        present = sorted(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
        if not present:
            raise SystemExit(f"no rank file under {ranks}")
        rank = present[-1]
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_gib * (1 << 30) / total))
    report = {"rank": rank, "replays": replays, "device": torch.cuda.get_device_name(), "free_GiB_at_start":
              round(free / 2**30, 1), "layer_sets": [list(s) for s in layer_sets], "shapes": [list(s) for s in shapes],
              "builds": {}}
    F = None
    for layers in layer_sets:
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        F, net, caches, target, draft = build(ranks, ranks, rank, layers, max_seqs=max_seqs, kv_gib=kv_gib)
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
                print(json.dumps({"layers": list(layers), "graph": key, "wall_us": graphs[key]["wall_us"],
                                  "device_us": graphs[key]["device_us"], "launches": graphs[key]["launches"]}),
                      flush=True)
        report["builds"][",".join(map(str, layers))] = {"counts": counts(F, layers), "build_s": round(built, 1),
                                                        "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                                        "graphs": {k: v for k, v in graphs.items() if "error" not in v},
                                                        "errors": {k: v for k, v in graphs.items() if "error" in v}}
        target.close()
        draft.close()
        del net, caches, target, draft
        torch.cuda.empty_cache()
    report["step"] = assemble(report["builds"], F)
    text = json.dumps(report, indent=1)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
    print(json.dumps({k: {m: v[m]["step_48"] for m in ("wall_us", "device_us", "launches")}
                      for k, v in report["step"].items()}, indent=1), flush=True)
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", default=None)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    run(a.output, a.ranks)
