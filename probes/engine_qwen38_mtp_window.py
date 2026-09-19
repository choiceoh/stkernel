"""The MTP head's draft graph at every context bucket, scored against windowed, arms interleaved (probe, single-GPU lane).

`Qwen38Net.mtp_window` (fleet --mtp-window SINK,RECENT; Windowed-MTP, arXiv 2607.21535) has the head attend its first
SINK and last RECENT groups of four positions instead of scoring every group it sees and taking the top 512. What that
removes grows with the context: the index scoring reads every group's key and the selection ranks every group, three
times a draft at K=3 (the observation and the chain's two). The attention itself reads the same 2,048 positions or
fewer either way. This builds the served net for one rank of TP=4 on one GPU from the rank file's weights -- one GDN
layer for the target (the draft graph is the head's alone, whatever the layers) -- captures the draft graphs the fleet
boot captures (decode_graphs.DraftGraphs, K=3, one row) once for every arm from the same net, seats each bucket at its
end, and replays the arms in turns: `ROUNDS` rounds of `REPLAYS` replays, the arms' order reversed every other round,
so a lane beside a serving rank loads every arm alike (q38win-0919a ran the arms one process after another beside the
operator's Qwen window on srv4, and the dense families that no window touches moved by 3 ms between arms). Per arm and
bucket: the median wall, and one profiled round's device time, launches and families.

    scored        the served head (its QSA selection)
    window-511    SINK 1, RECENT 511: the same 2,048 positions a row attends at most, chosen by recency
    window-127    SINK 1, RECENT 127: a quarter of them

    python3 probes/engine_kernel_check.py --lanes qwen38_mtp_window --ranks /home/choiceoh/models/st-qwen38-tep4 \\
        --output /cache/qwen38-mtp-window.json                                    (the queue's single-GPU lane)

Acceptance is not measured here -- a window changes what the head proposes, never what the target keeps; that is a
fleet window's.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = {"scored": None, "window-511": (1, 511), "window-127": (1, 127)}
LAYERS = (4,)                              # one GDN layer: the target is not what is timed
SPEC_K = 3                                 # the operator's K (fleet --spec-k 3)
REPLAYS = 50
ROUNDS = 6
MAX_GIB = 5.0                              # this process's own ceiling: the lane's budget beside production


def build(ranks: Path, rank: int, *, max_seqs: int = 1):
    """The served net over LAYERS and caches for one row at the served ceiling -> (F, net, caches, kv GiB)."""
    import dataclasses
    import torch
    from engine.base.arena import Arena
    from engine.base.params import total_bytes
    from engine.profiles.qwen38 import facts, lanes as lane_tables, mtp_side
    from engine.profiles.qwen38.caches import Qwen38Caches, cache_capacity, layout, snapshot_layout
    from engine.profiles.qwen38.fleet import rank_loader
    from engine.profiles.qwen38.net import Qwen38Net
    from probes.engine_qwen38_step import OneRankComm, ZeroPLETable

    F = dataclasses.replace(facts.load(ranks), spec_k=SPEC_K)
    net = Qwen38Net(F, OneRankComm(rank), lane_tables.served(), layers=list(LAYERS), mtp=True, mtp_experts="bf16")
    specs = net.specs()
    need = -(-(F.max_position + 2 * SPEC_K) // F.block) * max_seqs + 2       # every bucket's row, and the null block
    kv_gib, nb = 0.25, 0
    while nb < need:
        kv_gib *= 1.25
        nb, snapshots = cache_capacity(F, net.layers, kv_gib, max_seqs, 0.05, mtp=True)
    snapshots = min(snapshots, 9)
    arena = Arena(total_bytes(specs) + 256 * (len(specs) + 64) + layout(F, net.layers, mtp=True).nbytes(nb, max_seqs)
                  + snapshots * snapshot_layout(F, net.layers)[0])
    loader = rank_loader(ranks / f"rank{rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout)
    side = {s.name for s in net.side_specs()}
    views = loader.load([s.name for s in specs if s.name not in side], arena=arena)
    if side:
        views.update(rank_loader(mtp_side.path(mtp_side.DIRS["bf16"], rank, "bf16"),
                                 expected_layout=mtp_side.LAYOUTS["bf16"]).load(sorted(side), arena=arena))
    net.bind(views)
    if net._ple is not None:
        net.attach_ple(ZeroPLETable(F.ple_rows_per_rank, F.ple_head_dim, float(net._ple_scale)),
                       max_rows=max_seqs * (F.spec_k + 1))
    net.prepare_dense(None)
    caches = Qwen38Caches(arena, F, net.layers, nb, max_seqs, snapshots, mtp=True)
    torch.cuda.synchronize()
    return F, net, caches, round(kv_gib, 3)


def timed(graph, replays: int) -> float:
    """Microseconds a replay, over `replays` back to back."""
    import torch
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / replays


def measure(ranks: Path, rank: int, arms=tuple(ARMS), *, rounds: int = ROUNDS, replays: int = REPLAYS) -> dict:
    """Every arm's draft graphs from one net in this process, each bucket's arms replayed in turns."""
    import torch
    from engine.base import kernel_shape
    from engine.profiles.qwen38 import facts
    from engine.profiles.qwen38.decode_graphs import DraftGraphs
    from probes.engine_qwen38_step import replay_profile, seat
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1.0, MAX_GIB * (1 << 30) / total))
    kernel_shape.bind_recorded(ranks, ranks / "config.json", lambda: facts.load(ranks).kernel_shape())
    began = time.perf_counter()
    F, net, caches, kv_gib = build(ranks, rank)
    graphs = {}
    for arm in arms:                                  # the window is read while a graph is captured, never after
        net.mtp_window = ARMS[arm]
        graphs[arm] = DraftGraphs(net, caches, 1, SPEC_K + 1, k=SPEC_K, ceiling=F.max_position)
    net.mtp_window = None
    built = time.perf_counter() - began
    buckets = {}
    for blocks in graphs[arms[0]].buckets:
        shape = (1, SPEC_K + 1, blocks)
        walls = {arm: [] for arm in arms}
        for turn in range(rounds):
            for arm in (arms if turn % 2 == 0 else tuple(reversed(arms))):
                seat(graphs[arm], caches, F, shape)
                walls[arm].append(timed(graphs[arm].graphs.graphs[shape], replays))
        context = blocks * F.block - (SPEC_K + 1) - SPEC_K
        row = {"context": context}
        for arm in arms:
            seat(graphs[arm], caches, F, shape)
            profiled = replay_profile(graphs[arm], shape, replays)
            row[arm] = {"wall_us": round(statistics.median(walls[arm]), 1),
                        "wall_spread_us": round(max(walls[arm]) - min(walls[arm]), 1),
                        "device_us": profiled["device_us"], "launches": profiled["launches"],
                        "qsa_us": profiled["families"].get("qsa", {}).get("us", 0.0),
                        "families": profiled["families"]}
        buckets[str(blocks)] = row
        print(json.dumps({"blocks": blocks, "context": context,
                          **{arm: [row[arm]["wall_us"], row[arm]["device_us"], row[arm]["launches"], row[arm]["qsa_us"]]
                             for arm in arms}}), flush=True)
    for g in graphs.values():
        g.close()
    return {"rank": rank, "layers": list(LAYERS), "spec_k": SPEC_K, "rounds": rounds, "replays": replays,
            "arms": {arm: ARMS[arm] for arm in arms}, "kv_gib": kv_gib, "build_s": round(built, 1),
            "free_GiB_at_start": round(free / 2**30, 1), "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "buckets": buckets}


def saving(report: dict) -> dict:
    """Per bucket, the scored head less each window: median wall, device time, launches, the QSA family."""
    out = {}
    for blocks, row in report["buckets"].items():
        scored = row.get("scored")
        if scored is None:
            continue
        out[blocks] = {"context": row["context"],
                       **{arm: {m: round(scored[m] - row[arm][m], 1) for m in ("wall_us", "device_us", "launches", "qsa_us")}
                          for arm in report["arms"] if arm != "scored"}}
    return out


def run(output=None, ranks=None, *, rank: "int | None" = None, arms=tuple(ARMS)) -> dict:
    """The lane: every arm in this one process, then each window less the scored head per bucket."""
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        present = sorted(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
        if not present:
            raise SystemExit(f"no rank file under {ranks}")
        rank = present[-1]
    report = measure(ranks, rank, arms)
    report["saving_us"] = saving(report)
    print(json.dumps({"saving_us": report["saving_us"]}, indent=1), flush=True)
    text = json.dumps(report, indent=1)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--rank", type=int, default=None)
    a = ap.parse_args()
    run(a.output, a.ranks, rank=a.rank)
