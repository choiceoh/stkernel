"""The MTP head's draft graph at every context bucket, scored against windowed (probe, single-GPU lane).

`Qwen38Net.mtp_window` (fleet --mtp-window SINK,RECENT; Windowed-MTP, arXiv 2607.21535) has the head attend its first
SINK and last RECENT groups of four positions instead of scoring every group it sees and taking the top 512. What that
removes grows with the context: the index scoring reads every group's key and the selection ranks every group, three
times a draft at K=3 (the observation and the chain's two). The attention itself reads the same 2,048 positions or
fewer either way. This builds the served net for one rank of TP=4 on one GPU from the rank file's weights -- one GDN
layer for the target (the draft graph is the head's alone, whatever the layers) -- captures the draft graphs the fleet
boot captures (decode_graphs.DraftGraphs, K=3, one row) at every bucket up to the served ceiling, seats each at the end
of its bucket, and replays it: wall, device time, launches and the QSA family's device time, per arm and bucket.

Each arm is its own process (a built net's weights stay referenced by the prepared lanes):

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
MAX_GIB = 5.0                              # this process's own ceiling: the lane's budget beside production


def build(ranks: Path, rank: int, window, *, max_seqs: int = 1):
    """The served net over LAYERS with the head windowed or not, caches for one row at the served ceiling, and the
    draft graphs -> (F, net, caches, draft)."""
    import dataclasses
    import torch
    from engine.base.arena import Arena
    from engine.base.params import total_bytes
    from engine.profiles.qwen38 import facts, lanes as lane_tables, mtp_side
    from engine.profiles.qwen38.caches import Qwen38Caches, cache_capacity, layout, snapshot_layout
    from engine.profiles.qwen38.decode_graphs import DraftGraphs
    from engine.profiles.qwen38.fleet import rank_loader
    from engine.profiles.qwen38.net import Qwen38Net
    from probes.engine_qwen38_step import OneRankComm, ZeroPLETable

    F = dataclasses.replace(facts.load(ranks), spec_k=SPEC_K)
    net = Qwen38Net(F, OneRankComm(rank), lane_tables.served(), layers=list(LAYERS), mtp=True, mtp_experts="bf16")
    net.mtp_window = window
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
    draft = DraftGraphs(net, caches, max_seqs, SPEC_K + 1, k=SPEC_K, ceiling=F.max_position)
    torch.cuda.synchronize()
    return F, net, caches, draft, round(kv_gib, 3)


def measure(ranks: Path, rank: int, arm: str) -> dict:
    """One arm in this process: every bucket's one-row draft graph, seated at the bucket's end and replayed."""
    import torch
    from engine.base import kernel_shape
    from engine.profiles.qwen38 import facts
    from probes.engine_qwen38_step import replay_profile, seat
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1.0, MAX_GIB * (1 << 30) / total))
    kernel_shape.bind_recorded(ranks, ranks / "config.json", lambda: facts.load(ranks).kernel_shape())
    began = time.perf_counter()
    F, net, caches, draft, kv_gib = build(ranks, rank, ARMS[arm])
    built = time.perf_counter() - began
    buckets = {}
    for blocks in draft.buckets:
        shape = (1, SPEC_K + 1, blocks)
        seat(draft, caches, F, shape)
        profiled = replay_profile(draft, shape, REPLAYS)
        context = blocks * F.block - (SPEC_K + 1) - SPEC_K
        buckets[str(blocks)] = {"context": context, "wall_us": profiled["wall_us"], "device_us": profiled["device_us"],
                                "launches": profiled["launches"],
                                "qsa_us": profiled["families"].get("qsa", {}).get("us", 0.0),
                                "families": profiled["families"]}
        print(json.dumps({"arm": arm, "blocks": blocks, "context": context, "wall_us": profiled["wall_us"],
                          "qsa_us": buckets[str(blocks)]["qsa_us"]}), flush=True)
    row = {"arm": arm, "window": ARMS[arm], "rank": rank, "kv_gib": kv_gib, "build_s": round(built, 1),
           "free_GiB_at_start": round(free / 2**30, 1), "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "buckets": buckets}
    draft.close()
    return row


def run(output=None, ranks=None, *, rank: "int | None" = None, arms=tuple(ARMS)) -> dict:
    """The lane: each arm measured in a process of its own, then every windowed arm less the scored one per bucket."""
    import subprocess
    import tempfile
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        present = sorted(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
        if not present:
            raise SystemExit(f"no rank file under {ranks}")
        rank = present[-1]
    report = {"rank": rank, "layers": list(LAYERS), "spec_k": SPEC_K, "replays": REPLAYS, "arms": {}, "failed": {}}
    with tempfile.TemporaryDirectory() as scratch:
        for arm in arms:
            out = Path(scratch) / f"{arm}.json"
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", arm, "--ranks", str(ranks),
                                   "--rank", str(rank), "--output", str(out)], cwd=str(ROOT))
            if done.returncode or not out.exists():
                report["failed"][arm] = f"rc={done.returncode}"
                continue
            report["arms"][arm] = json.loads(out.read_text())
    scored = report["arms"].get("scored")
    if scored is not None:
        report["saving_us"] = {
            arm: {blocks: {"context": b["context"],
                           "wall_us": round(scored["buckets"][blocks]["wall_us"] - b["wall_us"], 1),
                           "qsa_us": round(scored["buckets"][blocks]["qsa_us"] - b["qsa_us"], 1)}
                  for blocks, b in row["buckets"].items() if blocks in scored["buckets"]}
            for arm, row in report["arms"].items() if arm != "scored"}
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
    ap.add_argument("--arm", default=None, choices=tuple(ARMS), help="one arm, measured in this process")
    a = ap.parse_args()
    if a.arm is not None:
        row = measure(Path(a.ranks), a.rank, a.arm)
        Path(a.output).write_text(json.dumps(row) + "\n")
    else:
        run(a.output, a.ranks, rank=a.rank)
