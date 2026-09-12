"""What a captured decode step's kernels are, on real weights (45차 §90).

A decode step replays a CUDA graph, and the inside of one cannot be timed from python: `Event.record` is not
capturable, so a mark placed inside `net.forward` times the CAPTURE and never a replay -- which is why the
stage clock can only wrap a replay whole (base/stage_clock), and why `forward` is 63.5% of a step with no
breakdown behind it (45차 §81). CUPTI does see through a replay. This captures the step the fleet serves and
profiles its replays.

One rank, real weights: the collectives are the identity here, so the shares are this rank's arithmetic and
not the fabric's. The layer range is the whole model by default -- a slice measures a slice.

    bash probes/run_engine_probe.sh probes/engine_graph_profile.py --ranks DIR
    bash probes/run_engine_probe.sh probes/engine_graph_profile.py --ranks DIR --layers 0-4 --steps 16

Read the per-step column. `calls` is over the whole run, so `calls / steps` says how many launches a step
spends on that kernel -- which is the other half of the question: bytes or launches.
"""
import argparse
import re

import torch

from engine.base.instruments import Recorder
from engine.profiles.glm53 import facts
from engine.profiles.glm53.boot import build
from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.net import Step

# A kernel's name says which lane it came from; anything unclaimed is listed under its own name so a
# surprise cannot hide inside a bucket.
LANES = (("mHC", r"mhc"), ("MLA / DSA", r"mla|sparse|logits|kpool|indexer"), ("KDA", r"kda|conv"),
         ("MoE", r"moe|b12x|expert"), ("dense GEMM", r"gemm|cutlass|nvjet|sm90|sm100|sm121"),
         ("norm / elementwise", r"norm|elementwise|vectorized|copy|cat|fill"),
         ("collective", r"nccl|all_reduce|allgather|reduce_scatter"))


def lane_of(name: str) -> str:
    low = name.lower()
    for lane, pattern in LANES:
        if re.search(pattern, low):
            return lane
    return "other"


class IsolatedRank:
    """Rank 0 alone: every collective is the identity, so what is measured is this rank's arithmetic."""
    rank = 0
    world_size = 4

    def all_reduce(self, x):
        return x

    def all_reduce_max(self, x):
        return x

    def all_gather(self, x, dim=-1):
        return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--ckpt-meta", default=str(facts.CKPT))
    ap.add_argument("--layers", default="", help="a slice like 0-4; the whole model by default")
    ap.add_argument("--seqs", type=int, default=1, help="rows in the step: production serves one")
    ap.add_argument("--steps", type=int, default=16)
    a = ap.parse_args()

    F = facts.load(a.ckpt_meta)
    if a.layers:
        first, last = (int(v) for v in a.layers.split("-"))
        layers = list(range(first, last + 1))
    else:
        layers = list(range(F.layers))
    torch.manual_seed(13)
    _, net, caches, _, _ = build(IsolatedRank(), layers, served(), a.ranks, .25, max(2, a.seqs), False,
                                 Recorder("profile"), ckpt_meta=a.ckpt_meta)
    t = F.spec_k + 1
    print(f"weights loaded: {len(layers)} layers, {a.seqs} row(s) of {t} tokens", flush=True)

    graphs = Glm53DecodeGraphs(net, caches, max(2, a.seqs), t)
    slots = [caches.slots.take(i) for i in range(a.seqs)]
    for i in range(a.seqs):
        caches.pool.reserve(i, 4352)
    ids = [torch.randint(0, 30000, (t,), device="cuda") for _ in range(a.seqs)]
    step = Step.decode([(ids[i], 2048, i, slots[i]) for i in range(a.seqs)])
    caches.prepare(step)
    for _ in range(4):
        graphs.run(step)
    torch.cuda.synchronize()
    print("captured and warm", flush=True)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(a.steps):
            graphs.run(step)
        torch.cuda.synchronize()

    rows = []
    for event in prof.key_averages():
        total = getattr(event, "self_device_time_total", 0.0) or 0.0
        if total > 0:
            rows.append((event.key, event.count, total))
    rows.sort(key=lambda r: -r[2])
    device = sum(r[2] for r in rows)
    print(f"\n  a decode step's device time: {device / a.steps:.1f} us over {a.steps} replays\n")

    lanes = {}
    for name, calls, total in rows:
        lane = lanes.setdefault(lane_of(name), [0.0, 0])
        lane[0] += total
        lane[1] += calls
    print(f"  {'lane':22s}{'us/step':>10s}{'share':>8s}{'launches/step':>15s}")
    for lane, (total, calls) in sorted(lanes.items(), key=lambda kv: -kv[1][0]):
        print(f"  {lane:22s}{total / a.steps:10.1f}{total / device * 100:7.1f}%{calls / a.steps:15.1f}")

    print(f"\n  {'kernel':64s}{'us/step':>10s}{'launches/step':>15s}")
    for name, calls, total in rows[:40]:
        print(f"  {name[:64]:64s}{total / a.steps:10.1f}{calls / a.steps:15.1f}")


if __name__ == "__main__":
    main()
