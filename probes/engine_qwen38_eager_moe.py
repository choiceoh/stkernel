"""Qwen3.8's eager MoE at decode sizes on one GB10: after the boot's warm pass, no request compiles (probe, single-GPU lane).

An uncaptured step -- the MTP head observing a parked row's positions, a step no graph admits -- dispatches only this
rank's (token, route) pairs, one route a pair (lanes.moe, compact). Up to eight pairs run the b12x micro kernel, whose
artifact is keyed by the pair count m AND the capacity r of the workspace the dispatcher has grown so far
(`micro_m{m}_…_t1_r{r}`). The boot's prefill passes never reach that path, so on 2026-09-19 the K=3 window's first
requests compiled six of them mid-request -- m 1-4 at r 2 and r 4, the capacities set by the order the counts arrived
in -- and the first decode step took 6.8 s (measurements/qwen38_serve_window_20260919). engine/profiles/qwen38/warmup
.eager_moe now grows the workspace to its ceiling first and runs every count below it.

This builds the served net for ONE rank from the rank file's own weights (one layer, no graphs: the eager path is the
only one exercised), wraps the dispatcher's micro getter to see every (m, top-k, capacity) it is asked for and whether
it added a kernel, and then:

    warm        warmup.eager_moe(net): what the boot runs -- the kernels it adds, each at one capacity
    requests    compact launches in an order a boot would not choose (3 7 1 5 8 2 6 4 pairs), then the router's own
                routes for 1..4 random rows, twice -- NO kernel may be added here
    capacity    each count again from an emptied workspace cache, so its capacity is its own count (the pre-warm
                situation), against the ceiling's output for the same inputs: whether the bytes a request gets depend
                on the capacity -- i.e. on the order earlier requests came in. This part compiles its kernels.

    python3 probes/engine_kernel_check.py --lanes qwen38_eager_moe --ranks /home/choiceoh/models/st-qwen38-tep4 \\
        --output /cache/qwen38-eager-moe.json                                     (the queue's single-GPU lane)

Correctness and compile counts only; no timing is a claim.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAYER = 4                                  # a GDN layer's MoE: every layer's experts have the same shapes
REQUEST_ORDER = (3, 7, 1, 5, 8, 2, 6, 4)
ROUTED_ROWS = (1, 2, 3, 4)


class Asked:
    """The dispatcher's micro getter, wrapped: every call's (m, top-k, capacity) and whether it added a kernel."""

    def __init__(self, md):
        self.md, self.calls, self.phase = md, [], "setup"
        getter = md._get_micro_kernel

        def wrapped(state_E, weight_E, m, k, n, num_topk, max_rows, *args, **kwargs):
            before = len(md._MICRO_KERNEL_CACHE)
            began = time.perf_counter()
            try:
                return getter(state_E, weight_E, m, k, n, num_topk, max_rows, *args, **kwargs)
            finally:
                self.calls.append(dict(phase=self.phase, m=m, topk=num_topk, capacity=max_rows,
                                       added=len(md._MICRO_KERNEL_CACHE) > before,
                                       seconds=round(time.perf_counter() - began, 3)))
        md._get_micro_kernel = wrapped

    def added(self, phase):
        return sorted({(c["m"], c["topk"], c["capacity"]) for c in self.calls if c["phase"] == phase and c["added"]})

    def asked(self, phase):
        return sorted({(c["m"], c["topk"], c["capacity"]) for c in self.calls if c["phase"] == phase})


def summarize(asked: Asked) -> dict:
    """The record's verdicts, from the getter's log alone (a unit test holds this without a GPU)."""
    warm, requests = asked.asked("warm"), asked.asked("requests")
    return {"warm_added": asked.added("warm"), "warm_asked": warm, "requests_asked": requests,
            "requests_added": asked.added("requests"),
            "one_capacity": sorted({cap for _, topk, cap in warm + requests if topk == 1}),
            "requests_within_warm": set(requests) <= set(warm)}


def run(output=None, ranks=None, *, rank: "int | None" = None, layer: int = LAYER) -> dict:
    import torch
    from engine.base import kernel_shape
    from engine.kernels.b12x import moe_dispatch as md
    from engine.profiles.qwen38 import facts
    from engine.profiles.qwen38.warmup import EAGER_PAIRS, eager_moe, eager_routes
    from probes.engine_qwen38_prefill import build

    assert torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        rank = sorted(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))[-1]
    _, shape_source = kernel_shape.bind_recorded(ranks, ranks / "config.json", lambda: facts.load(ranks).kernel_shape())
    asked = Asked(md)
    F, net, _caches = build(ranks, rank, (layer,), tokens=256)
    experts = net._experts[f"L{layer}."]
    w13 = net.p[f"L{layer}.moe.w13"]
    local, dev = w13.shape[0], w13.device
    gen = torch.Generator(device="cpu").manual_seed(0)

    def launch(m, x=None):
        ids, weights = eager_routes(m, first_expert=net.first_expert, local=local, experts=F.experts, topk=F.topk_experts)
        if x is None:
            x = (torch.randn(m, F.hidden, generator=gen) * 0.5).to(torch.bfloat16)
        out = experts(x.to(dev), ids.to(dev), weights.to(dev), compact=True)
        torch.cuda.synchronize()
        return x, out

    asked.phase = "warm"
    began = time.perf_counter()
    paid = eager_moe(net)
    warm_s = round(time.perf_counter() - began, 2)

    asked.phase = "requests"
    kept = {}
    for m in REQUEST_ORDER:
        kept[m] = launch(m)
    for _ in range(2):                                      # the router's own routes, as a parked row's head takes them
        for n in ROUTED_ROWS:
            x = (torch.randn(n, F.hidden, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
            scores = torch.mm(x, net.p[f"L{layer}.moe.gates"].t())
            ids, weights = net.lanes.route(scores[:, :F.experts], F.topk_experts)
            experts(x, ids, weights, compact=True)
    torch.cuda.synchronize()
    record = summarize(asked)

    asked.phase = "capacity"
    capacity = {}
    for m in range(1, EAGER_PAIRS + 1):
        md._WORKSPACE_CACHE.clear()                        # the pre-warm situation: this count's own capacity
        x, at_ceiling = kept[m]
        _, own = launch(m, x)
        capacity[m] = {"bytes_equal": bool(torch.equal(own.view(torch.int16), at_ceiling.view(torch.int16))),
                       "largest": round(float((own.float() - at_ceiling.float()).abs().max()), 6)}

    record.update(lane="qwen38_eager_moe", rank=rank, layer=layer, kernel_shape=shape_source, warm_s=warm_s, warm_paid=paid,
                  capacity=capacity, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  calls=asked.calls)
    text = json.dumps(record, default=list)
    print(json.dumps({k: v for k, v in record.items() if k != "calls"}, default=list), flush=True)
    if output:
        Path(output).write_text(text + "\n")
    if record["requests_added"] or not record["requests_within_warm"] or len(record["one_capacity"]) != 1:
        raise RuntimeError(f"after the warm pass a request still compiled: {record['requests_added']}, "
                           f"capacities {record['one_capacity']}")
    return record
