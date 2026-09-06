#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The shared expert's two MK GEMMs UNDER the routed MoE kernel -- the way
the decode step runs them (28차 trace reading: the 30-45 us class, 84-86
launches/step across the 42 MoE layers, 3.5 ms, 2.4 ms of it exposed).

Serving forks the shared expert (gate_up [1024 x 4096], down [4096 x 512])
onto an aux stream beside the routed b12x MoE call of the same layer, so a
persistent 48-block MK launch competes with the MoE kernel for SMs: its
blocks trickle in as MoE blocks retire, and on the global-quant path the
grid barrier then holds every resident block for the last one to arrive;
launched first, its 48 blocks hold every SM until they are done and the MoE
waits. The standalone bench cannot show that. This probe captures ONE graph
with the fork -- stream A: the served b12x wrapper at U=40 (the boot's
unique-expert count, DRAM-cold: 142 MB a call); stream B: the pair -- in
both issue orders, and times the union against each part alone:

  exposed = span(moe || pair) - span(moe)

Row: the v2 non-persistent lane (one block per unit, no barrier -- the
served lane since 32차 and, since 34차 §8, the GEMM segment's only kernel:
the persistent v1 kernel, the barrier-free local-quant kernel and the
fewer-blocks control row this probe once compared are gone). What differs
from the step, and why the x42 line is a projection, not a
step number: the step's main stream runs the router GEMM + topk before the
MoE kernel (the pair's first GEMM overlaps those, not the MoE), the
activation between the two GEMMs is a real kernel here replaced by a copy,
and one layer replayed back to back stands in for 42 layers each followed
by a join and attention.

    bash probes/run_mk_probe.sh probes/mk_gemm_concurrent_probe.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
os.environ.setdefault("VLLM_GLM53_MK_MHC", "0")
os.environ.setdefault("VLLM_GLM53_MK_MLA", "0")
os.environ.setdefault("VLLM_GLM53_MK_PDL", "1")
sys.path.insert(0, os.environ.get("MK_PKG_PATH",
                                  "/usr/local/lib/python3.12/dist-packages"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

# the served b12x fixture -- ONE definition, moe_decode_stream_probe's
from moe_decode_stream_probe import (  # noqa: E402
    DEV, E, HID, T, _routing, expert_set, served_wrapper)

U = 40                                    # unique experts a layer (27차)
MOE_LAYERS = 42                           # layers 3..44 carry a shared expert
SHARED_N = 1024                           # shared expert gate_up rows / rank
SHARED_K = 512                            # down-proj K / rank
REPS = 20
NREPLAY = 5                               # replays per event bracket
ROWS = (("v2 non-persistent",),)


def _capture(fn_a, fn_b, order: str):
    """One graph: fork, A on stream a, B on stream b, join. `order` is the
    issue order inside the capture (A first or B first)."""
    a = torch.cuda.Stream()
    b = torch.cuda.Stream()
    # warm both paths outside the capture (allocations, the MK arm, PDL attrs)
    for _ in range(2):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=a):
        fork = torch.cuda.Event()
        fork.record(a)
        b.wait_event(fork)
        if order == "moe-first":
            fn_a()
            with torch.cuda.stream(b):
                fn_b()
        else:
            with torch.cuda.stream(b):
                fn_b()
            fn_a()
        join = torch.cuda.Event()
        join.record(b)
        a.wait_event(join)
    torch.cuda.synchronize()
    return g


def _time(g, reps: int = REPS) -> float:
    """us per replay, median over reps of NREPLAY back-to-back replays: the
    graph-launch gap (which grows with the node count) amortises instead of
    sitting whole inside a bracket around one replay on an idle stream."""
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        g.replay()  # the stream is busy when the start event lands
        s.record()
        for _ in range(NREPLAY):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / NREPLAY)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    from vllm.model_executor.layers import glm53_megakernel as mk

    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()} T={T} U={U} layers={MOE_LAYERS}")
    wrapper = served_wrapper()
    w13, sf13, w2, sf2 = expert_set(torch.Generator().manual_seed(1))
    ones = torch.ones(E, dtype=torch.float32, device=DEV)
    ids, w = _routing(U)
    x = torch.randn(T, HID, dtype=torch.bfloat16, device=DEV) * 0.5
    out = torch.empty(T, HID, dtype=torch.bfloat16, device=DEV)

    def moe():
        wrapper.run(x, w13, sf13, w2, sf2, ids, w, w1_alpha=ones,
                    w2_alpha=ones, fc2_input_scale=ones, out=out)

    mk.maybe_arm()
    assert mk._ARMED["gemm"], "MK-GEMM did not arm"
    ext = mk._EXT
    # the shared expert's pair: gate_up [1024 x 4096] on x, then down
    # [4096 x 512] on the (untouched: the activation is not the point)
    # first 512 columns of that output. bg=True: the serving call site
    # marks these launches background (v2's lr_slot keys on it).
    p_up = mk.build_mk_weight_w4(
        torch.randn(SHARED_N, HID, dtype=torch.bfloat16, device=DEV) * 0.05)
    p_dn = mk.build_mk_weight_w4(
        torch.randn(HID, SHARED_K, dtype=torch.bfloat16, device=DEV) * 0.05)
    xs = torch.randn(T, HID, dtype=torch.bfloat16, device=DEV)
    h = torch.empty(T, SHARED_K, dtype=torch.bfloat16, device=DEV)

    def pair():
        up = mk._gemm_call(xs, p_up, SHARED_N, bg=True)
        h.copy_(up[:, :SHARED_K])
        mk._gemm_call(h, p_dn, HID, bg=True)

    def nothing():
        pass

    print(f"{'row':<44}{'us':>9}{'exposed_us':>12}")
    t_moe = _time(_capture(moe, nothing, "moe-first"))
    print(f"{'moe alone (U=40)':<44}{t_moe:>9.1f}")
    rows = {}
    with mk._gemm_probe_scope():  # the split override restored after
        for (label,) in ROWS:
            plan_up = ext.gemm2_plan(T, SHARED_N, HID)
            plan_dn = ext.gemm2_plan(T, HID, SHARED_K)
            print(f"  plans (ksr, units, blocks/SM): up={plan_up} down={plan_dn}")
            t_pair = _time(_capture(nothing, pair, "moe-first"))
            print(f"{'pair alone ' + label:<44}{t_pair:>9.1f}")
            for order in ("moe-first", "pair-first"):
                t = _time(_capture(moe, pair, order))
                rows[(label, order)] = t - t_moe
                print(f"{'moe || pair ' + label + ' ' + order:<44}{t:>9.1f}"
                      f"{t - t_moe:>12.1f}")
    # the projection onto the step: the exposed pair times the MoE layers
    for (label,) in ROWS:
        a = rows[(label, "moe-first")]
        b = rows[(label, "pair-first")]
        print(f"{label}: exposed per layer moe-first {a:.1f} / pair-first {b:.1f} us"
              f" -> x{MOE_LAYERS} = {MOE_LAYERS * a / 1e3:.2f} / "
              f"{MOE_LAYERS * b / 1e3:.2f} ms/step (projection, see the docstring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
