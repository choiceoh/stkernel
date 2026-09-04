#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The shared expert's two MK GEMMs UNDER the routed MoE kernel -- the way
the decode step runs them (28차: 86 launches/step in the 30-45 us class,
3.5 ms, 2.4 ms of it exposed on the critical path).

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

  exposed = span(moe || pair) - span(moe)      what the step actually pays

for the global path (VLLM_GLM53_MK_LOCALQ=0) and the local-quant kernel
(2) on its launched grid: `units` blocks (32 here), then 16 and 8
(VLLM_GLM53_MK_LQ_GRID) -- fewer blocks leave more SMs to the MoE.

    bash probes/run_mk_probe.sh probes/mk_gemm_concurrent_probe.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
os.environ.setdefault("VLLM_GLM53_MK_MHC", "0")
os.environ.setdefault("VLLM_GLM53_MK_KDA", "0")
os.environ.setdefault("VLLM_GLM53_MK_MLA", "0")
os.environ.setdefault("VLLM_GLM53_MK_PDL", "1")
sys.path.insert(0, os.environ.get("MK_PKG_PATH",
                                  "/usr/local/lib/python3.12/dist-packages"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

# the served b12x fixture -- ONE definition, moe_decode_stream_probe's
from moe_decode_stream_probe import (  # noqa: E402
    DEV, E, HID, INTER, T, TOPK, _routing, _weight_set)

U = 40                                    # unique experts a layer (27차)
SHARED_N = 1024                           # shared expert gate_up rows / rank
SHARED_K = 512                            # down-proj K / rank
REPS = 20
NREPLAY = 5                               # replays per event bracket


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
    from flashinfer.fused_moe import B12xMoEWrapper
    from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout
    from vllm.model_executor.layers import glm53_megakernel as mk

    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()} T={T} topk={TOPK} U={U}")
    wrapper = B12xMoEWrapper(
        num_experts=E, top_k=TOPK, hidden_size=HID, intermediate_size=INTER,
        use_cuda_graph=True, max_num_tokens=64, num_local_experts=E,
        activation="swigluoai_uninterleave", swiglu_alpha=1.0,
        swiglu_beta=0.0, swiglu_limit=10.0)
    gen = torch.Generator().manual_seed(1)
    w13, w2, s13, s2 = _weight_set(gen)
    sf13 = flashinfer_convert_sf_to_mma_layout(
        s13.reshape(E * 2 * INTER, HID // 16), m=2 * INTER, k=HID, num_groups=E)
    sf2 = flashinfer_convert_sf_to_mma_layout(
        s2.reshape(E * HID, INTER // 16), m=HID, k=INTER, num_groups=E)
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
    # marks these launches background, which is what LOCALQ=1 keys on.
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

    print(f"{'row':<40}{'us':>9}{'exposed_us':>12}")
    ext.set_probe(0, 0, -1)
    t_moe = _time(_capture(moe, nothing, "moe-first"))
    print(f"{'moe alone (U=40)':<40}{t_moe:>9.1f}")
    rows = {}
    # (localq knob, lq launch-grid cap): the global kernel, then the local
    # kernel on its units, on 16 and on 8 blocks
    for lq, cap in ((0, 0), (1, 0), (1, 16), (1, 8)):
        ext.set_probe(0, lq, cap)
        plan_up = ext.gemm_plan(T, SHARED_N, HID, 1)
        plan_dn = ext.gemm_plan(T, HID, SHARED_K, 1)
        tag = f"lq={lq}" + (f" grid<={cap}" if cap else "")
        print(f"  plans (grid, ksr, units, localq, lgrid): up={plan_up} "
              f"down={plan_dn}")
        t_pair = _time(_capture(nothing, pair, "moe-first"))
        print(f"{'pair alone ' + tag:<40}{t_pair:>9.1f}")
        for order in ("moe-first", "pair-first"):
            t = _time(_capture(moe, pair, order))
            rows[(lq, cap, order)] = t - t_moe
            print(f"{'moe || pair ' + tag + ' ' + order:<40}{t:>9.1f}"
                  f"{t - t_moe:>12.1f}")
    ext.set_probe(-1, -1, -1)
    # the number that maps onto the step: 43 layers x the exposed pair
    for (lq, cap) in ((0, 0), (1, 0), (1, 16), (1, 8)):
        a = rows[(lq, cap, "moe-first")]
        b = rows[(lq, cap, "pair-first")]
        tag = f"lq={lq}" + (f" grid<={cap}" if cap else "")
        print(f"{tag}: exposed per layer moe-first {a:.1f} / pair-first {b:.1f} us"
              f" -> x43 = {43 * a / 1e3:.2f} / {43 * b / 1e3:.2f} ms/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
