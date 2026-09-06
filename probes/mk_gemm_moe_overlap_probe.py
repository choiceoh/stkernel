#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The shared expert's two MK GEMMs UNDER the routed MoE kernel -- the way
the decode step runs them, for the persistent lane and the v2 lane.

Serving forks the shared expert (gate_up [1024 x 4096], down [4096 x 512])
onto a side stream beside the routed b12x MoE call of the same layer. The
09-04 armed trace (30차) showed what that costs the persistent kernel: its
48 x 69.6 KB blocks cannot share an SM with the MoE kernel's 90 KB blocks,
so down lands one block at a time as MoE blocks retire and the publish
barrier holds every landed block for the last one -- 135 us in the step
where it is 18 alone, and the pair exposes ~28 us per layer (MoE start
delayed by gate_up, down finishing after MoE). deep_gemm's independent
blocks ran the same down INSIDE the MoE tail (09-01 stock trace: 36 us,
all of it under the MoE kernel). The v2 lane is built to do the same.

One graph with the fork -- stream A: the served b12x wrapper at U=40 (the
boot's unique-expert count, DRAM-cold: 142 MB a call); stream B: the pair
-- in both issue orders, timing the union against each part alone:

  exposed = span(moe || pair) - span(moe)      what the step actually pays

    bash probes/run_megakernel_bench.sh   # ... with the probe swapped in, see
    # the module README's probe recipe (run_mk_probe pattern)
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

import torch  # noqa: E402

# the served MoE fixture -- routing, weight sets, geometry -- is the go/no-go
# probe's; one definition of the C=1 decode MoE call for both probes
from moe_decode_stream_probe import (  # noqa: E402
    DEV, E, HID, INTER, T, TOPK, _routing, _weight_set)

U = 40                                    # unique experts a layer (27차)
SHARED_N = 1024                           # shared expert gate_up rows / rank
SHARED_K = 512                            # down-proj K / rank
REPS = 20


def _capture(fn_a, fn_b, order: str):
    """One graph: fork, A on stream a, B on stream b, join. `order` is the
    issue order inside the capture (A first or B first)."""
    a = torch.cuda.Stream()
    b = torch.cuda.Stream()
    for _ in range(2):  # warm both paths outside the capture
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
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
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
    # the wrapper returns its output; the `out=` keyword exists only on the
    # deployed b12x overlay, and this probe runs against whichever wrapper
    # the container has (the bench runner mounts the megakernel files only)
    print(f"wrapper {type(wrapper).__module__}")

    def moe():
        wrapper.run(x, w13, sf13, w2, sf2, ids, w, w1_alpha=ones,
                    w2_alpha=ones, fc2_input_scale=ones)

    mk.maybe_arm()
    assert mk._ARMED["gemm"], "MK-GEMM did not arm"
    ext = mk._EXT
    p_up = mk.build_mk_weight_w4(
        torch.randn(SHARED_N, HID, dtype=torch.bfloat16, device=DEV) * 0.05)
    p_dn = mk.build_mk_weight_w4(
        torch.randn(HID, SHARED_K, dtype=torch.bfloat16, device=DEV) * 0.05)
    xs = torch.randn(T, HID, dtype=torch.bfloat16, device=DEV)
    h = torch.empty(T, SHARED_K, dtype=torch.bfloat16, device=DEV)

    def pair():
        up = mk._gemm_call(xs, p_up, SHARED_N)
        h.copy_(up[:, :SHARED_K])
        mk._gemm_call(h, p_dn, HID)

    def nothing():
        pass

    print(f"plans (ksr, units, blocks/SM): up={list(ext.gemm2_plan(T, SHARED_N, HID))} "
          f"down={list(ext.gemm2_plan(T, HID, SHARED_K))}")
    print(f"{'row':<36}{'us':>9}{'exposed_us':>12}")
    t_moe = _time(_capture(moe, nothing, "moe-first"))
    print(f"{'moe alone (U=40)':<36}{t_moe:>9.1f}")
    # one lane since 34차 §8: the v2 non-persistent kernel is the GEMM
    # segment's only kernel (the persistent v1 row this probe compared
    # against is gone with it)
    rows = {}
    t_pair = _time(_capture(nothing, pair, "moe-first"))
    print(f"{'pair alone v2':<36}{t_pair:>9.1f}")
    for order in ("moe-first", "pair-first"):
        t = _time(_capture(moe, pair, order))
        rows[order] = t - t_moe
        print(f"{'moe || pair v2 ' + order:<36}{t:>9.1f}"
              f"{t - t_moe:>12.1f}")
    # the number that maps onto the step: 42 layers x the exposed pair
    worst = max(rows.values())
    best = min(rows.values())
    print(f"v2: exposed per layer {best:.1f}..{worst:.1f} us -> "
          f"x42 = {42 * best / 1e3:.2f}..{42 * worst / 1e3:.2f} ms/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
