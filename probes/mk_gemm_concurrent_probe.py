#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The shared expert's two MK GEMMs UNDER the routed MoE kernel -- the way
the decode step runs them (28차: 86 launches/step in the 30-45 us class,
3.5 ms, 2.4 ms of it exposed on the critical path).

Serving forks the shared expert (gate_up [1024 x 4096], down [4096 x 512])
onto an aux stream beside the routed b12x MoE call of the same layer, so a
persistent 48-block MK launch competes with the MoE kernel for SMs: its
blocks trickle in as MoE blocks retire, and on the global-quant path the
grid barrier then holds every resident block for the last one to arrive.
The standalone bench cannot show that. This probe captures ONE graph with
the fork -- stream A: the served b12x wrapper at U=40 (the boot's unique-
expert count, DRAM-cold: 142 MB a call); stream B: the pair -- in both issue
orders, and times the union span against each part alone, for the local
(VLLM_GLM53_MK_LOCALQ=1) and global (0) paths:

  exposed = span(moe || pair) - span(moe)      what the step actually pays

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

import torch  # noqa: E402

DEV = "cuda"
E, TOPK, HID, INTER = 288, 8, 4096, 512   # per-rank GLM-5.3 TP=4 geometry
T = 8                                     # C=1 verify batch (k=7 + 1)
U = 40                                    # unique experts a layer (27차)
SHARED_N = 1024                           # shared expert gate_up rows / rank
SHARED_K = 512                            # down-proj K / rank
REPS = 20


def _routing(U: int):
    gen = torch.Generator().manual_seed(U)
    pool = torch.randperm(E, generator=gen)[:U]
    flat = torch.arange(T * TOPK) % U
    ids = pool[flat].view(T, TOPK).to(torch.int32)
    w = torch.rand(T, TOPK, generator=gen, dtype=torch.float32)
    w = w / w.sum(dim=1, keepdim=True)
    return ids.to(DEV), w.to(DEV)


def _weight_set(gen):
    w13 = torch.randint(0, 256, (E, 2 * INTER, HID // 2), dtype=torch.uint8,
                        generator=gen).to(DEV)
    w2 = torch.randint(0, 256, (E, HID, INTER // 2), dtype=torch.uint8,
                       generator=gen).to(DEV)
    s13 = (torch.rand(E, 2 * INTER, HID // 16, generator=gen) * 0.05 + 0.01)
    s2 = (torch.rand(E, HID, INTER // 16, generator=gen) * 0.05 + 0.01)
    return (w13, w2, s13.to(torch.float8_e4m3fn).to(DEV),
            s2.to(torch.float8_e4m3fn).to(DEV))


def _capture(fn_a, fn_b, order: str):
    """One graph: fork, A on stream a, B on stream b, join. `order` is the
    issue order inside the capture (A first or B first)."""
    a = torch.cuda.Stream()
    b = torch.cuda.Stream()
    cur = torch.cuda.current_stream()
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
    del cur
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
    out = torch.empty(T, HID, dtype=torch.bfloat16, device=DEV)

    def moe():
        wrapper.run(x, w13, sf13, w2, sf2, ids, w, w1_alpha=ones,
                    w2_alpha=ones, fc2_input_scale=ones, out=out)

    mk.maybe_arm()
    assert mk._ARMED["gemm"], "MK-GEMM did not arm"
    ext = mk._EXT
    # the shared expert's pair: gate_up [1024 x 4096] on x, then down
    # [4096 x 512] on the (untouched: the activation is not the point)
    # first 512 columns of that output
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

    plan_up = ext.gemm_plan(T, SHARED_N, HID)
    plan_dn = ext.gemm_plan(T, HID, SHARED_K)
    print(f"plans (grid, ksr, units, localq): up={plan_up} down={plan_dn}")
    print(f"{'row':<34}{'us':>9}{'exposed_us':>12}")
    rows = {}
    t_moe = _time(_capture(moe, nothing, "moe-first"))
    print(f"{'moe alone (U=40)':<34}{t_moe:>9.1f}")
    for lq in (0, 1):
        ext.set_probe(0, lq)
        t_pair = _time(_capture(nothing, pair, "moe-first"))
        print(f"{'pair alone lq=' + str(lq):<34}{t_pair:>9.1f}")
        for order in ("moe-first", "pair-first"):
            t = _time(_capture(moe, pair, order))
            rows[(lq, order)] = t - t_moe
            print(f"{'moe || pair lq=' + str(lq) + ' ' + order:<34}{t:>9.1f}"
                  f"{t - t_moe:>12.1f}")
    ext.set_probe(0, 1)
    # the number that maps onto the step: 43 layers x the exposed pair
    for lq in (0, 1):
        worst = max(rows[(lq, o)] for o in ("moe-first", "pair-first"))
        best = min(rows[(lq, o)] for o in ("moe-first", "pair-first"))
        print(f"lq={lq}: exposed per layer {best:.1f}..{worst:.1f} us -> "
              f"x43 = {43 * best / 1e3:.2f}..{43 * worst / 1e3:.2f} ms/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
