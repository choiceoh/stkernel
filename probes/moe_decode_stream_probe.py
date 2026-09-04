#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Go/no-go for MK_SEG_MOE: how fast does the SERVED b12x MoE kernel stream
expert bytes at the decode shape, against what this part can stream.

The 21차 trace put the routed MoE at 31 ms/step (44 pct) and called it
"bandwidth floor" -- but that 190 GB/s was back-computed from a GUESSED
~40 unique experts per layer, and the MK W4 lane streams 244 GB/s in-loop
on the same part. So the floor claim is circular. This probe measures the
kernel directly: the b12x wrapper the serving path constructs (same
geometry, same dispatch overlay, same static/micro cutover), at C=1 decode
(8 tokens x top-8 = 64 routed pairs -> the STATIC backend), with the number
of unique experts U controlled, weights DRAM-cold by rotating 8 full expert
sets (8 x 1 GB, L2 is 24 MB), CUDA-graph replay timed with events.

Bytes per unique expert per rank (GLM-5.3 TP=4, nvfp4): w13 [1024 x 2048 B]
2.10 MB + sf 0.26 MB + w2 [4096 x 256 B] 1.05 MB + sf 0.13 MB = 3.54 MB.

Rows:
  b12x U=..    us per MoE call, effective GB/s = U x 3.54 MB / t
  lane         the MK W4 GEMM streaming 12 different [6416 x 4096] packs in
               one graph (161 MB, PDL on): the rate a persistent MK segment
               would stream the same bytes at
  torch read   fp32 sum over 161 MB: a library read-only reference
  warm gemm    one n=6416 launch cold vs after its pack was pulled into L2
               (what an AR-wait prefetch of a consumer's pack buys)

Verdict rule (strategy doc): b12x >= 90 pct of the lane rate closes the
MK_SEG_MOE axis; below it, the gap x 31 ms is the segment's ceiling.

    bash probes/run_mk_probe.sh probes/moe_decode_stream_probe.py
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
E, TOPK, HID, INTER = 288, 8, 4096, 512   # per-rank intermediate 2048 / 4
T = 8                                     # C=1 verify batch (k=7 + 1)
SETS = 8                                  # weight sets rotated per replay
BYTES_PER_EXPERT = 1024 * 2048 + 1024 * 256 + 4096 * 256 + 4096 * 32


def _graph(fn, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        fn()
    torch.cuda.synchronize()
    return g


def _time_graph(g, reps: int) -> float:
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / reps   # us per replay


def _routing(U: int):
    """[T, TOPK] int32 with exactly U distinct experts, 8 distinct per token."""
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
    return w13, w2, s13.to(torch.float8_e4m3fn).to(DEV), s2.to(torch.float8_e4m3fn).to(DEV)


def served_wrapper():
    """The b12x wrapper serving constructs: exact GLM TP geometry, static
    workspace only. ONE definition -- the concurrent probe builds on it."""
    from flashinfer.fused_moe import B12xMoEWrapper

    return B12xMoEWrapper(
        num_experts=E, top_k=TOPK, hidden_size=HID, intermediate_size=INTER,
        use_cuda_graph=True, max_num_tokens=64, num_local_experts=E,
        activation="swigluoai_uninterleave", swiglu_alpha=1.0,
        swiglu_beta=0.0, swiglu_limit=10.0)


def expert_set(gen):
    """One DRAM-cold set of expert weights in the wrapper's layouts:
    (w13, sf13, w2, sf2)."""
    from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout

    w13, w2, s13, s2 = _weight_set(gen)
    sf13 = flashinfer_convert_sf_to_mma_layout(
        s13.reshape(E * 2 * INTER, HID // 16), m=2 * INTER, k=HID,
        num_groups=E)
    sf2 = flashinfer_convert_sf_to_mma_layout(
        s2.reshape(E * HID, INTER // 16), m=HID, k=INTER, num_groups=E)
    return w13, sf13, w2, sf2


def main() -> int:
    from vllm.model_executor.layers import glm53_megakernel as mk

    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()} sets={SETS} T={T} topk={TOPK}")
    stream = torch.cuda.Stream()

    wrapper = served_wrapper()
    gen = torch.Generator().manual_seed(1)
    sets = [expert_set(gen) for _ in range(SETS)]
    ones = torch.ones(E, dtype=torch.float32, device=DEV)
    x = torch.randn(T, HID, dtype=torch.bfloat16, device=DEV) * 0.5
    out = torch.empty(T, HID, dtype=torch.bfloat16, device=DEV)
    print(f"{'row':<14}{'us/call':>9}{'MB':>8}{'GB/s':>8}")

    rates = {}
    for U in (8, 16, 24, 32, 40, 48, 56, 64):
        ids, w = _routing(U)

        def moe_all():
            for w13, sf13, w2, sf2 in sets:
                wrapper.run(x, w13, sf13, w2, sf2, ids, w, w1_alpha=ones,
                            w2_alpha=ones, fc2_input_scale=ones, out=out)

        g = _graph(moe_all, stream)
        us = _time_graph(g, 20) / SETS
        mb = U * BYTES_PER_EXPERT / 1e6
        rates[U] = mb * 1e6 / us / 1e3
        print(f"{'b12x U=' + str(U):<14}{us:>9.1f}{mb:>8.1f}{rates[U]:>8.0f}")
        del g

    # ---- the MK W4 lane on the same class of bytes: 12 packs of [6416 x
    # 4096] (13.4 MB each, 161 MB per replay), PDL on, one graph
    mk.maybe_arm()
    assert mk._ARMED["gemm"], "MK-GEMM did not arm"
    packs = [mk.build_mk_weight_w4(
        torch.randn(6416, HID, dtype=torch.bfloat16, device=DEV) * 0.05)
        for _ in range(12)]
    xg = torch.randn(T, HID, dtype=torch.bfloat16, device=DEV)

    def lane_all():
        for p in packs:
            mk._gemm_call(xg, p, 6416)

    g = _graph(lane_all, stream)
    us = _time_graph(g, 20) / len(packs)
    nb = packs[0][0].numel() + packs[0][1].numel()
    lane = nb / us / 1e3
    print(f"{'lane n=6416':<14}{us:>9.1f}{nb / 1e6:>8.1f}{lane:>8.0f}")
    del g

    # ---- a library read-only reference over the same 161 MB
    big = torch.randn(len(packs) * nb // 4, dtype=torch.float32, device=DEV)
    g = _graph(lambda: big.sum(), stream)
    us = _time_graph(g, 20)
    print(f"{'torch sum':<14}{us:>9.1f}{big.numel() * 4 / 1e6:>8.1f}"
          f"{big.numel() * 4 / us / 1e3:>8.0f}")
    del g

    # ---- what an L2 prefetch of a consumer's pack buys: one n=6416 launch,
    # cold (after a 2 x 24 MB flush) vs after the pack was read once
    flush = torch.empty(48 << 20, dtype=torch.int8, device=DEV)
    drain = torch.zeros(16 << 20, dtype=torch.float32, device=DEV)
    p = packs[0]
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    cold, warm = [], []
    for i in range(20):
        flush.zero_()
        drain.sum()
        drain.sum()
        xg.add_(0)
        if i % 2:
            p[0].view(torch.int32).sum()   # pull nibbles into L2 (clean)
            p[1].view(torch.int32).sum()   # and the group scales
        s.record()
        mk._gemm_call(xg, p, 6416)
        e.record()
        torch.cuda.synchronize()
        (warm if i % 2 else cold).append(s.elapsed_time(e) * 1e3)
    cold.sort()
    warm.sort()
    print(f"{'gemm cold':<14}{cold[len(cold) // 2]:>9.1f}{nb / 1e6:>8.1f}"
          f"{nb / cold[len(cold) // 2] / 1e3:>8.0f}")
    print(f"{'gemm L2-warm':<14}{warm[len(warm) // 2]:>9.1f}{nb / 1e6:>8.1f}"
          f"{nb / warm[len(warm) // 2] / 1e3:>8.0f}")

    best = max(rates.values())
    print(f"b12x best {best:.0f} GB/s = {100 * best / lane:.0f} pct of the "
          f"lane's {lane:.0f} GB/s; U=40 -> {rates[40]:.0f} GB/s "
          f"({100 * rates[40] / lane:.0f} pct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
