#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The served b12x static MoE kernel vs the decode-streaming v2 kernel
(``moe_static_kernel_v2``), at the decode shape, DRAM-cold, graph-replayed --
and the v2 per-CTA timeline (stamps) that says where the time goes.

Shape: C=1 decode = 8 tokens x top-8 = 64 routed pairs -> the STATIC backend
(the micro cutover is 40 pairs), U unique experts per layer controlled by the
routing (serving: ~40). Bytes per unique expert per rank (GLM-5.3 TP=4,
nvfp4): w13 2.10 MB + sf 0.26 + w2 1.05 + sf 0.13 = 3.54 MB.

DRAM-cold: one 1 GB expert set, and each replay walks ROT disjoint expert
subsets (ROT x U x 3.54 MB >= 8 x L2), so a call's experts were last touched
ROT-1 calls ago.

Rows:
  stock U=..        us per MoE call, effective GB/s (27차 reference: U=40 ->
                    720 us / 197 GB/s)
  v2[cfg] U=..      the same with the v2 kernel config `cfg`
  numerics          max|v2 - stock| on the same input vs max|stock - stock'|
                    (the bf16 atomic scatter order is the only nondeterminism;
                    the two kernels quantize and accumulate identically)
  stamps            v2 timeline: frontend, per-item FC1 / quant / FC2, items
                    per CTA, and the tail each CTA idles after its last item

    bash probes/run_mk_probe.sh probes/b12x_static_probe.py [--configs 1,m32,f3,g2,...]
        [--us 8,16,24,32,40,48,64] [--reps 20]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("MK_PKG_PATH",
                                  "/usr/local/lib/python3.12/dist-packages"))

import torch  # noqa: E402

DEV = "cuda"
E, TOPK, HID, INTER = 288, 8, 4096, 512   # per-rank intermediate 2048 / 4
T = 8                                     # C=1 verify batch (k=7 + 1)
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


def _routings(U: int, T: int = T):
    """ROT disjoint expert subsets of size U; per subset [T, TOPK] ids with 8
    distinct experts per token, all inside the subset, plus normalized weights."""
    gen = torch.Generator().manual_seed(1000 + U + 7919 * T)
    perm = torch.randperm(E, generator=gen)
    rot = min(E // U, 8)
    out = []
    for r in range(rot):
        pool = perm[r * U:(r + 1) * U]
        rows = []
        for t in range(T):
            rows.append(pool[torch.randperm(U, generator=gen)[:TOPK]])
        ids = torch.stack(rows).to(torch.int32)
        w = torch.rand(T, TOPK, generator=gen, dtype=torch.float32)
        w = w / w.sum(dim=1, keepdim=True)
        out.append((ids.to(DEV), w.to(DEV)))
    return out


def _weight_set(gen):
    w13 = torch.randint(0, 256, (E, 2 * INTER, HID // 2), dtype=torch.uint8,
                        generator=gen).to(DEV)
    w2 = torch.randint(0, 256, (E, HID, INTER // 2), dtype=torch.uint8,
                       generator=gen).to(DEV)
    s13 = (torch.rand(E, 2 * INTER, HID // 16, generator=gen) * 0.05 + 0.01)
    s2 = (torch.rand(E, HID, INTER // 16, generator=gen) * 0.05 + 0.01)
    return w13, w2, s13.to(torch.float8_e4m3fn).to(DEV), s2.to(torch.float8_e4m3fn).to(DEV)


def served_wrapper():
    from flashinfer.fused_moe import B12xMoEWrapper

    return B12xMoEWrapper(
        num_experts=E, top_k=TOPK, hidden_size=HID, intermediate_size=INTER,
        use_cuda_graph=True, max_num_tokens=80, num_local_experts=E,
        activation="swigluoai_uninterleave", swiglu_alpha=1.0,
        swiglu_beta=0.0, swiglu_limit=10.0)


def expert_set(gen):
    from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout

    w13, w2, s13, s2 = _weight_set(gen)
    sf13 = flashinfer_convert_sf_to_mma_layout(
        s13.reshape(E * 2 * INTER, HID // 16), m=2 * INTER, k=HID,
        num_groups=E)
    sf2 = flashinfer_convert_sf_to_mma_layout(
        s2.reshape(E * HID, INTER // 16), m=HID, k=INTER, num_groups=E)
    return w13, sf13, w2, sf2


def _stamp_summary(st: torch.Tensor, label: str) -> None:
    """st: [grid, STAMP_SLOTS] int64 ns."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_static_kernel_v2 as k2

    s = st.cpu()
    grid = s.shape[0]
    t0 = s[:, 0]
    t1 = s[:, 1]
    t_end = s[:, k2.STAMP_MMA_END]
    n_items = s[:, k2.STAMP_MMA_END + 1]
    base = int(t0.min())
    kernel_end = int(t_end.max())
    fc1, quant, fc2, item_total = [], [], [], []
    dma_fc1, dma_fc2, dma_lead = [], [], []
    for b in range(grid):
        for i in range(min(int(n_items[b]), k2.STAMP_ITEMS)):
            o = 2 + 5 * i
            a, f1, q, f2 = (int(s[b, o + j]) for j in range(4))
            if 0 in (a, f1, q, f2):
                continue
            fc1.append((f1 - a) / 1e3)
            quant.append((q - f1) / 1e3)
            fc2.append((f2 - q) / 1e3)
            item_total.append((f2 - a) / 1e3)
            d = k2.STAMP_DMA_BASE + 3 * i
            d0, d1, d2 = (int(s[b, d + j]) for j in range(3))
            if 0 not in (d0, d1, d2):
                dma_fc1.append((d1 - d0) / 1e3)
                dma_fc2.append((d2 - d1) / 1e3)
                dma_lead.append((f1 - d1) / 1e3)   # MMA FC1 done after DMA issued
    tail = [(kernel_end - int(t_end[b])) / 1e3 for b in range(grid)]
    front = [(int(t1[b]) - int(t0[b])) / 1e3 for b in range(grid)]
    start_skew = [(int(t0[b]) - base) / 1e3 for b in range(grid)]
    med = statistics.median
    counts = sorted(int(x) for x in n_items.tolist())
    print(f"  stamps[{label}] span {(kernel_end - base) / 1e3:.1f} us | frontend med "
          f"{med(front):.1f} max {max(front):.1f} | start skew max {max(start_skew):.1f} | "
          f"items/CTA {counts[0]}..{counts[-1]} (mean {statistics.mean(counts):.2f})")
    if item_total:
        print(f"  stamps[{label}] per item (med/max us): total {med(item_total):.1f}/"
              f"{max(item_total):.1f} = FC1 {med(fc1):.1f}/{max(fc1):.1f} + quant "
              f"{med(quant):.1f}/{max(quant):.1f} + FC2 {med(fc2):.1f}/{max(fc2):.1f}")
    if dma_fc1:
        print(f"  stamps[{label}] DMA issue (med us): FC1 {med(dma_fc1):.1f} FC2 "
              f"{med(dma_fc2):.1f}; MMA FC1 done - DMA FC1 issued {med(dma_lead):.1f}")
    print(f"  stamps[{label}] idle tail per CTA (us): med {med(tail):.1f} max {max(tail):.1f} "
          f"mean {statistics.mean(tail):.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs",
                    default="1,m32,f3,g2,m64,f2,g2,m32,f2,g4,d,m32,f2,g4,s,m32,f2,g4,d,s")
    ap.add_argument("--us", default="8,16,24,32,40,48,64")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--full-sweep", default="all",
                    help="v2 config specs that get the full U sweep, or 'all' "
                         "(others: U=40,64)")
    args = ap.parse_args()
    # config specs are themselves comma-separated: split on ",m" boundaries
    raw = args.configs
    specs = []
    for part in raw.replace(",m", "|m").replace(",1", "|1").split("|"):
        part = part.strip()
        if part:
            specs.append(part)
    us_list = [int(u) for u in args.us.split(",")]
    full = (set(specs) if args.full_sweep == "all"
            else set(args.full_sweep.replace(",m", "|m").split("|")))

    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    torch.cuda.init()
    print(f"device {torch.cuda.get_device_name()} T={T} topk={TOPK} reps={args.reps} "
          f"configs={specs}")
    stream = torch.cuda.Stream()
    wrapper = served_wrapper()
    gen = torch.Generator().manual_seed(1)
    w13, sf13, w2, sf2 = expert_set(gen)
    ones = torch.ones(E, dtype=torch.float32, device=DEV)
    x = torch.randn(T, HID, dtype=torch.bfloat16, device=DEV) * 0.5
    out = torch.empty(T, HID, dtype=torch.bfloat16, device=DEV)
    routings = {U: _routings(U) for U in us_list}

    def make_io(t):
        return (torch.randn(t, HID, dtype=torch.bfloat16, device=DEV) * 0.5,
                torch.empty(t, HID, dtype=torch.bfloat16, device=DEV))

    def call(ids, w, xx=x, oo=out):
        wrapper.run(xx, w13, sf13, w2, sf2, ids, w, w1_alpha=ones,
                    w2_alpha=ones, fc2_input_scale=ones, out=oo)

    def bench(U):
        rot = routings[U]

        def all_calls():
            for ids, w in rot:
                call(ids, w)

        g = _graph(all_calls, stream)
        us = _time_graph(g, args.reps) / len(rot)
        del g
        mb = U * BYTES_PER_EXPERT / 1e6
        return us, mb, mb * 1e6 / us / 1e3

    def eager_out(ids, w, xx=x, oo=out):
        call(ids, w, xx, oo)
        torch.cuda.synchronize()
        return oo.clone()

    print(f"{'row':<22}{'us/call':>9}{'MB':>8}{'GB/s':>8}")
    md._STATIC_V2_OVERRIDE = None
    stock = {}
    for U in us_list:
        stock[U] = bench(U)
        print(f"{'stock U=' + str(U):<22}{stock[U][0]:>9.1f}{stock[U][1]:>8.1f}{stock[U][2]:>8.0f}")
    ids40, w40 = routings[40 if 40 in routings else us_list[-1]][0]
    # numerics cases: the served shape, a C=4-like shape, and the static
    # backend's largest shape (640 pairs), where an expert spans 3 m-tiles
    cases = [("T=8 U=40", ids40, w40, x, out)]
    for t, u in ((32, 16), (80, 8)):
        ids_u, w_u = _routings(u, t)[0]
        xx, oo = make_io(t)
        cases.append((f"T={t} U={u}", ids_u, w_u, xx, oo))
    refs = {}
    for case, ids_, w_, xx, oo in cases:
        ref_a = eager_out(ids_, w_, xx, oo)
        ref_b = eager_out(ids_, w_, xx, oo)
        noise = (ref_a.float() - ref_b.float()).abs().max().item()
        scale = ref_a.float().abs().max().item()
        refs[case] = (ref_a, noise, scale)
        print(f"numerics stock vs stock' [{case}] max|diff| {noise:.3e} "
              f"(max|out| {scale:.3e})")

    results = {}
    for spec in specs:
        cfg = md._parse_glm53_static_v2(spec)
        label = f"v2[{spec}]"
        try:
            md._STATIC_V2_OVERRIDE = cfg
            md._STATIC_V2_KERNEL_CACHE.clear()
            for case, ids_, w_, xx, oo in cases:
                ref_a, noise, scale = refs[case]
                v2_out = eager_out(ids_, w_, xx, oo)
                diff = (v2_out.float() - ref_a.float()).abs().max().item()
                verdict = "PASS" if diff <= max(4 * noise, 1e-2 * scale) else "FAIL"
                print(f"numerics {label} vs stock [{case}] max|diff| {diff:.3e} "
                      f"-> {verdict}")
            sweep = us_list if spec in full else [u for u in (40, 64) if u in routings]
            for U in sweep:
                r = bench(U)
                results[(spec, U)] = r
                s = stock[U]
                print(f"{label + ' U=' + str(U):<22}{r[0]:>9.1f}{r[1]:>8.1f}{r[2]:>8.0f}"
                      f"   vs stock {s[0]:.1f} us ({100 * (s[0] - r[0]) / s[0]:+.1f}%)")
            if cfg.get("stamps"):
                key = (48, str(torch.device(DEV, 0)))
                st = None
                for k_, v_ in md._STATIC_V2_STAMPS.items():
                    st = v_
                if st is not None:
                    st.zero_()
                    call(ids40, w40)
                    torch.cuda.synchronize()
                    _stamp_summary(st, spec)
        except Exception as exc:  # noqa: BLE001 -- one config must not end the run
            print(f"{label} ERROR: {type(exc).__name__}: {str(exc)[:400]}")
        finally:
            md._STATIC_V2_OVERRIDE = None

    best = None
    for (spec, U), r in results.items():
        if U == 40 and (best is None or r[0] < best[1]):
            best = (spec, r[0], r[2])
    if best is not None and 40 in stock:
        print(f"best v2 at U=40: {best[0]} {best[1]:.1f} us ({best[2]:.0f} GB/s) vs stock "
              f"{stock[40][0]:.1f} us ({stock[40][2]:.0f} GB/s): "
              f"{100 * (stock[40][0] - best[1]) / stock[40][0]:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
