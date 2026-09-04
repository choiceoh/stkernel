#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53 DFlash2 drafter GEMMs: what the tail of a decode step pays, arm by arm.

Where the number comes from. The armed 2026-09-03 trace (srv2, rank 0,
MK-GEMM/MHC/MLA armed) cut at the runner's prep kernel: a decode step is a
~59 ms target forward plus a 7.5-8.5 ms TAIL that the forward's own
annotation does not cover -- target head 836 us + AllGather 0.4 ms, then the
DFlash2 drafter: fc 792 us (bf16 cutlass, [7 x 20480] x [20480 x 4096],
replicated, 168 MB at 212 GB/s), five layers of sharded bf16 projections
(~560 us of GEMM per layer, at the bf16 DRAM floor), eleven one-shot
all-reduces, and the fp8 draft head 812 us. The drafter runs TP=4 in that
boot (the all-reduce after o_proj/down_proj and the 12 MB qkv streams say so),
so its per-rank weight bytes are a quarter of the checkpoint's. The GPU is
95% busy through the tail and the next step's prep starts when it ends, so
every microsecond of it is step time: the drafter's bf16 bytes (~3.6 ms of a
66 ms step, 733 MB/rank) are the largest kernel-side lever left in decode.

What this probe does. At the fleet shapes (read from the drafter config, TP
from --tp) it times every arm the image already has, per shape and summed
per step:

  bf16    F.linear on the bf16 weight -- what serves today (cutlass wmma)
  fp8     the target's production pair: _quantize_fp8_block_padded +
          _fp8_dense_gemm (deepgemm, fp8 x fp8, 128x128 block scales)
  fp4     _quantize_w4 + _fp8_fp4_dense_gemm (deepgemm, fp8 x e2m1)
  mk_w4   the megakernel W4 lane, build_mk_weight_w4 + _gemm_call. The lane
          takes K <= 4096, so the fc (K = 20480) runs as K-chunks of 4096
          summed in fp32: five launches where a K = 20480 kernel would be one
          (the chunked number is the ceiling of the wrapper, not the lane)

Weights are cycled -- NW distinct copies, enough that even the smallest arm's
bytes overflow the 24 MB L2 -- inside one replayed CUDA graph, so every
launch streams from DRAM the way a step's run of layers does (ledger trap 4:
back-to-back launches on ONE weight time the L2, not the kernel). Numerics
are the relative error against an fp32 matmul of the same bf16 weight, gated
by class (bf16 / fp8 5e-2, e2m1 0.15); the deliverable is TIMING. Any
drafter arm's adoption gate is acceptance (pos-1 within 2 pct of control),
and only a boot measures that.

Run inside the glm53 image with the composed overlays mounted; the megakernel
bench runner already does exactly that, so point it at this file:

    sed 's#megakernel_glm53_bench.py#drafter_fc_check.py#' \\
        probes/run_megakernel_bench.sh > /tmp/run_drafter.sh
    bash /tmp/run_drafter.sh [--tp 4] [--ms 7] [--config /models/.../config.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

# The megakernel driver reads its knobs at import; the runner forwards a
# caller's VLLM_GLM53_* but sets none itself, so the lane arms here unless
# the caller says otherwise (same convention as megakernel_glm53_bench.py).
os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
os.environ.setdefault("VLLM_GLM53_MK_MHC", "0")
os.environ.setdefault("VLLM_GLM53_MK_KDA", "0")
os.environ.setdefault("VLLM_GLM53_MK_MLA", "0")
os.environ.setdefault("VLLM_GLM53_MK_PDL", "1")

# Fleet checkpoint (glm53-redhat-nvfp4's GLM-5.3-Flash-DFlash2 drafter):
# hidden 4096, 5 layers, 32 q / 8 kv heads of 128, intermediate 12288,
# conv kernel_projection [1024, 4096] x2 per layer, fc [4096, 5 x 4096].
# #231's lesson: measure the fleet shape, never a synthetic stand-in --
# --config overrides every one of these from the drafter's config.json.
DEF_LAYERS, DEF_HIDDEN, DEF_M = 5, 4096, 7
DEF_HEADS, DEF_KV_HEADS, DEF_HEAD_DIM, DEF_INTER = 32, 8, 128, 12288
DEF_CONV_GROUP, DEF_CONV_TAPS = 16, 2   # kernel_projection: hidden/group x taps x 2
DEF_VOCAB = 154880
STEP_MS = 66.0   # armed 09-03 trace, prep-anchor cut, median of the two clean steps
L2_BYTES = 24 << 20
MIN_STREAM = 96 << 20  # cycled bytes per arm must be at least this, 4x the L2
NW_MIN, NW_MAX = 6, 48
MK_KMAX = 4096   # the lane's K contract (_mk_gemm_eligible)
TOL = {"bf16": 5e-2, "fp8": 5e-2, "fp4": 0.15, "mk_w4": 0.15}  # e2m1: by-design 0.15


def _load_overlay():
    for name in ("glm53_fp8_dense", "glm53_megakernel"):
        spec = importlib.util.find_spec(f"vllm.model_executor.layers.{name}")
        if spec is None:
            print(f"!! {name} not mounted in this image -- nothing to measure")
            sys.exit(2)
    from vllm.model_executor.layers import glm53_megakernel as mk
    from vllm.model_executor.layers import glm53_fp8_dense as fd
    return mk, fd


def _graph_us(fn_of_i, n: int, reps: int) -> float:
    """Mean us/launch over `reps` replays of a graph holding n launches, one
    per cycled weight. A replay returns nothing, so bitwise determinism is
    checked on live launches by the caller."""
    for i in range(3):
        fn_of_i(i % n)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        with torch.cuda.graph(g, stream=st):
            for i in range(n):
                fn_of_i(i)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / (reps * n) * 1e3


def _launch_bitwise(fn) -> bool:
    o1 = fn()
    torch.cuda.synchronize()
    o2 = fn()
    torch.cuda.synchronize()
    return bool(torch.equal(o1, o2))


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    den = ref.float().norm()
    if not torch.isfinite(den) or den == 0:
        return float("inf")
    return float((got.float() - ref.float()).norm() / den)


def _shapes(cfg: dict, tp: int, with_head: bool) -> list[tuple[str, int, int, int]]:
    """(name, N, K, count per step) at the per-rank shapes of a TP=tp drafter.
    Column-parallel linears shard N, row-parallel ones shard K, and the
    replicated ones (fc, the conv kernel_projections) are read whole."""
    h, L = cfg["hidden"], cfg["layers"]
    hd, nh, nkv, inter = cfg["head_dim"], cfg["heads"], cfg["kv_heads"], cfg["inter"]
    kproj_n = h // cfg["conv_group"] * cfg["conv_taps"] * 2
    out = [
        ("fc", h, L * h, 1),
        ("qkv_proj", (nh + 2 * nkv) * hd // tp, h, L),
        ("o_proj", h, nh * hd // tp, L),
        ("gate_up_proj", 2 * inter // tp, h, L),
        ("down_proj", h, inter // tp, L),
        ("kernel_projection", kproj_n, h, 2 * L),
    ]
    if with_head:
        out.append(("lm_head", cfg["vocab"] // tp, h, 1))
    return out


def _mk_chunks(mk, w: torch.Tensor):
    """W4 packs of the K-chunks of w (one when K <= 4096) -- built by the
    LANE's own chunker, not a copy of it. This probe is the number the
    K-chunked lane was adopted on, so its packs and its chunk width have to
    be the served ones; a second chunk loop here would keep measuring an old
    width after the driver moved (build_mk_weight_w4_kchunks / MK_GEMM_KMAX),
    with the table still labelled MK W4. The width is asserted, not assumed:
    the x-slicing below (xc) has to agree with it."""
    lane_kmax = getattr(mk, "MK_GEMM_KMAX", MK_KMAX)
    assert lane_kmax == MK_KMAX, (
        f"the lane's K contract is {lane_kmax}, this probe slices at {MK_KMAX}")
    return mk.build_mk_weight_w4_kchunks(w)


def _pack_bytes(p) -> int:
    return sum(t.numel() * t.element_size() for t in p if isinstance(t, torch.Tensor))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="drafter config.json (hidden_size, num_hidden_layers, heads...)")
    ap.add_argument("--tp", type=int, default=4,
                    help="drafter tensor parallel size (the 09-03 boot ran 4)")
    ap.add_argument("--ms", type=int, nargs="*", default=[DEF_M],
                    help="M values to time, e.g. --ms 7 8 16")
    ap.add_argument("--step-ms", type=float, default=STEP_MS)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shapes", default=None,
                    help="comma list to restrict, e.g. fc,gate_up_proj")
    ap.add_argument("--head", action="store_true",
                    help="also time the vocab/tp head shape (reference only: "
                         "the served head is the fp8_lm_head W8A16 lane)")
    args = ap.parse_args()
    cfg = {"hidden": DEF_HIDDEN, "layers": DEF_LAYERS, "heads": DEF_HEADS,
           "kv_heads": DEF_KV_HEADS, "head_dim": DEF_HEAD_DIM, "inter": DEF_INTER,
           "conv_group": DEF_CONV_GROUP, "conv_taps": DEF_CONV_TAPS, "vocab": DEF_VOCAB}
    if args.config:
        c = json.load(open(args.config))
        c = c.get("text_config", c)
        cfg.update(hidden=int(c["hidden_size"]),
                   layers=int(c.get("num_hidden_layers", DEF_LAYERS)),
                   heads=int(c.get("num_attention_heads", DEF_HEADS)),
                   kv_heads=int(c.get("num_key_value_heads", DEF_KV_HEADS)),
                   head_dim=int(c.get("head_dim", DEF_HEAD_DIM)),
                   inter=int(c.get("intermediate_size", DEF_INTER)),
                   vocab=int(c.get("vocab_size", DEF_VOCAB)))
        dc = c.get("dflash_config") or {}
        cfg.update(conv_group=int(dc.get("conv_group_size", DEF_CONV_GROUP)),
                   conv_taps=int(dc.get("conv_kernel_size", DEF_CONV_TAPS)))
    torch.manual_seed(0)
    dev = "cuda"
    mk, fd = _load_overlay()
    mk.maybe_arm()
    mk_armed = bool(mk._ARMED.get("gemm"))
    print(f"drafter tp={args.tp} hidden={cfg['hidden']} layers={cfg['layers']} "
          f"mk_gemm armed={mk_armed}")

    shapes = _shapes(cfg, args.tp, args.head)
    if args.shapes:
        keep = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in keep]
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    print(f"{'shape':<18}{'N':>6}{'K':>6}{'M':>3} {'arm':<7}{'us':>8}{'GB/s':>6}"
          f"  rel_err  (x per step)")
    per_step = {}  # (m, arm) -> us summed over the step's launches
    for name, N, K, count in shapes:
        w_bytes = {"bf16": N * K * 2, "fp8": N * K + (N // 128 + 1) * (K // 128 + 1) * 4,
                   "fp4": N * K // 2 + N * (K // 16), "mk_w4": None}
        # cycle enough copies that the SMALLEST arm's bytes overflow L2
        nw = max(NW_MIN, min(NW_MAX, -(-MIN_STREAM // (N * K // 2))))
        ws = [(torch.randn(N, K, dtype=torch.float32, device=dev) * 0.02
               ).to(torch.bfloat16) for _ in range(nw)]
        fp8 = [fd._quantize_fp8_block_padded(w) for w in ws]
        fp4 = []
        for packed_sf in (True, False):
            try:
                fp4 = [fd._quantize_w4(w, packed_sf=packed_sf) for w in ws]
                break
            except Exception as e:
                print(f"  fp4 pack packed_sf={packed_sf}: unavailable ({e!r})"[:120])
        mkp = []
        if mk_armed:
            try:
                mkp = [_mk_chunks(mk, w) for w in ws]
                w_bytes["mk_w4"] = sum(_pack_bytes(p) for p in mkp[0])
            except Exception as e:
                print(f"  mk_w4 pack: unavailable ({e!r})"[:120])
        for m in args.ms:
            x = (torch.randn(m, K, device=dev) * 1.5).to(torch.bfloat16)
            ref = x.float() @ ws[0].float().t()
            arms = []

            def a_bf16(i):
                return torch.nn.functional.linear(x, ws[i])

            def a_fp8(i):
                q, s, rows, cols = fp8[i]
                return fd._fp8_dense_gemm(x, q, s, rows, cols)

            def a_fp4(i):
                return fd._fp8_fp4_dense_gemm(x, fp4[i][0], fp4[i][1])

            def a_mk(i):
                packs = mkp[i]
                if len(packs) == 1:
                    # serving routes a single pack straight to the kernel
                    return mk._gemm_call(x, packs[0], N)
                # the served multi-chunk path: one launch per chunk, the
                # column slices taken contiguous INSIDE the timed region (the
                # lane pays that copy; a pre-sliced xc hid it) and the fp32
                # sum the lane does -- not a re-implementation of it
                out = mk._gemm_kchunks(x, packs, N)
                if out is None:
                    raise RuntimeError("the lane refused these chunks "
                                       "(_gemm_kchunks returned None)")
                return out

            cands = [("bf16", a_bf16)]
            cands.append(("fp8", a_fp8))
            if fp4:
                cands.append(("fp4", a_fp4))
            if mkp:
                cands.append(("mk_w4", a_mk))
            for label, fn in cands:
                try:
                    r = _rel(fn(0), ref)
                except Exception as e:
                    print(f"{name:<18}{N:>6}{K:>6}{m:>3} {label:<7}{'SKIP':>8}  -- {e!r}"[:118])
                    continue
                check(r <= TOL[label], f"{name} M={m} {label} rel_err {r:.2e} <= {TOL[label]}")
                bit = _launch_bitwise(lambda: fn(0))
                if not bit:
                    fails.append(f"{name} M={m} {label} not launch-bitwise")
                t = _graph_us(fn, nw, args.iters)
                nb = w_bytes[label] + x.numel() * 2 + m * N * 2
                chunk = f" [{len(mkp[0])}xK]" if label == "mk_w4" and len(mkp[0]) > 1 else ""
                print(f"{name:<18}{N:>6}{K:>6}{m:>3} {label:<7}{t:>8.1f}{nb / t / 1e3:>6.0f}"
                      f"  {r:.2e}  (x{count}){chunk}{'' if bit else ' [LAUNCH DIFFERS!]'}")
                arms.append((label, t))
                per_step[(m, label)] = per_step.get((m, label), 0.0) + t * count
            if name == "lm_head":
                # not part of the drafter sum: the served head is another lane
                for label, t in arms:
                    per_step[(m, label)] -= t * count
        del ws, fp8, fp4, mkp
        torch.cuda.empty_cache()

    print(f"\nper-step drafter GEMM sum ({cfg['layers']} layers x (qkv, o, gate_up, "
          f"down, 2 x kernel_projection) + fc), vs a {args.step_ms:.0f} ms step:")
    for m in args.ms:
        base = per_step.get((m, "bf16"))
        if base is None:
            continue
        for label in ("bf16", "fp8", "fp4", "mk_w4"):
            t = per_step.get((m, label))
            if t is None:
                continue
            prize = base - t
            print(f"  M={m} {label:<7}{t / 1000:7.2f} ms/step"
                  + (f"  ({prize / 1000:+.2f} ms = {100 * prize / (args.step_ms * 1000):+.1f}%"
                     f" of the step vs bf16)" if label != "bf16" else "  (serves today)"))
    if fails:
        print("\nFAIL:", *fails, sep="\n  ")
    print("VERDICT:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
