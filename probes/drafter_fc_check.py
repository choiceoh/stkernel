#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53 drafter fc projection: stock timing vs reachable alternatives.

STEP_KERNEL_MAP supplementary decomposition 3 names the drafter fc GEMM as a
split-K candidate: [M, K=5*hidden] x [K, N=hidden], one call per decode step,
deep_gemm fp8xfp4 in the current boot at 809 us -- 42 MB of fp4 weights read
at 52-104 GB/s, a quarter to a half of the W4 stream's measured floor
(~190 GB/s effective, megakernel ledger). While the host-idle hiding of the
2026-09-01 trace lasts it is off the critical path; the moment EXP-7 removes
that hiding it is the main body of the ~6% drafter exposure. This probe
measures the real shape cold-weights and every arm that already exists in the
fleet image, so the kernel-building decision is made on numbers, not on the
CUPTI figure.

Arms:
  stock    -- the production pair, glm53_fp8_dense._fp8_fp4_dense_gemm
              (per-token fp8 activation quant + deep_gemm fp8_fp4_gemm_nt)
  bf16 mm  -- torch.mm on the bf16 weight (the 168 MB eager path the 09-01
              trace caught in another boot)
  linear   -- F.linear NT layout of the same
  mk_w4    -- the megakernel W4 lane at this shape, best-effort: the lane was
              built and VERDICT-PASSed on K=4096 shapes; if it refuses
              K=20480 the SKIP line is the answer, not a failure

The weights here are built from a bf16 matrix by the same packer the dense
path uses -- by-design e2m1 error, so numerics are gated at the MK bench's
0.15, and TIMING is the deliverable. Run inside a glm53 image with the repo
mounted at /repo (deployment commit):

    docker run --rm --gpus all --entrypoint python3 \
      --mount type=bind,src=$REPO,dst=/repo,readonly glm53:v13-b12x \
      /repo/probes/drafter_fc_check.py [--config /models/.../config.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

# Fleet checkpoint defaults (glm53-redhat-nvfp4 drafter): 5 draft layers of
# hidden 4096 concatenated into one fc projection. #231's lesson: measure the
# fleet shape, never a synthetic stand-in -- pass --config to override.
DEF_LAYERS, DEF_HIDDEN, DEF_M = 5, 4096, 7
STEP_MS = 71.0   # ledger 09-01 trace median; --step-ms overrides
NW = 6           # weights cycled inside one replayed graph: 6 x 42 MB fp4
                 # = 252 MB, an order past GB10's 24 MB L2, so every launch
                 # streams from DRAM like a decode step's run of layers does
W4_GBPS = 190.0  # megakernel ledger: W4 stream effective DRAM rate
E2M1_TOL = 0.15  # MK bench's by-design gate for the e2m1 arms
BF16_TOL = 5e-2


def _load_overlay():
    for name in ("glm53_fp8_dense", "glm53_megakernel"):
        spec = importlib.util.find_spec(f"vllm.model_executor.layers.{name}")
        if spec is None:
            print(f"!! {name} not mounted in this image -- nothing to measure")
            sys.exit(2)
    from vllm.model_executor.layers import glm53_megakernel as mk
    from vllm.model_executor.layers.glm53_fp8_dense import (
        _fp8_fp4_dense_gemm, _quantize_w4)
    return mk, _fp8_fp4_dense_gemm, _quantize_w4


def _graph_us(fn_of_i, n: int = NW, reps: int = 20) -> float:
    """Mean us/launch over `reps` replays of a graph holding n launches, one
    per cycled weight. A replay returns nothing, so bitwise determinism is
    checked on live launches by the caller."""
    for i in range(3):
        fn_of_i(i)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="drafter config.json (hidden_size, num_hidden_layers)")
    ap.add_argument("--K", type=int, default=DEF_LAYERS * DEF_HIDDEN)
    ap.add_argument("--N", type=int, default=DEF_HIDDEN)
    ap.add_argument("--ms", type=int, nargs="*", default=[DEF_M],
                    help="M values to time, e.g. --ms 1 2 4 7 8 16 32")
    ap.add_argument("--step-ms", type=float, default=STEP_MS)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    N, K = args.N, args.K
    if args.config:
        c = json.load(open(args.config))
        c = c.get("text_config", c)
        hidden = int(c["hidden_size"])
        layers = int(c.get("num_hidden_layers", DEF_LAYERS))
        N, K = hidden, layers * hidden
    torch.manual_seed(0)
    dev = "cuda"
    mk, fp8_fp4_gemm, quantize_w4 = _load_overlay()
    print(f"shape: M x K={K} @ K x N={N} (drafter fc, one call/step); "
          f"{N * K // 2 / 1e6:.0f} MB fp4 / {N * K * 2 / 1e6:.0f} MB bf16")

    # One bf16 matrix as the packers' common source. The timed arms read the
    # packs; the bf16 arms keep their own full copy.
    w = (torch.randn(N, K, dtype=torch.float32, device=dev) * 0.02
         ).to(torch.bfloat16)
    ref_w = w.float()
    packs = []
    for packed_sf in (True, False):
        try:
            packs.append(quantize_w4(w, packed_sf=packed_sf))
        except Exception as e:
            print(f"  fp4 pack packed_sf={packed_sf}: unavailable ({e!r})")
    if not packs:
        print("!! no fp4 pack could be built -- stock arm impossible")
        return 2

    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    print(f"{'M':>4} {'arm':<16}{'us':>9}{'GB/s':>7}  rel_err")
    rows = []
    for m in args.ms:
        x = (torch.randn(m, K, device=dev) * 1.5).to(torch.bfloat16)
        ref = torch.mm(x.float(), ref_w)
        arms = []

        def arm_stock():
            return fp8_fp4_gemm(x, packs[0][0], packs[0][1])

        r = _rel(arm_stock(), ref)
        check(r <= E2M1_TOL, f"M={m} stock fp8xfp4 rel_err {r:.2e} <= {E2M1_TOL} (e2m1)")
        t = _graph_us(arm_stock, reps=args.iters)
        nb = N * K // 2 + x.numel() * 2 + m * N * 2
        bit = _launch_bitwise(arm_stock)
        print(f"{m:>4} {'stock fp8xfp4':<16}{t:>9.1f}{nb / t / 1e3:>7.0f}  {r:.2e}"
              f"{' [launch bitwise]' if bit else ' [LAUNCH DIFFERS!]'}")
        if not bit:
            fails.append(f"M={m} stock not launch-bitwise")
        arms.append(("stock fp8xfp4", t, nb))

        for label, fn in (
                ("bf16 mm", lambda: torch.mm(x, w.t())),
                ("linear NT", lambda: torch.nn.functional.linear(x, w))):
            r = _rel(fn(), ref)
            check(r <= BF16_TOL, f"M={m} {label} rel_err {r:.2e} <= {BF16_TOL} (bf16)")
            t = _graph_us(fn, reps=args.iters)
            nb = N * K * 2 + x.numel() * 2 + m * N * 2
            print(f"{m:>4} {label:<16}{t:>9.1f}{nb / t / 1e3:>7.0f}  {r:.2e}")
            arms.append((label, t, nb))

        try:
            p4 = mk.build_mk_weight_w4(w)
            arm_mk = lambda: mk._gemm_call(x, p4, N)  # noqa: E731
            r = _rel(arm_mk(), ref)
            check(r <= E2M1_TOL, f"M={m} mk_w4 rel_err {r:.2e} <= {E2M1_TOL} (e2m1)")
            t = _graph_us(arm_mk, reps=args.iters)
            nb = p4[0].numel() + p4[1].numel() + x.numel() * 2 + m * N * 2
            print(f"{m:>4} {'mk_w4':<16}{t:>9.1f}{nb / t / 1e3:>7.0f}  {r:.2e}")
            arms.append(("mk_w4", t, nb))
        except Exception as e:
            print(f"{m:>4} {'mk_w4':<16}{'SKIP':>9}  -- lane refuses K={K}: {e!r}"[:110])

        name, t, _ = min(arms, key=lambda kv: kv[1])
        rows.append((m, name, t, arms[0][1]))

    print("\nverdict (stock vs best arm; W4 DRAM bound "
          f"{N * K // 2 / W4_GBPS / 1e3:.0f} us at {W4_GBPS:.0f} GB/s):")
    for m, name, t, ts in rows:
        prize = ts - t
        tag = (" -- best arm is AT the W4 stream bound; a new split-K kernel "
               "has nothing left to take"
               if t <= N * K // 2 / W4_GBPS / 1e3 * 1.05 else "")
        print(f"  M={m}: stock {ts:.1f} us vs best {name} {t:.1f} us -> "
              f"{prize:.1f} us = {prize / (args.step_ms * 1000) * 100:.2f}% "
              f"of a {args.step_ms:.0f} ms step{tag}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
