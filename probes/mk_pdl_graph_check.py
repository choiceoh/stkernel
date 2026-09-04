#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Programmatic dependent launch (PDL) of the megakernel segments UNDER CUDA
graph capture -- the form serving actually replays.

The bench probe measures PDL on a stream (x2 pairs, MEASUREMENTS 2차: 17-19
pct less per launch); the drafter probe captured ONE MK op with PDL on and
replayed it bitwise. Neither captured a CHAIN of programmatic launches, and
that chain is what VLLM_GLM53_MK_PDL=1 puts into every decode graph: each MK
kernel triggers its dependents at entry and `griddepcontrol.wait`s before it
reads the previous kernel's output, so a chain is correct only if every
pre-wait read is of static data (weights) -- which is what this checks, on
the real kernels, in the real launch form.

Gates (one process = one PDL state, the .so caches the env once):
  chain    gemm -> gemm -> gemm and mhc -> gemm, eager == graph replay bitwise
  replay   two replays of the same graph agree bitwise (monotonic barriers)
  timing   per-launch us of a 24-launch graph (8 rounds x 3 different packs,
           DRAM-cold by rotation) -- run once with VLLM_GLM53_MK_PDL=1 and once
           with =0 and read the two rows side by side

Run inside the glm53 image with the composed overlays mounted:
    bash probes/run_mk_probe.sh probes/mk_pdl_graph_check.py
    VLLM_GLM53_MK_PDL=0 bash probes/run_mk_probe.sh probes/mk_pdl_graph_check.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
os.environ.setdefault("VLLM_GLM53_MK_MHC", "1")
os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
os.environ.setdefault("VLLM_GLM53_MK_KDA", "0")
os.environ.setdefault("VLLM_GLM53_MK_MLA", "0")
os.environ.setdefault("VLLM_GLM53_MK_PDL", "1")
sys.path.insert(0, os.environ.get("MK_PKG_PATH",
                                  "/usr/local/lib/python3.12/dist-packages"))

import torch  # noqa: E402

DEV = "cuda"
_FAIL = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        _FAIL.append(msg)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(
        torch.equal(a.view(torch.int16), b.view(torch.int16)))


def main() -> int:
    from vllm.model_executor.layers import glm53_megakernel as mk

    torch.cuda.init()
    ext = mk._build()
    pdl = bool(ext.pdl_enabled()) if hasattr(ext, "pdl_enabled") else None
    env_pdl = os.environ.get("VLLM_GLM53_MK_PDL")
    print(f"env VLLM_GLM53_MK_PDL={env_pdl} (.so reports {pdl})")
    mk.maybe_arm()
    check(mk._ARMED["gemm"], "MK-GEMM armed")
    check(mk._ARMED["mhc"], "MK-MHC armed")
    if not (mk._ARMED["gemm"] and mk._ARMED["mhc"]):
        return 1

    torch.manual_seed(0)
    T = 8
    # --- gemm -> gemm -> gemm: n = k = 4096 so each output feeds the next
    packs = [mk.build_mk_weight_w4(
        torch.randn(4096, 4096, dtype=torch.bfloat16, device=DEV) * 0.05)
        for _ in range(3)]
    x0 = torch.randn(T, 4096, dtype=torch.bfloat16, device=DEV)

    def chain_gemm(x):
        y = x
        for p in packs:
            y = mk._gemm_call(y, p, 4096)
        return y

    # --- mhc -> gemm: the MHC tail writes layer_input, the GEMM reads it
    H, HC, NOUT = mk.HIDDEN, mk.HC, mk.NOUT
    res = torch.randn(T, HC, H, dtype=torch.bfloat16, device=DEV) * 0.1
    pm = torch.rand(T, HC, dtype=torch.float32, device=DEV)
    cm = torch.rand(T, HC * HC, dtype=torch.float32, device=DEV).contiguous()
    fn = torch.randn(NOUT, HC * H, dtype=torch.float32, device=DEV) * 0.02
    hc_scale = torch.ones(3, dtype=torch.float32, device=DEV)
    hc_base = torch.zeros(NOUT, dtype=torch.float32, device=DEV)
    nw = torch.randn(H, dtype=torch.bfloat16, device=DEV)
    p_in = mk.build_mk_weight_w4(
        torch.randn(6416, 4096, dtype=torch.bfloat16, device=DEV) * 0.05)

    def chain_mhc(x):
        rc, pmc, cmc, li = mk._mhc_call(
            x, res, pm, cm, fn, hc_scale, hc_base, nw, T, 1e-6, 1e-6, 1e-6,
            1.0, 1e-6, mk.SINKHORN_SERVED)
        return mk._gemm_call(li, p_in, 6416), rc, pmc, cmc

    # eager references (warm the arm first)
    for _ in range(3):
        chain_gemm(x0)
        chain_mhc(x0)
    torch.cuda.synchronize()
    ref_g = chain_gemm(x0)
    ref_m = chain_mhc(x0)
    torch.cuda.synchronize()

    # capture both chains into one graph, replay twice
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            chain_gemm(x0)
            chain_mhc(x0)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=st):
            out_g = chain_gemm(x0)
            out_m = chain_mhc(x0)
    except Exception as e:  # noqa: BLE001
        check(False, f"CUDA-graph capture of the MK chains raised: {e!r}"[:300])
        return 1
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    first = (out_g.clone(), [t.clone() for t in out_m])
    g.replay()
    torch.cuda.synchronize()
    check(_same(first[0], ref_g), "gemm->gemm->gemm: graph replay == eager bitwise")
    check(all(_same(a, b) for a, b in zip(first[1], ref_m)),
          "mhc->gemm: graph replay == eager bitwise (out, residual, mixes)")
    check(_same(out_g, first[0]) and all(_same(a, b) for a, b in zip(out_m, first[1])),
          "second replay == first replay bitwise (monotonic barriers)")

    # --- timing: a 24-launch graph, DRAM-cold by rotation (8 x 3 packs of
    # 8.4 MB = 200 MB per replay, L2 is 24 MB), per-launch us
    rounds = 8
    rot = [mk.build_mk_weight_w4(
        torch.randn(4096, 4096, dtype=torch.bfloat16, device=DEV) * 0.05)
        for _ in range(rounds * 3)]
    xs = torch.randn(T, 4096, dtype=torch.bfloat16, device=DEV)

    def many():
        y = xs
        for p in rot:
            y = mk._gemm_call(y, p, 4096)
        return y

    with torch.cuda.stream(st):
        for _ in range(3):
            many()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, stream=st):
        many()
    torch.cuda.synchronize()
    for _ in range(5):
        g2.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    N = 40
    s.record()
    for _ in range(N):
        g2.replay()
    e.record()
    torch.cuda.synchronize()
    per = s.elapsed_time(e) * 1e3 / (N * len(rot))
    nb = rot[0][0].numel() + rot[0][1].numel()
    print(f"timing pdl={env_pdl}: {len(rot)}-launch graph, n=k=4096, "
          f"{per:.1f} us/launch, {nb / per / 1e3:.0f} GB/s over the pack bytes")
    print("VERDICT:", "PASS" if not _FAIL else f"FAIL ({len(_FAIL)})")
    return 0 if not _FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
