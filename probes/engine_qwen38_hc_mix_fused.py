#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 carry H2, measured before it is built: can ONE launch stand in for a site's gates + up GEMM + mix_mean?

engine/kernels/gated_residual.mix is four launches a site: cuBLAS's BF16 GEMM for down(+inject), a Triton launch for
the gates, cuBLAS's GEMM for up, a Triton launch for the streams' mean. H2 (and H1, the same idea for the down GEMM)
would fold a GEMM into a Triton launch to save launches -- 200 a step by the carry table's count. This probe holds a
prototype of that launch to the lane, byte for byte, and times both through a CUDA graph at the rows a captured step
had when it ran (SPEC_K was 1: two tokens a row, so 2, 4, 6, 8 rows of 10,240 channels; the profile serves 3 since).

What it found on an RTX 5050 (sm_120, triton 3.6, torch 2.11; 2026-09-18), which is why H2 was not built:

    the lane at 2..8 rows   28 us a site, of which the two GEMMs alone are 15.0 + 12.8 us: each reads its 6.5 MB of
                            weights once for every row, at about the card's memory bandwidth. The two Triton launches
                            H2 removes are under 1 us together inside a graph.
    the prototype           byte for byte the lane's output, 26 us at ONE row and 34 / 55 / 92 us at 2 / 4 / 8 rows: a
                            Triton matvec reads the weights again for each row.

    one read for every row  a second prototype with the step's rows in one program a channel block (what cuBLAS does)
                            was slower still, 40..107 us with a broadcast sum and a flat 188 us with tl.dot: a GEMM
                            Triton generates does not reach cuBLAS at this shape, whatever the rows. It is not kept.

So on that card the site is its two GEMMs' bytes, and what moves it is fewer bytes (H6: the mixers on block-scaled FP8),
not fewer launches; and a fold written in Triton loses to the lane on any card, because its GEMM does.

What one desktop card cannot say is what a launch costs on a GB10. There it was 1 us; the GLM campaign's ledger has a
launch's fixed cost inside a captured graph at 15-25 us, about 5 of them removable (MEASUREMENTS_ARCHIVE.md, the dense
GEMM entries). `headroom` below -- the lane less its two GEMMs alone -- is that number for this site: all a fold of
any kind could recover. If a GB10 reports tens of microseconds there, the fold is worth a CUDA segment of the kind
GLM's MK_SEG_MHC is, not a Triton launch. Run it through the single-GPU lane:

    bash probes/run_engine_probe.sh probes/engine_qwen38_hc_mix_fused.py

What the GB10 said (2026-09-19, measurements/qwen38_hc_headroom_20260919): headroom 2.0-2.8 us at 1..8 rows -- the
ledger's fixed cost is not at this site, and folding launches alone is worth 0.2-0.3 ms a step. READ ONLY THE HEADROOM:
this probe replays ONE 13.2 MB weight pair, so its GEMMs come out at 423 GB/s, above the card's 273 -- cache reads,
where a served step reads a hundred sites' distinct weights (the same two products are 61-66 us with the weights
rotated past 64 MB). And "a fold written in Triton loses on any card" above is true of this prototype's shape only: a
GEMV that reads each weight tile once for all rows beats cuBLAS on a GB10 and carries the fold (the skinny-GEMV work).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.kernels import gated_residual as hcr  # noqa: E402
# The report's writer lives under probes/ because that is what the single-GPU lane ships to its box (engine/, probes/,
# tests/ -- not bench/). Imported from bench/ this probe died there on this line (2026-09-19), and behind a guard its
# report was silently dropped on the one lane it is for.
from probes.probe_report import write_report  # noqa: E402

HC, HIDDEN, RANK = 4, 2560, 320                     # Qwen3.8's streams, hidden width and mixer rank (hc_lowrank)
ROWS = (1, 2, 4, 6, 8)                              # the rows this ran at: max_seqs 4 x (SPEC_K 1 + 1), 2026-09-18


@triton.jit
def _mix_fused(DI, UP, NORMED, OUT, sD, sN, sO, HC_F, R: tl.constexpr, BR: tl.constexpr, HID: tl.constexpr,
               HC: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    m = d < HID
    i = tl.arange(0, BR)
    mi = i < R
    x = tl.load(DI + r * sD + i, mask=mi, other=0.0)
    q = (x.to(tl.float32) / HC_F).to(x.dtype).to(tl.float32)
    gate = (q * tl.sigmoid(q)).to(x.dtype).to(tl.float32)              # silu rounds to BF16, as gated_residual._gates
    acc = tl.zeros([BD], dtype=tl.float32)
    for s in tl.static_range(HC):
        off = s * HID + d
        u = tl.load(UP + off[:, None] * R + i[None, :], mask=m[:, None] & mi[None, :], other=0.0).to(tl.float32)
        w = tl.sum(u * gate[None, :], axis=1).to(x.dtype)               # the up GEMM's BF16 output
        g = tl.sigmoid(w.to(tl.float32)).to(x.dtype).to(tl.float32)
        n = tl.load(NORMED + r * sN + off, mask=m, other=0.0).to(tl.float32)
        acc += (g * n).to(x.dtype).to(tl.float32)
    tl.store(OUT + r * sO + d, (acc / HC_F).to(OUT.dtype.element_ty), mask=m)


def fused_mix(normed, down, up, *, block: int, warps: int):
    """The closing mixer's form (no injection): the down GEMM, then gates + up + mean in one launch."""
    rows = normed.shape[0]
    di = torch.mm(normed, down.t())
    mixed = torch.empty(rows, HIDDEN, device=normed.device, dtype=normed.dtype)
    _mix_fused[(rows, triton.cdiv(HIDDEN, block))](di, up, normed, mixed, di.stride(0), normed.stride(0),
                                                   mixed.stride(0), float(HC), R=RANK,
                                                   BR=triton.next_power_of_2(RANK), HID=HIDDEN, HC=HC, BD=block,
                                                   num_warps=warps)
    return mixed


def graph_us(fn, replays: int) -> "tuple[float, torch.Tensor]":
    """Microseconds a replay, best of seven rounds: a captured step pays the device's time, not the host's."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / replays)
    return best * 1e6, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replays", type=int, default=200)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("needs a CUDA device: the question is a device's time inside a captured graph")
        return 2
    torch.manual_seed(0)
    dev, width = "cuda", HC * HIDDEN
    down = (torch.randn(RANK, width, device=dev) * 0.02).to(torch.bfloat16)
    up = (torch.randn(width, RANK, device=dev) * 0.02).to(torch.bfloat16)
    metrics, proof, rows_out = {}, {}, []
    for rows in ROWS:
        normed = torch.randn(rows, width, device=dev).to(torch.bfloat16)
        lane, (want, _) = graph_us(lambda: hcr.mix(normed, down, up, HC, inject=False), a.replays)
        gates = torch.randn(rows, RANK, device=dev).to(torch.bfloat16)
        gemms = (graph_us(lambda: torch.mm(normed, down.t()), a.replays)[0]
                 + graph_us(lambda: torch.mm(gates, up.t()), a.replays)[0])
        best, equal = float("inf"), True
        for block, warps in ((32, 4), (64, 4), (64, 8), (128, 4)):
            us, got = graph_us(lambda: fused_mix(normed, down, up, block=block, warps=warps), a.replays)
            best, equal = min(best, us), equal and bool(torch.equal(got, want))
        metrics[f"lane_us_rows{rows}"], metrics[f"fused_us_rows{rows}"] = round(lane, 2), round(best, 2)
        metrics[f"gemms_us_rows{rows}"] = round(gemms, 2)
        metrics[f"headroom_us_rows{rows}"] = round(lane - gemms, 2)
        proof[f"byte_equal_rows{rows}"] = equal
        rows_out.append((rows, lane, gemms, best, equal))
    print(f"{torch.cuda.get_device_name()}: a site's mixer, microseconds a replay")
    print("  rows   lane   its two GEMMs   headroom   fused prototype   byte-equal")
    for rows, lane, gemms, best, equal in rows_out:
        print(f"  {rows:4d} {lane:6.1f} {gemms:15.1f} {lane - gemms:10.1f} {best:17.1f}   {equal}")
    pays = [rows for rows, lane, _, best, _ in rows_out if rows >= 2 and best < lane]
    metrics["rows_where_fused_wins"] = len(pays)
    print(f"  the prototype beats the lane at {pays or 'no'} captured row count(s) (2..8)")
    write_report(metrics, proof, len(ROWS), torch.cuda.get_device_name())
    print(json.dumps(metrics))
    return 0 if all(proof.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
