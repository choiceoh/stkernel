"""Block-scaled FP8 x @ W.T for a decode step's handful of rows, in one launch (kernels, dense).

deep_gemm's recipe -- the rows quantized to e4m3 by 128-wide K blocks with power-of-two scales (fp8.quantize), the
weight e4m3 with power-of-two scales a 128 x 128 block, each K block's FP32 partial times the row's and the weight
block's scales, summed over the blocks -- with a step's rows padded to 16 for one FP8 tensor-core dot a K block, each
program owning BLOCK_N weight rows over the whole K. The inputs and scales are deep_gemm's own; only the order of the
FP32 sums differs, so an output moves by a BF16 step at most.

Qwen3.8's vocabulary head is the shape it is for: 62,080 x 2,560 a rank, 159 MB, read once by the verify step and once
by each of a K=3 draft chain's three steps. On a GB10 with production idle beside it (probes/engine_qwen38_head,
q38head-0919c, 1/2/4/8/16 rows) deep_gemm's call -- its sm120 GEMM and the scale conversions it runs every call -- took
873-893 us (178-182 GB/s); this kernel at its tile 688-709 us, 3-5% over a pure read of the same bytes (669-673 us);
max difference from deep_gemm 0.0037 of the largest logit, argmax identical on every row.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS = 16


def tile(rows: int) -> "tuple[int, int, int]":
    """(BLOCK_N, warps, stages): 32 weight rows a program, four stages in flight -- the fastest of five tiles at every
    row count from 1 to 16 (q38head-0919c; the others 1-5% behind)."""
    return (32, 4, 4)


@triton.jit
def _fp8_rows(XQ, XS, WQ, WS, OUT, M, N, so, K: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    live, cm = rows < M, cols < N
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for kb in range(K // 128):
        ks = kb * 128 + tl.arange(0, 128)
        xq = tl.load(XQ + rows[:, None] * K + ks[None, :], mask=live[:, None], other=0.0)
        wq = tl.load(WQ + cols[:, None] * K + ks[None, :], mask=cm[:, None], other=0.0)
        xs = tl.load(XS + rows * (K // 128) + kb, mask=live, other=0.0)
        ws = tl.load(WS + (cols // 128) * (K // 128) + kb, mask=cm, other=0.0)
        acc += tl.dot(xq, tl.trans(wq)) * xs[:, None] * ws[None, :]
    tl.store(OUT + rows[:, None] * so + cols[None, :], acc.to(OUT.dtype.element_ty), mask=live[:, None] & cm[None, :])


@triton.jit
def _w8a16_rows(X, WQ, WS, OUT, M, N, so, K: tl.constexpr, BLOCK_N: tl.constexpr, FP32_DOT: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    live, cm = rows < M, cols < N
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for kb in range(K // 128):
        ks = kb * 128 + tl.arange(0, 128)
        x = tl.load(X + rows[:, None] * K + ks[None, :], mask=live[:, None], other=0.0)
        w = tl.load(WQ + cols[:, None] * K + ks[None, :], mask=cm[:, None], other=0.0).to(tl.bfloat16)
        if FP32_DOT:                                             # the interpreter reads a BF16 dot's bits as integers
            x, w = x.to(tl.float32), w.to(tl.float32)
        ws = tl.load(WS + (cols // 128) * (K // 128) + kb, mask=cm, other=0.0)
        acc += tl.dot(x, tl.trans(w)) * ws[None, :]
    tl.store(OUT + rows[:, None] * so + cols[None, :], acc.to(OUT.dtype.element_ty), mask=live[:, None] & cm[None, :])


def project_bf16(x: torch.Tensor, weight: "tuple[torch.Tensor, torch.Tensor]", *,
                 out: "torch.Tensor | None" = None) -> torch.Tensor:
    """x [M <= 16, K] BF16 by weight (wq [N, K] e4m3, ws [N/128, K/128] fp32) -> [M, N] BF16, the rows NOT quantised: the
    e4m3 weight widened to BF16 exactly, a BF16 tensor-core dot a 128-wide K block, times the weight block's scale.
    The same bytes as `project` (the weight is the read) and one launch fewer (no activation quantisation); the product
    is the weight's alone, not the weight's and a block-128 FP8 rounding of the rows'. What a drafter's argmax reads
    (net.draft_tokens): the operator's rule of 2026-09-19 -- precision where it costs nothing and moves acceptance."""
    wq, ws = weight
    m, k = x.shape
    n = wq.shape[0]
    if (not 1 <= m <= MAX_ROWS or k % 128 or wq.shape[1] != k or n % 128 or tuple(ws.shape) != (n // 128, k // 128)
            or x.dtype != torch.bfloat16 or wq.dtype != torch.float8_e4m3fn or not x.is_contiguous()
            or not wq.is_contiguous() or not ws.is_contiguous()):
        raise ValueError(f"fp8_rows.project_bf16 takes 1..{MAX_ROWS} contiguous BF16 rows of the weight's K: x "
                         f"{tuple(x.shape)}, weight {tuple(wq.shape)}")
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    block_n, warps, stages = tile(m)
    _w8a16_rows[(triton.cdiv(n, block_n),)](x, wq, ws, out, m, n, out.stride(0), K=k, BLOCK_N=block_n,
                                            FP32_DOT=not x.is_cuda, num_warps=warps, num_stages=stages)
    return out


def project(q: torch.Tensor, scale: torch.Tensor, weight: "tuple[torch.Tensor, torch.Tensor]", *,
            out: "torch.Tensor | None" = None) -> torch.Tensor:
    """(q [M <= 16, K] e4m3, scale [M, K/128] fp32) by weight (wq [N, K] e4m3, ws [N/128, K/128] fp32) -> [M, N] BF16."""
    wq, ws = weight
    m, k = q.shape
    n = wq.shape[0]
    if (not 1 <= m <= MAX_ROWS or k % 128 or wq.shape[1] != k or n % 128 or tuple(ws.shape) != (n // 128, k // 128)
            or tuple(scale.shape) != (m, k // 128) or q.dtype != torch.float8_e4m3fn or wq.dtype != q.dtype
            or not q.is_contiguous() or not wq.is_contiguous() or not scale.is_contiguous() or not ws.is_contiguous()):
        raise ValueError(f"fp8_rows takes 1..{MAX_ROWS} block-128 FP8 rows of the weight's K: q {tuple(q.shape)}, "
                         f"weight {tuple(wq.shape)}")
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=q.device)
    elif tuple(out.shape) != (m, n) or out.dtype != torch.bfloat16 or out.stride(1) != 1:
        raise ValueError("fp8_rows writes packed BF16 [M, N]")
    block_n, warps, stages = tile(m)
    _fp8_rows[(triton.cdiv(n, block_n),)](q, scale, wq, ws, out, m, n, out.stride(0), K=k, BLOCK_N=block_n,
                                          num_warps=warps, num_stages=stages)
    return out


def qualify(device, *, n: int = 1024, k: int = 2560, rows=(1, 4, 16)) -> dict:
    """D3 before a boot serves: the kernel held to the FP32 product of its own dequantized inputs (the recipe's exact
    value) on `device` -> {rows: largest error over the largest magnitude}; raises past 2^-7 (one BF16 step), which a
    wrong scale, block or tile passes by orders."""
    from .fp8 import quantize
    gen = torch.Generator(device="cpu").manual_seed(0)
    w = (torch.randn(n, k, generator=gen) * 0.02).to(torch.bfloat16).to(device)
    from deep_gemm import per_block_cast_to_fp8
    wq, ws = per_block_cast_to_fp8(w.float(), use_ue8m0=True)
    wf = wq.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    out = {}
    for m in rows:
        x = torch.randn(m, k, generator=gen).to(torch.bfloat16).to(device)
        q, s = quantize(x)
        ref = (q.float() * s.repeat_interleave(128, 1)) @ wf.t()
        got = project(q, s, (wq, ws)).float()
        err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
        if not err <= 2.0 ** -7:
            raise RuntimeError(f"fp8_rows at {m} rows: error {err:.2e} of the largest magnitude against its recipe")
        out[m] = round(err, 6)
    return out


__all__ = ["MAX_ROWS", "tile", "project", "project_bf16", "qualify"]
