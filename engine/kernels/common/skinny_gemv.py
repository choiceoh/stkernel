"""A BF16 x @ W.T for a handful of rows that reads W once (kernels, common).

A decode step multiplies 1-16 rows by weights it reads once a step, so such a product's floor is the weight's bytes
over the memory's bandwidth. cuBLAS does not reach it at every shape: on a GB10 its gemv for Qwen3.8's router
[513, 2560] ran 1.43x (2 rows) to 1.95x (8 rows) slower than this kernel's first form, and its sm80 WMMA GEMM for the
hyper-connection mixer's down projection [324, 10240] 1.12-1.23x (probes/engine_qwen38_gemv, 2026-09-19, CUDA graphs
interleaved beside production). `linear_rows` takes the shapes `CONFIGS` names and hands every other call to torch.mm --
the same product in BF16 with FP32 accumulation, rounded once; only the order of the sums differs.

One launch: the rows pad to 16 for a tensor-core dot, a program keeps one [16, BLOCK_N] FP32 accumulator and walks its
share of K in BLOCK_K tiles. A narrow output over a long K splits K into SPLIT programs a column block; each stores its
FP32 partial and counts itself in on the block's arrival word, and the last to arrive sums the partials in split order
(the same order every call: a result does not depend on which program came last), rounds once, and resets the word to
zero for the next call. The arrival words are one int32 buffer a device, zeroed once outside any capture (`prepare`);
calls on one stream are ordered, so they share it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS = 16
MAX_BLOCKS = 4096                  # arrival words a device: column blocks of the widest split output

# (N, K) of W -> (BLOCK_N, BLOCK_K, SPLIT, warps, stages); absent -> torch.mm
CONFIGS = {
    (513, 2560): (32, 256, 1, 4, 3),      # Qwen3.8's router and shared gate (512 experts + 1)
    (324, 10240): (16, 256, 8, 4, 3),     # Qwen3.8's mixer down + inject (rank 320 + hc 4)
    (320, 10240): (16, 256, 8, 4, 3),     # its closing mixer's down (no inject)
}

_LOCKS: "dict[torch.device, torch.Tensor]" = {}


@triton.jit
def _skinny_gemv_kernel(X, W, OUT, PART, LOCKS, M, N, K, sx, sw, so, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        SPLIT: tl.constexpr, FP32_DOT: tl.constexpr):
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, 16)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    span = tl.cdiv(tl.cdiv(K, SPLIT), BLOCK_K) * BLOCK_K         # whole tiles a split: none straddles two programs
    k0 = pid_k * span
    k1 = tl.minimum(k0 + span, K)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k in range(k0, k1, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        kmask = ks < k1
        x = tl.load(X + rows[:, None] * sx + ks[None, :], mask=(rows[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * sw + ks[None, :], mask=(cols[:, None] < N) & kmask[None, :], other=0.0)
        if FP32_DOT:                                              # the interpreter reads a BF16 dot's bits as integers
            x, w = x.to(tl.float32), w.to(tl.float32)
        acc += tl.dot(x, tl.trans(w))
    keep = (rows[:, None] < M) & (cols[None, :] < N)
    if SPLIT == 1:
        tl.store(OUT + rows[:, None] * so + cols[None, :], acc.to(OUT.dtype.element_ty), mask=keep)
    else:
        at = rows[:, None] * N + cols[None, :]
        tl.store(PART + pid_k * M * N + at, acc, mask=keep)
        arrived = tl.atomic_add(LOCKS + pid_n, 1, sem="acq_rel")  # the partial above is visible to whoever sums
        if arrived == SPLIT - 1:
            total = tl.zeros((16, BLOCK_N), dtype=tl.float32)
            for s in range(SPLIT):
                total += tl.load(PART + s * M * N + at, mask=keep, other=0.0, cache_modifier=".cg")
            tl.store(OUT + rows[:, None] * so + cols[None, :], total.to(OUT.dtype.element_ty), mask=keep)
            tl.atomic_xchg(LOCKS + pid_n, 0)


def prepare(device) -> torch.Tensor:
    """The device's arrival words, zeroed -- before a graph captures a split call (a buffer made inside a capture
    would live in that graph's pool)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    locks = _LOCKS.get(device)
    if locks is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("skinny_gemv.prepare(device) before a graph captures a split GEMV")
        locks = _LOCKS[device] = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=device)
    return locks


def gemv(x: torch.Tensor, w: torch.Tensor, cfg: "tuple[int, int, int, int, int]",
         out: "torch.Tensor | None" = None) -> torch.Tensor:
    """x [M <= 16, K] BF16 @ w [N, K].T -> [M, N] BF16 under `cfg` (BLOCK_N, BLOCK_K, SPLIT, warps, stages); the last
    dimension of each operand contiguous."""
    m, k = x.shape
    n = w.shape[0]
    block_n, block_k, split, warps, stages = cfg
    if not 1 <= m <= MAX_ROWS or w.shape[1] != k:
        raise ValueError(f"skinny_gemv takes 1..{MAX_ROWS} rows of W's K: x {tuple(x.shape)}, w {tuple(w.shape)}")
    if out is None:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    blocks = triton.cdiv(n, block_n)
    if split > 1:
        if blocks > MAX_BLOCKS:
            raise ValueError(f"a split GEMV of {blocks} column blocks is past the {MAX_BLOCKS} arrival words")
        locks = prepare(x.device)
        partial = torch.empty(split, m, n, dtype=torch.float32, device=x.device)
    else:
        locks = partial = out
    _skinny_gemv_kernel[(blocks, split)](x, w, out, partial, locks, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                                         BLOCK_N=block_n, BLOCK_K=block_k, SPLIT=split, FP32_DOT=not x.is_cuda,
                                         num_warps=warps, num_stages=stages)
    return out


def linear_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T in BF16: this kernel for 1..16 rows of a CUDA shape `CONFIGS` names, torch.mm otherwise."""
    cfg = CONFIGS.get(tuple(w.shape))
    if (cfg is None or not x.is_cuda or not 1 <= x.shape[0] <= MAX_ROWS or x.dtype != torch.bfloat16
            or w.dtype != torch.bfloat16 or x.stride(-1) != 1 or w.stride(-1) != 1):
        return torch.mm(x, w.t())
    return gemv(x, w, cfg)


def qualify(device, rows=(1, 4, 16)) -> dict:
    """D3 before a boot serves: every shape in CONFIGS on `device`, held to the FP32 product -> {shape: largest error
    over the largest magnitude}; raises past 2^-6 (two BF16 steps), which a wrong tile, split or sum passes by orders.
    Twice a shape, so a split's arrival words are seen reset."""
    prepare(device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    out = {}
    for (n, k), cfg in CONFIGS.items():
        w = (torch.randn(n, k, generator=gen) * 0.02).to(torch.bfloat16).to(device)
        worst = 0.0
        for m in rows:
            x = torch.randn(m, k, generator=gen).to(torch.bfloat16).to(device)
            ref = x.float() @ w.float().t()
            for _ in range(2):
                got = gemv(x, w, cfg).float()
                worst = max(worst, float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)))
        if not worst <= 2.0 ** -6:
            raise RuntimeError(f"skinny_gemv [{n}, {k}] {cfg}: error {worst:.2e} of the largest magnitude")
        out[f"{n}x{k}"] = round(worst, 6)
    return out


__all__ = ["CONFIGS", "MAX_ROWS", "gemv", "linear_rows", "prepare", "qualify"]
