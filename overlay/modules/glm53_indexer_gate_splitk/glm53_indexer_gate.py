# SPDX-License-Identifier: Apache-2.0
"""deneb fork: the sparse indexer's fp32 head-gate projection as a
deterministic split-K Triton path (VLLM_GLM53_INDEXER_GATE_SPLITK).

`Indexer.forward` computes `weights = torch.mm(hidden_states.float(),
self._wp_fp32)` -- an [M, K=4096] x [K, N=index_n_heads] fp32 product per
full-attention layer, kept in fp32 on purpose (bf16 head-gates flip near-tie
pool rankings). For the small M of decode cuBLAS answers this shape with a
two-block `gemmSN` kernel that leaves 46 of the 48 SMs idle.

This path splits K across programs: program s loads one [BLOCK_K, N] weight
slice once, multiplies it against every row of x (bf16 or fp32, cast to fp32
in registers -- the same exact conversion `.float()` does) and writes its
[BLOCK_M, N] partial; a second kernel sums the partials in a fixed order.
No atomics and no memset, so the result is bitwise reproducible run to run
and identical on every TP rank (the indexer is replicated per rank, and the
ranks' top-k pool selections must agree). It is NOT bit-exact with cuBLAS:
both accumulate in fp32, only the summation order differs. Measured numbers
live in the module README and MEASUREMENTS.md, not here.

Routing: the split-K path is taken only when the knob is exactly "1" and the
shape is the small-M decode one (`splitk_applicable`); everything else --
prefill, larger verify batches, unexpected layouts -- keeps `torch.mm`.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

ENV = "VLLM_GLM53_INDEXER_GATE_SPLITK"
MAX_M = 16          # rows per program tile; also the routing cap (C=1: 8, C=2: 16)
MAX_N = 32          # index_n_heads of the fleet checkpoint; BN = next pow2 >= N
_BLOCK_K = 128      # K per program: 4096 / 128 = 32 programs on 48 SMs

_ANNOUNCED: set = set()


def gate_splitk_enabled() -> bool:
    """Exact opt-in: only the string "1" arms; anything else is stock."""
    return os.environ.get(ENV, "").strip() == "1"


@triton.jit
def _gate_splitk_partial_kernel(x_ptr, w_ptr, part_ptr, M, N, K, sxm, swk,
                                BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BN: tl.constexpr):
    # program s owns K-slice [s*BLOCK_K, (s+1)*BLOCK_K): one weight tile,
    # every row of x; partial[s, m, :] for m < BLOCK_M (rows >= M are zero)
    s = tl.program_id(0)
    offs_k = s * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BN)
    nmask = offs_n < N
    w = tl.load(w_ptr + offs_k[:, None] * swk + offs_n[None, :], mask=nmask[None, :], other=0.0)
    for m in tl.static_range(BLOCK_M):
        kv = tl.where(m < M, K, 0)                     # rows past M load zeros
        xv = tl.load(x_ptr + m * sxm + offs_k, mask=offs_k < kv, other=0.0).to(tl.float32)
        acc = tl.sum(xv[:, None] * w, axis=0)          # [BN], fixed tree order
        tl.store(part_ptr + (s * BLOCK_M + m) * BN + offs_n, acc)


@triton.jit
def _gate_splitk_reduce_kernel(part_ptr, out_ptr, N, som,
                               SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BN: tl.constexpr):
    # out[m, :] = sum_s partial[s, m, :] in split order 0..SPLIT-1 (deterministic)
    m = tl.program_id(0)
    offs_n = tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + (s * BLOCK_M + m) * BN + offs_n)
    tl.store(out_ptr + m * som + offs_n, acc, mask=offs_n < N)


def _bn_for(n: int) -> int:
    return max(8, triton.next_power_of_2(n))


def head_gate_splitk(x: torch.Tensor, w: torch.Tensor, block_k: int = _BLOCK_K) -> torch.Tensor:
    """fp32 [M, K] @ [K, N] for the shape `splitk_applicable` admits.

    x: bf16 or fp32, 2-D, unit inner stride; w: the stock `_wp_fp32` ([K, N]
    fp32, contiguous). Returns a fresh contiguous fp32 [M, N] tensor."""
    M, K = x.shape
    N = w.shape[1]
    if K % block_k:
        # The partial kernel reads w rows unmasked over [0, split*block_k): a K
        # the block does not tile would drop the tail and answer quietly wrong.
        raise ValueError(
            f"head_gate_splitk: K={K} is not a multiple of block_k={block_k}")
    bn = _bn_for(N)
    split = K // block_k
    part = torch.empty(split * MAX_M, bn, device=x.device, dtype=torch.float32)
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    _gate_splitk_partial_kernel[(split,)](
        x, w, part, M, N, K, x.stride(0), w.stride(0),
        BLOCK_M=MAX_M, BLOCK_K=block_k, BN=bn, num_warps=4)
    _gate_splitk_reduce_kernel[(M,)](
        part, out, N, out.stride(0), SPLIT=split, BLOCK_M=MAX_M, BN=bn, num_warps=1)
    return out


def splitk_applicable(x: torch.Tensor, w: torch.Tensor) -> bool:
    """The decode head-gate shape and layout the kernels are written for."""
    return (x.dim() == 2 and w.dim() == 2
            and 0 < x.shape[0] <= MAX_M
            and x.shape[1] == w.shape[0]
            and x.stride(1) == 1
            and x.dtype in (torch.bfloat16, torch.float32)
            and w.dtype == torch.float32 and w.is_contiguous()
            and 0 < w.shape[1] <= MAX_N
            and w.shape[0] % _BLOCK_K == 0)


def head_gate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The stock `torch.mm(x.float(), w)` unless the knob is on and the shape
    is the small-M decode one. Announces the routing once per shape so an
    armed boot that never takes the split-K path is visible in the log."""
    if gate_splitk_enabled():
        ok = splitk_applicable(x, w)
        key = (tuple(x.shape), tuple(w.shape), x.dtype, ok)
        if key not in _ANNOUNCED:
            _ANNOUNCED.add(key)
            (logger.info if ok else logger.warning)(
                "[indexer-gate] %s=1: x%s %s @ w%s -> %s", ENV, tuple(x.shape),
                x.dtype, tuple(w.shape), "split-K" if ok else "stock torch.mm (shape not admitted)")
        if ok:
            return head_gate_splitk(x, w)
    return torch.mm(x.float(), w)
