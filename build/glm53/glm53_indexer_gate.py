# SPDX-License-Identifier: Apache-2.0
"""deneb fork: the sparse indexer's fp32 head-gate projection as a split-K
Triton kernel (VLLM_GLM53_INDEXER_GATE_SPLITK).

`Indexer.forward` computes `weights = torch.mm(hidden_states.float(),
self._wp_fp32)` -- an [M, 4096] x [4096, 16] fp32 product per full-attention
layer, kept in fp32 on purpose (bf16 head-gates flip near-tie pool rankings).
cuBLAS answers that shape with a two-block `gemmSN` kernel: 47 us on an idle
GB10 (86 us under CUPTI in the 2026-09-01 serving trace), eleven times per
decode step, for 256 KB of weights. A split-K kernel that hands each
(row, K-slice) to one program and reduces with fp32 atomics runs the same
product in 7 us. Both accumulate in fp32; only the summation order differs
(measured max |diff| 3e-5 on values of magnitude ~64, i.e. ~5e-7 relative,
0 top-1 rank flips over the offline trials) -- not bit-exact, so this stays
an opt-in behind a numerics bracket.

At M > 16 (C >= 3 verify batches, prefill) cuBLAS is already fast, so the
kernel is used only for M <= 16; larger M keeps torch.mm.
"""
from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

ENV = "VLLM_GLM53_INDEXER_GATE_SPLITK"
MAX_M = 16
_SPLIT = 8
_BLOCK_K = 128


def gate_splitk_enabled() -> bool:
    """Exact opt-in: only the string "1" arms; anything else is stock."""
    return os.environ.get(ENV, "").strip() == "1"


@triton.jit
def _gate_splitk_kernel(x_ptr, w_ptr, out_ptr, K, N, sxm, swk, som,
                        SPLIT: tl.constexpr, BLOCK_K: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0)
    s = tl.program_id(1)
    kper = K // SPLIT
    k0 = s * kper
    offs_n = tl.arange(0, BN)
    nmask = offs_n < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for k in range(0, kper, BLOCK_K):
        offs_k = k0 + k + tl.arange(0, BLOCK_K)
        xv = tl.load(x_ptr + m * sxm + offs_k)
        wv = tl.load(w_ptr + offs_k[:, None] * swk + offs_n[None, :], mask=nmask[None, :], other=0.0)
        acc += tl.sum(xv[:, None] * wv, axis=0)
    tl.atomic_add(out_ptr + m * som + offs_n, acc, mask=nmask)


def head_gate_splitk(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 [M, K] @ [K, N] for N <= 16, K a multiple of SPLIT*BLOCK_K.

    x may be bf16 (cast to fp32 here, as the stock `.float()` does); w is the
    stock `_wp_fp32` ([K, N], fp32, contiguous)."""
    xf = x.float()
    M, K = xf.shape
    N = w.shape[1]
    out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    _gate_splitk_kernel[(M, _SPLIT)](
        xf, w, out, K, N, xf.stride(0), w.stride(0), out.stride(0),
        SPLIT=_SPLIT, BLOCK_K=_BLOCK_K, BN=16, num_warps=4)
    return out


def splitk_applicable(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (x.shape[0] <= MAX_M and w.dtype == torch.float32 and w.is_contiguous()
            and w.shape[1] <= 16 and w.shape[0] % (_SPLIT * _BLOCK_K) == 0)


def head_gate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The stock `torch.mm(x.float(), w)` unless the knob is on and the shape
    is the small-M decode one."""
    if gate_splitk_enabled() and splitk_applicable(x, w):
        return head_gate_splitk(x, w)
    return torch.mm(x.float(), w)
