# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
"""Single-sequence specialization of causal_conv.py's forward recurrence.

Keep its multiply/add order and SiLU expression; remove the batching maps and
mutable two-row state table. Each CTA owns one time tile and channel tile.
Only the last time tile publishes the final history, into separate storage.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _single_conv(X, W, S, Y, F, T: tl.constexpr, C: tl.constexpr,
                 XS: tl.constexpr, XC: tl.constexpr, WS: tl.constexpr, WC: tl.constexpr,
                 SS: tl.constexpr, SC: tl.constexpr, K: tl.constexpr,
                 HAS_STATE: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    start = tl.program_id(1) * BT
    history = ()
    weights = ()
    for j in tl.static_range(K):
        weights += (tl.load(W + c * WS + j * WC, c < C, other=0.),)
        if j < K - 1:
            pos = start - (K - 1) + j
            value = tl.load(X + pos * XS + c * XC, (c < C) & (pos >= 0), other=0.)
            if HAS_STATE:
                old = tl.load(S + c * SS + j * SC, (c < C) & (pos < 0), other=0.)
                # The old adapter assigned the history into an x.dtype table.
                value = tl.where(pos < 0, old.to(X.dtype.element_ty), value)
            history += (value,)
    for token in range(start, tl.minimum(start + BT, T)):
        current = tl.load(X + token * XS + c * XC, c < C, other=0.)
        values = history + (current,)
        acc = tl.zeros((BC,), tl.float32)
        for j in tl.static_range(K):
            acc += values[j] * weights[j]
        history = values[1:]
        acc = acc / (1 + tl.exp(-acc))
        tl.store(Y + token * C + c, acc, c < C)
    if start + BT >= T:
        for j in tl.static_range(K - 1):
            tl.store(F + c * (K - 1) + j, history[j], c < C)


def causal_conv1d_single(x, weight, initial_state=None):
    """Return SiLU depthwise conv [T,C] and independent history [C,K-1].

    Accept projection slices without making x contiguous. History is rounded
    to x.dtype, as in the former served adapter, and inputs are never mutated.
    """
    floating = (torch.float16, torch.bfloat16, torch.float32)
    if (x.ndim != 2 or weight.ndim != 2 or not x.is_cuda or
            x.dtype not in floating or weight.dtype not in floating or
            weight.device != x.device or weight.shape[0] != x.shape[1] or
            weight.shape[1] not in (2, 3, 4) or x.shape[1] == 0):
        raise ValueError("conv requires CUDA floating x [T,C], weight [C,K], C>0, K=2/3/4")
    t, c = x.shape
    k = weight.shape[1]
    if initial_state is not None and (
            initial_state.shape != (c, k - 1) or initial_state.device != x.device or
            initial_state.dtype not in floating):
        raise ValueError("conv history must be floating [C,K-1] on the input device")
    if t == 0:
        state = (torch.zeros(c, k - 1, device=x.device, dtype=x.dtype) if initial_state is None
                 else initial_state.to(x.dtype).clone())
        return torch.empty_like(x), state
    out = torch.empty((t, c), device=x.device, dtype=x.dtype)
    final = torch.empty((c, k - 1), device=x.device, dtype=x.dtype)
    ss, sc = (0, 0) if initial_state is None else initial_state.stride()
    # 6144 TP4 channels / 128 = 48 CTAs, covering the GB10's 48 SMs even at T=1.
    # This layout needs no shared memory or register spills on the pinned image.
    _single_conv[(triton.cdiv(c, 128), triton.cdiv(t, 8))](
        x, weight, initial_state, out, final, t, c, *x.stride(), *weight.stride(), ss, sc,
        k, initial_state is not None, 128, 8, num_warps=4, num_stages=2)
    return out, final
