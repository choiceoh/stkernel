"""Experimental KDA verification with state materialization after acceptance.

Verification leaves the canonical ring untouched and records FP32 update
factors. Commit replays only the accepted updates into the accepted final
row and any crossed prefix boundary. Callers must commit on the same stream
before another verification, snapshot, or consumer of this slot. This is a
kernel experiment, not a serving lane: the ordinary ring remains the default.
"""
import torch
import triton
import triton.language as tl

from .ring import _recurrent


def verify(q, k, v, g, beta, a_log, g_bias, ring, slot, context, lower_bound):
    return _recurrent(q, k, v, g, beta, a_log, g_bias, ring, slot, context,
                      lower_bound, deferred=True)


@triton.jit
def _commit(KEY, DECAY, UPDATE, RING, SLOT, CONTEXT, COUNT,
            T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
            R: tl.constexpr, SLOT_STRIDE: tl.constexpr,
            BK: tl.constexpr, BV: tl.constexpr, BLOCK: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    slot, context = tl.load(SLOT).to(tl.int64), tl.load(CONTEXT).to(tl.int64)
    count = tl.load(COUNT).to(tl.int64)
    # Zero accepted positions (finished/padded row) must not touch its slot.
    # Device metadata is trusted only within the declared commit contract.
    if count <= 0 or count > T:
        return
    kk, vv = tl.arange(0, BK), tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < V) & (kk[None, :] < K)
    base = slot * SLOT_STRIDE + head * K * V + kk[None, :] * V + vv[:, None]
    state = tl.load(RING + base + (tl.maximum(context - 1, 0) % R) * H * K * V,
                    mask & (context > 0), other=0).to(tl.float32)
    for i in range(count):
        offset = (i * H + head)
        key = tl.load(KEY + offset * K + kk, kk < K, other=0)
        decay = tl.load(DECAY + offset * K + kk, kk < K, other=0)
        update = tl.load(UPDATE + offset * V + vv, vv < V, other=0)
        # The verifier rounds decay multiplication before the dot product
        # which produces `update`; preserve that FP32 rounding here too.
        state = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                         args=[state, decay[None, :]], dtype=tl.float32,
                                         is_pure=True, pack=1)
        state += update[:, None] * key[None, :]
        position = context + i
        if i == count - 1 or (position + 1) % BLOCK == 0:
            tl.store(RING + base + (position % R) * H * K * V, state, mask)


def commit(factors, ring, slot, context, count, *, block):
    """Materialize a device-counted accepted prefix, including snapshot rows.

All three metadata inputs are CUDA integer singletons. Their owner guarantees
0 <= slot < slots, context >= 0, and 0 <= count <= verification width. A slot
may have only one uncommitted verification. Other ring rows remain unchanged.
"""
    if len(factors) != 3 or ring.ndim != 5 or not ring.is_cuda or ring.dtype != torch.float32:
        raise ValueError("deferred commit requires three factors and a CUDA FP32 ring")
    keys, decay, updates = factors
    if keys.ndim != 3:
        raise ValueError("deferred keys must have shape [T,H,K]")
    t, h, k = keys.shape
    v = ring.shape[-1]
    if (not 1 <= t <= ring.shape[1] or ring.shape[2:] != (h, k, v)
            or decay.shape != keys.shape or updates.shape != (t, h, v)
            or min(h, k, v) <= 0 or type(block) is not int or block <= 0
            or ring.stride()[1:] != (h*k*v, k*v, v, 1)
            or ring.stride(0) < ring.shape[1]*h*k*v):
        raise ValueError("deferred factors and ring geometry differ")
    if any(x.device != ring.device or x.dtype != torch.float32 or not x.is_contiguous() for x in factors):
        raise ValueError("deferred factors must be contiguous CUDA FP32 on the ring device")
    if any(x.device != ring.device or x.numel() != 1 or x.dtype not in (torch.int32, torch.int64)
           for x in (slot, context, count)):
        raise ValueError("deferred slot/context/count must be CUDA integer singletons")
    bv = 16 if h == 16 and k == v == 128 and t <= 6 else min(triton.next_power_of_2(v), 8)
    _commit[(triton.cdiv(v, bv), h)](
        keys, decay, updates, ring, slot, context, count, t, h, k, v,
        ring.shape[1], ring.stride(0), triton.next_power_of_2(k), bv, block,
        num_warps=1, num_stages=3)
