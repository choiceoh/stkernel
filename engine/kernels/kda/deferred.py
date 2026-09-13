"""Experimental KDA verification with state materialization after acceptance.

Verification leaves the canonical ring untouched and records FP32 update
factors. Commit replays only the accepted updates into the accepted final
row and any crossed prefix boundary. Callers must commit on the same stream
before another verification, snapshot, or consumer of this slot. This is a
kernel experiment with an explicit decode execution binding; the ordinary
ring remains the production default until the component and fleet gates pass.
"""
import torch
import triton
import triton.language as tl

from .ring import _recurrent


def verify(q, k, v, g, beta, a_log, g_bias, ring, slot, context, lower_bound):
    return _recurrent(q, k, v, g, beta, a_log, g_bias, ring, slot, context,
                      lower_bound, deferred=True)


def verify_rows(q, k, v, g, beta, a_log, g_bias, ring, slots, contexts, lower_bound, *, factors=None):
    """Verify equal-width rows, recording each row's factors in distinct storage."""
    if not isinstance(slots, torch.Tensor) or not isinstance(contexts, torch.Tensor) or slots.numel() != contexts.numel():
        raise ValueError("deferred rows require one device slot and context per row")
    return _recurrent(q, k, v, g, beta, a_log, g_bias, ring, slots, contexts,
                      lower_bound, deferred=True, rows=slots.numel(), factors=factors)


@triton.jit
def _commit_layers(KEY, DECAY, UPDATE, RING, OFFSETS, SLOT, CONTEXT, COUNT,
                   T: tl.constexpr, ROWS: tl.constexpr, H: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
                   SLOT_STRIDE: tl.constexpr, BLOCK: tl.constexpr, B: tl.constexpr):
    # Every CTA owns contiguous state cells. No reduction is needed during
    # materialization, so coalesce the full K,V matrix instead of the
    # verifier's strided value tiles. All layers commit in this one launch.
    tile, layer, row = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    count = tl.load(COUNT + row).to(tl.int64)
    if count <= 0 or count > T:
        return
    slot = tl.load(SLOT + row).to(tl.int64)
    context = tl.load(CONTEXT + row).to(tl.int64)
    cell = tile * B + tl.arange(0, B)
    mask = cell < H*K*V
    head, key, value = cell // (K*V), cell // V % K, cell % V
    base = tl.load(OFFSETS + layer) + slot * SLOT_STRIDE + cell
    state = tl.load(RING + base + (tl.maximum(context-1, 0) % R) * H*K*V,
                    mask & (context > 0), other=0)
    for i in range(count):
        factor = ((layer * ROWS + row) * T + i) * H + head
        k = tl.load(KEY + factor * K + key, mask, other=0)
        decay = tl.load(DECAY + factor * K + key, mask, other=0)
        update = tl.load(UPDATE + factor * V + value, mask, other=0)
        state = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                         args=[state, decay], dtype=tl.float32, is_pure=True, pack=1)
        state = tl.fma(update, k, state)
        position = context + i
        if i == count-1 or (position+1) % BLOCK == 0:
            tl.store(RING + base + (position % R) * H*K*V, state, mask)


class Batch:
    """Owned FP32 factors and one accepted-state commit for a model's layers.

    Ring views must share the arena and slot stride. This owner is retained
    across target and sampler graphs; a commit must precede its next verify.
    Different context-capacity graphs may share it when replay is serialized.
    """
    def __init__(self, rings, rows, tokens, *, block):
        self.rings = tuple(rings)
        if not self.rings or type(rows) is not int or rows <= 0 or type(tokens) is not int:
            raise ValueError("deferred batch needs layers, positive rows and an integer token width")
        ring = self.rings[0]
        if ring.ndim != 5 or not ring.is_cuda or ring.dtype != torch.float32:
            raise ValueError("deferred batch requires CUDA FP32 rings")
        slots, width, h, k, v = ring.shape
        if (min(slots, h, k, v) <= 0 or not 1 <= tokens <= width
                or type(block) is not int or block <= 0 or rows > slots
                or ring.stride()[1:] != (h*k*v, k*v, v, 1)
                or ring.stride(0) < width*h*k*v):
            raise ValueError("invalid deferred batch ring geometry")
        storage = ring.untyped_storage().data_ptr()
        for value in self.rings:
            if (value.shape != ring.shape or value.stride() != ring.stride()
                    or value.device != ring.device or value.dtype != ring.dtype
                    or value.untyped_storage().data_ptr() != storage):
                raise ValueError("deferred layers must share an arena and the same slot geometry")
        offsets = [(value.data_ptr()-ring.data_ptr()) // 4 for value in self.rings]
        # Compare within a slot; an overlapping layer would race another CTA.
        ordered = sorted(offsets)
        if any(b-a < width*h*k*v for a, b in zip(ordered, ordered[1:])) or ordered[-1]-ordered[0]+width*h*k*v > ring.stride(0):
            raise ValueError("deferred layer rings overlap within their slot")
        self.rows, self.tokens, self.block = rows, tokens, block
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=ring.device)
        self.factors = tuple(torch.empty((len(rings), rows*tokens, h, d), dtype=torch.float32, device=ring.device)
                             for d in (k, k, v))
        self.layer_factors = tuple(tuple(f[i] for f in self.factors) for i in range(len(rings)))

    @property
    def nbytes(self):
        return sum(x.numel()*x.element_size() for x in (*self.factors, self.offsets))

    def verify(self, layer, q, k, v, g, beta, a_log, g_bias, slots, contexts, lower_bound):
        out, _ = verify_rows(q, k, v, g, beta, a_log, g_bias, self.rings[layer], slots, contexts,
                             lower_bound, factors=self.layer_factors[layer])
        return out

    def commit(self, slots, contexts, counts):
        ring = self.rings[0]
        for x in (slots, contexts, counts):
            if (x.device != ring.device or x.dtype not in (torch.int32, torch.int64)
                    or x.shape != (self.rows,) or not x.is_contiguous()):
                raise ValueError("deferred commit requires contiguous integer vectors, one entry per row")
        _, width, h, k, v = ring.shape
        _commit_layers[(triton.cdiv(h*k*v, 256), len(self.rings), self.rows)](
            *self.factors, ring, self.offsets, slots, contexts, counts, self.tokens, self.rows,
            h, k, v, width, ring.stride(0), self.block, 256, num_warps=4, num_stages=1)


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
