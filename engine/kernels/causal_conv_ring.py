"""Short causal convolution with device-addressed, in-place raw-input history."""
import torch
import triton

from .causal_conv_single import _single_conv


def causal_conv1d_ring(x, weight, ring, slot, context):
    """Return dense SiLU convolution and write raw inputs into the chosen ring.

    Ring is [slots,C,R], dense within each slot with optional slot padding.
    Context zero and partial history mask unavailable positions. Only cells
    (context+i)%R in this slot are modified; final history is not materialized.
    One CTA time tile owns all 1..8 input tokens and a disjoint channel tile.
    The caller sizes R for its rollback window (GLM: T<=6, K=4, R=8).

    Slot/context are either two Python integers or CUDA integer singletons.
    The caller guarantees device values have 0<=slot<slots and context>=0;
    concurrent invocations must own different slots. No host read is needed.
    """
    floating = (torch.float16, torch.bfloat16, torch.float32)
    if (x.ndim != 2 or weight.ndim != 2 or not x.is_cuda or
            x.dtype not in floating or weight.dtype not in floating or
            weight.device != x.device or weight.shape[0] != x.shape[1] or
            weight.shape[1] not in (2, 3, 4) or x.shape[1] <= 0):
        raise ValueError("conv ring requires CUDA floating x [T,C] and weight [C,K], K=2/3/4")
    t, c = x.shape
    k = weight.shape[1]
    if (ring.ndim != 3 or ring.shape[0] <= 0 or ring.shape[1] != c or
            ring.device != x.device or ring.dtype != x.dtype or ring.shape[2] < k-1 or
            not 1 <= t <= min(8, ring.shape[2]) or ring.stride()[1:] != (ring.shape[2], 1) or
            ring.stride(0) < c*ring.shape[2]):
        raise ValueError("conv ring needs [slots,C,R] in x.dtype, dense rows and 1<=T<=min(8,R)")
    device_indices = isinstance(slot, torch.Tensor) and isinstance(context, torch.Tensor)
    if device_indices:
        if any(v.device != x.device or v.numel() != 1 or v.dtype not in (torch.int32, torch.int64)
               for v in (slot, context)):
            raise ValueError("slot/context must be CUDA integer singletons")
    elif (type(slot) is not int or type(context) is not int or not 0 <= slot < ring.shape[0] or context < 0):
        raise ValueError("slot/context must both be valid integers or CUDA singletons")
    lo = ring.data_ptr()
    hi = lo + ((ring.shape[0]-1)*ring.stride(0) + c*ring.shape[2])*ring.element_size()
    # Reject input aliases before any ring writes. Comparing bounding byte
    # ranges permits nonoverlapping weights in the same arena allocation.
    for v in (x, weight, *((slot, context) if device_indices else ())):
        end = v.data_ptr() + (1 + sum((n-1)*s for n,s in zip(v.shape,v.stride())))*v.element_size()
        if v.data_ptr() < hi and end > lo:
            raise ValueError("conv ring writes must not overlap inputs or device indices")
    out = torch.empty((t, c), device=x.device, dtype=x.dtype)
    _single_conv[(triton.cdiv(c, 128), 1)](
        x, weight, ring, out, None, t, c, *x.stride(), *weight.stride(), *ring.stride()[1:],
        k, True, 128, 8, ring_slot=slot, ring_context=context,
        RING_SIZE=ring.shape[2], RING_SLOT_STRIDE=ring.stride(0), RING_DEVICE_INDICES=device_indices,
        num_warps=4, num_stages=2)
    return out
