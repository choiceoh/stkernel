"""One-sequence KDA that owns writes to the engine's canonical rollback ring."""
import torch
import triton

from .fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel


def recurrent_kda_ring(q, k, v, g, beta, a_log, g_bias, ring, slot, context, lower_bound, *, round_seed=0):
    """Return dense output and store FP32 accumulators directly in the typed ring.

    Ring is [slots,R,HV,K,V], with dense per-slot rows and optional slot
    padding. Only rows (context + i) % R of the selected slot are modified.
    Context zero ignores old ring data. 1 <= T <= R preserves the same
    rollback window as the functional recurrent lane followed by ring writes.

    Slot/context are either two Python integers (eager) or two CUDA int32/
    int64 singletons (graph). The caller owns their values: slot must be in
    [0, slots), context nonnegative, and concurrent invocations must own
    different slots. Device values are never copied to the host.
    """
    return _recurrent(q, k, v, g, beta, a_log, g_bias, ring, slot, context, lower_bound, round_seed=round_seed)


def _recurrent(q, k, v, g, beta, a_log, g_bias, ring, slot, context, lower_bound, *, deferred=False, round_seed=0):
    if any(t.ndim != 4 for t in (q, k, v, g)):
        raise ValueError("ring KDA requires [1,T,H,D] inputs")
    b, t, h, kd = k.shape
    hv, vd = v.shape[2:]
    inputs = (q, k, v, g, beta)
    if (not q.is_cuda or b != 1 or min(h, hv, kd, vd) <= 0 or hv % h or
            q.shape != k.shape or v.shape[:2] != (1, t) or g.shape != (1, t, hv, kd) or
            beta.shape != (1, t, hv) or
            any(x.device != q.device or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                for x in inputs)):
        raise ValueError("ring KDA requires compatible CUDA floating Q/K, V, gate and scalar beta")
    if (ring.ndim != 5 or ring.shape[0] <= 0 or ring.shape[2:] != (hv, kd, vd) or
            not 1 <= t <= ring.shape[1] or ring.device != q.device or ring.dtype not in (torch.float32, torch.float16) or
            ring.stride()[1:] != (hv*kd*vd, kd*vd, vd, 1) or
            ring.stride(0) < ring.shape[1]*hv*kd*vd):
        raise ValueError("ring must be FP32/FP16 [slots,R,HV,K,V] with dense rows and 1 <= T <= R")
    for x, size in ((a_log, h), (g_bias, h*kd)):
        if x.device != q.device or x.dtype != torch.float32 or x.numel() != size or not x.is_contiguous():
            raise ValueError("ring gate parameters must be contiguous FP32 on the input device")
    if lower_bound is None:
        raise ValueError("ring KDA requires a bounded gate")
    device_indices = isinstance(slot, torch.Tensor) and isinstance(context, torch.Tensor)
    if device_indices:
        if any(x.device != q.device or x.numel() != 1 or x.dtype not in (torch.int32, torch.int64)
               for x in (slot, context)):
            raise ValueError("slot and context must be CUDA integer singletons")
    elif (type(slot) is not int or type(context) is not int or not 0 <= slot < ring.shape[0] or context < 0):
        raise ValueError("slot and context must both be valid integers or CUDA singletons")
    # A write cannot race a Q/K/gate read by another value tile. Arena views
    # may legitimately share storage with read-only weights or other fields,
    # so compare bounding byte ranges instead of their whole storage owner.
    # This conservatively rejects reads from padding between this ring's slots.
    ring_lo, ring_hi = ring.data_ptr(), ring.data_ptr() + (
        (ring.shape[0]-1)*ring.stride(0) + ring.shape[1]*hv*kd*vd)*ring.element_size()
    for x in (*inputs, a_log, g_bias, *((slot, context) if device_indices else ())):
        hi = x.data_ptr() + (1 + sum((n-1)*s for n,s in zip(x.shape,x.stride())))*x.element_size()
        if x.data_ptr() < ring_hi and hi > ring_lo:
            raise ValueError("ring writes must not overlap inputs or device indices")
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    factors = (torch.empty((t, hv, kd), device=q.device, dtype=torch.float32),
               torch.empty((t, hv, kd), device=q.device, dtype=torch.float32),
               torch.empty((t, hv, vd), device=q.device, dtype=torch.float32)) if deferred else (None, None, None)
    bk, bv = triton.next_power_of_2(kd), min(triton.next_power_of_2(vd), 8)
    # BV=16 at seven tokens changed rollback results in the GPU exact gate.
    if h == hv == 16 and kd == vd == 128 and t <= 6:
        bv = 16
    strides = tuple(x.stride()[1:] for x in inputs) if any(not x.is_contiguous() for x in inputs) else None
    fused_recurrent_gated_delta_rule_fwd_kernel[(1, triton.cdiv(vd, bv), hv)](
        q=q, k=k, v=v, g=g, beta=beta, o=out, h0=ring, ht=ring,
        cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
        a_log=a_log, g_bias=g_bias, scale=kd**-0.5, N=1, T=t,
        B=1, H=h, HV=hv, K=kd, V=vd, BK=bk, BV=bv,
        stride_init_state_token=ring.stride(1), stride_final_state_token=ring.stride(1),
        stride_indices_seq=1, stride_indices_tok=1,
        INPLACE_FINAL_STATE=False, IS_BETA_HEADWISE=False, USE_QK_L2NORM_IN_KERNEL=True,
        IS_KDA=True, SIGMOID_BETA=True, COMPUTE_GATE=True, SAFE_GATE=True,
        LOWER_BOUND=lower_bound, STATE_KV=True, INPUT_STRIDES=strides,
        ring_slot=slot, ring_context=context, RING_SIZE=ring.shape[1],
        RING_SLOT_STRIDE=ring.stride(0), RING_DEVICE_INDICES=device_indices,
        deferred_keys=factors[0], deferred_decay=factors[1], deferred_updates=factors[2],
        DEFERRED_STATE=deferred, ROUND_SEED=round_seed,
        num_warps=1, num_stages=3)
    return (out, factors) if deferred else out
