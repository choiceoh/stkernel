"""One-sequence KDA that owns writes to the engine's canonical rollback ring.

Three gate forms reach the same launch. The fused entries (`recurrent_kda_ring`, `recurrent_kda_ring_rows`) compute
KDA's per-channel gate inside the kernel from raw projections. The GDN entries (`recurrent_gdn_ring`,
`recurrent_gdn_ring_rows`) compute GatedDeltaNet's per-head decay inside it from its raw projection (HEAD_GATE:
engine/kernels/gdn.gates' arithmetic), reading the projection's a and b columns through their strides. The decay
entries (`recurrent_decay_ring`, `recurrent_decay_ring_rows`) take a log-decay computed outside it -- one value per
head read through a stride-0 channel axis (engine/kernels/linear_decay.per_channel), or one per channel -- with the
in-kernel gate off: glue (cells.GLUE), the recurrence being the same with every channel of a head sharing the decay
(engine/modules/linear_attention).
"""
import torch
import triton

from .fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel

_CELL_SEEN = {}
# Probe hook (probes/engine_qwen38_kda.py): force every ring launch's value tile BV -- a power of two no wider than
# next_power_of_2(V) -- to sweep a cell under measurement; kda.py's fused_recurrent_kda_fwd has its own. None keeps the
# rule in `_recurrent`. Measure, then fix the cell in the rule; never an env read (engine/kernels/README.md, D11).
_BV_OVERRIDE: int | None = None


def _forced_bv(vd: int) -> int:
    """`_BV_OVERRIDE`, held to the launch's value width before it replaces the rule's tile."""
    bv, widest = _BV_OVERRIDE, triton.next_power_of_2(vd)
    if type(bv) is not int or bv <= 0 or bv & (bv - 1) or bv > widest:
        raise ValueError(f"_BV_OVERRIDE must be a power of two up to {widest} (V={vd}), got {bv!r}")
    return bv


def _check_cell(decay: bool = False, *, head_gate: bool = False) -> None:
    """The fused entries compute KDA's per-channel gate (COMPUTE_GATE) inside the kernel; a kernel shape whose linear
    attention keeps its decay per head (GDN) cannot run them and takes the GDN entries (its own gate in the kernel) or
    the decay entries; the GDN entries refuse a per-channel cell. Checked once per bound shape and entry form."""
    from engine.base.kernel_shape import bound
    from engine.kernels.cells import FUSED_GATE_DECAY
    shape = bound()
    if _CELL_SEEN.get((decay, head_gate)) is not shape:
        if shape.linear is None:
            raise ValueError("the bound kernel shape declares no linear attention; the ring KDA lane does not apply")
        if not (decay or head_gate) and shape.linear.decay != FUSED_GATE_DECAY:
            raise ValueError("the ring KDA lane fuses KDA's per-channel gate; a per-head decay cell "
                             "(linear.decay == 'head') runs recurrent_gdn_ring on its projection, or "
                             "recurrent_decay_ring on a precomputed decay")
        if head_gate and shape.linear.decay == FUSED_GATE_DECAY:
            raise ValueError("recurrent_gdn_ring computes GatedDeltaNet's per-head decay; a per-channel (KDA) decay "
                             "cell runs the fused entries")
        _CELL_SEEN[(decay, head_gate)] = shape


def recurrent_kda_ring(q, k, v, g, beta, a_log, g_bias, ring, slot, context, lower_bound):
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
    return _recurrent(q, k, v, g, beta, a_log, g_bias, ring, slot, context, lower_bound)


def recurrent_kda_ring_rows(q, k, v, g, beta, a_log, g_bias, ring, slots, contexts, lower_bound):
    """`recurrent_kda_ring` over every row of a decode step in one launch (45차, the C=4 question).

    The inputs hold the rows back to back along T: [1, rows*t, ...], row i at tokens [i*t, (i+1)*t).
    `slots` and `contexts` are CUDA integer vectors of `rows` entries, one per row, in that order --
    each program reads its own row's pair, so the arithmetic and the ring writes are those of `rows`
    separate one-row launches (the tests pin the storage equal byte for byte). Rows must own
    distinct slots. A captured graph replays it with whatever the vectors hold at replay time."""
    if not (isinstance(slots, torch.Tensor) and isinstance(contexts, torch.Tensor)) or slots.numel() != contexts.numel():
        raise ValueError("rows need one CUDA slot and one CUDA context per row")
    return _recurrent(q, k, v, g, beta, a_log, g_bias, ring, slots, contexts, lower_bound, rows=slots.numel())


def recurrent_gdn_ring(q, k, v, a, b, A_log, dt_bias, ring, slot, context):
    """`recurrent_kda_ring` for GatedDeltaNet: its per-head decay computed inside the kernel (HEAD_GATE).

    `a` and `b` are the raw decay and beta projections per value head [1,T,HV] -- the in_proj columns, read through
    their strides without a copy; A_log and dt_bias are contiguous FP32 [HV]. The kernel computes engine/kernels/gdn.gates'
    decay, -exp(A_log) * softplus(a + dt_bias) in FP32, and sigmoids b in b's dtype, as `recurrent_decay_ring` does on the decay and
    raw beta that launch hands it: the same outputs and ring writes, one launch fewer. Q/K are l2-normalised in the
    kernel; the ring, slot and context contract is `recurrent_kda_ring`'s."""
    return _recurrent(q, k, v, a, b, A_log, dt_bias, ring, slot, context, None, head_gate=True)


def recurrent_gdn_ring_rows(q, k, v, a, b, A_log, dt_bias, ring, slots, contexts):
    """`recurrent_kda_ring_rows` for GatedDeltaNet (see `recurrent_gdn_ring`): the rows of a decode step back to back
    along T, one CUDA slot and context per row."""
    if not (isinstance(slots, torch.Tensor) and isinstance(contexts, torch.Tensor)) or slots.numel() != contexts.numel():
        raise ValueError("rows need one CUDA slot and one CUDA context per row")
    return _recurrent(q, k, v, a, b, A_log, dt_bias, ring, slots, contexts, None, rows=slots.numel(), head_gate=True)


def recurrent_decay_ring(q, k, v, decay, beta, ring, slot, context):
    """`recurrent_kda_ring` for a log-decay computed outside the kernel -- GDN's (glue, cells.GLUE).

    `decay` is the natural-log decay (<= 0) per head [1,T,HV], read through a stride-0 channel axis without a copy, or
    per channel [1,T,HV,K]. Beta holds raw logits: GDN rounds sigmoid to beta's dtype before the FP32 recurrence;
    KDA keeps its FP32 sigmoid. Q/K are l2-normalised in the kernel, and
    the ring, slot and context contract is `recurrent_kda_ring`'s. The launch and its ring writes are the fused entry's
    with the in-kernel gate off, the recurrence fused_recurrent_kda(compute_gate=False) computes on those beta values."""
    return _recurrent(q, k, v, decay, beta, None, None, ring, slot, context, None, decay=True)


def recurrent_decay_ring_rows(q, k, v, decay, beta, ring, slots, contexts):
    """`recurrent_kda_ring_rows` for a log-decay computed outside the kernel (see `recurrent_decay_ring`): the rows of
    a decode step back to back along T, one CUDA slot and context per row."""
    if not (isinstance(slots, torch.Tensor) and isinstance(contexts, torch.Tensor)) or slots.numel() != contexts.numel():
        raise ValueError("rows need one CUDA slot and one CUDA context per row")
    return _recurrent(q, k, v, decay, beta, None, None, ring, slots, contexts, None, rows=slots.numel(), decay=True)


def _recurrent(q, k, v, g, beta, a_log, g_bias, ring, slot, context, lower_bound, *, deferred=False, rows=1, factors=None,
               decay=False, head_gate=False):
    if decay and head_gate:
        raise ValueError("a ring launch takes a decay or GatedDeltaNet's projection, not both")
    _check_cell(decay, head_gate=head_gate)
    from engine.base.kernel_shape import bound
    # Qwen's beta is sigmoid(b) in b's dtype before the FP32 recurrence (gdn.gates and the model's GatedDeltaNet).
    # Widening b before sigmoid changed the state solely on decode/verify, even with identical quantised inputs.
    # The per-channel KDA contract keeps its existing FP32 sigmoid.
    round_beta = (head_gate or decay) and bound().linear.decay == "head"
    if head_gate:
        if lower_bound is not None or deferred:
            raise ValueError("the GDN entries take no gate bound and keep no deferred state")
        if not isinstance(g, torch.Tensor) or g.ndim != 3 or k.ndim != 4 or g.shape != (*k.shape[:2], v.shape[2]):
            raise ValueError("GatedDeltaNet's decay projection is per value head [1,T,HV]")
    if decay:
        if a_log is not None or g_bias is not None or lower_bound is not None:
            raise ValueError("the decay entries take the decay itself, no gate parameters")
        if not isinstance(g, torch.Tensor) or g.ndim not in (3, 4) or k.ndim != 4 or g.shape[:2] != k.shape[:2]:
            raise ValueError("a ring decay is per head [1,T,HV] or per channel [1,T,HV,K]")
        if g.ndim == 3:
            from engine.kernels.linear_decay import per_channel
            g = per_channel(g, k.shape[-1])
    if any(t.ndim != 4 for t in (q, k, v, *(() if head_gate else (g,)))):
        raise ValueError("ring KDA requires [1,T,H,D] inputs")
    b, t, h, kd = k.shape
    hv, vd = v.shape[2:]
    inputs = (q, k, v, g, beta)
    if (not q.is_cuda or b != 1 or min(h, hv, kd, vd) <= 0 or hv % h or
            q.shape != k.shape or v.shape[:2] != (1, t) or g.shape != ((1, t, hv) if head_gate else (1, t, hv, kd)) or
            beta.shape != (1, t, hv) or
            any(x.device != q.device or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                for x in inputs)):
        raise ValueError("ring KDA requires compatible CUDA floating Q/K, V, gate and scalar beta")
    if type(rows) is not int or rows <= 0 or t % rows:
        raise ValueError("rows must divide the token count: every row of a step holds the same tokens")
    t = t // rows                                                    # tokens per row from here on
    if (ring.ndim != 5 or ring.shape[0] <= 0 or ring.shape[2:] != (hv, kd, vd) or
            not 1 <= t <= ring.shape[1] or ring.device != q.device or ring.dtype not in (torch.float32, torch.float16) or
            ring.stride()[1:] != (hv*kd*vd, kd*vd, vd, 1) or
            ring.stride(0) < ring.shape[1]*hv*kd*vd):
        raise ValueError("ring must be FP32/FP16 [slots,R,HV,K,V] with dense rows and 1 <= T <= R")
    if head_gate:
        for x in (a_log, g_bias):
            if (not isinstance(x, torch.Tensor) or x.device != q.device or x.dtype != torch.float32 or x.shape != (hv,)
                    or not x.is_contiguous()):
                raise ValueError("GatedDeltaNet's A_log and dt_bias must be contiguous FP32 [HV] on the input device")
    elif not decay:
        for x, size in ((a_log, h), (g_bias, h*kd)):
            if x.device != q.device or x.dtype != torch.float32 or x.numel() != size or not x.is_contiguous():
                raise ValueError("ring gate parameters must be contiguous FP32 on the input device")
        if lower_bound is None:
            raise ValueError("ring KDA requires a bounded gate")
    device_indices = isinstance(slot, torch.Tensor) and isinstance(context, torch.Tensor)
    if device_indices:
        if any(x.device != q.device or x.numel() != rows or x.dtype not in (torch.int32, torch.int64)
               or not x.is_contiguous() for x in (slot, context)):
            raise ValueError("slot and context must be contiguous CUDA integer vectors, one entry per row")
    elif rows != 1:
        raise ValueError("rows need device slot and context vectors")
    elif (type(slot) is not int or type(context) is not int or not 0 <= slot < ring.shape[0] or context < 0):
        raise ValueError("slot and context must both be valid integers or CUDA singletons")
    # A write cannot race a Q/K/gate read by another value tile. Arena views
    # may legitimately share storage with read-only weights or other fields,
    # so compare bounding byte ranges instead of their whole storage owner.
    # This conservatively rejects reads from padding between this ring's slots.
    ring_lo, ring_hi = ring.data_ptr(), ring.data_ptr() + (
        (ring.shape[0]-1)*ring.stride(0) + ring.shape[1]*hv*kd*vd)*ring.element_size()
    for x in (*inputs, *(() if decay else (a_log, g_bias)), *((slot, context) if device_indices else ())):
        hi = x.data_ptr() + (1 + sum((n-1)*s for n,s in zip(x.shape,x.stride())))*x.element_size()
        if x.data_ptr() < ring_hi and hi > ring_lo:
            raise ValueError("ring writes must not overlap inputs or device indices")
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    if deferred:
        if ring.dtype != torch.float32:
            raise ValueError("deferred recurrence preserves only the FP32 state contract")
        shapes = ((rows*t, hv, kd), (rows*t, hv, kd), (rows*t, hv, vd))
        if factors is None:
            factors = tuple(torch.empty(s, device=q.device, dtype=torch.float32) for s in shapes)
        if len(factors) != 3 or any(x.shape != s or x.device != q.device or x.dtype != torch.float32
                                   or not x.is_contiguous() for x, s in zip(factors, shapes)):
            raise ValueError("deferred factors need contiguous FP32 [rows*T,H,D] storage")
        spans = [(x.data_ptr(), x.data_ptr()+x.numel()*x.element_size()) for x in factors]
        for i, (lo, hi) in enumerate(spans):
            reads = (*inputs, *(() if decay else (a_log, g_bias)), *((slot, context) if device_indices else ()))
            forbidden = [(ring_lo, ring_hi), *spans[:i]]
            forbidden += [(x.data_ptr(), x.data_ptr()+(1+sum((n-1)*s for n, s in zip(x.shape, x.stride())))*x.element_size())
                          for x in reads]
            if any(lo < end and start < hi for start, end in forbidden):
                raise ValueError("deferred factor writes must not overlap ring, inputs or another factor")
    else:
        if factors is not None:
            raise ValueError("factor storage belongs only to deferred recurrence")
        factors = (None, None, None)
    bk, bv = triton.next_power_of_2(kd), min(triton.next_power_of_2(vd), 8)
    # BV=16 at seven tokens changed rollback results in the GPU exact gate.
    if h == hv == 16 and kd == vd == 128 and t <= 6:
        bv = 16
    if _BV_OVERRIDE is not None:
        bv = _forced_bv(vd)
    strides = tuple(x.stride()[1:] for x in inputs) if any(not x.is_contiguous() for x in inputs) else None
    # rows > 1: the kernel's sequence axis (program i_n) is the row -- sequence i_n starts at token i_n * T of the
    # flat inputs and reads its own slot/context entry; B stays 1 because the only use of B*T is the K-block
    # offset of the output, and K is one block here (BK = K)
    fused_recurrent_gated_delta_rule_fwd_kernel[(1, triton.cdiv(vd, bv), rows * hv)](
        q=q, k=k, v=v, g=g, beta=beta, o=out, h0=ring, ht=ring,
        cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
        a_log=a_log, g_bias=g_bias, scale=kd**-0.5, N=rows, T=t,
        B=1, H=h, HV=hv, K=kd, V=vd, BK=bk, BV=bv,
        stride_init_state_token=ring.stride(1), stride_final_state_token=ring.stride(1),
        stride_indices_seq=1, stride_indices_tok=1,
        INPLACE_FINAL_STATE=False, IS_BETA_HEADWISE=False, USE_QK_L2NORM_IN_KERNEL=True,
        # the decay and GDN entries: KDA's gate is off and LOWER_BOUND unread, pinned to fused_recurrent_kda's -5.0
        # specialization; the GDN entries compute GatedDeltaNet's decay from a_log (A_log) and g_bias (dt_bias)
        IS_KDA=True, SIGMOID_BETA=True, COMPUTE_GATE=not (decay or head_gate), SAFE_GATE=True,
        LOWER_BOUND=-5.0 if decay or head_gate else lower_bound, STATE_KV=True, INPUT_STRIDES=strides,
        ring_slot=slot, ring_context=context, RING_SIZE=ring.shape[1],
        RING_SLOT_STRIDE=ring.stride(0), RING_DEVICE_INDICES=device_indices,
        RING_INDEX_STRIDE=1 if rows > 1 else 0,
        deferred_keys=factors[0], deferred_decay=factors[1], deferred_updates=factors[2],
        DEFERRED_STATE=deferred, HEAD_GATE=head_gate, ROUND_BETA=round_beta,
        num_warps=1, num_stages=3)
    return (out, factors) if deferred else out
