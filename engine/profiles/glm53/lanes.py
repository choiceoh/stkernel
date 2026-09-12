"""The kernel lanes GLM-5.3 runs on (profile), bound two ways.

    reference()   the engine's torch references in modules/* -- each one
                  judged against its served kernel in probes/ (44th ledger):
                  KDA chunk 6.3e-3, conv exact, mHC pre/post exact-ish,
                  MLA 2.2e-3, indexer logits 2.4e-3, kpool byte-identical
    served()      ST's kernels in engine/kernels, with direct library
                  dependencies on Triton, TileLang, DeepGEMM and FlashInfer
                  utilities. All or nothing: a lane that will not import
                  raises, the boot dies (D3) -- there is no per-lane
                  fallback to the reference.

The model (net.py) calls only these names; everything else it does is
plain torch on views. One contract per lane, spelled in the docstrings.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Lanes:
    name: str
    conv_prefill: object      # (x [T,C] bf16, w [C,K] f32, state [C,K-1] | None) -> (y [T,C] bf16, state' [C,K-1])
    kda_chunk: object         # (q,k,v [1,T,H,D] bf16, g_raw [1,T,H,D] bf16, beta_raw [1,T,H] bf16 (logits: the lane sigmoids), A_log [H] f32,
                              #  dt_bias [H*D] f32, state0 [1,H,D,D] f32 | None, lower_bound, states_at=None) -> (o [1,T,H,D] bf16,
                              #  state [1,H,D,D] f32, [k, v] layout); with `states_at` (ascending indices of kda_chunk_tokens-wide
                              #  kernel chunks) a third value: the fp32 states [n,H,D,D] at the START of those chunks (45차 §23 marks)
    kda_recurrent: object     # same inputs for a decode/verify step (T <= spec_k+1) -> (o [1,T,H,D], states [T,H,D,D] f32: after EVERY token)
    mhc_pre: object           # (res [T,hc,H] bf16, fn, scale, base, rms_eps, hc_eps, post_mult, sinkhorn, norm_w [H], norm_eps)
                              #   -> (post [T,hc,1] f32, comb [T,hc,hc] f32, x [T,H] bf16 = rmsnorm(sum_i pre_i res_i) * norm_w)
    mhc_post: object          # (x [T,H] bf16, res [T,hc,H], post, comb) -> res' [T,hc,H] bf16
    indexer_logits: object    # (q8 [T,h,128] e4m3 (rotated), k8 [N,128] e4m3, k_scale [N] f32, w [T,h] f32 (q scale folded),
                              #  ke [T] int32: keys [0, ke[m]) count for query m) -> [T,N] f32, garbage past ke
    kpool_compress: object    # (k [P,kp,128] bf16, score [P,kp,128] bf16, ape [kp,128] f32) -> (fp8 [P,128], scale [P,1] f32)
    mla_sparse: object        # (q_abs [T,H,512] bf16, latent [S,512] e4m3, slots [T,W] int32 (valid prefix), valid [T] int32,
                              #  scale, ckv_scale) -> [T,H,512] bf16
    moe: object               # (x [T,H] bf16, sel [T,k] int32, w [T,k] f32, w13 [E,2I,H/2] u8 [up|gate], w13_sf [E, 2I*H/16] e4m3 (interleaved),
                              #  w2 [E,H,I/2] u8, w2_sf [E, H*I/16] e4m3, limit, *, scales=None) -> [T,H] bf16: this rank's routed partial
                              #  scales=None: folded Red Hat; ModelOptScales: separate NVIDIA multipliers. E=1/k=1 also serves dense MLPs.
    indexer_quant: object     # contiguous [R,128] bf16 -> Hadamard-rotated [R,128] e4m3, per-row pow2 [R,1] f32 scale
    pool_slots: object        # (pool ids [T,G] int32, seq_lens [T] int32, pool size, block row | None, block size/stride,
                              #  layer offset, out [T,G*pool+pool-1], counts [T]) -> None; descending token positions, mapped valid prefix
    kda_output_norm: object   # (core/gate [T,H,D] bf16, weight [D] bf16/f32, eps) -> [T,H,D] bf16; FP32 RMS norm and sigmoid gate
    moe_prepare: object = None  # (w13, w13_sf, w2, w2_sf, top_k, limit, *, scales=None) -> None, once per bound layer BEFORE any capture:
                              #  the served lane's weight views (in-place tile-major relayout, packed SF6 owner); reference: None
    graph_resources: object = None  # () -> external workspace owners to retain until the captured graphs close
    kda_chunk_tokens: int = 64      # the kernel chunk `states_at` indexes: a mark inside a prefill chunk sits on a multiple of it
    kda_recurrent_ring: object = None  # recurrent inputs, then (ring [slots,R,H,K,V] f32, slot, context, lower_bound)
                                     # -> output only; writes each token state into the selected ring. None uses the functional lane.
    conv_ring: object = None  # (x [T,C], w [C,K], ring [slots,C,R], slot, context) -> y; writes raw inputs into ring, T<=8


def swiglu_clamped(g: torch.Tensor, u: torch.Tensor, limit: float) -> torch.Tensor:
    """GLM's gated activation everywhere (dense, shared, routed): the served
    dense path is SiluAndMulWithClamp and the served b12x lane runs
    'swigluoai_uninterleave' with alpha 1, beta 0, limit 10 -- the same thing."""
    g = g.float().clamp(max=limit)
    u = u.float().clamp(-limit, limit)
    return (torch.nn.functional.silu(g) * u).to(torch.bfloat16)


def reference() -> Lanes:
    from engine.modules.causal_conv import causal_conv1d
    from engine.modules.hyper_connection import mhc_pre, mhc_post
    from engine.modules.linear_attention import gated_delta_rule, kda_gate, kda_output_norm
    from engine.modules.moe import expert_gemm
    from engine.modules.sparse_attention import mla_sparse_mqa
    from engine.modules.sparse_indexer import fwht128_quant, indexer_logits, kpool_compress, pool_slots

    def conv_prefill(x, w, state):
        return causal_conv1d(x, w, None, state, "silu")

    def kda_chunk(q, k, v, g_raw, beta_raw, A_log, dt_bias, state0, lower_bound, states_at=None):
        g = kda_gate(g_raw, A_log, dt_bias, lower_bound, safe_gate=True)
        beta = torch.sigmoid(beta_raw.float())
        if not states_at:
            return gated_delta_rule(q, k, v, g, beta, state0, scale=q.shape[-1] ** -0.5, qk_l2norm=True, decay_per_channel=True)
        # the reference is a token-by-token recurrence: the state at a chunk's start is exactly the state after the
        # piece before it, so the pieces are run one after another and each piece's final state is that mark
        outs, states, state, lo = [], [], state0, 0
        for hi in [c * 64 for c in states_at] + [q.shape[1]]:
            if hi > lo:
                o, state = gated_delta_rule(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi], state,
                                            scale=q.shape[-1] ** -0.5, qk_l2norm=True, decay_per_channel=True)
                outs.append(o)
            if len(states) < len(states_at):
                states.append(state[0] if state is not None else torch.zeros(q.shape[2], q.shape[-1], v.shape[-1], device=q.device))
            lo = hi
        return torch.cat(outs, dim=1), state, torch.stack(states)

    def kda_recurrent(q, k, v, g_raw, beta_raw, A_log, dt_bias, state0, lower_bound):
        """The recurrence one token at a time, keeping every state: what a
        verify step needs so a rejected draft rolls back by position."""
        g = kda_gate(g_raw, A_log, dt_bias, lower_bound, safe_gate=True)
        beta = torch.sigmoid(beta_raw.float())
        t = q.shape[1]
        outs, states, state = [], [], state0
        for i in range(t):
            o, state = gated_delta_rule(q[:, i:i + 1], k[:, i:i + 1], v[:, i:i + 1], g[:, i:i + 1], beta[:, i:i + 1],
                                        state, scale=q.shape[-1] ** -0.5, qk_l2norm=True, decay_per_channel=True)
            outs.append(o); states.append(state[0])
        return torch.cat(outs, dim=1), torch.stack(states)

    def logits(q8, k8, k_scale, w, ke):
        return indexer_logits(q8.float(), k8.float() * k_scale[:, None], w)     # relu(c x) = c relu(x): scales fold

    def pre(res, fn, scale, base, rms_eps, hc_eps, post_mult, sinkhorn, norm_w, norm_eps):
        post, comb, x = mhc_pre(res, fn, scale, base, rms_eps, hc_eps, hc_eps, post_mult, sinkhorn)
        xf = x.float()
        x = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm_eps)).to(x.dtype) * norm_w
        return post, comb, x

    def moe(x, sel, w, w13, w13_sf, w2, w2_sf, limit, *, scales=None):
        """W4A4 from interleaved scales and optional ModelOpt multipliers.
        Red Hat's folded encoding uses unit weight/activation global scales;
        NVIDIA uses its calibrated per-expert FP32 dequantization scales."""
        from engine.modules.nvfp4_sf import unswizzle_sf
        from engine.modules.expert_layout import W13_K_IN_BYTES, W2_K_IN_BYTES, row_major_expert
        if any(getattr(t,"_st_sf6_consumed",False) for t in (w13_sf,w2_sf)):
            raise ValueError("raw scale storage was retired; reload rank weights for the reference lane")
        E, two_i, half_h = w13.shape
        i_local, hidden = two_i // 2, half_h * 2
        one = torch.ones((), device=x.device)
        out = torch.zeros(x.shape[0], hidden, dtype=torch.float32, device=x.device)
        for e in sel.unique().tolist():
            rows, k = (sel == e).nonzero(as_tuple=True)
            s13 = unswizzle_sf(w13_sf[e].view(torch.uint8), two_i, hidden // 16).view(torch.float8_e4m3fn)
            s2 = unswizzle_sf(w2_sf[e].view(torch.uint8), hidden, i_local // 16).view(torch.float8_e4m3fn)
            # CuTe keeps FC1 accumulators in FP32 through the activation,
            # then rounds the activation to BF16 before its FP4 quantization.
            xe = x[rows].float()
            w13e, w2e = row_major_expert(w13, e, W13_K_IN_BYTES), row_major_expert(w2, e, W2_K_IN_BYTES)   # served bind may have tiled the arena
            w1, a1, w2g, a2 = ((one, one, one, one) if scales is None else
                               (scales.weight13[e], scales.input13[e], scales.weight2[e], scales.input2[e]))
            u = expert_gemm(xe, w13e[:i_local], s13[:i_local], w1, a1, quantize_act=True)
            g = expert_gemm(xe, w13e[i_local:], s13[i_local:], w1, a1, quantize_act=True)
            y = expert_gemm(swiglu_clamped(g, u, limit), w2e, s2, w2g, a2, quantize_act=True)
            out.index_add_(0, rows, y.float() * w[rows, k][:, None])
        return out.to(x.dtype)

    return Lanes("reference", conv_prefill, kda_chunk, kda_recurrent, pre, mhc_post, logits, kpool_compress,
                 mla_sparse_mqa, moe, fwht128_quant, pool_slots, kda_output_norm)


MOE_STATIC_STOCK = "stock"          # the §15~18 judged default of STK_moe_static
MOE_STATIC_PRODUCTION = "t,r,sf6,q0"  # native TP4 decode and prefill recipe


def parse_moe_static(value: str) -> "tuple[str | None, bool]":
    """Explicit probe specification -> (b12x static lane, TP SF6 Q0 flag).
    Cells are the dispatcher's (u, t, r, sf6, f<n>, g<n>, ...); q0 is the engine's token."""
    tokens = [t.strip() for t in str(value).split(",") if t.strip()]
    if tokens in ([], [MOE_STATIC_STOCK], ["0"], ["off"]):
        return None, False
    q0 = "q0" in tokens
    spec = ",".join(t for t in tokens if t != "q0")
    if q0 and "sf6" not in tokens:
        raise ValueError("STK_moe_static: q0 needs the t,r,sf6 cells")
    return (spec or None), q0


def served(reference_for: "tuple[str, ...]" = (), *, tp=None, moe_static: str = MOE_STATIC_PRODUCTION, consume_scales: bool = False,
           mla_prefill: str = "tile32") -> Lanes:
    """Bind the ST kernel package without an overlay or vLLM installation.

    `reference_for` names lanes DECLARED to run on the torch reference in
    this table ("expert" and/or "kda_recurrent"). The table's name says so,
    boot prints it, proof can demand it: a declared choice, not a fallback
    (D3). Anything not named must bind or the call raises.

    `tp` explicitly owns dispatch for this table's lifetime. Bound tables may
    run only inside that LocalTP invocation. Omit it for direct fleet calls
    and warmup; constructing another table never rebinds an existing one.

    `moe_static` / `mla_prefill` are the profile's declared D11 knobs
    (boot.declared: STK_mla_prefill; MoE is fixed for serving), applied to the kernel
    package here, once, before anything binds or arms.
    """
    expert_lane = "reference" if "expert" in reference_for else "b12x"
    from engine.kernels.kda import chunk_kda_with_fused_gate, fused_recurrent_kda
    from engine.kernels.kda.output import kda_output_norm
    from engine.kernels.kda.ring import recurrent_kda_ring
    from engine.kernels.causal_conv_single import causal_conv1d_single as conv_prefill
    from engine.kernels.causal_conv_ring import causal_conv1d_ring
    from engine.kernels.mhc import mhc_pre_tilelang, mhc_post_tilelang
    from engine.kernels.deep_gemm import fp8_fp4_mqa_logits
    from engine.kernels.kpool import compress_pool_keys, fwht128_quant_fp8
    from engine.kernels import mla as mk
    from engine.kernels.indexer import pool_slots
    mk.configure_prefill(mla_prefill)
    ref = reference()

    def kda_chunk(q, k, v, g_raw, beta_raw, A_log, dt_bias, state0, lower_bound, states_at=None):
        t = q.shape[1]
        out = torch.empty_like(v)
        result = chunk_kda_with_fused_gate(
            q=q, k=k, v=v, raw_g=g_raw, beta=torch.sigmoid(beta_raw.float()), A_log=A_log.view(1, 1, -1, 1), g_bias=dt_bias,
            initial_state=state0.transpose(-1, -2).contiguous() if state0 is not None else None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device=q.device),
            safe_gate=True, lower_bound=lower_bound, out=out, states_at=list(states_at) if states_at else None)
        # The kernel stores [H,V,K]; the engine/reference contract is [H,K,V].
        if states_at:
            o, state, states = result
            return o, state.transpose(-1, -2).contiguous(), states.transpose(-1, -2).contiguous()
        o, state = result
        return o, state.transpose(-1, -2).contiguous()

    def kda_recurrent(q, k, v, g_raw, beta_raw, A_log, dt_bias, state0, lower_bound):
        # Dense one-sequence form, separate output states: no NULL slot indices,
        # no in-place overwrite of the initial state needed by rejected drafts.
        # Keep the same initial-state specialization as graph replay, whose
        # history reader supplies zeros at context 0. Omitting that load lets
        # Triton change reduction order and splits eager/graph state bytes.
        initial = state0 if state0 is not None else torch.zeros(
            1, v.shape[2], k.shape[-1], v.shape[-1], device=q.device, dtype=torch.float32)
        return fused_recurrent_kda(
            q, k, v, g_raw, beta_raw, scale=k.shape[-1] ** -0.5, initial_state=initial,
            inplace_final_state=False, use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True, a_log=A_log, g_bias=dt_bias, compute_gate=True,
            lower_bound=lower_bound, state_layout="kv")

    def pre(res, fn, scale, base, rms_eps, hc_eps, post_mult, sinkhorn, norm_w, norm_eps):
        if fn.data_ptr() % 16:
            raise ValueError("mHC weight is not TMA-aligned; regenerate rank files with the aligned RankWriter")
        post, comb, x = mhc_pre_tilelang(res, fn, scale, base, rms_eps, hc_eps, hc_eps, post_mult,
                                       sinkhorn, 1, norm_w, norm_eps)
        return post, comb, x

    def post(x, res, p, comb):
        return mhc_post_tilelang(x, res, p, comb)

    def logits(q8, k8, k_scale, w, ke):
        t = q8.shape[0]
        return fp8_fp4_mqa_logits((q8, None), (k8, k_scale.contiguous()), w.contiguous(),
                                  torch.zeros(t, device=q8.device, dtype=torch.int32), ke.contiguous(), clean_logits=False)

    def mla(q_abs, latent, slots, valid, scale, ckv_scale):
        mk.maybe_arm()
        if not mk._ARMED.get("mla"):
            raise RuntimeError("ST MLA lane did not pass its boot self-test")
        cache = latent.view(torch.uint8)
        # the lane is built for this fleet's 16 heads per rank; at world 1 the 64 heads go through in fours (MQA: heads are independent)
        parts = [mk.mla_decode(q_abs[:, i:i + mk.MLA_H].contiguous(), cache, slots, valid, scale, ckv_scale)
                 for i in range(0, q_abs.shape[1], mk.MLA_H)]
        return torch.cat(parts, dim=1)

    if "kda_recurrent" in reference_for:
        kda_recurrent = ref.kda_recurrent
        recurrent_kda_ring = None
    moe_prepare = None
    graph_resources = None
    if expert_lane == "reference":
        moe = ref.moe
    else:
        from engine.kernels.b12x import b12x_fused_moe
        from engine.kernels.b12x import moe_dispatch as md
        graph_resources = md.cached_workspace_owners
        from engine.modules.nvfp4_sf import mma_sf_view
        spec, q0 = parse_moe_static(moe_static)
        md.configure_static_v2(spec)                # refuses once views exist: the layout choice is per process
        md.configure_tp_sf6_q0(q0)
        ones = {}
        prepared = {}                               # weight + derived scale pointers -> stable views and quantizer/epilogue arguments

        def views_for(w13, w13_sf, w2, w2_sf, top_k, limit, *, in_place, scales=None):
            """The dispatcher's weight views for one layer, built once. `in_place` (bind time) re-lays the
            arena bytes tile-major when the spec says t -- no second copy of 45 layers; a call without a
            bind (probes) copies instead so the caller's row-major tensors stay what they were."""
            key = (w13.data_ptr(), w13_sf.data_ptr(), w2.data_ptr(), w2_sf.data_ptr())
            if scales is not None:
                key += tuple(v.data_ptr() for v in (scales.alpha13, scales.input13, scales.alpha2, scales.input2))
            got = prepared.get(key)
            if got is not None:
                return got
            E, n, k = w13.shape[0], w13.shape[1] // 2, w13.shape[2] * 2
            if E not in ones:
                ones[E] = torch.ones(E, device=w13.device, dtype=torch.float32)
            alpha13, alpha2, quant13, quant2 = ((ones[E], ones[E], None, ones[E]) if scales is None else
                (scales.alpha13, scales.alpha2, scales.input13, scales.input2))
            sf13 = mma_sf_view(w13_sf, w13.shape[1], k)
            sf2 = mma_sf_view(w2_sf, w2.shape[1], w2.shape[2] * 2)
            geometry = dict(num_experts=E, num_local_experts=E, hidden_size=k, intermediate_size=n, num_topk=int(top_k),
                            quant_mode="nvfp4", activation="swigluoai_uninterleave", swiglu_limit=float(limit),
                            activation_precision="fp4")
            tiled = md.static_v2_weights_layout(**geometry)
            reform = md.static_v2_weights_reform_sf_pack(**geometry)
            sf_pack = md.static_v2_weights_sf_pack(**geometry)
            if tiled and in_place:
                md.tile_expert_weights_inplace(w13, w2)     # one transient copy of this layer, then the arena IS tile-major
            views = md._get_weight_views(w1_fp4=w13, w1_blockscale=sf13, w2_fp4=w2, w2_blockscale=sf2,
                                         w1_alphas=alpha13, w2_alphas=alpha2, n=n, k=k,
                                         activation_precision="fp4", quant_mode="nvfp4",
                                         tiled=tiled, sf_pack=sf_pack, reform_sf_pack=reform,
                                         packed_only=bool(tiled and reform and not sf_pack))   # sf6: packed scales only, no converted raw copies
            if in_place and consume_scales and views.packed_only:
                md.consume_packed_scale_storage(views,w13_sf,w2_sf)
            # _weight_views tells dispatch the alpha is final. In particular,
            # it must not fold the input scale into alpha a second time.
            prepared[key] = (views, sf13, sf2, alpha13, alpha2, quant13, quant2)
            return prepared[key]

        def moe_prepare(w13, w13_sf, w2, w2_sf, top_k, limit, *, scales=None):
            views_for(w13, w13_sf, w2, w2_sf, top_k, limit, in_place=True, scales=scales)

        def moe(x, sel, w, w13, w13_sf, w2, w2_sf, limit, *, scales=None):
            """Packed W4A4 with prepared b12x quantizer/epilogue scales.
            Red Hat uses unit scales; ModelOpt passes a for quantization and
            a*w for each GEMM. The clamped activation is common to both."""
            E = w13.shape[0]
            views, sf13, sf2, a13, a2, q13, q2 = views_for(
                w13, w13_sf, w2, w2_sf, sel.shape[1], limit, in_place=False, scales=scales)
            # The ST caller owns the output allocation, including the graph
            # memory pool during capture; the b12x API requires an explicit out.
            output = torch.empty_like(x, memory_format=torch.contiguous_format)
            return b12x_fused_moe(x=x.contiguous(), output=output,
                                  w1_weight=w13, w1_weight_sf=sf13, w2_weight=w2, w2_weight_sf=sf2,
                                  token_selected_experts=sel.contiguous(), token_final_scales=w.contiguous(),
                                  num_experts=E, num_local_experts=E, top_k=sel.shape[1],
                                  w1_alpha=a13, w2_alpha=a2, fc2_input_scale=q2, input_global_scale=q13,
                                  activation="swigluoai_uninterleave", swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=float(limit),
                                  activation_precision="fp4", quant_mode="nvfp4", _weight_views=views)

    def on_main(fn):
        """Served kernels run on the main thread: DeepGEMM's JIT runtime raises
        CUDA_ERROR_INVALID_VALUE from a worker (probes/mhc_lane_isolate.py), and
        triton's autotuner is not thread-safe either. base/comm.LocalTP hands
        the call over; on the fleet (one rank per process) it is a direct call."""
        if tp is None:
            return fn
        def run(*a, **k):
            return tp.on_main(fn, *a, **k)
        return run

    name = "served" + (f" (reference: {', '.join(reference_for)})" if reference_for else "")
    table = Lanes(name, *(on_main(f) for f in (conv_prefill, kda_chunk, kda_recurrent, pre, post, logits, compress_pool_keys, mla, moe,
                                            fwht128_quant_fp8, pool_slots, kda_output_norm)),
                  moe_prepare=None if moe_prepare is None else on_main(moe_prepare),
                  graph_resources=graph_resources,
                  kda_recurrent_ring=None if recurrent_kda_ring is None else on_main(recurrent_kda_ring),
                  conv_ring=None if "conv_prefill" in reference_for else on_main(causal_conv1d_ring))
    # 45차 §21 bisect: any other lane named in `reference_for` runs on the torch reference in this table
    # (the served output is garbage while every self-consistency judge passes -- which lane, if any, is found by
    # swapping them one at a time; "expert" and "kda_recurrent" are the two the kernels already know how to declare).
    known = {"expert", "kda_recurrent"}
    fields = {f for f in Lanes.__dataclass_fields__ if f not in ("name", "moe_prepare")}
    unknown = [n for n in reference_for if n not in known and n not in fields]
    if unknown:
        raise ValueError(f"reference_for names no lane: {unknown}; lanes are {sorted(fields)} (plus 'expert')")
    swapped = {n: getattr(ref, n) for n in reference_for if n in fields and n != "kda_recurrent"}
    if swapped:
        from dataclasses import replace
        table = replace(table, **swapped)
    return table


def _selfcheck() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lanes = reference()
    x = torch.randn(4, 8, device=dev, dtype=torch.bfloat16)
    y = swiglu_clamped(x[:, :4] * 100, x[:, 4:] * 100, 10.0)
    assert y.abs().max().item() <= 10.0 * 10.0 + 1e-3 and y.dtype == torch.bfloat16
    conv_w = torch.randn(8, 4, device=dev)
    y1, st = lanes.conv_prefill(x, conv_w, None)
    y2a, st_a = lanes.conv_prefill(x[:2], conv_w, None); y2b, st_b = lanes.conv_prefill(x[2:], conv_w, st_a)
    assert torch.allclose(y1.float(), torch.cat([y2a, y2b]).float(), atol=2e-2) and torch.equal(st, st_b)
    print(f"  lanes: reference table bound ({', '.join(f for f in Lanes.__dataclass_fields__ if f != 'name')}); "
          "clamped swiglu, conv state carry across a chunk boundary OK")


if __name__ == "__main__":
    _selfcheck()
