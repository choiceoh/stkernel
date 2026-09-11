"""The nine kernel lanes GLM-5.3 runs on (profile), bound two ways.

    reference()   the engine's torch references in modules/* -- each one
                  judged against its served kernel in probes/ (44th ledger):
                  KDA chunk 6.3e-3, conv exact, mHC pre/post exact-ish,
                  MLA 2.2e-3, indexer logits 2.4e-3, kpool byte-identical
    served()      the kernels that serve today, from engine/kernels (ours,
                  packaged with their real dependencies: Triton, TileLang,
                  DeepGEMM, FlashInfer's CuTe-DSL utilities; no vLLM). All
                  or nothing: a lane that will not import
                  raises, the boot dies (D3) -- there is no per-lane
                  fallback to the reference.

The model (net.py) calls only these nine names; everything else it does is
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
                              #  dt_bias [H*D] f32, state0 [1,H,D,D] f32 | None, lower_bound) -> (o [1,T,H,D] bf16, state [1,H,D,D] f32, [k, v] layout)
    kda_recurrent: object     # same inputs for a decode/verify step (T <= spec_k+1) -> (o [1,T,H,D], states [T,H,D,D] f32: after EVERY token)
    mhc_pre: object           # (res [T,hc,H] bf16, fn, scale, base, rms_eps, hc_eps, post_mult, sinkhorn, norm_w [H], norm_eps)
                              #   -> (post [T,hc,1] f32, comb [T,hc,hc] f32, x [T,H] bf16 = rmsnorm(sum_i pre_i res_i) * norm_w)
    mhc_post: object          # (x [T,H] bf16, res [T,hc,H], post, comb) -> res' [T,hc,H] bf16
    indexer_logits: object    # (q8 [T,h,128] e4m3 (rotated), k8 [N,128] e4m3, k_scale [N] f32, w [T,h] f32 (q scale folded),
                              #  ke [T] int32: keys [0, ke[m]) count for query m) -> [T,N] f32, garbage past ke
    kpool_compress: object    # (k [P,kp,128] bf16, score [P,kp,128] bf16, ape [kp,128] f32) -> (fp8 [P,128], scale [P,1] f32)
    mla_sparse: object        # (q_abs [T,H,512] bf16, latent [S,512] e4m3, slots [T,W] int32 (valid prefix), valid [T] int32,
                              #  scale, ckv_scale) -> [T,H,512] bf16
    moe: object               # (x [T,H] bf16, sel [T,k] int32, w [T,k] f32, w13 [E,2I,H/2] u8 rows [up; gate], w13_sf [E, 2I*H/16] e4m3 (folded, interleaved),
                              #  w2 [E,H,I/2] u8, w2_sf [E, H*I/16] e4m3, limit) -> [T,H] bf16: this rank's routed partial (shared expert excluded)


_TP = {"active": None}


def bind_tp(tp) -> None:
    """Tell the served lanes which LocalTP run (if any) owns the main thread."""
    _TP["active"] = tp


def _active_tp():
    return _TP["active"]


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
    from engine.modules.linear_attention import gated_delta_rule, kda_gate
    from engine.modules.moe import expert_gemm
    from engine.modules.sparse_attention import mla_sparse_mqa
    from engine.modules.sparse_indexer import indexer_logits, kpool_compress

    def conv_prefill(x, w, state):
        return causal_conv1d(x, w, None, state, "silu")

    def kda_chunk(q, k, v, g_raw, beta_raw, A_log, dt_bias, state0, lower_bound):
        g = kda_gate(g_raw, A_log, dt_bias, lower_bound, safe_gate=True)
        return gated_delta_rule(q, k, v, g, torch.sigmoid(beta_raw.float()), state0, scale=q.shape[-1] ** -0.5,
                                qk_l2norm=True, decay_per_channel=True)

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

    def moe(x, sel, w, w13, w13_sf, w2, w2_sf, limit):
        """Per selected expert: unswizzle its folded scales, dequantise, W4A4 with
        the kernel's DYNAMIC activation quant (per-16 scales under a global of 1,
        no calibrated input scale -- what the served SM12x lane does)."""
        from engine.modules.nvfp4_sf import unswizzle_sf
        E, two_i, half_h = w13.shape
        i_local, hidden = two_i // 2, half_h * 2
        one = torch.ones((), device=x.device)
        out = torch.zeros(x.shape[0], hidden, dtype=torch.float32, device=x.device)
        for e in sel.unique().tolist():
            rows, k = (sel == e).nonzero(as_tuple=True)
            s13 = unswizzle_sf(w13_sf[e].view(torch.uint8), two_i, hidden // 16).view(torch.float8_e4m3fn)
            s2 = unswizzle_sf(w2_sf[e].view(torch.uint8), hidden, i_local // 16).view(torch.float8_e4m3fn)
            xe = x[rows]
            u = expert_gemm(xe, w13[e, :i_local], s13[:i_local], one, one, quantize_act=True)     # w13 rows are [up; gate]
            g = expert_gemm(xe, w13[e, i_local:], s13[i_local:], one, one, quantize_act=True)
            y = expert_gemm(swiglu_clamped(g, u, limit), w2[e], s2, one, one, quantize_act=True)
            out.index_add_(0, rows, y.float() * w[rows, k][:, None])
        return out.to(x.dtype)

    return Lanes("reference", conv_prefill, kda_chunk, kda_recurrent, pre, mhc_post, logits, kpool_compress,
                 mla_sparse_mqa, moe)


def served(reference_for: "tuple[str, ...]" = ()) -> Lanes:
    """Bound in the ST image (engine/runtime; probes/run_engine_probe.sh runs there).

    `reference_for` names lanes DECLARED to run on the torch reference in
    this table ("expert" and/or "kda_recurrent"). The table's name says so,
    boot prints it, proof can demand it: a declared choice, not a fallback
    (D3). Anything not named must bind or the call raises."""
    expert_lane = "reference" if "expert" in reference_for else "b12x"
    from engine.kernels.kda import chunk_kda_with_fused_gate, fused_recurrent_kda   # fla fork (kernels/SOURCES.json names every origin)
    from engine.kernels.causal_conv import causal_conv1d_fn                            # judged: probes/conv_check.py
    from engine.kernels.mhc import mhc_pre_tilelang, mhc_post_tilelang                 # TileLang mixing + DeepGEMM prenorm, plain functions
    from engine.kernels.deep_gemm import fp8_fp4_mqa_logits                            # the DeepGEMM library, promoted out of vLLM's tree
    from engine.kernels.kpool import kpool_compress_and_write_cache                    # byte-identical to engine.modules.sparse_indexer's
    from engine.kernels import mla as mk                                               # the megakernel's MLA lane, its .cu unchanged
    ref = reference()

    def conv_prefill(x, w, state):
        """probes/conv_check.py's call, verbatim: a two-row state table with
        our row at index 1 -- the served kernel treats cache index 0 as the
        null block and silently skips a sequence that names it."""
        t, c = x.shape
        table = torch.zeros(2, c, w.shape[1] - 1, device=x.device, dtype=x.dtype)
        if state is not None:
            table[1] = state
        y = causal_conv1d_fn(x.T, w, None, table, torch.tensor([0, t], device=x.device, dtype=torch.int32),
                             cache_indices=torch.tensor([1], device=x.device, dtype=torch.int32),
                             has_initial_state=torch.tensor([state is not None], device=x.device), activation="silu")
        y = y.T if y.shape[0] == c else y
        return y, table[1]

    def kda_chunk(q, k, v, g_raw, beta_raw, A_log, dt_bias, state0, lower_bound):
        t = q.shape[1]
        out = torch.empty_like(v)
        o, state = chunk_kda_with_fused_gate(
            q=q, k=k, v=v, raw_g=g_raw, beta=torch.sigmoid(beta_raw.float()), A_log=A_log.view(1, 1, -1, 1), g_bias=dt_bias,
            initial_state=state0.transpose(-1, -2).contiguous() if state0 is not None else None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device=q.device),
            safe_gate=True, lower_bound=lower_bound, out=out)
        # The kernel stores [H,V,K]; the engine/reference contract is [H,K,V].
        return o, state.transpose(-1, -2).contiguous()

    def kda_recurrent(q, k, v, g_raw, beta_raw, A_log, dt_bias, state0, lower_bound):
        # Dense one-sequence form, separate output states: no NULL slot indices,
        # no in-place overwrite of the initial state needed by rejected drafts.
        initial = (state0.transpose(-1, -2).contiguous() if state0 is not None
                   else torch.zeros(1, v.shape[2], v.shape[-1], k.shape[-1], device=q.device, dtype=torch.float32))
        out, states = fused_recurrent_kda(
            q, k, v, g_raw, beta_raw, scale=k.shape[-1] ** -0.5, initial_state=initial,
            inplace_final_state=False, use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True, a_log=A_log, g_bias=dt_bias, compute_gate=True, lower_bound=lower_bound)
        return out, states.transpose(-1, -2).contiguous()

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

    def kpool(k, score, ape):
        pn = k.shape[0]
        dummy = torch.zeros(1, 64, 132, device=k.device, dtype=torch.uint8)
        res = kpool_compress_and_write_cache(dummy, k, score, ape, torch.arange(pn, device=k.device, dtype=torch.int64),
                                             k.shape[1], return_compressed=True, write_cache=False)
        q8 = res[0].view(torch.uint8).reshape(pn, -1)[:, :128].contiguous().view(torch.float8_e4m3fn)
        return q8, res[1].reshape(pn, 1).float()

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
    if expert_lane == "reference":
        moe = ref.moe
    else:
        from engine.kernels.b12x import b12x_fused_moe                                      # ours (CuTe-DSL), FlashInfer supplies the JIT/utilities
        from engine.modules.nvfp4_sf import mma_sf_view
        ones = {}
        scale_views = {}                            # stable arena aliases, one pair per bound MoE layer

        def moe(x, sel, w, w13, w13_sf, w2, w2_sf, limit):
            """The served call (flashinfer_b12x_moe._apply_*): packed nibbles, folded
            interleaved scales, alpha 1, fc2 input scale 1, no input scale (dynamic
            per-block activation quant), clamped SiLU spelled the kernel's way."""
            E = w13.shape[0]
            if E not in ones:
                ones[E] = torch.ones(E, device=x.device, dtype=torch.float32)
            key = (w13_sf.data_ptr(), w2_sf.data_ptr())
            if key not in scale_views:
                scale_views[key] = (mma_sf_view(w13_sf, w13.shape[1], w13.shape[2] * 2),
                                    mma_sf_view(w2_sf, w2.shape[1], w2.shape[2] * 2))
            sf13, sf2 = scale_views[key]
            return b12x_fused_moe(x=x.contiguous(), w1_weight=w13, w1_weight_sf=sf13, w2_weight=w2, w2_weight_sf=sf2,
                                  token_selected_experts=sel.contiguous(), token_final_scales=w.contiguous(),
                                  num_experts=E, num_local_experts=E, top_k=sel.shape[1],
                                  w1_alpha=ones[E], w2_alpha=ones[E], fc2_input_scale=ones[E], input_global_scale=None,
                                  activation="swigluoai_uninterleave", swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=float(limit),
                                  activation_precision="fp4", quant_mode="nvfp4")

    def on_main(fn):
        """Served kernels run on the main thread: DeepGEMM's JIT runtime raises
        CUDA_ERROR_INVALID_VALUE from a worker (probes/mhc_lane_isolate.py), and
        triton's autotuner is not thread-safe either. base/comm.LocalTP hands
        the call over; on the fleet (one rank per process) it is a direct call."""
        def run(*a, **k):
            tp = _active_tp()
            return fn(*a, **k) if tp is None else tp.on_main(fn, *a, **k)
        return run

    name = "served" + (f" (reference: {', '.join(reference_for)})" if reference_for else "")
    return Lanes(name, *(on_main(f) for f in (conv_prefill, kda_chunk, kda_recurrent, pre, post, logits, kpool, mla, moe)))


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
