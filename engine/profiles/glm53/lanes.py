"""The eight kernel lanes GLM-5.3 runs on (profile), bound two ways.

    reference()   the engine's torch references in modules/* -- each one
                  judged against its served kernel in probes/ (44th ledger):
                  KDA chunk 6.3e-3, conv exact, mHC pre/post exact-ish,
                  MLA 2.2e-3, indexer logits 2.4e-3, kpool byte-identical
    served()      the kernels that serve today: ours, in this repo's
                  overlay, imported by the path they are MOUNTED at inside
                  the glm53 image (vLLM's namespace is the mount point, not
                  their home). All or nothing: a lane that will not import
                  raises, the boot dies (D3) -- there is no per-lane
                  fallback to the reference.

The model (net.py) calls only these eight names; everything else it does is
plain torch on views. One contract per lane, spelled in the docstrings.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Lanes:
    name: str
    conv_prefill: object      # (x [T,C] bf16, w [C,K] f32, state [C,K-1] | None) -> (y [T,C] bf16, state' [C,K-1])
    kda_chunk: object         # (q,k,v [1,T,H,D] bf16, g_raw [1,T,H,D] bf16, beta [1,T,H] f32 sigmoided, A_log [H] f32,
                              #  dt_bias [H*D] f32, state0 [1,H,D,D] f32 | None, lower_bound) -> (o [1,T,H,D] bf16, state [1,H,D,D] f32)
    mhc_pre: object           # (res [T,hc,H] bf16, fn, scale, base, rms_eps, hc_eps, post_mult, sinkhorn, norm_w [H], norm_eps)
                              #   -> (post [T,hc,1] f32, comb [T,hc,hc] f32, x [T,H] bf16 = rmsnorm(sum_i pre_i res_i) * norm_w)
    mhc_post: object          # (x [T,H] bf16, res [T,hc,H], post, comb) -> res' [T,hc,H] bf16
    indexer_logits: object    # (q8 [T,h,128] e4m3 (rotated), k8 [N,128] e4m3, k_scale [N] f32, w [T,h] f32 (q scale folded),
                              #  ke [T] int32: keys [0, ke[m]) count for query m) -> [T,N] f32, garbage past ke
    kpool_compress: object    # (k [P,kp,128] bf16, score [P,kp,128] bf16, ape [kp,128] f32) -> (fp8 [P,128], scale [P,1] f32)
    mla_sparse: object        # (q_abs [T,H,512] bf16, latent [S,512] e4m3, slots [T,W] int32 (valid prefix), valid [T] int32,
                              #  scale, ckv_scale) -> [T,H,512] bf16
    expert: object            # (x [n,H] bf16, w13 [2I,H/2] u8, w13_s [2I,H/16] e4m3, w13_mult [2] f32, a13_mult [] f32,
                              #  w2 [H,I/2] u8, w2_s [H,I/16] e4m3, w2_mult [], a2_mult [], limit) -> [n,H] bf16 (this rank's partial)


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

    def kda_chunk(q, k, v, g_raw, beta, A_log, dt_bias, state0, lower_bound):
        g = kda_gate(g_raw, A_log, dt_bias, lower_bound, safe_gate=True)
        return gated_delta_rule(q, k, v, g, beta, state0, scale=q.shape[-1] ** -0.5,
                                qk_l2norm=True, decay_per_channel=True)

    def logits(q8, k8, k_scale, w, ke):
        return indexer_logits(q8.float(), k8.float() * k_scale[:, None], w)     # relu(c x) = c relu(x): scales fold

    def pre(res, fn, scale, base, rms_eps, hc_eps, post_mult, sinkhorn, norm_w, norm_eps):
        post, comb, x = mhc_pre(res, fn, scale, base, rms_eps, hc_eps, hc_eps, post_mult, sinkhorn)
        xf = x.float()
        x = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm_eps)).to(x.dtype) * norm_w
        return post, comb, x

    def expert(x, w13, w13_s, w13_mult, a13_mult, w2, w2_s, w2_mult, a2_mult, limit):
        half = w13.shape[0] // 2
        g = expert_gemm(x, w13[:half], w13_s[:half], w13_mult[0], a13_mult, quantize_act=True)
        u = expert_gemm(x, w13[half:], w13_s[half:], w13_mult[1], a13_mult, quantize_act=True)
        h = swiglu_clamped(g, u, limit)
        return expert_gemm(h, w2, w2_s, w2_mult, a2_mult, quantize_act=True)

    return Lanes("reference", conv_prefill, kda_chunk, pre, mhc_post, logits, kpool_compress,
                 mla_sparse_mqa, expert)


def served(expert_lane: str = "b12x") -> Lanes:
    """Bound inside the glm53 image (probes/* run there the same way).
    `expert_lane="reference"` is for the lane judge only (check.py says so
    out loud): the b12x expert lane is not bound yet."""
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate          # ours: overlay/modules/glm53_kernels/kda.py
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn                # served op (judged: probes/conv_check.py)
    import vllm.model_executor.layers.mhc  # noqa: F401  registers torch.ops.vllm.mhc_*_tilelang (ours: overlay dsv4_mhc_tilelang)
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits                                             # served DeepGEMM op
    from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_compress_and_write_cache        # served op (byte-identical to ours)
    from vllm.model_executor.layers import glm53_megakernel as mk                                   # ours: overlay/modules/glm53_megakernel

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

    def kda_chunk(q, k, v, g_raw, beta, A_log, dt_bias, state0, lower_bound):
        t = q.shape[1]
        out = torch.empty_like(v)
        o, state = chunk_kda_with_fused_gate(
            q=q, k=k, v=v, raw_g=g_raw, beta=beta, A_log=A_log.view(1, 1, -1, 1), g_bias=dt_bias,
            initial_state=state0, output_final_state=True, use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device=q.device),
            safe_gate=True, lower_bound=lower_bound, out=out)
        return o, state

    def pre(res, fn, scale, base, rms_eps, hc_eps, post_mult, sinkhorn, norm_w, norm_eps):
        post, comb, x = torch.ops.vllm.mhc_pre_tilelang(res, fn, scale, base, rms_eps, hc_eps, hc_eps, post_mult,
                                                        sinkhorn, 1, norm_w, norm_eps)
        return post, comb, x

    def post(x, res, p, comb):
        return torch.ops.vllm.mhc_post_tilelang(x, res, p, comb)

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
            raise RuntimeError("megakernel MLA lane did not arm (VLLM_GLM53_MK_MLA?)")
        cache = latent.view(torch.uint8)
        # the lane is built for this fleet's 16 heads per rank; at world 1 the 64 heads go through in fours (MQA: heads are independent)
        parts = [mk.mla_decode(q_abs[:, i:i + mk.MLA_H].contiguous(), cache, slots, valid, scale, ckv_scale)
                 for i in range(0, q_abs.shape[1], mk.MLA_H)]
        return torch.cat(parts, dim=1)

    if expert_lane == "reference":
        expert = reference().expert
    else:
        def expert(*a, **k):
            raise NotImplementedError("the b12x expert lane eats moe_sf_pack-swizzled packs; it is bound through the served layer (44th ledger), not here yet")

    return Lanes("served" if expert_lane != "reference" else "served (experts: reference)", conv_prefill, kda_chunk, pre, post, logits, kpool, mla, expert)


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
