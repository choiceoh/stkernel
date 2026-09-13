# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ST mHC pre/post: TileLang mixing and standalone DeepGEMM prenorm."""
import torch
from . import tilelang_kernels  # Required compiler dependency, checked when the lane binds.

_HC_PRENORM_MIN_M = 8

def _deneb_hc_prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits):
    """37차: deep_gemm's tf32 prenorm GEMM, with M padded to 8 rows when it is
    smaller.

    The one place a served M below 8 has ever reached this GEMM is a k=5
    speculative boot (6 tokens per request); chain 13's K5 boot spun there
    (rank 0, CPU 200%, 19 min, no new JIT files) and the M=8/16/24 decode
    batches of k=7 and every prefill M (arbitrary, e.g. 27) run through it
    daily. So M < 8 -- and only that -- is served as the proven M=8 shape:
    zero rows appended, GEMM, the live rows copied back. Both outputs are
    row-wise (out = x @ fn^T, sqrsum = |x|^2 per row), so the padding rows are
    inert and never read. Cost at k=5, C=1: one 196 KB zero-copy and two
    ~30 KB copies per layer; at k=7 this branch is never taken.
    """
    from engine.kernels.deep_gemm import tf32_hc_prenorm_gemm

    m = x.shape[0]
    if m >= _HC_PRENORM_MIN_M:
        tf32_hc_prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits)
        return
    x_pad = torch.zeros((_HC_PRENORM_MIN_M,) + tuple(x.shape[1:]),
                        dtype=x.dtype, device=x.device)
    x_pad[:m].copy_(x)
    mul_pad = torch.empty((out_mul.shape[0], _HC_PRENORM_MIN_M) + tuple(out_mul.shape[2:]),
                          dtype=out_mul.dtype, device=out_mul.device)
    sq_pad = torch.empty((out_sqrsum.shape[0], _HC_PRENORM_MIN_M),
                         dtype=out_sqrsum.dtype, device=out_sqrsum.device)
    tf32_hc_prenorm_gemm(x_pad, fn, mul_pad, sq_pad, n_splits)
    out_mul.copy_(mul_pad[:, :m])
    out_sqrsum.copy_(sq_pad[:, :m])



def _deneb_parse_bigfuse(raw: str):
    """Parse "h_blk[,post_n_thr]" -> (h_blk, post_thr|None), else None.

    dsv4 R3: big_fuse h_blk=4096 with n_thr 96/160 (GLM stock 96 already in
    the winning set, so only h_blk is exposed). dsv4 R2: mhc_post prefill
    (n_thr 512, h_blk 4096) beat its stock (128, 1024) by +3.6% at M=4096 --
    post_thr is that second field."""
    try:
        parts = [int(v.strip()) for v in raw.split(",")]
    except Exception:
        return None
    if len(parts) == 1:
        h_blk, post_thr = parts[0], None
    elif len(parts) == 2:
        h_blk, post_thr = parts
    else:
        return None
    if h_blk not in (1024, 2048, 4096):
        return None
    if post_thr is not None and post_thr not in (128, 256, 512):
        return None
    return h_blk, post_thr



def _deneb_bigfuse_hblk(num_tokens: int, hidden_size: int):
    """h_blk for the prefill big_fuse kernels, or None to run stock."""
    if _DENEB_BIGFUSE is None or num_tokens <= 64:
        return None
    if hidden_size % _DENEB_BIGFUSE[0]:
        return None
    return _DENEB_BIGFUSE[0]



def _deneb_bigfuse_post(num_tokens: int):
    """(n_thr, h_blk) for the prefill mhc_post kernel, or None for stock."""
    if _DENEB_BIGFUSE is None or num_tokens <= 64:
        return None
    post_thr = _DENEB_BIGFUSE[1]
    if post_thr is None:
        return None
    return post_thr, _DENEB_BIGFUSE[0]



def mhc_pre_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;
        norm_weight: optional RMSNorm weight, shape (hidden_size,), dtype
            torch.bfloat16. When provided, RMSNorm is fused into the
            layer_input write path of the big_fuse kernel.
        norm_eps: epsilon for the fused RMSNorm; only consulted when
            norm_weight is given.

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """
    from .tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from engine.kernels.deep_gemm import tf32_hc_prenorm_gemm
    from triton import cdiv

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    block_k = 64
    block_m = 64
    n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    residual_2d = residual_flat.view(num_tokens, hc_mult * hidden_size)
    _deneb_hc_prenorm_gemm(residual_2d, fn, gemm_out_mul, gemm_out_sqrsum, n_splits)

    _bf = _deneb_bigfuse_hblk(num_tokens, hidden_size)
    _bf_kw = {"h_blk": _bf} if _bf else {}
    # Prefill tails change split-K even though the row dimension is dynamic.
    # Infer the split extent from the input tensor so one compiled pre-map
    # serves every tail. Small decode keeps its existing specialization.
    fused_splits = 0 if num_tokens > 64 else n_splits
    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            fused_splits,
            hc_mult,
            **_bf_kw,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            fused_splits,
            hc_mult,
            **_bf_kw,
        )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )



def mhc_post_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from .tilelang_kernels import (
        mhc_post_tilelang as _mhc_post_kernel,
    )

    out = torch.empty_like(residual)
    _post_kw = {}
    # deneb fork: prefill-only retune (dsv4 R2); decode M never passes the gate
    _post = _deneb_bigfuse_post(residual.shape[0])
    if _post is not None:
        _post_kw = {"n_thr": _post[0], "h_blk": _post[1]}
    _mhc_post_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
        **_post_kw,
    )
    return out


# D11 (2026-09-12, ST): the dsv4-era big-fuse override (ST_GLM53_MHC_BIGFUSE) was
# never adopted for GLM (production unset = stock); nothing here reads the env.
_DENEB_BIGFUSE = None
