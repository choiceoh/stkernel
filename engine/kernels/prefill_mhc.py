# SPDX-License-Identifier: Apache-2.0
"""Private GLM prefill post/prenorm fusion with existing lossless BF16 weights.

The previous BF16 post rounding remains before GEMM and squared sum. Only
weights already exactly representable in BF16 qualify; no weights are
requantized and no additional persistent weight pack is allocated. Existing
TileLang Sinkhorn/pre-map/RMSNorm follows the new prenorm outputs. Different
reduction order requires GPU numerical and consumer quality qualification.
"""
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["M"])
def _post_prenorm(
    Comb, Residual, Post, X, Fn, ResidualOut, GemmOut, Sqrsum,
    M, HIDDEN: tl.constexpr, NSPLITS: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
):
    """Reuse rounded post-map tiles for prenorm, across BM prompt rows.

    The residual remains materialized for the following pre-map and future
    layers. Only its separate prenorm read is removed. Lossless BF16-origin weights
    reuse the existing decode coefficient pack and BF16 tensor cores; the original BF16 post rounding stays before both the GEMM
    and squared sum. Different split/reduction order remains a quality gate.
    """
    # 64-bit row addressing: at m = 131072 the residual offset
    # (row * 16384 + 3 * 4096 + 4095) already equals INT32_MAX, so 32-bit
    # addressing has zero margin at the 128K rung and wraps past it into
    # negative offsets. The private dispatcher additionally bounds m at 32768.
    row = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    split = tl.program_id(1)
    col = tl.arange(0, BN)
    h_tile = tl.arange(0, BK)
    h_tiles_per_split: tl.constexpr = tl.cdiv(tl.cdiv(HIDDEN, BK), NSPLITS)
    mul = tl.zeros((BM, BN), dtype=tl.float32)
    sqr = tl.zeros((BM,), dtype=tl.float32)

    for tile in range(h_tiles_per_split):
        h = (split * h_tiles_per_split + tile) * BK + h_tile
        mask = (row[:, None] < M) & (h[None, :] < HIDDEN)
        residual_base = row[:, None] * (4 * HIDDEN) + h[None, :]
        # Each of the four output channels uses these same five vectors.
        r0 = tl.load(Residual + residual_base, mask, other=0).to(tl.float32)
        r1 = tl.load(Residual + residual_base + HIDDEN, mask, other=0).to(tl.float32)
        r2 = tl.load(Residual + residual_base + 2 * HIDDEN, mask, other=0).to(tl.float32)
        r3 = tl.load(Residual + residual_base + 3 * HIDDEN, mask, other=0).to(tl.float32)
        x = tl.load(X + row[:, None] * HIDDEN + h[None, :], mask, other=0).to(tl.float32)

        for hc in range(4):
            p = tl.load(Post + row * 4 + hc, row < M, other=0)
            c0 = tl.load(Comb + row * 16 + hc, row < M, other=0)
            c1 = tl.load(Comb + row * 16 + 4 + hc, row < M, other=0)
            c2 = tl.load(Comb + row * 16 + 8 + hc, row < M, other=0)
            c3 = tl.load(Comb + row * 16 + 12 + hc, row < M, other=0)
            # Match mhc_post's output-channel orientation and FMA sequence.
            updated = p[:, None] * x
            updated = tl.fma(c0[:, None], r0, updated)
            updated = tl.fma(c1[:, None], r1, updated)
            updated = tl.fma(c2[:, None], r2, updated)
            updated = tl.fma(c3[:, None], r3, updated)
            rounded = updated.to(tl.bfloat16)
            tl.store(ResidualOut + residual_base + hc * HIDDEN, rounded, mask)
            rounded_f32 = tl.where(h[None, :] < HIDDEN, rounded.to(tl.float32), 0.0)
            sqr += tl.sum(rounded_f32 * rounded_f32, axis=1)
            weights = tl.load(
                Fn + (col[None, :] * HIDDEN + h[:, None]) * 4 + hc,
                (col[None, :] < 24) & (h[:, None] < HIDDEN), other=0,
            )
            mul = tl.dot(rounded, weights, mul)

    tl.store(
        GemmOut + (split * M + row[:, None]) * 24 + col[None, :],
        mul, (row[:, None] < M) & (col[None, :] < 24),
    )
    tl.store(Sqrsum + split * M + row, sqr, row < M)


def post_pre(x, residual, post, comb, packed_fn, scale, base, norm,
             rms_eps, hc_eps, post_mult, sinkhorn):
    """Private eager 65..32768-row specialization; caller owns the checked pack."""
    m, hidden = x.shape
    if (not 64 < m <= 32768 or hidden != 4096 or residual.shape != (m, 4, hidden)
            or x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16
            or packed_fn.shape != (24, hidden, 4) or packed_fn.dtype != torch.bfloat16
            or post.numel() != m * 4 or comb.numel() != m * 16
            or post.dtype != torch.float32 or comb.dtype != torch.float32
            or scale.shape != (3,) or base.shape != (24,) or norm.shape != (hidden,)
            or norm.dtype != torch.bfloat16 or scale.dtype != torch.float32
            or base.dtype != torch.float32):
        raise ValueError('prefill MHC requires exact GLM BF16-origin coefficient geometry')
    tensors = (x, residual, post, comb, packed_fn, scale, base, norm)
    if any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError('prefill MHC requires contiguous tensors on one device')
    if not x.is_cuda or torch.cuda.is_current_stream_capturing():
        raise ValueError('private prefill MHC is eager CUDA only pending qualification')
    from engine.kernels.mhc.tilelang_kernels import mhc_pre_big_fuse_with_norm_tilelang
    # Static bounded split policy, without per-call device property queries.
    # Each split covers disjoint H tiles and all four residual channels.
    splits = max(1, min(8, 48 // triton.cdiv(m, 32)))
    updated = torch.empty_like(residual)
    mul = torch.empty((splits, m, 24), device=x.device, dtype=torch.float32)
    sqr = torch.empty((splits, m), device=x.device, dtype=torch.float32)
    next_post = torch.empty((m, 4, 1), device=x.device, dtype=torch.float32)
    next_comb = torch.empty((m, 4, 4), device=x.device, dtype=torch.float32)
    layer_input = torch.empty_like(x)
    _post_prenorm[(triton.cdiv(m, 32), splits)](
        comb, residual, post, x, packed_fn, updated, mul, sqr,
        m, hidden, splits, 32, 128, 32, num_warps=4)
    mhc_pre_big_fuse_with_norm_tilelang(
        mul, sqr, scale, base, updated, next_post.view(m, 4), next_comb.view(m, 16), layer_input, norm,
        hidden, rms_eps, hc_eps, hc_eps, post_mult, sinkhorn, rms_eps, 0, 4)
    return updated, next_post, next_comb, layer_input
