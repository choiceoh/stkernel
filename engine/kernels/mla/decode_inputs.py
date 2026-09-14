"""RMS-normalize, cast and scatter decode latents without intermediate tensors."""
import torch
import triton
import triton.language as tl


@triton.jit
def _latent_norm_write(X, W, LATENT, TABLE, CTX, XS: tl.constexpr, LS: tl.constexpr,
                       TS0: tl.constexpr, TS1: tl.constexpr, BLOCK: tl.constexpr,
                       BLOCK_STRIDE: tl.constexpr, OFFSET: tl.constexpr, TOKENS: tl.constexpr,
                       EPS, D: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, D)
    # Match common.norm_rope._norm: same 512-wide, four-warp reduction,
    # BF16 rounding after normalization and again after multiplication.
    x = tl.load(X + r * XS + d).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / D + EPS)
    w = tl.load(W + d).to(tl.float32)
    value = ((x * scale).to(tl.bfloat16) * w.to(tl.bfloat16)).to(tl.bfloat16)
    seq = r // TOKENS
    pos = tl.load(CTX + seq) + r % TOKENS
    page = tl.load(TABLE + seq * TS0 + (pos // BLOCK) * TS1)
    slot = page.to(tl.int64) * BLOCK_STRIDE + OFFSET + pos % BLOCK
    tl.store(LATENT + slot * LS + d, value.to(tl.float8e4nv))


def latent_norm_write(x, weight, latent, table, block, block_stride, offset, contexts, tokens, eps):
    if (type(tokens) is not int or tokens != 8 or x.ndim != 2
            or x.shape != (contexts.numel() * tokens, 512) or contexts.ndim != 1
            or not 1 <= contexts.numel() <= 4 or contexts.dtype != torch.int64
            or not contexts.is_contiguous() or weight.shape != (512,)
            or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
            or x.stride(1) != 1 or x.stride(0) < 512 or not weight.is_contiguous()
            or latent.ndim != 2 or latent.shape[1] != 512 or latent.stride(1) != 1
            or latent.stride(0) < 512 or latent.dtype != torch.float8_e4m3fn
            or table.ndim != 2 or table.shape[0] != contexts.numel() or table.shape[1] == 0
            or table.dtype != torch.int32 or any(t.device != x.device for t in (weight, latent, table, contexts))
            or type(block) is not int or block <= 0 or type(block_stride) is not int
            or type(offset) is not int or offset < 0 or block_stride < offset + block):
        raise ValueError('latent norm write requires the bound K=7 BF16/FP8 mapped-cache layout')
    if not x.is_cuda:
        raise ValueError('latent norm write requires CUDA')
    _latent_norm_write[(len(x),)](x, weight, latent, table, contexts, x.stride(0), latent.stride(0),
                                 *table.stride(), block, block_stride, offset, tokens, eps, 512, num_warps=4)
