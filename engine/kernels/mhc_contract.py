"""Experimental GLM mHC terminal consumer and direct drafter feature writer.

Only a terminal consumer may discard the four post-map channels. Intermediate
layers still need that residual carry. Each channel is rounded to BF16 before
the FP32 mean, just as in the served TileLang post -> float -> mean -> BF16
path. A column view of the final five-layer feature tensor avoids its concat.
This probe is not bound into serving until the same-runtime GPU gate passes.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["OUT_STRIDE"])
def _contract(X, Residual, Post, Comb, Out, OUT_STRIDE,
              H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(X + row * H + col).to(tl.float32)
    base = row * (4 * H) + col
    r0 = tl.load(Residual + base).to(tl.float32)
    r1 = tl.load(Residual + base + H).to(tl.float32)
    r2 = tl.load(Residual + base + 2 * H).to(tl.float32)
    r3 = tl.load(Residual + base + 3 * H).to(tl.float32)
    total = tl.full((B,), 0.0, tl.float32)
    for channel in tl.static_range(4):
        post = tl.load(Post + row * 4 + channel)
        c0 = tl.load(Comb + row * 16 + channel)
        c1 = tl.load(Comb + row * 16 + 4 + channel)
        c2 = tl.load(Comb + row * 16 + 8 + channel)
        c3 = tl.load(Comb + row * 16 + 12 + channel)
        # Match the existing post kernel's product then four ordered FMAs.
        value = post * x
        value = tl.fma(c0, r0, value)
        value = tl.fma(c1, r1, value)
        value = tl.fma(c2, r2, value)
        value = tl.fma(c3, r3, value)
        rounded = value.to(tl.bfloat16).to(tl.float32)
        # Torch's four-element, non-fastest-dimension mean uses a serial
        # accumulator combine in this runtime. Do not use a tree tl.sum.
        total = total + rounded
    tl.store(Out + row * OUT_STRIDE + col, total * 0.25)


def contract(x, residual, post, comb, *, out=None):
    """Write [N,4096] BF16 to new storage or an owned feature column view.

Inputs are contiguous CUDA tensors. Output rows may have a wider stride,
allowing five layer outputs to share a [N,5*4096] destination without concat.
The caller owns that destination; no input storage may overlap its span.
"""
    if (x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != 4096
            or residual.shape != (x.shape[0], 4, 4096)
            or post.shape != (x.shape[0], 4, 1)
            or comb.shape != (x.shape[0], 4, 4)):
        raise ValueError("mHC contraction requires X[N,4096], residual[N,4,4096], post[N,4,1], comb[N,4,4]")
    tensors = (x, residual, post, comb)
    for value, dtype in zip(tensors, (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32)):
        if value.device != x.device or not value.is_cuda or value.dtype != dtype or not value.is_contiguous():
            raise ValueError("mHC contraction requires contiguous CUDA inputs with BF16 values and FP32 mixes")
    if out is None:
        out = torch.empty_like(x)
    if (out.shape != x.shape or out.device != x.device or out.dtype != x.dtype
            or out.stride(1) != 1 or out.stride(0) < 4096):
        raise ValueError("mHC contraction output requires nonoverlapping BF16 rows with contiguous columns")
    begin = out.data_ptr()
    end = begin + ((out.shape[0] - 1) * out.stride(0) + 4096) * out.element_size()
    if any(begin < value.data_ptr() + value.numel() * value.element_size()
           and value.data_ptr() < end for value in tensors):
        raise ValueError("mHC contraction output must not overlap an input")
    _contract[(x.shape[0], 8)](x, residual, post, comb, out, out.stride(0), 4096, 512,
                              num_warps=4)
    return out
