# SPDX-License-Identifier: Apache-2.0
"""Two-launch NVFP4 activation scale/alpha construction, without an abs copy.

The input and scale recipe are unchanged. Only contiguous BF16 prefill inputs
are admitted; the caller keeps the ordinary PyTorch recipe for other layouts.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(X, Partial, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(Partial + tl.program_id(0), tl.max(tl.abs(x), 0))


@triton.jit
def _finish(Partial, WeightScale, GlobalScale, Alpha, COUNT: tl.constexpr,
            ALPHA_SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    amax = tl.max(tl.load(Partial + offsets, offsets < COUNT, other=0), 0)
    amax = tl.maximum(amax, 1.0e-12)
    # The stock recipe is `448.0 * 6.0 / amax` and `alpha_scale / (x_gs * w_gs)`
    # with PYTHON floats on the left: torch's __rtruediv__ computes
    # reciprocal(tensor) * scalar (two roundings), not one IEEE division.
    # 39차 gate: div_rn(2688, amax) read 637.1556 against stock's 637.1555.
    # Mirror the two roundings exactly: correctly rounded reciprocal, then
    # the multiply (enable_fp_fusion=False keeps them separate).
    scale = tl.div_rn(1.0, amax) * 2688.0
    product = scale * tl.load(WeightScale)
    alpha = tl.div_rn(1.0, product) * ALPHA_SCALE
    tl.store(GlobalScale, scale)
    tl.store(Alpha, alpha)


def activation_scale_alpha(x, weight_scale, alpha_scale):
    if (not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.numel() == 0 or weight_scale.dtype != torch.float32
            or weight_scale.device != x.device or weight_scale.numel() != 1):
        raise ValueError("NVFP4 fused scale requires contiguous BF16 and one FP32 scale")
    blocks = triton.cdiv(x.numel(), 8192)
    partial = torch.empty(blocks, device=x.device, dtype=torch.float32)
    scale = torch.empty(1, device=x.device, dtype=torch.float32)
    alpha = torch.empty_like(scale)
    _partials[(blocks,)](x, partial, N=x.numel(), BLOCK=8192)
    _finish[(1,)](partial, weight_scale, scale, alpha, COUNT=blocks,
                  ALPHA_SCALE=alpha_scale, BLOCK=triton.next_power_of_2(blocks),
                  enable_fp_fusion=False)
    return scale, alpha
