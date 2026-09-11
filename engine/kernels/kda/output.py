"""KDA per-head RMS normalization and sigmoid output gate."""
import math

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _output_norm(X, G, W, Y, ROWS: tl.constexpr, D: tl.constexpr,
                 EPS: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
                 PRECISE: tl.constexpr = True):
    row = tl.program_id(0)*BR+tl.arange(0,BR)
    col = tl.arange(0,BD)
    mask = (row[:,None]<ROWS)&(col[None,:]<D)
    x = tl.load(X+row[:,None]*D+col[None,:],mask,other=0).to(tl.float32)
    g = tl.load(G+row[:,None]*D+col[None,:],mask,other=0).to(tl.float32)
    w = tl.load(W+col,col<D,other=0).to(tl.float32)
    variance = tl.sum(x*x,1)/D
    scale = tl.rsqrt(variance+EPS)
    if PRECISE:
        # libdevice.div_rn is lowered with .ftz by the pinned Triton runtime.
        # A gate near -88 has a representable FP32/BF16 subnormal sigmoid;
        # preserve it through an explicit non-FTZ rounded PTX division.
        gate = tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=(tl.full((),1.,tl.float32),1.+libdevice.exp(-g)),
            dtype=tl.float32,is_pure=True,pack=1)
    else:
        gate = tl.sigmoid(g)
    y = ((x*scale[:,None])*w[None,:])*gate
    tl.store(Y+row[:,None]*D+col[None,:],y,mask)


def kda_output_norm(x, gate, weight, eps=1e-6):
    """Contiguous BF16 [...,D] inputs and [D] weight -> BF16 [...,D]."""
    if (x.dtype != torch.bfloat16 or gate.dtype != x.dtype or x.shape != gate.shape
            or not x.is_cuda or gate.device != x.device or weight.device != x.device
            or x.ndim < 2 or weight.shape != (x.shape[-1],)
            or weight.dtype not in (torch.bfloat16, torch.float32)
            or not all(t.is_contiguous() for t in (x,gate,weight))):
        raise ValueError("KDA output norm requires contiguous CUDA BF16 inputs and BF16/FP32 [D] weight")
    d = x.shape[-1]
    if not 1 <= d <= 512 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("KDA output norm requires 1 <= D <= 512 and positive epsilon")
    rows = x.numel()//d
    output = torch.empty_like(x)
    if rows:
        # One warp per row keeps the 128-wide reduction order aligned with
        # torch. libdevice exp/div preserve its BF16 sigmoid rounding; faster
        # grouped-row/approximate variants remain explicit probe options only.
        _output_norm[(rows,)](x,gate,weight,output,rows,d,eps,triton.next_power_of_2(d),1,
                             PRECISE=True,num_warps=1,enable_fp_fusion=False)
    return output
