"""KDA per-head RMS normalization and sigmoid output gate."""
import math

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

# The dense megakernel's activation scale (kernels.cu mk_act_scale): amax * (1/448) with the reciprocal
# folded in FP32, floored at 1e-30. Both constants are the FP32 values that source spells.
INV_448 = tl.constexpr(1.0 / 448.0)
SCALE_FLOOR = tl.constexpr(1.0e-30)


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


@triton.jit
def _output_norm_pack(X, G, W, Y, WORDS, SCALES, ROWS: tl.constexpr, HEADS: tl.constexpr, D: tl.constexpr,
                      EPS: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr, WIDE: tl.constexpr = False):
    # The normalization is _output_norm's, statement for statement (same shapes, masks and reductions),
    # so Y keeps its bytes. One program is one (token, head): at D=128 a head is exactly one 128-column
    # K block of the o_proj input [tokens, heads*128], so the block's amax is this program's own row.
    row = tl.program_id(0)*BR+tl.arange(0,BR)
    col = tl.arange(0,BD)
    mask = (row[:,None]<ROWS)&(col[None,:]<D)
    x = tl.load(X+row[:,None]*D+col[None,:],mask,other=0).to(tl.float32)
    g = tl.load(G+row[:,None]*D+col[None,:],mask,other=0).to(tl.float32)
    w = tl.load(W+col,col<D,other=0).to(tl.float32)
    variance = tl.sum(x*x,1)/D
    scale = tl.rsqrt(variance+EPS)
    gate = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=(tl.full((),1.,tl.float32),1.+libdevice.exp(-g)),
        dtype=tl.float32,is_pure=True,pack=1)
    y = (((x*scale[:,None])*w[None,:])*gate).to(tl.bfloat16)
    tl.store(Y+row[:,None]*D+col[None,:],y,mask)
    # The consumer cell's pack (kernels.cu mk_input_pack_kernel), from Y's own BF16 bytes: a row scale per
    # 128 columns, then four E4M3 bytes per 4-column lane at that kernel's interleaved offset.
    v = y.to(tl.float32).reshape(32, 4)
    amax = tl.max(tl.max(tl.abs(v), 1), 0)
    sc = tl.maximum(amax*tl.full((),INV_448,tl.float32), tl.full((),SCALE_FLOOR,tl.float32))
    rcp = tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                    args=(tl.full((),1.,tl.float32),sc), dtype=tl.float32, is_pure=True, pack=1)
    pairs = (v*rcp).reshape(32, 2, 2)
    first, second = tl.split(pairs)          # columns (0, 2) and (1, 3) of each lane
    c0, c2 = tl.split(first)
    c1, c3 = tl.split(second)
    # mk_f32x4_to_e4m3: SATFINITE, round-to-nearest, x0 in the low byte. The native pair converter puts
    # its second operand in the low byte, exactly as the megakernel helpers call it.
    lo = tl.inline_asm_elementwise("{ .reg .b16 q; cvt.rn.satfinite.e4m3x2.f32 q, $2, $1; mov.b32 $0, {q, 0}; }",
                                   "=r,f,f", [c0, c1], dtype=tl.int32, is_pure=True, pack=1)
    hi = tl.inline_asm_elementwise("{ .reg .b16 q; cvt.rn.satfinite.e4m3x2.f32 q, $2, $1; mov.b32 $0, {0, q}; }",
                                   "=r,f,f", [c2, c3], dtype=tl.int32, is_pure=True, pack=1)
    token = tl.program_id(0)//HEADS
    block = tl.program_id(0)%HEADS
    lane = tl.arange(0,32)
    if WIDE:
        # Sixteen rows: mk_wide_input_pack_kernel's layout, the sixteen-row CTA's input -- each K block
        # reserves 32 rows of 128 bytes in natural order, then a row scale at [block*32+token].
        tl.store(WORDS+(block*32+token)*32+lane, lo|hi)
        tl.store(SCALES+block*32+token, sc)
    else:
        q = lane>>3
        word = lane&7
        ks = ((word>>1)-q)&3
        offset = block*1024+ks*256+(token*4+q)*8+(word&1)*4
        tl.store(WORDS+offset//4, lo|hi)
        tl.store(SCALES+block*8+token, sc)


def kda_output_norm(x, gate, weight, eps=1e-6, *, pack=None):
    """Contiguous BF16 [...,D] inputs and [D] weight -> BF16 [...,D].

    `pack`: storage for the o_proj cell's input pack of an [8 or 16, heads, 128] step, laid out as
    engine.kernels.dense.producer_pack_nbytes says: the C1 cell's own pack at 8 rows, the wide pack of the
    sixteen-row CTA at 16. The output is unchanged; the pack is the one the bound cell would otherwise
    launch for itself, so the cell reads it in place (kernels.cu run_gemm_bound_input)."""
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
    if pack is not None:
        from engine.kernels.dense import producer_pack_nbytes
        heads = x.shape[1] if x.ndim == 3 else 0
        steps = x.shape[0] if x.ndim == 3 else 0
        if (x.ndim != 3 or steps not in (8, 16) or d != 128 or not 1 <= heads <= 32 or pack.dtype != torch.uint8
                or pack.device != x.device or not pack.is_contiguous() or pack.data_ptr() % 8
                or pack.numel() != producer_pack_nbytes(steps, heads * d)):
            raise ValueError("an o_proj input pack needs an [8 or 16, heads, 128] step and its aligned cell layout")
        words_bytes = heads * (1024 if steps == 8 else 32 * 128)
        words = pack[:words_bytes].view(torch.int32)
        scales = pack[words_bytes:].view(torch.float32)
        _output_norm_pack[(rows,)](x,gate,weight,output,words,scales,rows,heads,d,eps,128,1,
                                  WIDE=steps == 16,num_warps=1,enable_fp_fusion=False)
        return output
    if rows:
        # One warp per row keeps the 128-wide reduction order aligned with
        # torch. libdevice exp/div preserve its BF16 sigmoid rounding; faster
        # grouped-row/approximate variants remain explicit probe options only.
        _output_norm[(rows,)](x,gate,weight,output,rows,d,eps,triton.next_power_of_2(d),1,
                             PRECISE=True,num_warps=1,enable_fp_fusion=False)
    return output


kda_output_norm.producer_pack = True
