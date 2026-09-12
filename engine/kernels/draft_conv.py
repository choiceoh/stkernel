"""The drafter's grouped causal tap mix, in one launch (45차 §83).

DFlash puts a short causal convolution over the block's rows on either side of attention and of the MLP: four a
layer, twenty a block, a hundred in the five blocks a proposal step. Written with torch ops each one is fifteen
dispatches and materialises a [rows, taps, groups, group] coefficient -- 98 KiB a call, 9.8 MiB a step -- to
read it once and drop it.

    out[r, c] = sum_tap (base[tap, c] + delta[r, tap, c // group]) * x[r - tap, c]    , zero where r % block < tap

`block` is how the rows are cut into independent blocks: the taps look back inside one and never into the one
before it, which is what lets a step propose for every row at once.

The coefficient is the thing worth keeping in registers: it is the sum of a weight the model owns and a number
the row's own projection produced, so it cannot be prepared ahead, only fused.

Every term is bit-identical to the torch form's: the coefficient is rounded to the weights' dtype where the
torch add rounds it, and one product of two bf16 numbers rounds the same either way. What differs is the sum --
this one runs in fp32 and rounds once, where the torch form rounds the running total at every tap. A row whose
only live tap is the first is therefore bit-identical to the old answer; past that the gap is bounded by the
roundings the old form does and this one does not -- measured at most 2 half-steps of the summed term
magnitudes at the two taps production runs, 3 at four taps -- and it falls on the kernel's side.

Only the drafter reads this, so a moved last bit moves a draft and never a served token: the target's logits
decide what is emitted and the block verification decides what is kept.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _taps(X, DELTA, BASE, OUT, sX, sDr, sDt, sDg, sO, width, BLOCK: tl.constexpr,
          GROUP: tl.constexpr, T: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0)
    within = r % BLOCK
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    live = c < width
    acc = tl.zeros([BC], tl.float32)
    for tap in tl.static_range(T):
        # the coefficient is rounded to the weights' dtype exactly as the torch form's add is, so the terms
        # of the sum are identical to it; the round trip is what forces that rounding in a fp32 register
        coeff = (tl.load(BASE + tap * width + c, mask=live, other=0.0).to(tl.float32) +
                 tl.load(DELTA + r * sDr + tap * sDt + (c // GROUP) * sDg, mask=live, other=0.0).to(tl.float32)
                 ).to(BASE.dtype.element_ty).to(tl.float32)
        # a row nearer the block's start than the tap reads nothing: this is the boundary the torch form pads for
        x = tl.load(X + (r - tap) * sX + c, mask=live & (within >= tap), other=0.0).to(tl.float32)
        acc += coeff * x
    tl.store(OUT + r * sO + c, acc.to(OUT.dtype.element_ty), mask=live)


def tap_mix(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, group: int, block: "int | None" = None):
    """x [rows, width], delta [rows, taps, width // group], base [taps, width] -> [rows, width].

    `block` is the length of an independent block of rows; the default is one block covering every row."""
    if x.ndim != 2 or delta.ndim != 3 or delta.shape[0] != x.shape[0]:
        raise ValueError("the tap mix takes x [rows, width] and delta [rows, taps, groups]")
    rows, width = x.shape
    taps = delta.shape[1]
    block = rows if block is None else block
    if width % group or delta.shape[2] != width // group or base.numel() != taps * width:
        raise ValueError("the tap mix's base and delta must cover every group of every tap")
    if x.dtype != delta.dtype or x.dtype != base.dtype:
        raise ValueError("the tap mix reads one dtype: the coefficient is rounded as the model's weights are")
    if block < 1 or rows % block:
        raise ValueError("the rows must divide into whole blocks the taps stay inside of")
    if not x.is_cuda:
        return _by_torch(x, delta, base, group, block)
    # What the block hands over is `coeff[:, 0]` of its projection's [rows, 2, taps, groups] output -- a view.
    # The kernel reads it where it lies, so the step does not copy a coefficient it is about to consume once.
    src = x if x.stride(1) == 1 else x.contiguous()
    flat = base.reshape(taps, width)
    out = torch.empty_like(src)
    span = 512 if width >= 512 else triton.next_power_of_2(width)
    if rows:
        _taps[(rows, triton.cdiv(width, span))](
            src, delta, flat.contiguous() if flat.stride(1) != 1 else flat, out,
            src.stride(0), delta.stride(0), delta.stride(1), delta.stride(2), out.stride(0), width,
            BLOCK=block, GROUP=group, T=taps, BC=span, num_warps=4)
    return out


def _by_torch(x, delta, base, group, block=None):
    """The form this replaces, kept as the reference it is judged against."""
    import torch.nn.functional as Fn
    rows, width = x.shape
    taps, block = delta.shape[1], rows if block is None else block
    n, g = rows // block, width // group
    blocks = x.view(n, block, g, group)
    coeff = base.reshape(1, 1, taps, g, group) + delta.view(n, block, taps, g, 1)
    valid = torch.arange(block, device=x.device)[:, None] >= torch.arange(taps, device=x.device)[None, :]
    out = coeff[:, :, 0] * blocks
    for tap in range(1, taps):
        shifted = Fn.pad(blocks[:, :-tap], (0, 0, 0, 0, tap, 0))
        out = out + coeff[:, :, tap] * shifted * valid[:, tap].view(1, block, 1, 1)
    return out.reshape(rows, width)
