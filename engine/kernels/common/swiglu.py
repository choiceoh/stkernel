"""The gated MLP's activation in one launch (45차 §86).

`Fn.silu(gate) * up` over a fused [rows, 2 * inter] projection is two elementwise launches reading three
tensors to write one. The drafter does it once a layer, five layers a block, five blocks a proposal step --
twenty-five times, on eighteen thousand numbers each -- so what it costs is the launches, not the arithmetic.

The rounding is the torch form's: silu is computed in fp32 and rounded to the input's dtype, as
`torch.nn.functional.silu` does on a bf16 input, and the product with `up` rounds again. That makes this bit
identical to the two ops it replaces rather than merely close.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu(X, OUT, sX, sO, width, out_width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = c < width
    gate = tl.load(X + row * sX + c, live, other=0.0).to(tl.float32)
    up = tl.load(X + row * sX + width + c, live, other=0.0)
    gate = (gate * tl.sigmoid(gate)).to(up.dtype)
    # columns past the activation's width are zeros: a padded consumer's input (dense.PaddedDenseLinear) written here
    tl.store(OUT + row * sO + c, tl.where(live, gate * up, tl.zeros_like(up)), c < out_width)


def swiglu(fused: torch.Tensor, *, pad_to: "int | None" = None) -> torch.Tensor:
    """[rows, 2 * inter] gate-then-up -> [rows, inter], `silu(gate) * up`; CUDA columns must be contiguous. `pad_to`
    widens the output with zero columns in the same launch -- the input a PaddedDenseLinear pads to -- instead of a
    separate pad after it."""
    if fused.ndim != 2 or fused.shape[1] % 2:
        raise ValueError("the gated activation takes one [rows, 2 * inter] projection")
    rows, width = fused.shape[0], fused.shape[1] // 2
    out_width = width if pad_to is None else pad_to
    if out_width < width:
        raise ValueError(f"the gated activation cannot pad {width} columns to {out_width}")
    if not fused.is_cuda:
        gate, up = fused.chunk(2, -1)
        out = torch.nn.functional.silu(gate) * up
        return torch.nn.functional.pad(out, (0, out_width - width)) if out_width > width else out
    if fused.stride(1) != 1:
        raise ValueError("CUDA SwiGLU needs contiguous projection columns")
    out = torch.empty(rows, out_width, device=fused.device, dtype=fused.dtype)
    block = 1024 if out_width >= 1024 else triton.next_power_of_2(out_width)
    if rows:
        _swiglu[(rows, triton.cdiv(out_width, block))](fused, out, fused.stride(0), out.stride(0), width, out_width,
                                                       BLOCK=block, num_warps=4)
    return out
