"""Block-scaled fp8 / fp4 quantisation and the GEMMs that consume it (module),
as torch reference semantics read off DSv4.1's kernel.py.
"""
from __future__ import annotations

import torch


FP8_MAX = 448.0


FP4_MAX = 6.0


FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def _pow2_round(scale: torch.Tensor) -> torch.Tensor:
    """kernel.py:36 fast_round_scale -- 2^ceil(log2(scale)), exactly."""
    out = torch.exp2(torch.ceil(torch.log2(scale.clamp_min(torch.finfo(torch.float32).tiny))))
    return torch.where(scale > 0, out, torch.ones_like(out))


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None,
              scale_dtype: torch.dtype = torch.float32, inplace: bool = False):
    """Block-wise fp8. Returns (fp8 values, per-block scale)."""
    n = x.size(-1)
    assert n % block_size == 0, (n, block_size)
    z = x.contiguous().float()
    blocks = z.unflatten(-1, (n // block_size, block_size))
    amax = blocks.abs().amax(dim=-1)
    scale = amax / FP8_MAX
    if scale_fmt is not None or scale_dtype == torch.float8_e8m0fnu:
        scale = _pow2_round(scale)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    y = (blocks / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).flatten(-2)
    if inplace:
        x.copy_(y.to(torch.float8_e4m3fn).float().unflatten(
            -1, (n // block_size, block_size)).mul(scale.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return y.to(torch.float8_e4m3fn), scale.to(scale_dtype)


def _fp4_encode(values: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 index for each value, as uint8 nibbles."""
    table = FP4_TABLE.to(values.device)
    idx = (values.unsqueeze(-1) - table).abs().argmin(dim=-1)
    # index 8 is a second zero; the encoder never emits it, matching convert.py
    # which only ever DECODES it.
    return idx.to(torch.uint8)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False,
                  scale_dtype: torch.dtype = torch.float8_e8m0fnu):
    """Block-wise fp4 e2m1, two values per byte. Returns (packed, scale)."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    n = x.size(-1)
    assert n % block_size == 0 and n % 2 == 0
    z = x.contiguous().float()
    blocks = z.unflatten(-1, (n // block_size, block_size))
    scale = (blocks.abs().amax(dim=-1) / FP4_MAX)
    if scale_dtype == torch.float8_e8m0fnu:
        scale = _pow2_round(scale)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    scaled = (blocks / scale.unsqueeze(-1)).flatten(-2)
    if inplace:
        table = FP4_TABLE.to(x.device)
        deq = table[_fp4_encode(scaled).long()]
        x.copy_(deq.unflatten(-1, (n // block_size, block_size))
                .mul(scale.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    nib = _fp4_encode(scaled).unflatten(-1, (n // 2, 2))
    packed = (nib[..., 0] | (nib[..., 1] << 4))      # even -> low nibble
    return packed.view(torch.float4_e2m1fn_x2), scale.to(scale_dtype)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    table = FP4_TABLE.to(packed.device)
    low = table[(raw & 0x0F).long()]
    high = table[((raw >> 4) & 0x0F).long()]
    return torch.stack([low, high], dim=-1).flatten(-2)


def _expand(scale: torch.Tensor, block: int, width: int) -> torch.Tensor:
    return scale.float().repeat_interleave(block, dim=-1)[..., :width]


def fp8_gemm(a, a_s, b, b_s, scale_dtype: torch.dtype = torch.float32,
             block_size: int = 128):
    """C[M,N] = A[M,K] @ B[N,K]^T, per-block scales on both sides.

    The two sides are NOT shaped alike and the wrapper asserts it:
    `a_s` is [M, K/block] -- one scale per row per K-block -- while `b_s` is
    [ceil(N/block), K/block], one scale per BLOCK OF ROWS. Treating b_s as
    per-row is the bug that reads the shared expert's scale as if it had a row
    per output (dsv41_shapes.py says the same thing about the loader).
    """
    k, n = a.size(-1), b.size(0)
    af = (a.float().unflatten(-1, (k // block_size, block_size))
          * a_s.float().unsqueeze(-1)).flatten(-2)
    rows = b_s.float().repeat_interleave(block_size, dim=0)[:n]
    bf = (b.float().unflatten(-1, (k // block_size, block_size))
          * rows.unsqueeze(-1)).flatten(-2)
    return (af @ bf.T).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype: torch.dtype = torch.float32,
             act_block_size: int = 128):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T with B packed two per byte."""
    k = a.size(-1)
    af = a.float().unflatten(-1, (k // act_block_size, act_block_size))
    af = (af * a_s.float().unsqueeze(-1)).flatten(-2)
    bf = _unpack_fp4(b).unflatten(-1, (k // 32, 32))
    bf = (bf * b_s.float().unsqueeze(-1)).flatten(-2)
    return (af @ bf.T).to(torch.get_default_dtype())
