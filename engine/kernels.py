"""The six kernels DSv4.1 needs, in torch, so the engine can run at all.

`inference/kernel.py` is TileLang and tilelang is not installed on this fleet.
These are not a port of it -- they are its stated semantics, written in torch,
read off the kernel sources line by line and cited where the reading is not
obvious. They are SLOW ON PURPOSE: an oracle first (CHARTER D4/D14), a
performance path second. `engine/shapes.py` says which shapes are legal; this
says what the numbers are.

Where each one comes from:

    act_quant           kernel.py:98   block-wise fp8, optional pow2 scale
    fp4_act_quant       kernel.py:184  block-wise fp4 e2m1, packed two per byte
    fp8_gemm            kernel.py:277  C[M,N] = A[M,K] @ B[N,K]^T, blocked scales
    fp4_gemm            kernel.py:562  same with B packed [N, K//2]
    sparse_attn         kernel.py:311  gather top-k, online softmax, sink term
    hc_split_sinkhorn   kernel.py:407  sigmoid gates + Sinkhorn-normalised comb

Two details are load-bearing and easy to get wrong:

  the -1 sentinel is not a mask.  `sparse_attn` seeds its running max with
    -1e30 rather than -inf precisely so a row whose indices are ALL -1 gives a
    zero output instead of NaN. The kernel's own comment says so. Reproducing
    the -inf version passes every small test and fails on the first unreachable
    query.

  fp4 packing puts the EVEN element in the low nibble.  convert.py's
    `low = x & 0x0F; high = (x >> 4) & 0x0F` then `stack([low, high])` fixes the
    order; swapping them still round-trips through this file and disagrees with
    every stored weight.
"""
from __future__ import annotations

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0

# convert.py:13 -- index -> value for e2m1, sign in the high bit of the nibble.
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


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """kernel.py:311, without the online-softmax staging (torch does it once).

    Everything the contract in dsv41_sparse_contract.py names is here: int32
    ids, -1 and only -1 as the sentinel, a gather with no upper bound (so the
    caller's range check is the only one), and the finite -1e30 seed that keeps
    an all-(-1) row at zero instead of NaN.
    """
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = topk_idxs != -1
    gathered = kv.gather(
        1, idx.clamp_min(0).reshape(b, -1, 1).expand(-1, -1, d)
    ).reshape(b, m, -1, d)                                   # [b, m, topk, d]
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    row_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    weights = torch.exp(scores - row_max)
    denom = weights.sum(dim=-1) + torch.exp(attn_sink.float().view(1, 1, h) - row_max.squeeze(-1))
    out = torch.einsum("bmhk,bmkd->bmhd", weights, gathered.float()) / denom.unsqueeze(-1)
    return out.to(q.dtype)


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """kernel.py:407, term for term."""
    b, s, _ = mixes.shape
    hc = hc_mult
    m = mixes.float().view(-1, (2 + hc) * hc)
    base, scale = hc_base.float(), hc_scale.float()
    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (m[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return (pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc))


def install():
    """Publish these as the `kernel` module the reference imports."""
    import sys
    import types

    mod = types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                 "sparse_attn", "hc_split_sinkhorn"):
        setattr(mod, name, globals()[name])
    sys.modules["kernel"] = mod
    return mod
