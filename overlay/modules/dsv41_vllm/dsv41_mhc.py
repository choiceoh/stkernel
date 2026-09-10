"""V4.1's Hyper-Connection mixing, in the pairing V4.1 actually uses.

## The difference that matters

V4 and V4.1 both carry `hc_mult` residual copies and both mix them with
sinkhorn-normalised coefficients. They pair those coefficients DIFFERENTLY,
and the difference is at every sublayer, not only at the head.

The reference says it in one line (`Block.forward`):

    Each sub-block's own `hc_mixes` produces the mix for the *next* one, so
    attention uses what the previous layer's FFN produced and the FFN uses
    what this attention produced.

So a sublayer collapses its input with the PREVIOUS sublayer's `pre`, and
hands its own `pre` forward. The image's V4 kernel does not:

    mhc_pre_tilelang(residual, fn, ...) -> (post_mix, comb_mix, layer_input)

It takes no `pre_mix` argument, so the `layer_input` it returns is collapsed
with the pre it just computed from the same `residual` -- V4's pairing.
Upstream's V4.1 branch needed a new kernel for exactly this reason, and named
it for the difference:

    mhc_pre_delayed_tilelang(residual, fn, ..., pre_mix=...) 
        -> (post_mix, res_mix, x, pre)

Running V4.1's weights through V4's pairing produces tokens. It does not
produce V4.1.

## What is here

The mixing math in torch, transcribed from the reference -- `hc_mixes`, the
sinkhorn split, `hc_pre`, `hc_post` -- so the pairing can be assembled
correctly and checked without a GPU. `hc_split_sinkhorn` lives in the
checkpoint's `kernel.py` as TileLang, which is not installed here; its torch
equivalent is written in the kernel's own comments, and that is what this
transcribes:

    pre[j]     = sigmoid(mixes[j]      * hc_scale[0] + hc_base[j]) + eps
    post[j]    = 2 * sigmoid(mixes[j+hc] * hc_scale[1] + hc_base[j+hc])
    comb[j,k]  = mixes[2hc + j*hc + k] * hc_scale[2] + hc_base[2hc + j*hc + k]
    comb       = softmax(comb, -1) + eps
    comb       = comb / (comb.sum(-2) + eps)
    repeat sinkhorn_iters - 1:
        comb   = comb / (comb.sum(-1) + eps)
        comb   = comb / (comb.sum(-2) + eps)

Why the first iteration is asymmetric: the softmax has already normalised the
rows, so a row pass immediately after it would divide by 1 + hc*eps and do
nothing. The kernel skips it and starts on the columns. Running the symmetric
pair from the start instead agrees to float32 rounding (6e-8 at one iteration,
1.2e-7 at twenty), which is worth knowing precisely because it is the sort of
difference one would otherwise assume matters.

What IS load-bearing is which axis the loop ENDS on. The body is row then
column, so the result has unit COLUMNS and only approximately unit rows. How
approximately depends on the inputs, and "20 iterations is plenty" is a claim
about them rather than about the number -- measured on a wide input:

    iters   |row sum - 1|   |col sum - 1|
        1        8.2e-01         3.3e-05
        4        1.4e-01         1.2e-06
        8        5.0e-02         1.1e-06
       20        6.7e-03         1.1e-06

On a narrow input it converges to doubly stochastic well before 20 and the
ending axis stops mattering, which is how the first version of the probe here
concluded -- wrongly -- that it never did.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor,
                      hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    """[..., (2 + hc) * hc] -> (pre [..., hc], post [..., hc], comb [..., hc, hc])."""
    hc = hc_mult
    lead = mixes.shape[:-1]
    flat = mixes.reshape(-1, (2 + hc) * hc).float()
    base = hc_base.float()
    scale = hc_scale.float()

    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(flat[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (flat[:, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(-1, hc, hc)

    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return (pre.reshape(*lead, hc), post.reshape(*lead, hc),
            comb.reshape(*lead, hc, hc))


def hc_mixes(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
             hc_base: torch.Tensor, *, hc_mult: int, norm_eps: float,
             sinkhorn_iters: int, hc_eps: float):
    """The pre/post/comb coefficients for one sublayer.

    `x` is the FULL hc stream [..., hc, d]; the statistic is taken over the
    flattened hc*d vector, one per token -- not per copy. Normalising per copy
    reads the same way and gives different coefficients.
    """
    flat = x.flatten(-2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, hc_fn.float()) * rsqrt
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult,
                             sinkhorn_iters, hc_eps)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the hc copies, weighted. [..., hc, d] x [..., hc] -> [..., d].

    fp32 multiply and fp32 sum, then back to x's dtype. Upstream's Triton
    version passes `enable_fp_fusion=False` to keep the multiply and the sum
    separate, which is what the torch reference does; fusing them into an FMA
    is a different number.
    """
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=-2)
    return y.to(x.dtype)


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
            comb: torch.Tensor) -> torch.Tensor:
    """Expand back to hc copies and mix the residual in through `comb`.

        out[k] = post[k] * x + sum_j comb[j, k] * residual[j]

    The sum is over comb's FIRST index. Summing over the second instead is a
    transpose of a nearly doubly-stochastic matrix -- every shape checks out,
    every value stays finite, and the residual streams are permuted.
    """
    y = (post.unsqueeze(-1) * x.unsqueeze(-2)
         + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3))
    return y.type_as(x)


def sublayer_pair(residual, pre_mix, hc_fn, hc_scale, hc_base, sublayer,
                  norm, *, hc_mult, norm_eps, sinkhorn_iters, hc_eps):
    """One sublayer, in V4.1's pairing. Returns (out, own_pre).

        mixes from THIS input      -> (own_pre, post, comb)
        collapse with the CARRIED pre_mix, not own_pre
        norm, sublayer
        expand back through post/comb

    `own_pre` goes to the NEXT sublayer. The whole difference between V4 and
    V4.1 is that `pre_mix` and `own_pre` are two different tensors here; V4's
    fused kernel collapses with the one it just computed, which is the same
    call site with `own_pre` substituted for `pre_mix`.
    """
    own_pre, post, comb = hc_mixes(residual, hc_fn, hc_scale, hc_base,
                                   hc_mult=hc_mult, norm_eps=norm_eps,
                                   sinkhorn_iters=sinkhorn_iters,
                                   hc_eps=hc_eps)
    x = hc_pre(residual, pre_mix)
    x = norm(x)
    x = sublayer(x)
    return hc_post(x, residual, post, comb), own_pre


def identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """The one-hot mix the first sublayer is handed: copy 0, weight 1."""
    pre = x.new_zeros(*x.shape[:-2], hc_mult, dtype=torch.float32)
    pre[..., 0] = 1.0
    return pre
