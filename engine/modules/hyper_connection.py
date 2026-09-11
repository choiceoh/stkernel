"""Hyper-connection mixes (module): the family DSv4.1, GLM-5.3 and Qwen3.8 share.

    hc_split_sinkhorn   DSv4.1's kernel, term for term (sigmoid gates + Sinkhorn comb)
    mhc_pre / mhc_post  GLM-5.3's mHC, term for term from vLLM's kernels/mhc/torch.py --
                        the reference its served TileLang fork (ours) is held to.
                        probes/mhc_check.py judges these against torch.ops.vllm.mhc_*_tilelang
                        inside the glm53 image.

The two differ in where the RMS normalisation sits (mHC normalises the mix
logits by the flattened stream's RMS) and in the pre-mix epsilon; the comb is
the same softmax + Sinkhorn in both.
"""
from __future__ import annotations

import torch


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


def mhc_pre(residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
            rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
            sinkhorn_repeat: int):
    """residual [.., hc, hidden] bf16; fn [(2+hc)*hc, hc*hidden] fp32; returns
    (post_mix [.., hc, 1], comb_mix [.., hc, hc], layer_input [.., hidden])."""
    hc, hidden = residual.shape[-2], residual.shape[-1]
    outer = residual.shape[:-2]
    r = residual.reshape(-1, hc, hidden)
    x = r.reshape(r.shape[0], hc * hidden).float()
    mixes = x @ fn.float().t()
    mixes = mixes * torch.rsqrt(x.square().sum(-1, keepdim=True) / (hc * hidden) + rms_eps)
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
    post = torch.sigmoid(mixes[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]) * hc_post_mult_value
    comb = mixes[:, 2 * hc:].view(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc:].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = (pre.unsqueeze(-1) * r.float()).sum(1).to(torch.bfloat16)
    return post.view(*outer, hc, 1), comb.view(*outer, hc, hc), layer_input.view(*outer, hidden)


def mhc_post(x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor,
             comb_res_mix: torch.Tensor) -> torch.Tensor:
    """residual_new[j] = sum_i comb[i, j] * residual[i] + post[j] * x."""
    mixed = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    return (mixed + post_layer_mix.float() * x.unsqueeze(-2).float()).to(residual.dtype)
