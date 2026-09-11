"""Hyper-connection mixes: sigmoid gates plus a Sinkhorn-normalised comb (module),
term for term from DSv4.1's hc_split_sinkhorn_kernel. Qwen3.8's hc is next.
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
