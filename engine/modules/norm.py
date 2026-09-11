"""RMSNorm with vLLM's fused-residual calling convention (module).

glm53_model calls both forms:
    h = norm(h)                          -> normalised h
    h, residual = norm(h, residual)      -> (normalised(h + residual), h + residual)
The second is the pre-norm block's "add then norm" fused into one call; the
returned residual is the SUM, which the next block adds to. Getting that
wrong drifts the residual stream by one block per layer and nothing errors.
"""
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.get_default_dtype()), requires_grad=False)
        self.eps = eps

    def _norm(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x)
        summed = x + residual
        return self._norm(summed), summed


class FusedRMSNormGated(RMSNorm):
    """KDA's output norm (kda.py:797 forward_native): rmsnorm(x) * act(g),
    act = silu for "swish"/"silu", sigmoid otherwise. GLM's o_norm is this."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__(hidden_size, eps)
        self.activation = activation

    def forward(self, x, g, residual=None, prenorm=False):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        gf = g.float()
        gate = gf * torch.sigmoid(gf) if self.activation in ("swish", "silu") else torch.sigmoid(gf)
        return (normed * gate).to(x.dtype)


def _selfcheck() -> None:
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev):
        n = RMSNorm(64, eps=1e-6); n.weight.data.uniform_(0.5, 1.5)
    x, r = torch.randn(3, 64, device=dev), torch.randn(3, 64, device=dev)
    ref = torch.nn.functional.rms_norm((x + r).float(), (64,), n.weight.float(), 1e-6).to(x.dtype)
    y, res = n(x, r)
    assert torch.allclose(y, ref, atol=2e-2, rtol=2e-2) and torch.equal(res, x + r)
    assert torch.allclose(n(x), torch.nn.functional.rms_norm(x.float(), (64,), n.weight.float(), 1e-6).to(x.dtype), atol=2e-2, rtol=2e-2)
    with torch.device(dev):
        gn = FusedRMSNormGated(64, 1e-6, "silu"); gn.weight.data.uniform_(0.5, 1.5)
    g = torch.randn(3, 64, device=dev)
    ref = (torch.nn.functional.rms_norm(x.float(), (64,), gn.weight.float(), 1e-6) * torch.nn.functional.silu(g.float())).to(x.dtype)
    assert torch.allclose(gn(x, g), ref, atol=2e-2, rtol=2e-2)
    print("  norm: RMSNorm plain, fused-residual, and KDA's gated (silu) forms == torch OK")


if __name__ == "__main__":
    _selfcheck()
