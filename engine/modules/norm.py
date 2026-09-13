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


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """weight * (x * rsqrt(mean(x^2) + eps)): the norm in fp32, rounded to x's dtype BEFORE the weight (T5LayerNorm as
    transformers writes it for glm5_next, deepseek_v3 and inkling, cast for cast)."""
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rmsnorm_unit_offset(x: torch.Tensor, weight: torch.Tensor, eps: float, group: "int | None" = None) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * (1 + weight), in fp32, back to x's dtype. `group`: normalise each `group`-wide slice
    of the last axis on its own (Qwen3.8's hyper-connection and PLE norms over hc streams; transformers qwen4_exp
    Qwen4ExpTextRMSNorm, cast for cast)."""
    xf = x.float()
    if group is not None:
        xf = xf.reshape(*xf.shape[:-1], -1, group)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        out = out.flatten(-2)
    return (out * (1.0 + weight.float())).type_as(x)


def rmsnorm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float, activation: str) -> torch.Tensor:
    """weight * rmsnorm(x) * act(gate), act "silu" or "sigmoid" -- a linear-attention output norm (Qwen3.8's GDN:
    transformers qwen4_exp Qwen4ExpTextRMSNormGated, cast for cast: the norm rounds to x's dtype before the weight, the
    gate is applied in fp32, the product rounds once)."""
    if activation not in ("silu", "swish", "sigmoid"):
        raise ValueError(f"rmsnorm_gated takes a silu or sigmoid gate, not {activation!r}")
    dtype = x.dtype
    xf = x.to(torch.float32)
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = weight * normed.to(dtype)
    g = gate.to(torch.float32)
    return (out * (torch.nn.functional.silu(g) if activation != "sigmoid" else torch.sigmoid(g))).to(dtype)


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
