"""The engram write into the residual stream, with the lookup taken out.

The reference `Engram.forward` opens with

    kv = self.wkv(self.embed(hash_ids).flatten(-2))

so the table read and the arithmetic that consumes it are one expression. That
is fine when the table is in memory and fatal when it is on an SSD: the read
would then happen at the layer, where layer 1 has one layer of compute in front
of it and 4.19 ms p95 of unhidden stall behind it.

So the seam is here. `gate_and_write` takes the readout ALREADY GATHERED -- by
dsv41_engram.ShardEmbedding, whose result is bit-identical to `self.embed(...)`
-- and does everything after it. The caller issues the read at layer 0 and calls
this at the layer. Nothing else about the computation moves.

Everything below is the reference's arithmetic, and the parts that look like
detail are the parts that are: `rstd` is a PRODUCT of two rsqrt terms taken per
(token, hc copy) over `dim` rather than jointly over the copies; the gate is a
SIGNED square root, `copysign(|dot|.clamp_min(1e-6).sqrt(), dot)`, which the
reference notes matches the training kernel; and a masked token gets gate 0
rather than being skipped, so it passes through with its residual untouched.
probes/dsv41_engram_gate_diff.py holds all of it to bit equality against the
reference class run on the same inputs.
"""

from __future__ import annotations

# `Engram.clamp_value` in the reference. Small, and load-bearing: without it a
# dot of exactly zero would take sqrt'(0) = inf into the gradient, and at
# inference it is what keeps the signed sqrt from collapsing the sign.
CLAMP_VALUE = 1e-6


def gate_and_write(x, readout, wkv_weight, q_weight, k_weight, *,
                   hc_mult: int, dim: int, eps: float, token_mask=None):
    """x: [..., hc_mult, dim]; readout: [..., n_hash_cols, head_dim].

    Returns x updated in the reference's dtype, which is x's.
    """
    import torch
    import torch.nn.functional as F

    # the reference's `self.wkv(self.embed(hash_ids).flatten(-2))`
    kv = F.linear(readout.flatten(-2), wkv_weight)
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    # q and k only ever appear as a product; forming it once is the reference's
    # own note, not an optimization applied on top of it
    weight = q_weight.float() * k_weight.float()

    h = x.float()
    rstd = (torch.rsqrt(h.square().mean(-1) + eps)
            * torch.rsqrt(key.square().mean(-1) + eps))
    dot = (h * weight * key).sum(-1) * rstd * dim ** -0.5
    gate = torch.sigmoid(
        torch.copysign(dot.abs().clamp_min(CLAMP_VALUE).sqrt(), dot))
    if token_mask is not None:
        # zeroing the gate rather than skipping the position: the residual is
        # still written, it just receives nothing
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
