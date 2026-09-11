"""Rotary embedding (module), the `get_rope(...)` factory glm53_model calls.

GLM-5.3 as served uses NO rope: qk_rope_head_dim is 0 (mla_use_nope) and the
KDA layers have none, so `rotary_emb` is None on every served layer. This
exists so the model file's construction path resolves and for Qwen3.8's
partial (0.25) interleaved mrope later; only the neox/non-neox base form is
implemented and checked here.
"""
from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position, base, is_neox_style=True):
        super().__init__()
        self.head_size, self.rotary_dim, self.is_neox = head_size, rotary_dim, is_neox_style
        inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv)                                # [max_pos, rotary_dim/2]
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _apply(self, x, cos, sin):                                  # x [.., heads, head_size]
        r = x[..., : self.rotary_dim].float(); rest = x[..., self.rotary_dim:]
        if self.is_neox:
            x1, x2 = r.chunk(2, dim=-1)
            rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        else:
            x1, x2 = r[..., 0::2], r[..., 1::2]
            rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return torch.cat([rot.to(x.dtype), rest], dim=-1)

    def forward(self, positions, q, k):                             # positions [tokens]
        cos = self.cos[positions].unsqueeze(-2); sin = self.sin[positions].unsqueeze(-2)
        return self._apply(q, cos, sin), self._apply(k, cos, sin)


def get_rope(head_size, max_position, rope_parameters=None, is_neox_style=True, rotary_dim=None):
    if head_size == 0:
        return None                                                 # GLM-5.3 served: no rope
    base = (rope_parameters or {}).get("rope_theta", 10000.0)
    return RotaryEmbedding(head_size, rotary_dim or head_size, max_position, base, is_neox_style)


def _selfcheck() -> None:
    assert get_rope(0, 1024) is None
    rope = get_rope(16, 64, {"rope_theta": 10000.0}, is_neox_style=True)
    q = torch.randn(5, 2, 16); k = torch.randn(5, 2, 16); pos = torch.arange(5)
    rq, rk = rope(pos, q, k)
    # rotation preserves norms and position 0 is the identity
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-4) and torch.allclose(rq[0], q[0], atol=1e-5)
    # relative property: <rope(q,m), rope(k,n)> depends only on m-n
    a = (rope(torch.tensor([3]), q[:1], k[:1])[0][0, 0] * rope(torch.tensor([1]), q[:1], k[:1])[1][0, 0]).sum()
    b = (rope(torch.tensor([7]), q[:1], k[:1])[0][0, 0] * rope(torch.tensor([5]), q[:1], k[:1])[1][0, 0]).sum()
    assert torch.allclose(a, b, atol=1e-4)
    print("  rotary: get_rope(0)->None (GLM served), norm-preserving, relative-position property OK")


if __name__ == "__main__":
    _selfcheck()
