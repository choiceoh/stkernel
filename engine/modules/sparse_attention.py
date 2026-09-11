"""Top-k gathered attention with a sink term (module), torch reference semantics
of DSv4.1's sparse_attn_kernel -- -1 is the only sentinel, the running max
is seeded at -1e30 so an all-(-1) row is zero rather than NaN.
"""
from __future__ import annotations

import torch


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
