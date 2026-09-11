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


def mla_sparse_mqa(q_abs: torch.Tensor, kv_c: torch.Tensor, topk_slots: torch.Tensor,
                   valid: torch.Tensor, scale: float, ckv_scale: float = 1.0) -> torch.Tensor:
    """GLM-5.3's sparse MLA in its MQA form, as the served lanes compute it
    (flashinfer_mla_sparse_sm90.py: `mla_decode(q, cache, slots, lens, scale,
    ckv_scale)` and the FlashInfer page_size=1 wrapper):

        q_abs   [T, H, 512]   queries already absorbed through W_UK, bf16
        kv_c    [S, 512]      the latent cache, fp8 e4m3 (x ckv_scale) or bf16
        slots   [T, K] int32  per-token top-k cache slots from the indexer
        valid   [T] int32     how many of the K are real (the rest are padding)

    Every head attends the SAME latent row set (MQA); causality is already in
    the indexer's selection, so there is no mask beyond `valid`. No sink.
    The output is the latent-space context [T, H, 512]; W_UV is applied by
    the caller (the wrapper's un-absorb), not here.
    """
    t, h, d = q_abs.shape
    # Read selected rows before dequantizing. A paged cache can be many GiB;
    # converting all of it per layer needlessly scales work with the arena.
    rows = kv_c[topk_slots.long().clamp_min(0)].float()                # [T, K, 512]
    if kv_c.dtype != torch.bfloat16:
        rows = rows * ckv_scale
    active = torch.arange(topk_slots.shape[1], device=q_abs.device)[None, :] < valid[:, None]
    rows = rows.masked_fill(~active[:, :, None], 0)
    scores = torch.einsum("thd,tkd->thk", q_abs.float(), rows) * scale
    k = topk_slots.shape[1]
    mask = torch.arange(k, device=q_abs.device)[None, :] >= valid[:, None]   # [T, K] padding
    scores = scores.masked_fill(mask[:, None, :], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("thk,tkd->thd", p, rows).to(q_abs.dtype)


def _selfcheck_mla() -> None:
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    T, H, D, S, K = 5, 4, 512, 300, 32
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(S, D, device=dev, dtype=torch.bfloat16)
    slots = torch.randint(0, S, (T, K), device=dev, dtype=torch.int32)
    valid = torch.tensor([32, 17, 1, 32, 9], device=dev, dtype=torch.int32)
    out = mla_sparse_mqa(q, kv, slots, valid, scale=D ** -0.5)
    # dense reference over exactly the valid rows, per token
    for t in range(T):
        rows = kv[slots[t, : valid[t]].long()].float()
        sc = (q[t].float() @ rows.T) * D ** -0.5
        ref = torch.softmax(sc, -1) @ rows
        assert torch.allclose(out[t].float(), ref, atol=2e-2, rtol=2e-2), t
    # fp8 cache with a scale must agree with the bf16 cache scaled the same way
    kv8 = (kv.float() / 4.0).to(torch.float8_e4m3fn)
    out8 = mla_sparse_mqa(q, kv8, slots, valid, scale=D ** -0.5, ckv_scale=4.0)
    ref8 = mla_sparse_mqa(q, (kv8.float() * 4.0).to(torch.bfloat16), slots, valid, scale=D ** -0.5)
    assert torch.allclose(out8.float(), ref8.float(), atol=5e-2, rtol=5e-2)
    print("  sparse_attention: mla_sparse_mqa == dense attention over valid slots; fp8 cache x ckv_scale OK")


if __name__ == "__main__":
    _selfcheck_mla()
