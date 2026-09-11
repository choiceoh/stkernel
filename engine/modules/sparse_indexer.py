"""The sparse indexer's scoring (module): the feature DSv4.1 (CED), GLM-5.3
(kpool) and Qwen3.8 (QSA) share -- a small side attention that decides which
positions the real attention may read.

Scoring, as the served DeepGEMM op `fp8_fp4_mqa_logits` computes it
(vllm/utils/deep_gemm.py:515, GLM's indexer; DSv4.1's Indexer.forward is the
same formula in bf16):

    logits[m, n] = sum_h  weights[m, h] * relu( q[m, h, :] . k[n, :] )

with q per-token-scaled fp8 (its scale folded into `weights`), k fp8 with a
per-row scale, and one shared key per position (MQA). The top-k over `n`
is then taken per query. `indexer_logits` is the bf16 reference of that
formula; probes/indexer_check.py judges it against the served op.

Pooling (GLM's kpool: `index_kpool` consecutive keys become one, gated) and
the tail rule live beside it once read off the served kernels.
"""
from __future__ import annotations

import torch


def indexer_logits(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """q [M, H, D], k [N, D], weights [M, H] fp32 -> logits [M, N] fp32."""
    s = torch.einsum("mhd,nd->mhn", q.float(), k.float()).relu_()
    return torch.einsum("mhn,mh->mn", s, weights.float())


def topk_positions(logits: torch.Tensor, k: int, valid: "torch.Tensor | None" = None) -> torch.Tensor:
    """Per-query top-k position ids, -1 padded; `valid[m]` masks positions >= it."""
    m, n = logits.shape
    if valid is not None:
        mask = torch.arange(n, device=logits.device)[None, :] >= valid[:, None]
        logits = logits.masked_fill(mask, float("-inf"))
    kk = min(k, n)
    vals, idx = logits.topk(kk, dim=-1)
    idx = idx.masked_fill(torch.isinf(vals), -1).to(torch.int32)
    if kk < k:
        idx = torch.cat([idx, torch.full((m, k - kk), -1, dtype=torch.int32, device=logits.device)], -1)
    return idx


def _selfcheck() -> None:
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    M, H, D, N = 4, 32, 128, 300
    q = torch.randn(M, H, D, device=dev, dtype=torch.bfloat16); k = torch.randn(N, D, device=dev, dtype=torch.bfloat16)
    w = torch.rand(M, H, device=dev)
    lg = indexer_logits(q, k, w)
    # per-element formula, one query
    ref = sum(w[0, h] * torch.relu(q[0, h].float() @ k.float().T) for h in range(H))
    assert torch.allclose(lg[0], ref, atol=1e-2, rtol=1e-3)
    idx = topk_positions(lg, 8, valid=torch.tensor([300, 100, 5, 0], device=dev))
    assert idx.shape == (M, 8) and (idx[1] < 100).all() and (idx[2, 5:] == -1).all() and (idx[3] == -1).all()
    print("  sparse_indexer: logits == sum_h w_h relu(q_h.k), top-k with valid masks and -1 padding OK")


if __name__ == "__main__":
    _selfcheck()
