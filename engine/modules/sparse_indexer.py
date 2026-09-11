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

Pooling, GLM's kpool (kpool_compress.py `_kpool_softmax_rotate_write_cache_kernel`):
one program per pool of `kpool` consecutive keys --

    p[slot, :] = softmax_slot( slot_score[slot] + ape[slot, :] )      per channel
    pooled     = sum_slot p[slot, :] * k[slot, :]
    key        = fp8( hadamard128(pooled) )                          per-row absmax, ue8m0 scale

and the fp8 step is `fwht128_quant_fp8`: butterflies in fp32 with the exact
1/sqrt(128), round to bf16, absmax clamp 1e-4, scale = exp2(ceil(log2(absmax/448))),
clamp +-448. Both are below, judged in probes/indexer_check.py.
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


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard over the last dim (128), scaled by 1/sqrt(128), fp32 butterflies."""
    assert x.shape[-1] == 128
    h = x.float()
    n = 128; step = 1
    while step < n:
        h = h.view(*h.shape[:-1], n // (2 * step), 2, step)
        a, b = h[..., 0, :], h[..., 1, :]
        h = torch.stack([a + b, a - b], dim=-2).reshape(*h.shape[:-3], n)
        step *= 2
    return h * 0.08838834764831845


def fwht128_quant(rows: torch.Tensor):
    """(fp8 [R, 128], scale [R, 1] fp32): rotate, round to bf16, absmax quant with pow2 scale."""
    x = hadamard128(rows).to(torch.bfloat16).float()
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    y = (x / scale).clamp(-448.0, 448.0)
    return y.to(torch.float8_e4m3fn), scale


def kpool_compress(k: torch.Tensor, slot_score: torch.Tensor, ape: torch.Tensor):
    """k [P, kpool, 128] bf16, slot_score [P, kpool] (bf16/fp32), ape [kpool, 128] fp32
    -> (pooled fp8 [P, 128], scale [P, 1]) -- one compressed key per pool."""
    score = slot_score.float()[..., None] + ape.float()[None]             # [P, kpool, 128]
    prob = torch.softmax(score, dim=1)
    pooled = (prob * k.float()).sum(dim=1)                                # [P, 128]
    return fwht128_quant(pooled)


def _selfcheck_pool() -> None:
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Hadamard is orthogonal: H H^T = I after the 1/sqrt(128) scale
    x = torch.randn(5, 128, device=dev)
    assert torch.allclose(hadamard128(hadamard128(x)), x, atol=1e-4)
    P, kp = 7, 4
    k = torch.randn(P, kp, 128, device=dev, dtype=torch.bfloat16); sc = torch.randn(P, kp, device=dev); ape = torch.randn(kp, 128, device=dev)
    q8, s = kpool_compress(k, sc, ape)
    assert q8.shape == (P, 128) and s.shape == (P, 1) and (s == torch.exp2(torch.log2(s))).all()
    # a uniform gate (score 0, ape 0) is a plain mean
    q_mean, s_mean = kpool_compress(k, torch.zeros(P, kp, device=dev), torch.zeros(kp, 128, device=dev))
    ref = fwht128_quant(k.float().mean(1))
    assert torch.equal(q_mean.view(torch.uint8), ref[0].view(torch.uint8))
    print("  sparse_indexer: hadamard128 orthogonal, kpool_compress pow2 scale, uniform gate == mean OK")


if __name__ == "__main__":
    _selfcheck_pool()
