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


def topk_positions(logits: torch.Tensor, k: int, valid: "torch.Tensor | None" = None, inplace: bool = False) -> torch.Tensor:
    """Per-query top-k position ids, -1 padded; `valid[m]` masks positions >= it.
    `inplace` masks the caller's logits instead of copying them (a [T, N] fp32 copy is the
    prefill indexer's largest transient)."""
    m, n = logits.shape
    if valid is not None:
        mask = torch.arange(n, device=logits.device)[None, :] >= valid[:, None]
        logits = logits.masked_fill_(mask, float("-inf")) if inplace else logits.masked_fill(mask, float("-inf"))
    kk = min(k, n)
    # The caller wants the SET, not an order: pool_slots writes the winners in descending token position, so a
    # sorted top-k is a sort paid for nothing. With `valid` the padding test is a position compare on the int32
    # winners instead of isinf over the fp32 values the sort returned -- two fewer passes over [rows, k].
    top = logits.topk(kk, dim=-1, sorted=False)
    idx = top.indices.to(torch.int32)
    idx = (idx.masked_fill(idx >= valid[:, None].to(torch.int32), -1) if valid is not None
           else idx.masked_fill(torch.isinf(top.values), -1))
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
    """k [P, kpool, 128] bf16, slot_score [P, kpool, 128] PER-CHANNEL gate score
    (a [P, kpool] score broadcasts), ape [kpool, 128] fp32
    -> (pooled fp8 [P, 128], scale [P, 1]) -- one compressed key per pool.
    The softmax is over the pool's slots, separately per channel: the served
    kernel keeps max_score/prob as (BLOCK_D,) vectors."""
    score = slot_score.float()
    if score.dim() == 2:
        score = score[..., None]
    score = score + ape.float()[None]                                     # [P, kpool, 128]
    prob = torch.softmax(score, dim=1)
    pooled = (prob * k.float()).sum(dim=1)                                # [P, 128]
    # The served pool kernel materializes BF16 before the Hadamard transform,
    # as well as after it. Keep both rounding boundaries in the reference.
    return fwht128_quant(pooled.to(torch.bfloat16))


def select_with_tail(pool_ids: torch.Tensor, seq_lens: torch.Tensor, pool_size: int) -> torch.Tensor:
    """GLM's `expand_pools_and_append_tail`: selected pool ids [rows, topk/pool]
    (-1 = none) -> token ids [rows, topk + pool_size - 1] int32: every selected
    pool expands to its `pool_size` tokens, then the incomplete trailing pool
    (`index_kpool_always_select_tail`: the newest tokens are always attended)
    is appended, -1 padded."""
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    dev = pool_ids.device
    offs = torch.arange(pool_size, device=dev, dtype=torch.int64)
    ids = pool_ids.to(torch.int64)
    tokens = (ids[..., None] * pool_size + offs).reshape(rows, topk)
    seq = seq_lens.to(torch.int64)
    pool_len = seq // pool_size
    # a selected pool is real only if it is a COMPLETE pool of this sequence:
    # the served kernel masks ids >= seq_len // pool_size to -1 (rows with
    # seq 3 and 4 exposed this: every "selected" pool came back -1)
    invalid = (ids < 0) | (ids >= pool_len[:, None])
    tokens = tokens.masked_fill(invalid[..., None].expand(-1, -1, pool_size).reshape(rows, topk), -1)
    tail_start = pool_len * pool_size
    tail_count = seq - tail_start                                          # in [0, pool_size)
    t_offs = torch.arange(pool_size - 1, device=dev, dtype=torch.int64)
    tail = tail_start[:, None] + t_offs[None, :]
    tail = tail.masked_fill(t_offs[None, :] >= tail_count[:, None], -1)
    return torch.cat([tokens, tail], dim=1).to(torch.int32)


def indexer_slots(tokens, block_table, block_size, block_stride, layer_offset, out, counts):
    """Write descending-position latent slots and valid-prefix counts in place.

    A None block table is an identity map. Otherwise each block occupies
    `block_stride` latent rows and this layer starts at `layer_offset` rows.
    """
    positions = tokens.sort(dim=1, descending=True).values
    counts.copy_((positions >= 0).sum(1).to(torch.int32))
    if block_table is None:
        slots = positions
    else:
        safe = positions.clamp_min(0)
        blocks = block_table[(safe // block_size).long()]
        slots = blocks * block_stride + layer_offset + safe % block_size
    out.copy_(slots.masked_fill(positions < 0, -1))


def pool_slots(pool_ids, seq_lens, pool_size, block_table, block_size, block_stride,
               layer_offset, out, counts, tokens: int = 1):
    """Expanded-token oracle for the compressed-pool slot finalization lane.

    A 2-D block table [sequences, blocks] is a captured decode step's: rows come `tokens`
    to a sequence in order, and each sequence's rows are finalized against its own block row."""
    if block_table is not None and block_table.ndim == 2:
        if tokens <= 0 or block_table.shape[0] * tokens != pool_ids.shape[0]:
            raise ValueError("one block row per `tokens` query rows")
        for i in range(block_table.shape[0]):
            sl = slice(i * tokens, (i + 1) * tokens)
            indexer_slots(select_with_tail(pool_ids[sl], seq_lens[sl], pool_size), block_table[i],
                          block_size, block_stride, layer_offset, out[sl], counts[sl])
        return
    indexer_slots(select_with_tail(pool_ids, seq_lens, pool_size), block_table,
                  block_size, block_stride, layer_offset, out, counts)


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
    ref = fwht128_quant(k.float().mean(1).to(torch.bfloat16))
    assert torch.equal(q_mean.view(torch.uint8), ref[0].view(torch.uint8))
    out = select_with_tail(torch.tensor([[3, 0, -1], [1, 2, 5]], device=dev), torch.tensor([15, 24], device=dev), 4)
    assert out.shape == (2, 12 + 3)
    # seq 15: pools 0..2 complete, pool 3 (tokens 12-15) is NOT -- token 15 does not exist --
    # so a selected id 3 is masked and tokens 12..14 arrive through the tail instead
    assert out[0].tolist() == [-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 12, 13, 14], out[0].tolist()
    assert out[1, 12:].tolist() == [-1, -1, -1]                                                      # seq 24: no tail
    short = select_with_tail(torch.tensor([[3, 0, 1]], device=dev), torch.tensor([4], device=dev), 4)
    assert short[0].tolist() == [-1] * 4 + [0, 1, 2, 3] + [-1] * 4 + [-1] * 3, short[0].tolist()   # seq 4: only pool 0 exists
    print("  sparse_indexer: hadamard128 orthogonal, kpool_compress pow2 scale, uniform gate == mean, tail expansion OK")


if __name__ == "__main__":
    _selfcheck_pool()
