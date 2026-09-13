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
                   valid: torch.Tensor, scale: float, ckv_scale: float = 1.0, *, out=None) -> torch.Tensor:
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
    result = torch.einsum("thk,tkd->thd", p, rows).to(q_abs.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def gqa_sparse(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor,
               valid: torch.Tensor, scale: float) -> torch.Tensor:
    """Sparse grouped-query attention over selected positions, no sink -- the math Qwen3.8's attention computes
    (overlay/modules/qwen38_qsa/ops_qsa.py: qsa_sparse_paged_attention, without its paging):

        q        [T, H, D]      queries after RoPE
        k_cache  [S, G, D]      keys after RoPE, G KV heads per position, H a multiple of G
        v_cache  [S, G, D]
        slots    [T, K] int32   selected positions, the valid prefix first
        valid    [T] int32      how many of the K are real

    Query head h reads KV head h // (H // G). Causality is in the selection; the only mask is `valid`, and a row with
    no valid position is zero (as the sparse kernels leave it) rather than NaN. fp32 inside, q's dtype outside.
    """
    t, h, d = q.shape
    g = k_cache.shape[1]
    if h % g or k_cache.shape != v_cache.shape or k_cache.shape[2] != d:
        raise ValueError("gqa_sparse takes q [T,H,D] and K/V caches [S,G,D] with H a multiple of G")
    idx = slots.long().clamp_min(0)
    heads = torch.arange(h, device=q.device) // (h // g)
    keys = k_cache[idx].float()[:, :, heads]                                # [T, K, H, D]
    values = v_cache[idx].float()[:, :, heads]
    scores = torch.einsum("thd,tkhd->thk", q.float(), keys) * scale
    padding = torch.arange(slots.shape[1], device=q.device)[None, :] >= valid[:, None]   # [T, K]
    scores = scores.masked_fill(padding[:, None, :], float("-inf"))
    p = torch.softmax(scores, dim=-1).masked_fill((valid <= 0)[:, None, None], 0.0)
    return torch.einsum("thk,tkhd->thd", p, values).to(q.dtype)


class GatedSparseAttention:
    """Full attention with a sigmoid output gate over the positions a QSA indexer selects, as a token mixer
    (engine/base/composition.Feature): Qwen3.8's full-attention layer (transformers qwen4_exp Qwen4ExpTextAttention and
    Qwen4ExpTextQSAIndexer, eager attention).

    q_proj gives the query and its gate [heads, 2*head_dim]; q and k are RMS-normalised per head (unit-offset weight)
    and rotated on their first `rotary_dim` channels; the indexer (index_qk_proj -> index heads' queries and one raw key
    per position) selects each query's positions (modules/sparse_indexer.qsa_select); softmax(q.k * head_dim^-0.5) over
    them in fp32, KV heads repeated to the query heads; the output times sigmoid(gate); o_proj. No sink: Qwen3.8's QSA
    refuses one (vllm qwen3_8_flash_next nvidia/qsa.py). Per sequence it carries K and V (post-rotation) and the
    indexer's raw keys for every position.

    `weights(layer, name)`: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, indexer.index_qk_proj,
    indexer.q_layernorm, indexer.k_layernorm -- the transformers names under `self_attn.`."""

    def __init__(self, *, heads: int, kv_heads: int, head_dim: int, rotary_dim: int, theta: float, eps: float,
                 index_heads: int, index_head_dim: int, budget: int, ratio: int, weights, mrope_section=None,
                 dtype: str = "bfloat16"):
        if heads % kv_heads or budget % ratio or rotary_dim > min(head_dim, index_head_dim):
            raise ValueError("gated sparse attention: heads a multiple of kv_heads, budget of whole blocks, "
                             "rotary_dim within both head widths")
        self.heads, self.kv_heads, self.head_dim, self.rotary_dim, self.theta = heads, kv_heads, head_dim, rotary_dim, theta
        self.eps, self.index_heads, self.index_head_dim, self.budget, self.ratio = eps, index_heads, index_head_dim, budget, ratio
        self.weights, self.mrope_section, self.dtype = weights, mrope_section, dtype

    def __call__(self, layer, x, step, state):
        from engine.modules.norm import rmsnorm_unit_offset
        from engine.modules.rotary import apply_rope, rope_tables
        from engine.modules.sparse_indexer import qsa_select
        linear = torch.nn.functional.linear
        w = lambda name: self.weights(layer, name)
        out = None
        for s in step.segments:
            xs = x[s.start:s.start + s.length]
            t, total = xs.shape[0], s.ctx + s.length
            cos_all, sin_all = rope_tables(torch.arange(total, device=xs.device), self.rotary_dim, self.theta,
                                           xs.dtype, self.mrope_section)
            cos, sin = cos_all[s.ctx:], sin_all[s.ctx:]
            # the indexer: its queries at this segment's positions, its raw keys at every position so far
            iq, ik = torch.split(linear(xs, w("indexer.index_qk_proj")),
                                 [self.index_heads * self.index_head_dim, self.index_head_dim], dim=-1)
            iq = apply_rope(rmsnorm_unit_offset(iq.reshape(t, self.index_heads, self.index_head_dim),
                                                w("indexer.q_layernorm"), self.eps), cos, sin)
            state.put_rows(layer, "qsa_raw_keys", s.seq, ik)
            raw_keys = state.rows(layer, "qsa_raw_keys", s.seq, total)
            allowed = torch.zeros(t, total, dtype=torch.bool, device=xs.device)
            for j in range(t):
                allowed[j, qsa_select(iq[j], raw_keys, s.ctx + j, self.ratio, self.budget // self.ratio, cos_all,
                                      sin_all, w("indexer.k_layernorm"), self.eps)] = True
            # the attention over the selected positions
            query, gate = torch.chunk(linear(xs, w("q_proj")).view(t, self.heads, 2 * self.head_dim), 2, dim=-1)
            gate = gate.reshape(t, -1)
            query = apply_rope(rmsnorm_unit_offset(query, w("q_norm"), self.eps), cos, sin)
            key = apply_rope(rmsnorm_unit_offset(linear(xs, w("k_proj")).view(t, self.kv_heads, self.head_dim),
                                                 w("k_norm"), self.eps), cos, sin)
            value = linear(xs, w("v_proj")).view(t, self.kv_heads, self.head_dim)
            state.put_rows(layer, "attention_kv", s.seq, torch.stack([key, value], dim=1))       # [t, 2, G, D]
            kv = state.rows(layer, "attention_kv", s.seq, total)
            key, value = kv[:, 0], kv[:, 1]
            groups = self.heads // self.kv_heads
            k_all = key.repeat_interleave(groups, dim=1).transpose(0, 1)        # [heads, total, D]
            v_all = value.repeat_interleave(groups, dim=1).transpose(0, 1)
            scores = torch.matmul(query.transpose(0, 1), k_all.transpose(1, 2)) * self.head_dim ** -0.5
            scores = scores.masked_fill(~allowed[None], torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attended = torch.matmul(probs, v_all).transpose(0, 1).reshape(t, -1)
            ys = linear(attended * torch.sigmoid(gate), w("o_proj"))
            if out is None:
                out = xs.new_empty(x.shape[0], ys.shape[-1])
            out[s.start:s.start + s.length] = ys
        return out

    def cache_specs(self, layers):
        from engine.base.cache_spec import PagedSpec, _ITEMSIZE
        size = _ITEMSIZE[self.dtype]
        return [PagedSpec("attention kv", len(layers), 2 * self.kv_heads * self.head_dim * size,
                          f"[2, kv_heads, head_dim] k and v {self.dtype} per position",
                          key="attention_kv", dtype=self.dtype, shape=(2, self.kv_heads, self.head_dim)),
                PagedSpec("qsa raw keys", len(layers), self.index_head_dim * size,
                          f"[index_head_dim] {self.dtype} per position, pooled per block at selection",
                          key="qsa_raw_keys", dtype=self.dtype, shape=(self.index_head_dim,))]


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
