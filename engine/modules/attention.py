"""Softmax attention over a sequence's history (module): the family GLM-5.3 (MLA + DSA k-pool indexer), Qwen3.8
(gated GQA + QSA), DeepSeek-V3 / Kimi K3 / Ling-3.0 (dense MLA), MiniMax-M3 (GQA + MSA block indexer) and Inkling
(GQA with a sliding window or a relative position bias, short convs on k and v) share -- one feature, and the axes the
six differ on.

One computation: a query attends the positions a *selection* allows, softmax(q.k * scale + bias) v, the keys and
values (or the latent they expand from) kept once per position and layer (paged rows). Around it (`Attention`):

    form         "gqa"  q, k, v per head from x; kv_heads <= heads, a KV head serving heads // kv_heads query heads
                 "mla"  DeepSeek's latent: kv_a(x) -> [latent | k_rope]; the latent RMS-normalised and cached; k_nope
                        and v expanded from it by kv_b at attention time; q from x, or q_b(norm(q_a(x))) with q_lora;
                        q and k are [nope | rope]
    rotary_dim   0: no rotation (GLM-5.3's MLA, Kimi K3's NoPE, Inkling); GQA rotates the first rotary_dim channels
                 of a head (Qwen3.8 head_dim/4, MiniMax-M3 head_dim/2); MLA rotates its rope part (DeepSeek's 64)
    interleaved  False: neox halves (rotate_half); True: DeepSeek's interleaved pairs (config rope_interleave)
    qk_norm      None | "rms" (T5: the weight after the cast -- Inkling) | "rms_unit_offset" ((1 + w) in fp32 --
                 Qwen3.8, MiniMax-M3), per head before the rotation. MLA's latent norms are the form's, not this axis
    gate         None | "channel" (sigmoid(gate(x)) on every output channel -- Qwen3.8, Kimi K3) | "head" (Ling-3.0)
    scale        the softmax scale: head_dim^-0.5, (nope + rope)^-0.5, 1/head_dim (Inkling), yarn-mscaled (DeepSeek)
    select       the positions a query may attend, causal ones only: Causal (all), Window(w), or an indexer with its
                 own keys per position -- QSA (Qwen3.8: blocks of pooled raw keys), DSAKpool (GLM-5.3: gated pools of
                 keys, the incomplete tail always), MSA (MiniMax-M3: block max of raw key scores, local blocks always)
    sink         one extra logit per head in the softmax's denominator (DeepSeek-V4.1's attn_sink; the reference in
                 sparse_attention.sparse_attn)
    relative     Inkling's relative position bias: r(x) [heads, d_rel] against a bank [d_rel, extent], gathered at the
                 query-key distance and zero beyond it; `log_scaling` (n_floor, alpha) scales q and the bias by
                 1 + alpha * log(max((pos + 1) / n_floor, 1)) on Inkling's global layers
    kv_conv      Inkling's short convs (kernel k) on k and v before the norm: fp32 depthwise conv plus the input

What is NOT an axis: the softmax in fp32 rounded to the activation dtype, values in that dtype, masking with the
dtype's minimum (transformers' eager attention, every one of the six), the fp32 key/query rotation tables cast to the
activation dtype. Two places the reference follows the served kernels rather than transformers' eager path: a query
allowed no position attends nothing (zero, not uniform over every key), and an indexer breaks equal scores toward
the earlier pool or block (relu leaves many at zero; torch.topk's order there depends on how many candidates exist,
so a prefill in pieces would not select as the whole does). Nor is the weight layout: `named(scheme, source)` maps the family's canonical names onto each
checkpoint's (Qwen3.8's q_proj carries the gate beside each head's query; the binder splits it).

The reference is held to transformers on the CPU (tests/test_engine_attention_family.py, test_engine_composition.py):
qwen4_exp (GQA + QSA + gate), glm5_next (MLA + DSAKpool), deepseek_v3 (dense MLA, both rotations, with and without
q_lora), minimax_m3_vl (GQA + MSA), inkling (window + relative bias + k/v convs; the global layer's log scaling); the
sink to sparse_attention.sparse_attn. Kimi K3's and Ling-3.0's settings (the "kimi" scheme) have no local oracle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from engine.modules.norm import rmsnorm, rmsnorm_unit_offset
from engine.modules.rotary import apply_rope, rope_tables


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """DeepSeek's interleaved pairs (x[0::2], x[1::2]) rotated by the pair's angle and written as [even | odd]
    (transformers deepseek_v3 apply_rotary_pos_emb_interleave, op for op; the tables are the neox cat(freqs, freqs)).
    x [N, heads, R] or [N, R]; cos/sin [N, R]."""
    half = cos.shape[-1] // 2
    c, s = cos[..., :half], sin[..., :half]
    if x.ndim == 3:
        c, s = c[:, None, :], s[:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def _norm(kind: "str | None", x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if kind == "rms_unit_offset":
        return rmsnorm_unit_offset(x, weight, eps)
    if kind == "rms":
        return rmsnorm(x, weight, eps)
    raise ValueError(f"qk_norm is None, 'rms' or 'rms_unit_offset', not {kind!r}")


@dataclass
class Query:
    """What a selection sees of one segment: the feature, the segment, the step's tensors, the state."""
    feature: "Attention"
    layer: int
    seq: int
    xs: torch.Tensor                    # [t, hidden] the segment's inputs
    ctx: int                            # positions before the segment
    total: int                          # ctx + t
    state: object
    tables: "tuple[torch.Tensor, torch.Tensor] | None"     # (cos, sin) [total, rotary_dim] or None
    q_resid: "torch.Tensor | None"      # MLA's normalised low-rank query [t, q_lora] (DSA scores from it)

    @property
    def t(self) -> int:
        return self.xs.shape[0]

    def w(self, name: str) -> torch.Tensor:
        return self.feature.weights(self.layer, name)

    def positions(self) -> torch.Tensor:
        return torch.arange(self.ctx, self.total, device=self.xs.device)


class Causal:
    """Every position up to the query's."""
    def specs(self, feature, layers):
        return []

    def allowed(self, at: Query):
        return None


class Window:
    """Inkling's sliding window: the `window` positions ending at the query's (distance < window)."""
    def __init__(self, window: int):
        if window <= 0:
            raise ValueError(f"a window is positive, not {window}")
        self.window = window

    def specs(self, feature, layers):
        return []

    def allowed(self, at: Query):
        distance = at.positions()[:, None] - torch.arange(at.total, device=at.xs.device)[None, :]
        return (distance >= 0) & (distance < self.window)


class QSA:
    """Qwen3.8's indexer (transformers qwen4_exp Qwen4ExpTextQSAIndexer, op for op; modules/sparse_indexer.qsa_select):
    one raw key per position; every complete block of `ratio` visible positions pools its raw keys, is RMS-normalised
    (unit offset) and rotated at its first position; a block scores sum over the index heads of relu(q . key) / sqrt(D);
    the top budget/ratio blocks' positions and the tail of an incomplete block. The index queries are RMS-normalised
    and rotated like the attention's (the same tables).

    Weights: index_qk (index heads' queries and the key, fused), index_q_norm, index_k_norm."""
    def __init__(self, *, index_heads: int, index_head_dim: int, budget: int, ratio: int):
        if budget % ratio or budget <= 0:
            raise ValueError("QSA's budget is whole blocks of `ratio` positions")
        self.index_heads, self.index_head_dim, self.budget, self.ratio = index_heads, index_head_dim, budget, ratio

    def specs(self, feature, layers):
        from engine.base.cache_spec import PagedSpec, _ITEMSIZE
        return [PagedSpec("qsa raw keys", len(layers), self.index_head_dim * _ITEMSIZE[feature.dtype],
                          f"[index_head_dim] {feature.dtype} per position, pooled per block at selection",
                          key="qsa_raw_keys", dtype=feature.dtype, shape=(self.index_head_dim,))]

    def allowed(self, at: Query):
        from engine.modules.sparse_indexer import qsa_select
        if at.tables is None:
            raise ValueError("QSA rotates its index queries and keys: the attention needs a rotary_dim")
        cos_all, sin_all = at.tables
        t, eps = at.t, at.feature.eps
        iq, ik = torch.split(torch.nn.functional.linear(at.xs, at.w("index_qk")),
                             [self.index_heads * self.index_head_dim, self.index_head_dim], dim=-1)
        iq = apply_rope(rmsnorm_unit_offset(iq.reshape(t, self.index_heads, self.index_head_dim), at.w("index_q_norm"), eps),
                        cos_all[at.ctx:], sin_all[at.ctx:])
        at.state.put_rows(at.layer, "qsa_raw_keys", at.seq, ik)
        raw_keys = at.state.rows(at.layer, "qsa_raw_keys", at.seq, at.total)
        allowed = torch.zeros(t, at.total, dtype=torch.bool, device=at.xs.device)
        for j in range(t):
            allowed[j, qsa_select(iq[j], raw_keys, at.ctx + j, self.ratio, self.budget // self.ratio, cos_all, sin_all,
                                  at.w("index_k_norm"), eps)] = True
        return allowed


class DSAKpool:
    """GLM-5.3's DSA indexer with k-pool compression (transformers glm5_next Glm5NextTextIndexer, op for op): per
    position one key, LayerNorm(wk(x)), and a gate vector gate(x); positions group into pools of `kpool` from position
    0, a complete pool's key the per-channel softmax over its slots of gate + ape times the keys; the index queries
    wq_b(q_resid) score relu(q . pool_key / sqrt(D)) per head, summed with weights(x) / sqrt(heads); the top
    topk/kpool pools whose last position the query sees, expanded to their positions, then (`always_tail`) the
    positions after the last complete pool the query sees.

    Weights: index_wq_b, index_wk, index_k_norm, index_k_norm_bias, index_weights, index_gate, index_ape."""
    LAYERNORM_EPS = 1e-6                                   # nn.LayerNorm(head_dim, eps=1e-6) in the model

    def __init__(self, *, index_heads: int, index_head_dim: int, topk: int, kpool: int, always_tail: bool = True):
        if kpool < 1 or topk < kpool:
            raise ValueError("DSA k-pool: a pool of at least one position, a budget of at least one pool")
        self.index_heads, self.index_head_dim, self.topk, self.kpool = index_heads, index_head_dim, topk, kpool
        self.always_tail = always_tail

    def specs(self, feature, layers):
        from engine.base.cache_spec import PagedSpec, _ITEMSIZE
        return [PagedSpec("dsa index keys", len(layers), 2 * self.index_head_dim * _ITEMSIZE[feature.dtype],
                          f"[2, index_head_dim] {feature.dtype} per position: the key and its pool gate",
                          key="dsa_index_keys", dtype=feature.dtype, shape=(2, self.index_head_dim))]

    def allowed(self, at: Query):
        if at.q_resid is None:
            raise ValueError("DSA scores from MLA's low-rank query: the attention is form 'mla' with q_lora")
        linear = torch.nn.functional.linear
        t, D, Hi, kpool = at.t, self.index_head_dim, self.index_heads, self.kpool
        q = linear(at.q_resid, at.w("index_wq_b")).view(t, Hi, D)
        k = torch.nn.functional.layer_norm(linear(at.xs, at.w("index_wk")), (D,), at.w("index_k_norm"),
                                           at.w("index_k_norm_bias"), self.LAYERNORM_EPS)
        gates = linear(at.xs, at.w("index_gate"))
        at.state.put_rows(at.layer, "dsa_index_keys", at.seq, torch.stack([k, gates], dim=1))
        rows = at.state.rows(at.layer, "dsa_index_keys", at.seq, at.total)
        k_all, g_all = rows[:, 0], rows[:, 1]
        pos_q = at.positions()
        allowed = torch.zeros(t, at.total, dtype=torch.bool, device=at.xs.device)
        pools = at.total // kpool                                              # complete pools only
        if pools > 0:
            gk = k_all[:pools * kpool].view(pools, kpool, D)
            logits = g_all[:pools * kpool].view(pools, kpool, D).float() + at.w("index_ape").float()[None]
            pool_keys = (torch.softmax(logits, dim=1).to(gk.dtype) * gk).sum(dim=1)                 # [pools, D]
            scores = torch.relu(torch.matmul(q.float(), pool_keys.float().T) * D ** -0.5)          # [t, Hi, pools]
            weights = linear(at.xs, at.w("index_weights")).float() * Hi ** -0.5                     # [t, Hi]
            index_scores = torch.einsum("thp,th->tp", scores, weights)
            pool_end = torch.arange(pools, device=at.xs.device) * kpool + kpool - 1
            candidates = pool_end[None, :] <= pos_q[:, None]                                        # [t, pools]
            index_scores = index_scores.masked_fill(~candidates, torch.finfo(index_scores.dtype).min)
            select_k = min(self.topk // kpool, pools)
            # the top pools by score; equal scores (relu leaves many at zero) go to the earlier pool, so the choice
            # does not depend on how many pools exist yet (a prefill in pieces selects as the whole one does)
            chosen = torch.sort(index_scores, dim=-1, descending=True, stable=True).indices[:, :select_k]
            valid = candidates.gather(-1, chosen)
            for j in range(t):
                for p in chosen[j][valid[j]].tolist():
                    allowed[j, p * kpool:(p + 1) * kpool] = True
        if self.always_tail:
            for j in range(t):
                visible = int(pos_q[j]) + 1
                allowed[j, visible - visible % kpool:visible] = True
        return allowed


class MSA:
    """MiniMax-M3's lightning indexer (transformers minimax_m3_vl MiniMaxM3VLIndexer, op for op): per position one key,
    RMS-normalised (unit offset) and rotated like the attention's; index queries the same; a key's score q . k per
    index head, max-pooled over blocks of `block` positions from position 0; the query's own block and the
    `local_blocks` - 1 before it always win; the top `topk_blocks` blocks per index head, each index head serving
    heads // index_heads query heads in order.

    Weights: index_q, index_k, index_q_norm, index_k_norm."""
    def __init__(self, *, index_heads: int, index_head_dim: int, block: int, topk_blocks: int, local_blocks: int = 1):
        if block <= 0 or topk_blocks <= 0 or local_blocks < 0:
            raise ValueError("MSA: a positive block and budget, a nonnegative local count")
        self.index_heads, self.index_head_dim, self.block = index_heads, index_head_dim, block
        self.topk_blocks, self.local_blocks = topk_blocks, local_blocks

    def specs(self, feature, layers):
        from engine.base.cache_spec import PagedSpec, _ITEMSIZE
        return [PagedSpec("msa index keys", len(layers), self.index_head_dim * _ITEMSIZE[feature.dtype],
                          f"[index_head_dim] {feature.dtype} per position, normalised and rotated",
                          key="msa_index_keys", dtype=feature.dtype, shape=(self.index_head_dim,))]

    def allowed(self, at: Query):
        linear = torch.nn.functional.linear
        t, D, Hi, block, eps = at.t, self.index_head_dim, self.index_heads, self.block, at.feature.eps
        iq = rmsnorm_unit_offset(linear(at.xs, at.w("index_q")).view(t, Hi, D), at.w("index_q_norm"), eps)
        ik = rmsnorm_unit_offset(linear(at.xs, at.w("index_k")).view(t, 1, D), at.w("index_k_norm"), eps)
        if at.tables is not None:
            cos, sin = at.tables[0][at.ctx:], at.tables[1][at.ctx:]
            iq, ik = apply_rope(iq, cos, sin), apply_rope(ik, cos, sin)
        at.state.put_rows(at.layer, "msa_index_keys", at.seq, ik[:, 0])
        k_all = at.state.rows(at.layer, "msa_index_keys", at.seq, at.total)                           # [total, D]
        pos_q, pos_k = at.positions(), torch.arange(at.total, device=at.xs.device)
        scores = torch.matmul(iq.float().transpose(0, 1), k_all.float().T)                            # [Hi, t, total]
        scores = scores.masked_fill((pos_k[None, :] > pos_q[:, None])[None], float("-inf"))
        blocks = -(-at.total // block)
        pad = blocks * block - at.total
        if pad:
            scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
        block_scores = scores.view(Hi, t, blocks, block).amax(dim=-1)                                # [Hi, t, blocks]
        q_block = pos_q // block
        if self.local_blocks > 0:
            local = (q_block[:, None] - torch.arange(self.local_blocks, device=at.xs.device)[None, :]).clamp(min=0)
            block_scores.scatter_(-1, local[None].expand(Hi, -1, -1), float("inf"))
        ordered = torch.sort(block_scores, dim=-1, descending=True, stable=True)       # ties to the earlier block
        top_values, top_indices = ordered.values[..., :min(self.topk_blocks, blocks)], ordered.indices[..., :min(self.topk_blocks, blocks)]
        chosen = top_indices.masked_fill(top_values == float("-inf"), -1)                            # [Hi, t, k]
        keep = torch.zeros(Hi, t, blocks + 1, dtype=torch.bool, device=at.xs.device)
        keep.scatter_(-1, chosen.masked_fill(chosen < 0, blocks), True)
        keep = keep[..., :blocks].repeat_interleave(block, dim=-1)[..., :at.total]                   # [Hi, t, total]
        return keep.repeat_interleave(at.feature.heads // Hi, dim=0)


class Attention:
    """The family's layer as a token mixer (engine/base/composition.Feature): the axes of the module docstring.

    `weights(layer, name)` answers the canonical names (`named` maps a checkpoint's): gqa -- q, k, v, o, q_norm,
    k_norm; mla -- q | q_a, q_a_norm, q_b; kv_a, kv_norm, kv_b, o; gate; sink; r, rel_bank; k_conv, v_conv; and the
    selection's own. Per sequence it keeps, per position, the KV rows (gqa) or the latent rows (mla) and the
    selection's keys; with kv_conv, the convs' last inputs."""

    def __init__(self, *, form: str = "gqa", heads: int, head_dim: "int | None" = None, kv_heads: "int | None" = None,
                 latent: "int | None" = None, nope: "int | None" = None, rope: int = 0, v_dim: "int | None" = None,
                 q_lora: "int | None" = None, rotary_dim: int = 0, theta: float = 10000.0, interleaved: bool = False,
                 mrope_section=None, qk_norm: "str | None" = None, gate: "str | None" = None, scale: "float | None" = None,
                 select=None, sink: bool = False, relative: "tuple[int, int] | None" = None,
                 log_scaling: "tuple[float, float] | None" = None, kv_conv: int = 0, eps: float = 1e-6, weights=None,
                 dtype: str = "bfloat16"):
        if form not in ("gqa", "mla"):
            raise ValueError(f"form is 'gqa' or 'mla', not {form!r}")
        if form == "gqa":
            kv_heads = kv_heads or heads
            if head_dim is None or heads % kv_heads:
                raise ValueError("gqa: a head_dim, and heads a multiple of kv_heads")
            if rotary_dim > head_dim:
                raise ValueError("gqa rotates at most head_dim channels")
            self.q_dim = self.v_dim = head_dim
        else:
            if None in (latent, nope, v_dim) or (rotary_dim and rotary_dim != rope):
                raise ValueError("mla: latent, nope and v_dim widths; rotary_dim is 0 or the rope width")
            if kv_conv or relative is not None or qk_norm is not None:
                raise ValueError("mla has neither per-head qk norms, k/v convs nor a relative bias")
            self.q_dim, self.v_dim = nope + rope, v_dim
        if qk_norm not in (None, "rms", "rms_unit_offset"):
            raise ValueError(f"qk_norm is None, 'rms' or 'rms_unit_offset', not {qk_norm!r}")
        if gate not in (None, "channel", "head"):
            raise ValueError(f"gate is None, 'channel' or 'head', not {gate!r}")
        if rotary_dim < 0 or rotary_dim % 2:
            raise ValueError(f"a rotary width is even and nonnegative, not {rotary_dim}")
        if log_scaling is not None and relative is None:
            raise ValueError("log scaling belongs to the relative-bias attention")
        self.form, self.heads, self.head_dim, self.kv_heads = form, heads, head_dim, kv_heads
        self.latent, self.nope, self.rope, self.q_lora = latent, nope, rope, q_lora
        self.rotary_dim, self.theta, self.interleaved, self.mrope_section = rotary_dim, theta, interleaved, mrope_section
        self.qk_norm, self.gate, self.sink, self.relative, self.log_scaling = qk_norm, gate, sink, relative, log_scaling
        self.kv_conv, self.eps, self.weights, self.dtype = kv_conv, eps, weights, dtype
        self.scale = self.q_dim ** -0.5 if scale is None else scale
        self.select = Causal() if select is None else select

    # -- pieces ------------------------------------------------------------------------------------------------------
    def _rotate(self, x, cos, sin):
        return apply_rope_interleaved(x, cos, sin) if self.interleaved else apply_rope(x, cos, sin)

    def _conv(self, layer, seq, key, x, weight, state):
        """Inkling's short conv: fp32 depthwise causal conv (no activation) plus its input, rounded once."""
        from engine.modules.causal_conv import causal_conv1d
        xf = x.float()
        y, held = causal_conv1d(xf, weight.reshape(weight.shape[0], -1).float(), None, state.get(layer, key, seq), None)
        state.put(layer, key, seq, held)
        return (y.float() + xf).to(x.dtype)

    def _relative_bias(self, xs, w, pos_q, total):
        """[heads, t, total]: r(x) . bank at the query-key distance, zero outside [0, extent)."""
        d_rel, extent = self.relative
        t = xs.shape[0]
        rel = torch.nn.functional.linear(xs, w("r")).view(t, self.heads, d_rel)
        logits = torch.matmul(rel, w("rel_bank")).transpose(0, 1)                                   # [heads, t, extent]
        distance = pos_q[:, None] - torch.arange(total, device=xs.device)[None, :]                    # [t, total]
        index = distance.clamp(0, extent - 1)[None].expand(self.heads, -1, -1)
        bias = logits.gather(-1, index)
        return bias.masked_fill(((distance < 0) | (distance >= extent))[None], 0.0)

    def _softmax_with_sink(self, scores, sink):
        """exp(s - m) / (sum exp(s - m) + exp(sink - m)) in fp32 (sparse_attention.sparse_attn's arithmetic)."""
        s = scores.float()
        row_max = s.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        weights = torch.exp(s - row_max)
        denom = weights.sum(dim=-1, keepdim=True) + torch.exp(sink.float().view(-1, 1, 1) - row_max)
        return weights / denom

    # -- the feature --------------------------------------------------------------------------------------------------
    def __call__(self, layer, x, step, state):
        linear = torch.nn.functional.linear
        w = lambda name: self.weights(layer, name)
        H, eps = self.heads, self.eps
        out = None
        for s in step.segments:
            xs = x[s.start:s.start + s.length]
            t, ctx, total = xs.shape[0], s.ctx, s.ctx + s.length
            pos_q = torch.arange(ctx, total, device=xs.device)
            tables = None
            if self.rotary_dim:
                tables = rope_tables(torch.arange(total, device=xs.device), self.rotary_dim, self.theta, xs.dtype,
                                     self.mrope_section)
                cos, sin = tables[0][ctx:], tables[1][ctx:]
            q_resid = None
            if self.form == "gqa":
                G, D = self.kv_heads, self.head_dim
                q = linear(xs, w("q")).view(t, H, D)
                k, v = linear(xs, w("k")), linear(xs, w("v"))
                if self.kv_conv:
                    k = self._conv(layer, s.seq, "attention_k_conv", k, w("k_conv"), state)
                    v = self._conv(layer, s.seq, "attention_v_conv", v, w("v_conv"), state)
                k, v = k.view(t, G, D), v.view(t, G, D)
                if self.qk_norm:
                    q, k = _norm(self.qk_norm, q, w("q_norm"), eps), _norm(self.qk_norm, k, w("k_norm"), eps)
                if tables is not None:
                    q, k = self._rotate(q, cos, sin), self._rotate(k, cos, sin)
                state.put_rows(layer, "attention_kv", s.seq, torch.stack([k, v], dim=1))          # [t, 2, G, D]
                kv = state.rows(layer, "attention_kv", s.seq, total)
                groups = H // G
                keys = kv[:, 0].repeat_interleave(groups, dim=1).transpose(0, 1)                  # [H, total, D]
                values = kv[:, 1].repeat_interleave(groups, dim=1).transpose(0, 1)
            else:
                nope, rope, latent = self.nope, self.rope, self.latent
                if self.q_lora:
                    q_resid = rmsnorm(linear(xs, w("q_a")), w("q_a_norm"), eps)
                    q = linear(q_resid, w("q_b"))
                else:
                    q = linear(xs, w("q"))
                q = q.view(t, H, nope + rope)
                lat, k_rope = torch.split(linear(xs, w("kv_a")), [latent, rope], dim=-1)
                lat = rmsnorm(lat, w("kv_norm"), eps)
                if tables is not None:
                    q = torch.cat([q[..., :nope], self._rotate(q[..., nope:], cos, sin)], dim=-1)
                    k_rope = self._rotate(k_rope, cos, sin)
                state.put_rows(layer, "attention_latent", s.seq, torch.cat([lat, k_rope], dim=-1))  # [t, latent+rope]
                rows = state.rows(layer, "attention_latent", s.seq, total)
                expanded = linear(rows[:, :latent], w("kv_b")).view(total, H, nope + self.v_dim)
                k_rope_all = rows[:, latent:][:, None, :].expand(total, H, rope)
                keys = torch.cat([expanded[..., :nope], k_rope_all], dim=-1).transpose(0, 1)         # [H, total, nope+rope]
                values = expanded[..., nope:].transpose(0, 1)                                       # [H, total, v]
            qh = q.transpose(0, 1)                                                                  # [H, t, Dq]
            bias = None
            if self.relative is not None:
                bias = self._relative_bias(xs, w, pos_q, total)
                if self.log_scaling is not None:
                    n_floor, alpha = self.log_scaling
                    tau = (1.0 + alpha * torch.log(((pos_q + 1).float() / n_floor).clamp(min=1.0))).view(1, -1, 1)
                    qh = (qh.float() * tau).to(qh.dtype)
                    bias = (bias.float() * tau).to(bias.dtype)
            scores = torch.matmul(qh, keys.transpose(1, 2)) * self.scale
            if bias is not None:
                scores = scores + bias
            allowed = torch.arange(total, device=xs.device)[None, :] <= pos_q[:, None]             # causal [t, total]
            chosen = self.select.allowed(Query(self, layer, s.seq, xs, ctx, total, state, tables, q_resid))
            if chosen is not None:
                allowed = allowed & chosen
            if allowed.ndim == 2:
                allowed = allowed[None]
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            if self.sink:
                probs = self._softmax_with_sink(scores, w("sink")).to(qh.dtype)
            else:
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(qh.dtype)
            # a query allowed no position at all attends nothing (what the sparse kernels leave in its row:
            # sparse_attention.gqa_sparse, sparse_attn); transformers' eager path would spread it over every key
            probs = probs.masked_fill(~allowed.any(dim=-1, keepdim=True), 0.0)
            attended = torch.matmul(probs, values).transpose(0, 1)                                  # [t, H, Dv]
            if self.gate == "channel":
                attended = attended.reshape(t, -1) * torch.sigmoid(linear(xs, w("gate")))
            elif self.gate == "head":
                attended = attended * torch.sigmoid(linear(xs, w("gate"))).view(t, H, 1)
            ys = linear(attended.reshape(t, -1), w("o"))
            if out is None:
                out = xs.new_empty(x.shape[0], ys.shape[-1])
            out[s.start:s.start + s.length] = ys
        return out

    def cache_specs(self, layers):
        from engine.base.cache_spec import PagedSpec, SlotSpec, _ITEMSIZE
        size = _ITEMSIZE[self.dtype]
        if self.form == "gqa":
            specs = [PagedSpec("attention kv", len(layers), 2 * self.kv_heads * self.head_dim * size,
                               f"[2, kv_heads, head_dim] k and v {self.dtype} per position",
                               key="attention_kv", dtype=self.dtype, shape=(2, self.kv_heads, self.head_dim))]
            if self.kv_conv:
                width = self.kv_heads * self.head_dim
                for name in ("k", "v"):
                    specs.append(SlotSpec(f"attention {name} conv", len(layers), width * (self.kv_conv - 1) * 4,
                                          f"[kv_heads*head_dim, kernel-1] fp32: the {name} conv's last inputs",
                                          key=f"attention_{name}_conv", dtype="float32", shape=(width, self.kv_conv - 1)))
        else:
            specs = [PagedSpec("attention latent", len(layers), (self.latent + self.rope) * size,
                               f"[latent + rope] {self.dtype} per position: the normalised latent and the rotated key",
                               key="attention_latent", dtype=self.dtype, shape=(self.latent + self.rope,))]
        return specs + self.select.specs(self, layers)


# The family's canonical weight names on each checkpoint's: a string is a module's matrix (or a bare parameter), a
# tuple is concatenated along dim 0, a callable builds the tensor from `source` and the feature's dims.
def _qwen_q_half(part):
    def build(source, dims):
        heads, head_dim = dims["heads"], dims["head_dim"]
        return source("q_proj").view(heads, 2, head_dim, -1)[:, part].reshape(heads * head_dim, -1)
    return build


_MLA = {"q": "q_proj", "q_a": "q_a_proj", "q_a_norm": "q_a_layernorm", "q_b": "q_b_proj", "kv_a": "kv_a_proj_with_mqa",
        "kv_norm": "kv_a_layernorm", "kv_b": "kv_b_proj", "o": "o_proj"}
_GQA = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "q_norm": "q_norm", "k_norm": "k_norm"}
SCHEMES = {
    "qwen4_exp": {**_GQA, "q": _qwen_q_half(0), "gate": _qwen_q_half(1), "index_qk": "indexer.index_qk_proj",
                  "index_q_norm": "indexer.q_layernorm", "index_k_norm": "indexer.k_layernorm"},
    "glm5_next": {**_MLA, "index_wq_b": "indexer.wq_b", "index_wk": "indexer.wk", "index_k_norm": "indexer.k_norm",
                  "index_k_norm_bias": "indexer.k_norm.bias", "index_weights": "indexer.weights_proj",
                  "index_gate": "indexer.index_kpool_compress_gate", "index_ape": "indexer.index_kpool_compress_ape"},
    "deepseek_v3": dict(_MLA),
    "kimi": {**_MLA, "gate": "g_proj"},                                    # Kimi K3 / Ling-3.0: not held to an oracle
    "minimax_m3_vl": {**_GQA, "index_q": "indexer.q_proj", "index_k": "indexer.k_proj",
                      "index_q_norm": "indexer.q_norm", "index_k_norm": "indexer.k_norm"},
    "inkling": {**_GQA, "r": "r_proj", "rel_bank": "rel_logits_proj.proj", "k_conv": "k_sconv.conv1d",
                "v_conv": "v_sconv.conv1d"},
}


def named(scheme: str, source, **dims):
    """canonical name -> tensor over `source(checkpoint name)` (a module's matrix by its module name, a bare parameter
    by its own); `dims` (heads, head_dim) for the entries that carve a tensor; KeyError for a name the scheme or the
    checkpoint lacks."""
    table = SCHEMES[scheme]

    def get(name: str) -> torch.Tensor:
        if name not in table:
            raise KeyError(name)
        at = table[name]
        if callable(at):
            return at(source, dims)
        if isinstance(at, tuple):
            return torch.cat([source(n) for n in at], dim=0)
        return source(at)
    return get
