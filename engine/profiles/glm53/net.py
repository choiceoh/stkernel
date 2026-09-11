"""GLM-5.3-Flash as the engine runs it (profile): the composition.

45 layers, each under mHC (four residual streams, sigmoid gates, a Sinkhorn
comb), of a KDA linear-attention block or a sparse-MLA block (nope, kpool
indexer), then a dense MLP (layers 0-2) or the NVFP4 MoE (288 experts top-8
plus one shared, noaux_tc router). Vocab-parallel embed and head. TP=4 by
heads; every collective is one `comm.all_reduce` at a block's row-parallel
output, exactly where the served path reduces.

What is NOT here, by design: no config object, no forward context, no
weight loader, no cache spec discovery, no graph-capture breaks. The model
is plain functions over VIEWS (`bind` takes the rank file's tensors as the
loader carved them from the arena, specs.py says which) and over CACHES it
is handed (the `Caches` protocol below: per-layer flat latent/pool regions
and per-slot KDA state, addressed through the caller's block tables). A
step's metadata is its arguments. That is what makes a step recordable and
a decode step capturable (D7, D12).

Kernels come from a `Lanes` table (lanes.py): the same eight names bound to
the judged torch references or to the served kernels; the composition does
not know which. Everything the served files did between those eight calls
-- projections, splits, norms, the router, the pool bookkeeping, the tail,
absorbed MQA -- is written here once, from the served files' algebra.
"""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as Fn

from engine.modules.sparse_indexer import fwht128_quant, select_with_tail, topk_positions
from engine.profiles.glm53 import specs
from engine.profiles.glm53.facts import Facts
from engine.profiles.glm53.lanes import Lanes, swiglu_clamped

BF16, F32, E4M3 = torch.bfloat16, torch.float32, torch.float8_e4m3fn
O_NORM_EPS = 1e-6           # FusedRMSNormGated(head_dim, activation="sigmoid") default; a fact to hold the layer judge to
K_NORM_EPS = 1e-6           # indexer LayerNorm(head_dim, eps=1e-6)


class Caches(Protocol):
    """What a sequence's state lives in. Flat per-layer regions (the arena's),
    addressed by slot ids the caller's block tables produce."""
    def kda(self, layer: int, slot: int) -> "tuple[torch.Tensor, torch.Tensor]": ...   # conv [C, K-1] bf16, rec [Hl, D, D] f32 (views, in place)
    def latent(self, layer: int) -> torch.Tensor: ...            # [S, 512] e4m3, all slots of the box
    def pool_keys(self, layer: int) -> torch.Tensor: ...         # [P, 128] e4m3 (FWHT-rotated, per-row scaled)
    def pool_scales(self, layer: int) -> torch.Tensor: ...       # [P] f32
    def tail(self, layer: int, slot: int) -> torch.Tensor: ...   # [kpool, 2, 128] bf16: raw k (0) and gate score (1), ring by pos % kpool
    def token_slots(self, seq: int, positions: torch.Tensor) -> torch.Tensor: ...   # int32 latent slots
    def pool_slots(self, seq: int, pool_ids: torch.Tensor) -> torch.Tensor: ...     # int32 pool slots


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


class Glm53Net:
    def __init__(self, F: Facts, comm, lanes: Lanes, layers=None):
        self.F, self.comm, self.lanes = F, comm, lanes
        self.W, self.rank = comm.world_size, comm.rank
        self.layers = list(range(F.layers)) if layers is None else list(layers)
        self.Hl = F.heads // self.W                 # MLA heads on this rank
        self.Hk = F.kda_heads // self.W             # KDA heads on this rank
        self.vp = F.vocab // self.W
        self.p = None
        self.probe = None                           # probe(block, layer, out) after every block, for judges

    # -- binding ----------------------------------------------------------------
    def specs(self):
        return specs.all_specs(self.F, self.W, self.layers)

    def bind(self, views: dict) -> None:
        from engine.base.params import bind
        self.p = bind(self.specs(), views)

    # -- blocks -----------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        start = self.rank * self.vp
        local = ids - start
        mask = (local < 0) | (local >= self.vp)
        h = Fn.embedding(local.masked_fill(mask, 0), self.p["embed"]).masked_fill(mask[:, None], 0)
        return self.comm.all_reduce(h) if self.W > 1 else h

    def head(self, h: torch.Tensor) -> torch.Tensor:
        local = Fn.linear(h, self.p["head"])
        return self.comm.all_gather(local, dim=-1) if self.W > 1 else local

    def _hc_pre(self, L: int, res: torch.Tensor, side: str):
        F, p, n = self.F, self.p, f"L{L}."
        norm_w = p[n + ("in_norm" if side == "attn" else "post_norm")]
        return self.lanes.mhc_pre(res, p[n + f"hc.{side}_fn"], p[n + f"hc.{side}_scale"], p[n + f"hc.{side}_base"],
                                  F.rms_eps, F.hc_eps, F.post_mult, F.sinkhorn, norm_w, F.rms_eps)

    def _kda(self, L: int, x: torch.Tensor, ctx: int, slot: int, caches: Caches) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.kda."
        T = x.shape[0]; Hl, D = self.Hk, F.kda_dim
        proj = Fn.linear(x, p[n + "in_proj"])
        qkv, b, f_a, g_a = proj.split([3 * Hl * D, Hl, D, D], dim=-1)
        conv_state, rec = caches.kda(L, slot)
        y, conv_new = self.lanes.conv_prefill(qkv, p[n + "conv"], conv_state if ctx > 0 else None)
        conv_state.copy_(conv_new.to(conv_state.dtype))
        q, k, v = (t.reshape(1, T, Hl, D) for t in y.split(Hl * D, dim=-1))
        g_raw = Fn.linear(f_a, p[n + "f_b"]).reshape(1, T, Hl, D)
        g_out = Fn.linear(g_a, p[n + "g_b"]).reshape(T, Hl, D)
        beta = torch.sigmoid(b.float()).reshape(1, T, Hl)
        o, state = self.lanes.kda_chunk(q, k, v, g_raw, beta, p[n + "A_log"], p[n + "dt_bias"],
                                        rec[None] if ctx > 0 else None, F.lower_bound)
        rec.copy_(state[0])
        of = o[0].float()                                                             # o_norm: rmsnorm(o) * w * sigmoid(g)
        core = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + O_NORM_EPS) * p[n + "o_norm"].float()
                * torch.sigmoid(g_out.float())).to(x.dtype)
        return self.comm.all_reduce(Fn.linear(core.reshape(T, Hl * D), p[n + "o_proj"]))

    def _indexer(self, L: int, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor, ctx: int,
                 seq: int, slot: int, caches: Caches):
        """kpool indexer for a chunk starting at `ctx` (pool-aligned): writes this
        chunk's complete pools and its tail, then selects for every query the
        top-k complete pools before it plus the in-progress tail, as token slots."""
        F, p, n = self.F, self.p, f"L{L}.idx."
        T = x.shape[0]; kp, nh, d = F.kpool, F.idx_heads, F.idx_dim
        if ctx % kp:
            raise ValueError(f"prefill chunk starts at {ctx}: chunks start on a pool boundary (block {F.block} does)")
        q = Fn.linear(qr, p[n + "wq_b"]).view(T, nh, d)
        k = Fn.linear(x, p[n + "wk"])
        k = Fn.layer_norm(k.float(), (d,), p[n + "k_norm_w"], p[n + "k_norm_b"], K_NORM_EPS).to(x.dtype)
        w = x.float() @ p[n + "w_heads"].T                                           # fp32 head gate, as served
        gate = Fn.linear(x, p[n + "gate"])                                           # [T, 128] per-channel pool score
        q8, qs = fwht128_quant(q.reshape(-1, d))
        q_rot = q8.float().view(T, nh, d)
        w_eff = w * qs.view(T, nh) * F.idx_scale
        # -- this chunk's pools ------------------------------------------------
        n_full = T // kp
        if n_full:
            pk8, ps = self.lanes.kpool_compress(k[: n_full * kp].view(n_full, kp, d), gate[: n_full * kp].view(n_full, kp, d), p[n + "ape"])
            ids = ctx // kp + torch.arange(n_full, device=x.device)
            pslots = caches.pool_slots(seq, ids).long()
            caches.pool_keys(L)[pslots] = pk8
            caches.pool_scales(L)[pslots] = ps.view(-1)
        rem = T - n_full * kp
        if rem:
            tail = caches.tail(L, slot)
            tail[:rem, 0] = k[n_full * kp:]
            tail[:rem, 1] = gate[n_full * kp:]
        # -- selection ----------------------------------------------------------
        seq_lens = (positions + 1).to(torch.int32)
        n_cand = int(seq_lens[-1].item()) // kp                                     # complete pools before the last query
        if n_cand:
            cand = caches.pool_slots(seq, torch.arange(n_cand, device=x.device)).long()
            kd = caches.pool_keys(L)[cand].float() * caches.pool_scales(L)[cand][:, None]
            logits = self.lanes.indexer_logits(q_rot, kd, w_eff)                    # [T, n_cand]
            pool_ids = topk_positions(logits, F.topk // kp, valid=seq_lens // kp)
        else:
            pool_ids = torch.full((T, F.topk // kp), -1, dtype=torch.int32, device=x.device)
        tokens = select_with_tail(pool_ids, seq_lens, kp)                            # positions, -1 padded
        tokens = tokens.sort(dim=1, descending=True).values                          # valid prefix first (set semantics)
        valid = (tokens >= 0).sum(1).to(torch.int32)
        slots = caches.token_slots(seq, tokens.clamp_min(0).reshape(-1)).view_as(tokens).masked_fill(tokens < 0, -1)
        return slots.contiguous(), valid

    def _dsa(self, L: int, x: torch.Tensor, positions: torch.Tensor, ctx: int, seq: int, slot: int, caches: Caches):
        F, p, n = self.F, self.p, f"L{L}.mla."
        T = x.shape[0]; Hl = self.Hl
        q_a, kv_c = Fn.linear(x, p[n + "qkv_a"]).split([F.q_lora, F.kv_lora], dim=-1)
        qr = rmsnorm(q_a, p[n + "q_a_norm"], F.rms_eps)
        q = Fn.linear(qr, p[n + "q_b"]).view(T, Hl, F.qk_nope)
        kv_n = rmsnorm(kv_c, p[n + "kv_a_norm"], F.rms_eps)
        latent = caches.latent(L)
        latent[caches.token_slots(seq, positions).long()] = kv_n.to(E4M3)           # fp8 KV, scale 1 (no kv scales in the checkpoint)
        slots, valid = self._indexer(L, x, qr, positions, ctx, seq, slot, caches)
        kv_b = p[n + "kv_b"].view(Hl, F.qk_nope + F.v_dim, F.kv_lora)
        w_uk, w_uv = kv_b[:, : F.qk_nope, :], kv_b[:, F.qk_nope:, :]
        q_abs = torch.einsum("thd,hdc->thc", q, w_uk)                                # absorb W_UK: MQA over the latent
        ctx_lat = self.lanes.mla_sparse(q_abs.contiguous(), latent, slots, valid, F.mla_scale, 1.0)
        o = torch.einsum("thc,hvc->thv", ctx_lat, w_uv)                              # un-absorb W_UV
        return self.comm.all_reduce(Fn.linear(o.reshape(T, Hl * F.v_dim), p[n + "o_proj"]))

    def _dense(self, L: int, x: torch.Tensor) -> torch.Tensor:
        p, n = self.p, f"L{L}.mlp."
        g, u = Fn.linear(x, p[n + "gate_up"]).chunk(2, dim=-1)
        return self.comm.all_reduce(Fn.linear(swiglu_clamped(g, u, self.F.swiglu_limit), p[n + "down"]))

    def route(self, L: int, x: torch.Tensor):
        """noaux_tc: sigmoid scores fp32, select by score + bias, weight by the
        raw scores renormalised, times routed_scaling_factor."""
        F, p, n = self.F, self.p, f"L{L}.moe."
        s = torch.sigmoid(x.float() @ p[n + "gate"].float().T)
        sel = (s + p[n + "bias"]).topk(F.topk_experts, dim=-1).indices
        w = s.gather(-1, sel)
        return sel.to(torch.int32), w / w.sum(-1, keepdim=True) * F.routed_scale

    def _moe(self, L: int, x: torch.Tensor) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.moe."
        T = x.shape[0]
        sel, w = self.route(L, x)
        out = torch.zeros(T, F.hidden, dtype=F32, device=x.device)
        for e in sel.unique().tolist():
            rows, k = (sel == e).nonzero(as_tuple=True)
            y = self.lanes.expert(x[rows], p[n + "w13"][e], p[n + "w13_s"][e], p[n + "w13_mult"][e], p[n + "a13_mult"][e],
                                  p[n + "w2"][e], p[n + "w2_s"][e], p[n + "w2_mult"][e], p[n + "a2_mult"][e], F.swiglu_limit)
            out.index_add_(0, rows, y.float() * w[rows, k][:, None])
        g, u = Fn.linear(x, p[n + "sh_gate_up"]).chunk(2, dim=-1)
        out += Fn.linear(swiglu_clamped(g, u, F.swiglu_limit), p[n + "sh_down"]).float()
        return self.comm.all_reduce(out.to(x.dtype))

    # -- the step -------------------------------------------------------------------
    def prefill(self, ids: torch.Tensor, ctx: int, seq: int, slot: int, caches: Caches, finish: bool = True):
        """One chunk of one sequence: tokens `ids` at positions ctx..ctx+T-1.
        Returns the final hidden states [T, hidden] (post final norm) when
        `finish` (the chain ends at the model's last layer or the caller says
        so), else the raw mHC carry (res, post, comb, x) for inspection."""
        F = self.F
        T = ids.shape[0]
        positions = ctx + torch.arange(T, device=ids.device)
        x = self.embed(ids)
        res = x[:, None, :].expand(T, F.hc, F.hidden).contiguous()                   # hc_expand
        post = comb = None
        for L in self.layers:
            if post is not None:
                res = self.lanes.mhc_post(x, res, post, comb)
            post, comb, x = self._hc_pre(L, res, "attn")
            x = self._dsa(L, x, positions, ctx, seq, slot, caches) if F.is_dsa(L) else self._kda(L, x, ctx, slot, caches)
            if self.probe:
                self.probe("dsa" if F.is_dsa(L) else "kda", L, x)
            res = self.lanes.mhc_post(x, res, post, comb)
            post, comb, x = self._hc_pre(L, res, "ffn")
            x = self._moe(L, x) if F.is_moe(L) else self._dense(L, x)
            if self.probe:
                self.probe("moe" if F.is_moe(L) else "dense", L, x)
        if not finish:
            return res, post, comb, x
        res = self.lanes.mhc_post(x, res, post, comb)
        h = res.float().mean(1).to(x.dtype)                                           # hc_contract
        return rmsnorm(h, self.p["norm"], F.rms_eps)


def _selfcheck() -> None:
    from engine.base.comm import Comm
    from engine.profiles.glm53 import facts, lanes
    F = facts.load()
    net = Glm53Net(F, Comm.init(rank=0, world=1), lanes.reference(), layers=range(0, 5))
    names = [s.name for s in net.specs()]
    assert names[:3] == ["embed", "norm", "head"] and "L3.moe.w13" in names and "L4.kda.in_proj" in names
    assert net.Hl == 64 and net.Hk == 64 and net.vp == F.vocab
    print(f"  net: glm53 layers 0-4 at world 1 declares {len(names)} tensors; embed/norm/head + kda/dsa/moe/dense blocks composed OK")


if __name__ == "__main__":
    _selfcheck()
