"""GLM-5.3-Flash as the engine runs it (profile): the composition.

45 layers, each under mHC (four residual streams, sigmoid gates, a Sinkhorn
comb), of a KDA linear-attention block or a sparse-MLA block (nope, kpool
indexer), then a dense MLP (layers 0-2) or the NVFP4 MoE (288 experts top-8
plus one shared, noaux_tc router). Vocab-parallel embed and head. TP=4 by
heads is the shape of the code, not a parameter: every rank-local size is a
fact (facts.py) and `comm` must be one rank of four -- base/comm.Comm on the
fleet, base/comm.LocalTP on one box. Every collective is one
`comm.all_reduce` at a block's row-parallel output, exactly where the served
path reduces.

ONE step function. A step is a list of segments -- (seq, slot, ctx, length)
-- over a flat token array: a prefill chunk is one long segment, a decode
step is n short ones (1 + K draft tokens each). Nothing in the composition
distinguishes them; the lanes do (chunk vs recurrent KDA). Every per-sequence
state is addressed BY POSITION: the KDA conv inputs and recurrent states
live in rings indexed by `position % width`, the latent/pool/tail caches by
position. So a rejected draft is not rolled back -- it is overwritten by the
next step's writes at the same positions, and the state the next step starts
from is simply the one after position ctx-1. That is the whole spec-decode
state machine, and it is the same code for prefill.

What is NOT here, by design: no config object, no forward context, no
weight loader, no cache spec discovery, no graph-capture breaks. The model
is plain functions over VIEWS (`bind` takes the rank file's tensors as the
loader carved them from the arena, specs.py says which) and over CACHES it
is handed (the `Caches` protocol). Kernels come from a `Lanes` table
(lanes.py): the same nine names bound to judged torch references or to the
served kernels; the composition does not know which.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as Fn

from engine.modules.sparse_indexer import fwht128_quant, select_with_tail, topk_positions
from engine.profiles.glm53 import specs
from engine.profiles.glm53.facts import TP, Facts
from engine.profiles.glm53.lanes import Lanes, swiglu_clamped

BF16, F32, E4M3 = torch.bfloat16, torch.float32, torch.float8_e4m3fn
O_NORM_EPS = 1e-6           # FusedRMSNormGated(head_dim, activation="sigmoid") default; a fact to hold the layer judge to
K_NORM_EPS = 1e-6           # indexer LayerNorm(head_dim, eps=1e-6)


@dataclass(frozen=True)
class Segment:
    seq: int
    slot: int
    ctx: int                    # tokens already computed for this sequence: positions ctx.. are this step's
    start: int                  # first token of this segment in the step's flat arrays
    length: int


@dataclass(frozen=True)
class Step:
    ids: torch.Tensor           # [N] int64
    segments: "tuple[Segment, ...]"

    def __post_init__(self):
        if self.ids.ndim != 1 or self.ids.dtype != torch.int64 or not self.segments:
            raise ValueError("a step needs a flat int64 token vector and nonempty segments")
        end, seqs, slots = 0, set(), set()
        for s in self.segments:
            if s.start != end or s.length <= 0 or s.ctx < 0 or s.seq < 0 or s.slot <= 0:
                raise ValueError("segments must cover tokens contiguously with valid contexts and state slots")
            if s.seq in seqs or s.slot in slots:
                raise ValueError("a sequence and its state slot may appear only once per step")
            seqs.add(s.seq); slots.add(s.slot)
            end += s.length
        if end != self.ids.numel():
            raise ValueError("segment lengths must cover every token exactly once")

    @property
    def positions(self) -> torch.Tensor:
        return torch.cat([torch.arange(s.ctx, s.ctx + s.length, device=self.ids.device) for s in self.segments])

    @staticmethod
    def prefill(ids: torch.Tensor, ctx: int, seq: int, slot: int) -> "Step":
        return Step(ids, (Segment(seq, slot, ctx, 0, ids.shape[0]),))

    @staticmethod
    def decode(chunks: "list[tuple[torch.Tensor, int, int, int]]") -> "Step":
        """chunks: (ids, ctx, seq, slot) per sequence, 1 + K draft tokens each."""
        if not chunks:
            raise ValueError("decode needs at least one sequence")
        segs, start = [], 0
        for ids, ctx, seq, slot in chunks:
            segs.append(Segment(seq, slot, ctx, start, ids.shape[0])); start += ids.shape[0]
        return Step(torch.cat([c[0] for c in chunks]), tuple(segs))


class Caches(Protocol):
    """What a sequence's state lives in. Flat per-layer regions (the arena's),
    addressed by slot ids the caller's block tables produce, plus per-slot rings."""
    def kda(self, layer: int, slot: int) -> "tuple[torch.Tensor, torch.Tensor]": ...   # conv ring [C, conv-1+K] bf16 by pos % width; rec ring [K+1, Hl, D, D] f32 by pos % (K+1)
    def latent(self, layer: int) -> torch.Tensor: ...            # [S, 512] e4m3, all slots of the box
    def pool_keys(self, layer: int) -> torch.Tensor: ...         # [P, 128] e4m3 (FWHT-rotated, per-row scaled)
    def pool_scales(self, layer: int) -> torch.Tensor: ...       # [P] f32
    def tail(self, layer: int, slot: int) -> torch.Tensor: ...   # [kpool-1+K, 2, 128] bf16: enough raw keys/gates to reject K drafts
    def token_slots(self, layer: int, seq: int, positions: torch.Tensor) -> torch.Tensor: ...   # int32 absolute latent slots
    def pool_slots(self, layer: int, seq: int, pool_ids: torch.Tensor) -> torch.Tensor: ...     # int32 absolute pool slots


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


class Glm53Net:
    def __init__(self, F: Facts, comm, lanes: Lanes, layers=None):
        if comm.world_size != TP:
            raise ValueError(f"glm53 is written for TP={TP}; comm has world {comm.world_size}")
        self.F, self.comm, self.lanes = F, comm, lanes
        self.rank = comm.rank
        self.layers = list(range(F.layers)) if layers is None else list(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(not 0 <= L < F.layers for L in self.layers):
            raise ValueError("model layers must be nonempty, unique and inside the profile")
        self.Hl = F.heads_local                      # MLA heads on this rank (16)
        self.Hk = F.kda_heads_local                  # KDA heads on this rank (16)
        self.vp = F.vocab_local
        self.conv_ring = F.conv - 1 + F.spec_k       # conv inputs kept per slot: the window plus K drafts
        self.rec_ring = F.spec_k + 1                 # recurrent states kept per slot: one per draft position
        self.p = None
        self.probe = None                            # probe(block, layer, out) after every block, for judges

    # -- binding ----------------------------------------------------------------
    def specs(self):
        return specs.all_specs(self.F, self.layers)

    def bind(self, views: dict) -> None:
        from engine.base.params import bind
        self.p = bind(self.specs(), views)

    # -- embed / head -------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        start = self.rank * self.vp
        local = ids - start
        mask = (local < 0) | (local >= self.vp)
        h = Fn.embedding(local.masked_fill(mask, 0), self.p["embed"]).masked_fill(mask[:, None], 0)
        return self.comm.all_reduce(h)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.comm.all_gather(Fn.linear(h, self.p["head"]), dim=-1)

    # -- mHC -----------------------------------------------------------------------
    def _hc_pre(self, L: int, res: torch.Tensor, side: str):
        F, p, n = self.F, self.p, f"L{L}."
        norm_w = p[n + ("in_norm" if side == "attn" else "post_norm")]
        return self.lanes.mhc_pre(res, p[n + f"hc.{side}_fn"], p[n + f"hc.{side}_scale"], p[n + f"hc.{side}_base"],
                                  F.rms_eps, F.hc_eps, F.post_mult, F.sinkhorn, norm_w, F.rms_eps)

    # -- KDA ------------------------------------------------------------------------
    def _kda(self, L: int, x: torch.Tensor, step: Step, caches: Caches) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.kda."
        N = x.shape[0]; Hl, D, K = self.Hk, F.kda_dim, F.conv
        proj = Fn.linear(x, p[n + "in_proj"])
        qkv_all, b_all, f_a, g_a = proj.split([3 * Hl * D, Hl, D, D], dim=-1)
        g_raw_all = Fn.linear(f_a, p[n + "f_b"]).view(N, Hl, D)
        g_out = Fn.linear(g_a, p[n + "g_b"]).view(N, Hl, D)
        beta_all = b_all                                                             # raw logits: each lane sigmoids as its kernel wants
        core = torch.empty(N, Hl, D, dtype=x.dtype, device=x.device)
        wc, wr = self.conv_ring, self.rec_ring
        for s in step.segments:
            sl = slice(s.start, s.start + s.length)
            conv_ring, rec_ring = caches.kda(L, s.slot)
            # conv: the K-1 inputs before ctx from the ring (positions < 0 are zero), then this segment's inputs
            hist_pos = s.ctx + torch.arange(-(K - 1), 0, device=x.device)
            hist = conv_ring[:, hist_pos.clamp_min(0) % wc].masked_fill((hist_pos < 0)[None, :], 0)
            y, _ = self.lanes.conv_prefill(qkv_all[sl], p[n + "conv"], hist if getattr(step, "captured", False) or s.ctx > 0 else None)
            keep = min(s.length, wc)
            pos = s.ctx + torch.arange(s.length - keep, s.length, device=x.device)
            conv_ring[:, pos % wc] = qkv_all[sl][-keep:].T
            q, k, v = (t.reshape(1, s.length, Hl, D) for t in y.split(Hl * D, dim=-1))
            g_raw, beta = g_raw_all[sl][None], beta_all[sl][None]
            if getattr(step, "captured", False):
                state0 = rec_ring.index_select(0, ((s.ctx - 1) % wr).reshape(1))
                state0 = state0.masked_fill(s.ctx <= 0, 0)
            else:
                state0 = rec_ring[(s.ctx - 1) % wr][None] if s.ctx > 0 else None
            if s.length > wr:                                                       # a prefill chunk: only the final state is kept
                o, state = self.lanes.kda_chunk(q, k, v, g_raw, beta, p[n + "A_log"], p[n + "dt_bias"], state0, F.lower_bound)
                rec_ring[(s.ctx + s.length - 1) % wr] = state[0]
            else:                                                                   # a decode/verify step: one state per position
                o, states = self.lanes.kda_recurrent(q, k, v, g_raw, beta, p[n + "A_log"], p[n + "dt_bias"], state0, F.lower_bound)
                for i in range(s.length):
                    if getattr(step, "captured", False):
                        rec_ring.index_copy_(0, ((s.ctx + i) % wr).reshape(1), states[i:i+1])
                    else:
                        rec_ring[(s.ctx + i) % wr] = states[i]
            core[sl] = o[0]
        of = core.float()                                                              # o_norm: rmsnorm(o) * w * sigmoid(g)
        out = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + O_NORM_EPS) * p[n + "o_norm"].float()
               * torch.sigmoid(g_out.float())).to(x.dtype)
        return self.comm.all_reduce(Fn.linear(out.reshape(N, Hl * D), p[n + "o_proj"]))

    # -- sparse MLA + kpool indexer ------------------------------------------------------
    def _indexer(self, L: int, x: torch.Tensor, qr: torch.Tensor, step: Step, caches: Caches):
        """kpool indexer: per segment, complete this step's pools (pooling the
        tail ring's earlier tokens with the new ones), keep the new tail, then
        select for every query the top-k complete pools before it plus the
        in-progress tail, as latent slots (valid prefix first) and counts."""
        F, p, n = self.F, self.p, f"L{L}.idx."
        N = x.shape[0]; kp, nh, d = F.kpool, F.idx_heads, F.idx_dim
        q = Fn.linear(qr, p[n + "wq_b"]).view(N, nh, d)
        k = Fn.linear(x, p[n + "wk"])
        k = Fn.layer_norm(k.float(), (d,), p[n + "k_norm_w"], p[n + "k_norm_b"], K_NORM_EPS).to(x.dtype)
        w = x.float() @ p[n + "w_heads"].T                                           # fp32 head gate, as served
        gate = Fn.linear(x, p[n + "gate"])                                           # [N, 128] per-channel pool score
        q8, qs = fwht128_quant(q.reshape(-1, d))
        q8 = q8.view(N, nh, d)
        w_eff = (w * qs.view(N, nh) * F.idx_scale).contiguous()                     # q's scale folds into the head gate, as served
        width = F.topk + kp - 1
        slots_out = torch.full((N, width), -1, dtype=torch.int32, device=x.device)
        valid_out = torch.zeros(N, dtype=torch.int32, device=x.device)
        keys, scales = caches.pool_keys(L), caches.pool_scales(L)
        for s in step.segments:
            sl = slice(s.start, s.start + s.length)
            tail = caches.tail(L, s.slot)
            tail_width = kp - 1 + F.spec_k
            if tail.shape[0] != tail_width:
                raise ValueError(f"indexer tail needs {tail_width} positions to support draft rollback")
            end = s.ctx + s.length
            if getattr(step, "captured", False):
                from engine.profiles.glm53.decode_graphs import complete_pools
                n_cand = complete_pools(self, L, s, tail, k[sl], gate[sl], caches)
            else:
                pool0 = (s.ctx // kp) * kp                                              # the pool ctx sits in may be half-built
                lead = s.ctx - pool0                                                    # its earlier tokens are in the tail ring
                lead_pos = torch.arange(pool0, s.ctx, device=x.device)
                k_win = torch.cat([tail[lead_pos % tail_width, 0], k[sl]]) if lead else k[sl]
                g_win = torch.cat([tail[lead_pos % tail_width, 1], gate[sl]]) if lead else gate[sl]
                n_full = (end - pool0) // kp
                if n_full:
                    pk8, ps = self.lanes.kpool_compress(k_win[: n_full * kp].view(n_full, kp, d), g_win[: n_full * kp].view(n_full, kp, d), p[n + "ape"])
                    pslots = caches.pool_slots(L, s.seq, pool0 // kp + torch.arange(n_full, device=x.device)).long()
                    keys[pslots] = pk8
                    scales[pslots] = ps.view(-1)
                new_pos = torch.arange(s.ctx, end, device=x.device)
                # Repeated scatter indices have no defined last-writer order on
                # CUDA. Write each ring cell once, retaining only the newest window.
                keep = min(s.length, tail_width)
                tail[new_pos[-keep:] % tail_width, 0] = k[sl][-keep:]
                tail[new_pos[-keep:] % tail_width, 1] = gate[sl][-keep:]
                n_cand = end // kp
            new_pos = s.ctx + torch.arange(s.length, device=x.device)
            # -- selection -----------------------------------------------------------
            seq_lens = (new_pos + 1).to(torch.int32)
            if n_cand:
                cand = caches.pool_slots(L, s.seq, torch.arange(n_cand, device=x.device)).long()
                ke = seq_lens // kp
                logits = self.lanes.indexer_logits(q8[sl], keys[cand], scales[cand], w_eff[sl], ke)
                pool_ids = topk_positions(logits[:, :n_cand].float(), F.topk // kp, valid=ke)
            else:
                pool_ids = torch.full((s.length, F.topk // kp), -1, dtype=torch.int32, device=x.device)
            tokens = select_with_tail(pool_ids, seq_lens, kp)                        # positions, -1 padded
            tokens = tokens.sort(dim=1, descending=True).values                      # valid prefix first (set semantics)
            valid_out[sl] = (tokens >= 0).sum(1).to(torch.int32)
            ts = caches.token_slots(L, s.seq, tokens.clamp_min(0).reshape(-1)).view_as(tokens)
            slots_out[sl] = ts.masked_fill(tokens < 0, -1)
        return slots_out.contiguous(), valid_out

    def _dsa(self, L: int, x: torch.Tensor, step: Step, caches: Caches) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.mla."
        N = x.shape[0]; Hl = self.Hl
        q_a, kv_c = Fn.linear(x, p[n + "qkv_a"]).split([F.q_lora, F.kv_lora], dim=-1)
        qr = rmsnorm(q_a, p[n + "q_a_norm"], F.rms_eps)
        q = Fn.linear(qr, p[n + "q_b"]).view(N, Hl, F.qk_nope)
        kv_n = rmsnorm(kv_c, p[n + "kv_a_norm"], F.rms_eps)
        latent = caches.latent(L)
        for s in step.segments:                                                     # fp8 KV, scale 1 (no kv scales in the checkpoint)
            sl = slice(s.start, s.start + s.length)
            latent[caches.token_slots(L, s.seq, (s.ctx + torch.arange(s.length, device=x.device))).long()] = kv_n[sl].to(E4M3)
        slots, valid = self._indexer(L, x, qr, step, caches)
        kv_b = p[n + "kv_b"].view(Hl, F.qk_nope + F.v_dim, F.kv_lora)
        w_uk, w_uv = kv_b[:, : F.qk_nope, :], kv_b[:, F.qk_nope:, :]
        q_abs = torch.einsum("thd,hdc->thc", q, w_uk)                                # absorb W_UK: MQA over the latent
        ctx_lat = self.lanes.mla_sparse(q_abs.contiguous(), latent, slots, valid, F.mla_scale, 1.0)
        o = torch.einsum("thc,hvc->thv", ctx_lat, w_uv)                              # un-absorb W_UV
        return self.comm.all_reduce(Fn.linear(o.reshape(N, Hl * F.v_dim), p[n + "o_proj"]))

    # -- MLPs -----------------------------------------------------------------------------
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
        sel, w = self.route(L, x)
        out = self.lanes.moe(x, sel, w, p[n + "w13"], p[n + "w13_sf"], p[n + "w2"], p[n + "w2_sf"], F.swiglu_limit).float()
        g, u = Fn.linear(x, p[n + "sh_gate_up"]).chunk(2, dim=-1)
        out += Fn.linear(swiglu_clamped(g, u, F.swiglu_limit), p[n + "sh_down"]).float()
        return self.comm.all_reduce(out.to(x.dtype))

    # -- the step ---------------------------------------------------------------------------
    def forward(self, step: Step, caches: Caches, finish: bool = True, aux_layers=None):
        """One step: every segment's tokens through the chain. Returns the final
        hidden states [N, hidden] (post final norm) when `finish`, else the raw
        mHC carry (res, post, comb, x) for inspection. With `aux_layers`, also
        the contracted residual after each of those layers, concatenated
        [N, len * hidden] -- what the drafter reads (the served model's
        aux_hidden_states: hc_post then hc_contract after layer idx)."""
        F = self.F
        N = step.ids.shape[0]
        x = self.embed(step.ids)
        res = x[:, None, :].expand(N, F.hc, F.hidden).contiguous()                   # hc_expand
        post = comb = None
        aux = {}
        for L in self.layers:
            if post is not None:
                res = self.lanes.mhc_post(x, res, post, comb)
            post, comb, x = self._hc_pre(L, res, "attn")
            x = self._dsa(L, x, step, caches) if F.is_dsa(L) else self._kda(L, x, step, caches)
            if self.probe:
                self.probe("dsa" if F.is_dsa(L) else "kda", L, x)
            res = self.lanes.mhc_post(x, res, post, comb)
            post, comb, x = self._hc_pre(L, res, "ffn")
            x = self._moe(L, x) if F.is_moe(L) else self._dense(L, x)
            if self.probe:
                self.probe("moe" if F.is_moe(L) else "dense", L, x)
            if aux_layers and L in aux_layers:
                aux[L] = self.lanes.mhc_post(x, res, post, comb).float().mean(1).to(x.dtype)
        if not finish:
            return res, post, comb, x
        res = self.lanes.mhc_post(x, res, post, comb)
        h = rmsnorm(res.float().mean(1).to(x.dtype), self.p["norm"], F.rms_eps)      # hc_contract, final norm
        if aux_layers:
            return h, torch.cat([aux[L] for L in aux_layers], dim=-1)
        return h

    def prefill(self, ids: torch.Tensor, ctx: int, seq: int, slot: int, caches: Caches, finish: bool = True):
        return self.forward(Step.prefill(ids, ctx, seq, slot), caches, finish)


def _selfcheck() -> None:
    from engine.base.comm import Comm, LocalTP
    from engine.profiles.glm53 import facts, lanes
    F = facts.load()
    net = Glm53Net(F, LocalTP(4).rank(3), lanes.reference(), layers=range(0, 5))
    names = [s.name for s in net.specs()]
    assert names[:3] == ["embed", "norm", "head"] and "L3.moe.w13" in names and "L4.kda.in_proj" in names
    assert net.Hl == 16 and net.Hk == 16 and net.vp == 38720 and net.rank == 3 and (net.conv_ring, net.rec_ring) == (8, 6)
    try:
        Glm53Net(F, Comm.init(rank=0, world=1), lanes.reference()); raise AssertionError("world 1 must be refused")
    except ValueError:
        pass
    ids = torch.arange(10)
    st = Step.decode([(ids[:6], 100, 7, 1), (ids[6:], 40, 9, 2)])
    assert [tuple(s) for s in map(lambda s: (s.seq, s.slot, s.ctx, s.start, s.length), st.segments)] == [(7, 1, 100, 0, 6), (9, 2, 40, 6, 4)]
    assert st.positions.tolist() == list(range(100, 106)) + list(range(40, 44))
    print(f"  net: glm53 layers 0-4 declares {len(names)} tensors per rank (16 heads, vocab 38,720); rings conv 8 / rec 6; "
          "steps are segments over flat tokens; world != 4 refused OK")


if __name__ == "__main__":
    _selfcheck()
