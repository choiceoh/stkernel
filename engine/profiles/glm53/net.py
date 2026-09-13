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
(lanes.py): the same names bound to judged torch references or to the
served kernels; the composition does not know which.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

import torch
import torch.nn.functional as Fn

from engine.base.constants import fresh, iota
from engine.base.graph_labels import operation
from engine.modules.sparse_indexer import topk_positions
from engine.profiles.glm53 import specs
from engine.profiles.glm53.facts import TP, Facts
from engine.profiles.glm53.lanes import Lanes, swiglu_clamped

BF16, F32, E4M3 = torch.bfloat16, torch.float32, torch.float8_e4m3fn
O_NORM_EPS = 1e-5           # FusedRMSNormGated(head_dim, activation="sigmoid") constructor default
                            # Its lower-level rms_norm_gated helper defaults to 1e-6; GLM uses the class value.
                            # 45차 §23: the core's outputs are ~2e-4 rms (mean(x^2) ~ 7e-10), so the norm is eps-dominated -- 1e-6 scaled every KDA
                            # block by sqrt(10) (layer 0 measured 2.9x against vLLM's own model); with 1e-5 the block matches to rel 0.008
K_NORM_EPS = 1e-6           # indexer LayerNorm(head_dim, eps=1e-6)
SELECT_ROWS = 1024          # query rows per indexer selection pass: the [rows, candidates] fp32 logits are the prefill's
                            # largest transient (6,912 x 32,768 x 4 B = 0.84 GiB per DSA layer at 128K, x2 with a masked copy)


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
    patches: tuple = ()         # ((positions [n] int64 into ids, rows [n, hidden]) ...): rows that replace the embedding at
                                # those positions -- the vision tower's output at image placeholders (45차 §23 A7, vision.py)
    marks: tuple = ()           # ((position into ids, snapshot) ...): block boundaries inside a prefill segment whose position
                                # state the caches keep as a prefix snapshot (base/prefix.py): the KDA recurrence is cut there

    def __post_init__(self):
        if self.ids.ndim != 1 or self.ids.dtype != torch.int64 or not self.segments:
            raise ValueError("a step needs a flat int64 token vector and nonempty segments")
        for pos, rows in self.patches:
            if (pos.ndim != 1 or pos.dtype != torch.int64 or rows.ndim != 2 or rows.shape[0] != pos.numel()
                    or (pos.numel() and (int(pos.min()) < 0 or int(pos.max()) >= self.ids.numel()))):
                raise ValueError("patches are (positions inside the step, one row per position)")
        if self.marks:
            positions = [p for p, _ in self.marks]
            if (len(self.segments) != 1 or positions != sorted(set(positions)) or positions[0] <= 0
                    or positions[-1] >= self.ids.numel() or any(type(p) is not int or type(s) is not int for p, s in self.marks)):
                raise ValueError("marks are increasing positions strictly inside a single prefill segment, each with its snapshot")
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
    def prefill(ids: torch.Tensor, ctx: int, seq: int, slot: int, patches: tuple = (), marks: tuple = ()) -> "Step":
        return Step(ids, (Segment(seq, slot, ctx, 0, ids.shape[0]),), patches, marks)

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
    def kda(self, layer: int, slot: int) -> "tuple[torch.Tensor, torch.Tensor]": ...   # conv ring [C, conv-1+K] bf16 by pos % width; rec ring [K+1, Hl, D, D] f32/f16 by pos % (K+1)
    def latent(self, layer: int) -> torch.Tensor: ...            # [S, 512] e4m3, all slots of the box
    def pool_keys(self, layer: int) -> torch.Tensor: ...         # [P, 128] e4m3 (FWHT-rotated, per-row scaled)
    def pool_scales(self, layer: int) -> torch.Tensor: ...       # [P] f32
    def tail(self, layer: int, slot: int) -> torch.Tensor: ...   # [kpool-1+K, 2, 128] bf16: enough raw keys/gates to reject K drafts
    def token_slots(self, layer: int, seq: int, positions: torch.Tensor) -> torch.Tensor: ...   # int32 absolute latent slots
    def token_map(self, layer: int, seq: int) -> tuple: ...     # (block row | None for identity, block size, block stride, layer offset), in latent rows
    def pool_slots(self, layer: int, seq: int, pool_ids: torch.Tensor) -> torch.Tensor: ...     # int32 absolute pool slots


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


HEAD_NAME = "Glm5NextForCausalLM/lm_head"     # the pack store's name of the head's calibration (its FP8 GPTQ)


class Glm53Net:
    def __init__(self, F: Facts, comm, lanes: Lanes, layers=None):
        if comm.world_size != TP:
            raise ValueError(f"glm53 is written for TP={TP}; comm has world {comm.world_size}")
        self.F, self.comm, self.lanes = F, comm, lanes
        self._norm = lanes.rmsnorm or rmsnorm
        self._activation = lanes.swiglu or swiglu_clamped
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
        self.dense = {}
        self.shared_mlp = {}
        self.shared_overlap = None
        self._router_weights = {}
        self._router_tensorcore = set()
        self.prefill_transport = None
        self.mhc = None
        from engine.profiles.glm53.weights import WEIGHT_LAYOUT, MODELOPT_WEIGHT_LAYOUT
        self.weight_layout = getattr(F, 'weight_layout', WEIGHT_LAYOUT)
        if self.weight_layout not in (WEIGHT_LAYOUT, MODELOPT_WEIGHT_LAYOUT):
            raise ValueError('unsupported GLM weight layout')
        self.modelopt = self.weight_layout == MODELOPT_WEIGHT_LAYOUT
        self._experts = {}
        self._quant_scales = {}
        if self.modelopt:
            self._dense = self._dense_nvfp4
        self.probe = None                            # probe(block, layer, out) after every block, for judges

    # -- binding ----------------------------------------------------------------
    def specs(self):
        if self.modelopt:
            from engine.profiles.glm53.modelopt_weights import all_specs
            return all_specs(self.F, self.layers)
        return specs.all_specs(self.F, self.layers)

    def bind(self, views: dict) -> None:
        from engine.base.params import bind
        self.p = bind(self.specs(), views)
        prepare = getattr(self.lanes, "moe_prepare", None)
        F, p = self.F, self.p
        # Bind scales and weight views before capture. The served t cell may
        # relayout arena bytes in place; the reference can read that layout.
        for L in self.layers:
            if not F.is_moe(L) and not self.modelopt:
                continue
            n = f"L{L}." + ('moe.' if F.is_moe(L) else 'mlp.')
            kw = {}
            if self.modelopt:
                from engine.profiles.glm53.modelopt_scales import ModelOptScales
                scales = ModelOptScales.bind(*(p[n + s] for s in ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')),
                    experts=p[n + 'w13'].shape[0], device=p[n + 'w13'].device)
                self._quant_scales[L] = scales
                kw['scales'] = scales
            if prepare is not None:
                prepare(p[n + "w13"], p[n + "w13_sf"], p[n + "w2"], p[n + "w2_sf"],
                        F.topk_experts if F.is_moe(L) else 1, F.swiglu_limit, **kw)
            self._experts[L] = partial(self.lanes.moe, w13=p[n+'w13'], w13_sf=p[n+'w13_sf'],
                w2=p[n+'w2'], w2_sf=p[n+'w2_sf'], limit=F.swiglu_limit, **kw)

    def router_nbytes(self):
        """FP32 routing matrices, explicitly reserved apart from BF16 rank weights."""
        return sum(self.F.experts * self.F.hidden * 4 for layer in self.layers if self.F.is_moe(layer))

    def prepare_routers(self, arena):
        """Convert immutable BF16 router weights once, into budgeted arena rows.

        Rank files and their binding contract stay BF16. Every projection sees
        exactly the same FP32 values as the former per-step conversion.
        """
        if self._router_weights:
            raise RuntimeError('router weights were already prepared')
        for layer in self.layers:
            if self.F.is_moe(layer):
                weight = self.p[f'L{layer}.moe.gate']
                resident = arena.carve(weight.numel() * 4, f'router/{layer}').view(F32).view_as(weight)
                resident.copy_(weight)
                self._router_weights[layer] = resident

    @staticmethod
    def dense_weight_names(keys):
        """The rank-file key and original calibration name of each projection."""
        names = {
            "kda.in_proj": "self_attn.in_proj_qkvbfg_a", "kda.o_proj": "self_attn.o_proj",
            "mla.qkv_a": "self_attn.fused_qkv_a_proj", "mla.q_b": "self_attn.q_b_proj",
            "mla.o_proj": "self_attn.o_proj", "idx.wq_b": "self_attn.indexer.wq_b",
            "mlp.gate_up": "mlp.gate_up_proj", "mlp.down": "mlp.down_proj",
            "moe.sh_gate_up": "mlp.shared_experts.gate_up_proj", "moe.sh_down": "mlp.shared_experts.down_proj",
        }
        result = {}
        for key in keys:
            layer, _, suffix = key.partition(".")
            if suffix in names:
                result[key] = f"Glm5NextForCausalLM/model.layers.{layer[1:]}.{names[suffix]}"
        return result

    def smoothing_groups(self):
        """(norm key, dense consumer keys, bf16 consumer keys) of every norm output a dense weight reads: the channel
        smoothing of kernels/dense/smoothing divides the norm's weight and multiplies EVERY reader's columns -- a
        reader left out would see an input it was not prepared for. KDA: in_norm -> in_proj. MLA: in_norm -> qkv_a
        and the indexer's wk / gate (bf16); q_a_norm -> q_b and the indexer's wq_b. Dense MLP: post_norm -> gate_up.
        A MoE layer's post_norm feeds the routed experts and the router as well, which cannot take the factor: none."""
        F = self.F
        groups = []
        for L in self.layers:
            n = f"L{L}."
            if F.is_dsa(L):
                groups.append((n + "in_norm", [n + "mla.qkv_a"], [n + "idx.wk", n + "idx.gate"]))
                groups.append((n + "mla.q_a_norm", [n + "mla.q_b", n + "idx.wq_b"], []))
            else:
                groups.append((n + "in_norm", [n + "kda.in_proj"], []))
            if not F.is_moe(L):
                groups.append((n + "post_norm", [n + "mlp.gate_up"], []))
        return groups

    def smooth_inputs(self, amax_of) -> dict:
        """Fold the calibration's channel smoothing into the norms (kernels/dense/smoothing): `amax_of(store name)`
        gives a norm output's channel peaks or None. Returns {dense key: (smoothed weight, s_eff)} for the packs;
        the bf16 readers are rescaled in place, the norms divided in place."""
        from engine.kernels.dense.smoothing import fold, scales, smooth_weight
        names = self.dense_weight_names(self.p)
        smoothed = {}
        for norm_key, dense_keys, bf16_keys in self.smoothing_groups():
            if any(self.p.get(k) is None for k in dense_keys + bf16_keys + [norm_key]) or dense_keys[0] not in names:
                continue                                                    # a layer subset (a local boot), or a reader already retired
            amax = amax_of(names[dense_keys[0]])
            if amax is None:
                continue
            readers = [self.p[k] for k in dense_keys + bf16_keys]
            s_eff = fold(self.p[norm_key], scales(amax, readers))
            for k in dense_keys:
                smoothed[k] = (smooth_weight(self.p[k], s_eff), s_eff)
            for k in bf16_keys:
                self.p[k].copy_(smooth_weight(self.p[k], s_eff))
        return smoothed

    def prepare_dense(self, store=None, *, consume_weights=False):
        """Declare and prepare the same dense families as the fleet's MK path."""
        from engine.kernels.dense import DenseLinear
        self.dense = {}
        smoothed = self.smooth_inputs(store.amax) if store is not None else {}
        for key, name in self.dense_weight_names(self.p).items():
            weight = self.p[key]
            packed, smooth = smoothed.get(key, (weight, None))
            self.dense[key] = DenseLinear(packed, store=store, name=name, smooth=smooth)
            if consume_weights:
                self.dense[key].consume_weight(weight)
                self.p[key]=None
        from engine.kernels.dense import FP8Linear
        # the vocabulary head stays FP8 (45차: W4 there was folded); its fp8 rounding is GPTQ'd from its own calibration
        head_fp8 = store.pack_fp8(self.p["head"], HEAD_NAME) if (store is not None and store.calibrated(HEAD_NAME)) else None
        self.dense["head"] = FP8Linear(self.p["head"], quantized=head_fp8, name=HEAD_NAME)
        if consume_weights:
            self.dense["head"].consume_weight(self.p["head"])
            self.p["head"]=None
        from engine.kernels.dense.mhc import MHC
        self.mhc = MHC({key: weight for key, weight in self.p.items() if key.endswith(("hc.attn_fn","hc.ffn_fn"))})
        from engine.kernels.dense.shared_mlp import SharedMLP, SharedOverlap
        self.shared_mlp = {L: SharedMLP(self.dense[f"L{L}.moe.sh_gate_up"],
                                        self.dense[f"L{L}.moe.sh_down"], self.F.swiglu_limit)
                           for L in self.layers if self.F.is_moe(L)}
        if self.shared_mlp:
            self.shared_overlap = SharedOverlap(self.p["norm"].device)

    @operation("linear", name_arg=2)
    def linear(self, x, name):
        layer = self.dense.get(name)
        return layer(x) if layer is not None else Fn.linear(x, self.p[name])

    # -- embed / head -------------------------------------------------------------
    @operation("embed")
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        start = self.rank * self.vp
        local = ids - start
        mask = (local < 0) | (local >= self.vp)
        h = Fn.embedding(local.masked_fill(mask, 0), self.p["embed"]).masked_fill(mask[:, None], 0)
        return self.comm.all_reduce(h)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.comm.all_gather(self.head_local(h), dim=-1)

    @operation("head_local")
    def head_local(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h, "head")

    @operation("head_tokens")
    def head_tokens(self, h: torch.Tensor, decodable=None) -> torch.Tensor:
        from engine.modules.vocab import argmax
        return argmax(self.head_local(h), self.comm, self.rank * self.vp, decodable)

    # -- mHC -----------------------------------------------------------------------
    @operation("hc_pre", layer_arg=1)
    def _hc_pre(self, L: int, res: torch.Tensor, side: str):
        F, p, n = self.F, self.p, f"L{L}."
        norm_w = p[n + ("in_norm" if side == "attn" else "post_norm")]
        return self.lanes.mhc_pre(res, p[n + f"hc.{side}_fn"], p[n + f"hc.{side}_scale"], p[n + f"hc.{side}_base"],
                                  F.rms_eps, F.hc_eps, F.post_mult, F.sinkhorn, norm_w, F.rms_eps)

    # -- KDA ------------------------------------------------------------------------
    @operation("hc_post_pre", layer_arg=1)
    def _hc_post_pre(self, L, x, res, post, comb, side):
        if self.mhc is None or x.shape[0] > 64:
            res = self.lanes.mhc_post(x, res, post, comb)
            post, comb, x = self._hc_pre(L, res, side)
            return res, post, comb, x
        n, F, p = f"L{L}.", self.F, self.p
        return self.mhc(n+f"hc.{side}_fn",x,res,post,comb,p[n+f"hc.{side}_scale"],
                        p[n+f"hc.{side}_base"],p[n+("in_norm" if side=="attn" else "post_norm")],
                        F.rms_eps,F.hc_eps,F.post_mult,F.sinkhorn)

    @operation("kda", layer_arg=1)
    def _kda(self, L: int, x: torch.Tensor, step: Step, caches: Caches, reduce=None) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.kda."
        N = x.shape[0]; Hl, D, K = self.Hk, F.kda_dim, F.conv
        proj = self.linear(x, n + "in_proj")
        qkv_all, b_all, f_a, g_a = proj.split([3 * Hl * D, Hl, D, D], dim=-1)
        g_raw_all = self.linear(f_a, n + "f_b").view(N, Hl, D)
        g_out = self.linear(g_a, n + "g_b").view(N, Hl, D)
        beta_all = b_all                                                             # raw logits: each lane sigmoids as its kernel wants
        core = torch.empty(N, Hl, D, dtype=x.dtype, device=x.device)
        wc, wr = self.conv_ring, self.rec_ring
        captured = getattr(step, "captured", False)
        for s in step.segments:
            sl = slice(s.start, s.start + s.length)
            direct_ring = self.lanes.kda_recurrent_ring is not None and s.length <= wr
            direct_conv = direct_ring and self.lanes.conv_ring is not None and s.length <= min(8, wc)
            if captured:
                if direct_conv:
                    conv_ring, ring, physical = caches.kda_rings(L, s.slot)
                elif direct_ring:
                    hist, ring, physical = caches.kda_ring_history(L, s.slot, s.ctx)
                else:
                    hist, state0 = caches.kda_history(L, s.slot, s.ctx)
            else:
                conv_ring, rec_ring = caches.kda(L, s.slot)
                if not direct_conv:
                    hist_pos = s.ctx + torch.arange(-(K - 1), 0, device=x.device)
                    hist = conv_ring[:, hist_pos.clamp_min(0) % wc].masked_fill((hist_pos < 0)[None, :], 0)
                if direct_ring:
                    ring, physical = rec_ring[None], 0
                else:
                    state0 = rec_ring[(s.ctx - 1) % wr][None] if s.ctx > 0 else None
            if direct_conv:
                y = self.lanes.conv_ring(qkv_all[sl], p[n + "conv"], conv_ring if captured else conv_ring[None], physical, s.ctx)
            else:
                y, _ = self.lanes.conv_prefill(qkv_all[sl], p[n + "conv"], hist if captured or s.ctx > 0 else None)
                if captured:
                    caches.write_conv(L, s.slot, s.ctx, qkv_all[sl])
                else:
                    keep = min(s.length, wc)
                    pos = s.ctx + torch.arange(s.length - keep, s.length, device=x.device)
                    conv_ring[:, pos % wc] = qkv_all[sl][-keep:].T
            q, k, v = (t.reshape(1, s.length, Hl, D) for t in y.split(Hl * D, dim=-1))
            g_raw, beta = g_raw_all[sl][None], beta_all[sl][None]
            if direct_ring:
                o = self.lanes.kda_recurrent_ring(q, k, v, g_raw, beta, p[n + "A_log"], p[n + "dt_bias"],
                                                  ring, physical, s.ctx, F.lower_bound)
            elif s.length > wr:                                                     # a prefill chunk: only the final state is kept --
                # except at the step's marks (prefix snapshots at block boundaries inside the chunk): the lane hands out the
                # fp32 state at the start of the kernel chunks the marks sit on, out of the one uncut computation (45차 §23:
                # cutting the recurrence into 9 pieces is exact but costs +104% of the layer; the side output costs nothing
                # and is bit-identical to the cut's state)
                marks = [(m, snap) for m, snap in step.marks if 0 < m < s.length] if step.marks else []
                if marks:
                    unit = self.lanes.kda_chunk_tokens
                    if any(m % unit for m, _ in marks):
                        raise ValueError(f"a mark inside a prefill chunk must sit on a {unit}-token kernel chunk")
                    o, state, states = self.lanes.kda_chunk(q, k, v, g_raw, beta, p[n + "A_log"], p[n + "dt_bias"], state0, F.lower_bound,
                                                            states_at=[m // unit for m, _ in marks])
                    for (m, snap), st in zip(marks, states.unbind(0)):
                        caches.mark_kda(L, snap, st, qkv_all[sl][m - (K - 1):m])
                else:
                    o, state = self.lanes.kda_chunk(q, k, v, g_raw, beta, p[n + "A_log"], p[n + "dt_bias"], state0, F.lower_bound)
                rec_ring[(s.ctx + s.length - 1) % wr] = state[0]
            else:                                                                   # a decode/verify step: one state per position
                o, states = self.lanes.kda_recurrent(q, k, v, g_raw, beta, p[n + "A_log"], p[n + "dt_bias"], state0, F.lower_bound)
                if captured:
                    caches.write_rec(L, s.slot, s.ctx, states)
                else:
                    for i in range(s.length):
                        rec_ring[(s.ctx + i) % wr] = states[i]
            core[sl] = o[0]
        out = self.lanes.kda_output_norm(core, g_out, p[n + "o_norm"], O_NORM_EPS)
        return (reduce or self.comm.all_reduce)(self.linear(out.reshape(N, Hl * D), n + "o_proj"))

    # -- sparse MLA + kpool indexer ------------------------------------------------------
    @operation("indexer", layer_arg=1)
    def _indexer(self, L: int, x: torch.Tensor, qr: torch.Tensor, step: Step, caches: Caches):
        """kpool indexer: per segment, complete this step's pools (pooling the
        tail ring's earlier tokens with the new ones), keep the new tail, then
        select for every query the top-k complete pools before it plus the
        in-progress tail, as latent slots (valid prefix first) and counts."""
        F, p, n = self.F, self.p, f"L{L}.idx."
        N = x.shape[0]; kp, nh, d = F.kpool, F.idx_heads, F.idx_dim
        q = self.linear(qr, n + "wq_b").view(N, nh, d)
        k = self.linear(x, n + "wk")
        if self.lanes.layernorm is None:
            k = Fn.layer_norm(k.float(), (d,), p[n + "k_norm_w"], p[n + "k_norm_b"], K_NORM_EPS).to(x.dtype)
        else:
            k = self.lanes.layernorm(k, p[n + "k_norm_w"], p[n + "k_norm_b"], K_NORM_EPS)
        w = x.float() @ p[n + "w_heads"].T                                           # fp32 head gate, as served
        gate = self.linear(x, n + "gate")                                           # [N, 128] per-channel pool score
        q8, qs = self.lanes.indexer_quant(q.reshape(-1, d))
        q8 = q8.view(N, nh, d)
        w_eff = (w * qs.view(N, nh) * F.idx_scale).contiguous()                     # q's scale folds into the head gate, as served
        width = F.topk + kp - 1
        slots_out = torch.empty((N, width), dtype=torch.int32, device=x.device)
        valid_out = torch.empty(N, dtype=torch.int32, device=x.device)
        keys, scales = caches.pool_keys(L), caches.pool_scales(L)
        captured = getattr(step, "captured", False)
        index = iota if captured else fresh                 # see _dsa: kept only for the bounded set
        tail_width = kp - 1 + F.spec_k
        if captured:
            from engine.profiles.glm53.decode_graphs import complete_pools   # the profile's captured writer
            tails = caches.tails(L)
            if tails.shape[1] != tail_width:
                raise ValueError(f"indexer tail needs {tail_width} positions to support draft rollback")
            # Every segment's pools are completed before any selection runs. A segment
            # selects only from its own sequence's blocks and a step may not carry a
            # sequence twice, so the order the pools are written in changes nothing.
            rows = len(step.segments)
            width = step.tokens
            pooled = complete_pools(self, L, step.contexts, width, tails,
                                    k.view(rows, width, d), gate.view(rows, width, d), caches)
        for s in step.segments:
            sl = slice(s.start, s.start + s.length)
            if captured:
                n_cand = pooled
            else:
                tail = caches.tail(L, s.slot)
                if tail.shape[0] != tail_width:
                    raise ValueError(f"indexer tail needs {tail_width} positions to support draft rollback")
                end = s.ctx + s.length
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
            new_pos = s.ctx + index(s.length, x.device)
            # -- selection -----------------------------------------------------------
            seq_lens = (new_pos + 1).to(torch.int32)
            if n_cand:
                cand = caches.pool_slots(L, s.seq, index(n_cand, x.device)).long()
                pool_ids = self._select_pools(q8[sl], w_eff[sl], keys[cand], scales[cand], seq_lens // kp, n_cand, F.topk // kp)
            else:
                pool_ids = torch.full((s.length, F.topk // kp), -1, dtype=torch.int32, device=x.device)
            self.lanes.pool_slots(pool_ids, seq_lens, kp, *caches.token_map(L, s.seq),
                                  slots_out[sl], valid_out[sl])
        return slots_out.contiguous(), valid_out

    def _select_pools(self, q8, w_eff, keys, scales, ke, n_cand: int, k: int) -> torch.Tensor:
        """Top-k complete pools per query, in passes of SELECT_ROWS rows: every row's
        selection is independent, so the passes are exact and the transient is bounded."""
        rows = q8.shape[0]
        if rows <= SELECT_ROWS:
            logits = self.lanes.indexer_logits(q8, keys, scales, w_eff, ke)
            return topk_positions(logits[:, :n_cand].float(), k, valid=ke, inplace=True)
        out = torch.empty((rows, k), dtype=torch.int32, device=q8.device)
        for r0 in range(0, rows, SELECT_ROWS):
            r1 = min(rows, r0 + SELECT_ROWS)
            logits = self.lanes.indexer_logits(q8[r0:r1], keys, scales, w_eff[r0:r1], ke[r0:r1])
            out[r0:r1] = topk_positions(logits[:, :n_cand].float(), k, valid=ke[r0:r1], inplace=True)
        return out

    @operation("dsa", layer_arg=1)
    def _dsa(self, L: int, x: torch.Tensor, step: Step, caches: Caches, reduce=None) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.mla."
        N = x.shape[0]; Hl = self.Hl
        q_a, kv_c = self.linear(x, n + "qkv_a").split([F.q_lora, F.kv_lora], dim=-1)
        qr = self._norm(q_a, p[n + "q_a_norm"], F.rms_eps)
        q = self.linear(qr, n + "q_b").view(N, Hl, F.qk_nope)
        kv_n = self._norm(kv_c, p[n + "kv_a_norm"], F.rms_eps)
        latent = caches.latent(L)
        # A captured step asks for the same few lengths forever, so they come from the kept
        # constants; an eager prefill's follow the request and would grow that cache unbounded.
        index = iota if getattr(step, "captured", False) else fresh
        for s in step.segments:                                                     # fp8 KV, scale 1 (no kv scales in the checkpoint)
            sl = slice(s.start, s.start + s.length)
            latent[caches.token_slots(L, s.seq, (s.ctx + index(s.length, x.device))).long()] = kv_n[sl].to(E4M3)
        slots, valid = self._indexer(L, x, qr, step, caches)
        kv_b = p[n + "kv_b"].view(Hl, F.qk_nope + F.v_dim, F.kv_lora)
        w_uk, w_uv = kv_b[:, : F.qk_nope, :], kv_b[:, F.qk_nope:, :]
        q_abs = torch.einsum("thd,hdc->thc", q, w_uk)                                # absorb W_UK: MQA over the latent
        ctx_lat = self.lanes.mla_sparse(q_abs.contiguous(), latent, slots, valid, F.mla_scale, 1.0)
        o = torch.einsum("thc,hvc->thv", ctx_lat, w_uv)                              # un-absorb W_UV
        return (reduce or self.comm.all_reduce)(self.linear(o.reshape(N, Hl * F.v_dim), n + "o_proj"))

    # -- MLPs -----------------------------------------------------------------------------
    @operation("dense", layer_arg=1)
    def _dense(self, L: int, x: torch.Tensor, reduce=None) -> torch.Tensor:
        p, n = self.p, f"L{L}.mlp."
        g, u = self.linear(x, n + "gate_up").chunk(2, dim=-1)
        return (reduce or self.comm.all_reduce)(self.linear(self._activation(g, u, self.F.swiglu_limit), n + "down"))

    @operation("dense_nvfp4", layer_arg=1)
    def _dense_nvfp4(self, L: int, x: torch.Tensor, reduce=None) -> torch.Tensor:
        # Fixed one-expert routing: no router/selection or shared expert. These
        # buffers are owned by the graph pool during capture, like the MoE out.
        ids = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.int32)
        weights = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float32)
        return (reduce or self.comm.all_reduce)(self._experts[L](x, ids, weights))

    @operation("route", layer_arg=1)
    def route(self, L: int, x: torch.Tensor):
        """noaux_tc: sigmoid scores fp32, select by score + bias, weight by the
        raw scores renormalised, times routed_scaling_factor."""
        F, p, n = self.F, self.p, f"L{L}.moe."
        if self._router_weights and x.shape[0] <= F.spec_k + 1:
            from engine.kernels.glm_pointwise import router_logits
            logits = router_logits(x, p[n + "gate"])
            self._router_tensorcore.add(L)
        else:
            gate = self._router_weights.get(L, p[n + "gate"])
            logits = x.float() @ gate.float().T
        if self.lanes.route_weights is not None:
            return self.lanes.route_weights(logits, p[n + "bias"], F.topk_experts, F.routed_scale)
        s = torch.sigmoid(logits)
        sel = (s + p[n + "bias"]).topk(F.topk_experts, dim=-1).indices
        w = s.gather(-1, sel)
        return sel.to(torch.int32), w / w.sum(-1, keepdim=True) * F.routed_scale

    @operation("moe", layer_arg=1)
    def _moe(self, L: int, x: torch.Tensor, reduce=None) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.moe."
        # GPU component gate: C=1 wins; C=4 with reused routes regresses.
        # Keep the established shared chain for wider captured batches.
        if self.shared_overlap is not None and x.shape[0] <= F.spec_k + 1:
            def routed():
                sel, w = self.route(L, x)
                return self._experts[L](x, sel, w)
            joined = self.shared_overlap(self.shared_mlp[L], x, routed)
            return (reduce or self.comm.all_reduce)(joined)
        sel, w = self.route(L, x)
        out = self._experts[L](x, sel, w)
        g, u = self.linear(x, n + "sh_gate_up").chunk(2, dim=-1)
        shared = self.linear(self._activation(g, u, F.swiglu_limit), n + "sh_down")
        # Both lanes return BF16. Its add accumulates in FP32 and rounds once,
        # just like the former float() + float() followed by to(BF16).
        return (reduce or self.comm.all_reduce)(out + shared)

    # -- the step ---------------------------------------------------------------------------
    def forward(self, step: Step, caches: Caches, finish: bool = True, aux_layers=None, aux_ready=None):
        """One step: every segment's tokens through the chain. Returns the final
        hidden states [N, hidden] (post final norm) when `finish`, else the raw
        mHC carry (res, post, comb, x) for inspection. With `aux_layers`, also
        the contracted residual after each of those layers, concatenated
        [N, len * hidden] -- what the drafter reads (the served model's
        aux_hidden_states: hc_post then hc_contract after layer idx)."""
        F = self.F
        N = step.ids.shape[0]
        sp = self.prefill_transport if (finish and not self.probe and len(step.segments) == 1
                                       and N >= 128 and N % self.comm.world_size == 0
                                       and not getattr(step, "captured", False)) else None
        reduce = sp.reduce_scatter if sp else self.comm.all_reduce
        x = self.embed(step.ids)
        for pos, rows in step.patches:                                               # image rows in place of their placeholders
            x.index_copy_(0, pos, rows.to(x.dtype))
        if sp:
            x = x.chunk(self.comm.world_size, dim=0)[self.rank]
            N = x.shape[0]
        res = x[:, None, :].expand(N, F.hc, F.hidden).contiguous()                   # hc_expand
        post = comb = None
        aux = {}
        features = None
        for L in self.layers:
            if post is not None:
                res, post, comb, x = self._hc_post_pre(L, x, res, post, comb, "attn")
            else:
                post, comb, x = self._hc_pre(L, res, "attn")
            if sp:
                x = sp.all_gather(x.contiguous())
            x = self._dsa(L, x, step, caches, reduce) if F.is_dsa(L) else self._kda(L, x, step, caches, reduce)
            if self.probe:
                self.probe("dsa" if F.is_dsa(L) else "kda", L, x)
            res, post, comb, x = self._hc_post_pre(L, x, res, post, comb, "ffn")
            if sp:
                x = sp.all_gather(x.contiguous())
            x = self._moe(L, x, reduce) if F.is_moe(L) else self._dense(L, x, reduce)
            if self.probe:
                self.probe("moe" if F.is_moe(L) else "dense", L, x)
            if aux_layers and L in aux_layers:
                aux[L] = self.lanes.mhc_post(x, res, post, comb).float().mean(1).to(x.dtype)
                if aux_ready is not None and L == max(aux_layers):
                    if sp is not None:
                        raise ValueError("early draft observation belongs to decode, not SP prefill")
                    features = torch.cat([aux[l] for l in aux_layers], dim=-1)
                    aux_ready(features)
        if not finish:
            return res, post, comb, x
        res = self.lanes.mhc_post(x, res, post, comb)
        h = self._norm(res.float().mean(1).to(x.dtype), self.p["norm"], F.rms_eps)      # hc_contract, final norm
        if sp:
            h = self.comm.all_gather(h, dim=0)
        if aux_layers:
            if features is None:
                features = torch.cat([aux[L] for L in aux_layers], dim=-1)
            return h, self.comm.all_gather(features, dim=0) if sp else features
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
    assert net.Hl == 16 and net.Hk == 16 and net.vp == 38720 and net.rank == 3
    # The rings are sized from the draft width, so asserting their numbers only restates the arithmetic. What
    # can actually break is the lane the width is supposed to reach: `_kda` takes the direct conv ring only for
    # `s.length <= min(8, conv_ring)`, and a decode block is spec_k + 1 tokens. Past spec_k 7 that silently
    # falls back to the prefill convolution -- armed and not running, which is the failure this fleet knows.
    block = F.spec_k + 1
    assert block <= net.rec_ring, "the recurrent ring must hold one state per draft position"
    assert block <= min(8, net.conv_ring), (
        f"a decode block of {block} tokens is past the direct conv ring's {min(8, net.conv_ring)}: "
        "the KDA lane would fall back to the prefill convolution on every decode step")
    try:
        Glm53Net(F, Comm.init(rank=0, world=1), lanes.reference()); raise AssertionError("world 1 must be refused")
    except ValueError:
        pass
    ids = torch.arange(10)
    st = Step.decode([(ids[:6], 100, 7, 1), (ids[6:], 40, 9, 2)])
    assert [tuple(s) for s in map(lambda s: (s.seq, s.slot, s.ctx, s.start, s.length), st.segments)] == [(7, 1, 100, 0, 6), (9, 2, 40, 6, 4)]
    assert st.positions.tolist() == list(range(100, 106)) + list(range(40, 44))
    print(f"  net: glm53 layers 0-4 declares {len(names)} tensors per rank (16 heads, vocab 38,720); a decode "
          f"block of {block} rides rings conv {net.conv_ring} / rec {net.rec_ring}; steps are segments over "
          "flat tokens; world != 4 refused OK")


if __name__ == "__main__":
    _selfcheck()
