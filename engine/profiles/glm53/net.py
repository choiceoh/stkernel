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
from engine.kernels.decode_topk import select as select_native
from engine.modules.sparse_indexer import pin_pools_in_logits, tail_pin_pools, topk_positions
from engine.profiles.glm53 import specs
from engine.profiles.glm53.facts import TP, Facts
from engine.profiles.glm53.lanes import Lanes, swiglu_clamped

BF16, F32, E4M3 = torch.bfloat16, torch.float32, torch.float8_e4m3fn
O_NORM_EPS = 1e-5           # FusedRMSNormGated(head_dim, activation="sigmoid") constructor default
                            # Its lower-level rms_norm_gated helper defaults to 1e-6; GLM uses the class value.
                            # 45차 §23: the core's outputs are ~2e-4 rms (mean(x^2) ~ 7e-10), so the norm is eps-dominated -- 1e-6 scaled every KDA
                            # block by sqrt(10) (layer 0 measured 2.9x against vLLM's own model); with 1e-5 the block matches to rel 0.008
K_NORM_EPS = 1e-6           # indexer LayerNorm(head_dim, eps=1e-6)
def _torch_topk(logits, k):
    """The selection `engine.kernels.decode_topk` replaces, on logits whose horizon is already masked."""
    return torch.topk(logits, k, dim=-1, sorted=False).indices


SELECT_ROWS = 1024          # query rows per indexer selection pass: the [rows, candidates] fp32 logits are the prefill's
                            # largest transient (6,912 x 32,768 x 4 B = 0.84 GiB per DSA layer at 128K, x2 with a masked copy)
JOINED_SLICES = 20          # query rows a captured step's decode selection joins into one top-k: torch picks single- or
                            # multi-block selection from the slice count and size, and up to 20 slices one size rule
                            # (20,000 columns) decides for any count, so a joined C=2 block selects like each row's launch


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


def skip_route_weights(weights, tau: float, scale: float):
    """ACE's slot skip on one layer's routes: a slot whose normalised gate is below `tau` gets weight 0 unless it is the
    token's top-1, and the kept weights are renormalised to `scale`. The expert ids are left alone, so a skipped slot
    still reads its expert and adds 0 x its output -- the numbers of skipping it, without the bytes (a measurement arm)."""
    if not 0.0 <= tau < 1.0:
        raise ValueError("a skip threshold is a normalised gate in [0, 1)")
    p = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    keep = p >= tau
    keep.scatter_(-1, p.argmax(-1, keepdim=True), True)
    kept = torch.where(keep, weights, torch.zeros_like(weights))
    return kept / kept.sum(-1, keepdim=True).clamp_min(1e-20) * scale


from engine.modules.selection_capture import SelectionCapture, from_net


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
        self.cublas_readers = {}
        self.shared_mlp = {}
        self.shared_overlap = None
        self._router_layers = None                         # None until every native FP32 router is resident
        # A measurement arm's routed-slot skip (ACE, arXiv:2609.05228): None serves every routed slot. Read at call time,
        # so only eager steps see a value set after capture -- the capture sets it per document for prefill passes.
        self.route_skip = None
        self._router_weights = {}
        self._router_fp32 = set()
        # The aligned tail matches served selection/normalization on the same
        # logits. Admit the qualified K7 geometry; route() also requires a bound
        # C1/C2 width and resident FP32 weights/bias, with no route-slot skip.
        self.fused_decode_router = (F.hidden, F.experts, F.topk_experts, F.spec_k) == (4096, 288, 8, 7)
        self._router_fused_bias = {}
        self._router_fused_executed = set()
        self._decode_pairs = {}
        self.decode_fastpath_rows = ()
        self.decode_pairs_executed = set()
        self._query_pairs = {}
        self.decode_dsa_rows = ()
        self.decode_latents_executed = set()
        self.decode_pools_executed = set()
        self._indexer_head_gates = {}
        self.decode_indexer_gate_rows = ()
        self._decode_absorb = {}
        self.decode_absorb_rows = ()
        self.prefill_transport = None
        self.prefill_ffn_packets = False
        self.prefill_packet_executed = set()
        self.prefill_packet_planned = set()
        self.prefill_packet_peak_bytes = 0
        self.prefill_indexer_shards = False
        # A diagnostic dump of one selection per layer for tools/selection_reference.py;
        # inert unless ST_SELECTION_CAPTURE is set (engine/modules/selection_capture.py).
        self.selection_capture = SelectionCapture()
        self.prefill_indexer_executed = set()
        self.prefill_dense_prefix = False
        self.prefill_dense_prefix_executed = set()
        self.prefill_covered_queries_executed = set()
        self.prefill_absorb_tiles = False
        self.prefill_absorb_tiles_executed = set()
        self.mhc = None
        from engine.profiles.glm53.weights import WEIGHT_LAYOUT, MODELOPT_LAYOUTS, MODELOPT_BF16_DENSE_LAYOUT
        self.weight_layout = getattr(F, 'weight_layout', WEIGHT_LAYOUT)
        if self.weight_layout not in (WEIGHT_LAYOUT, *MODELOPT_LAYOUTS):
            raise ValueError('unsupported GLM weight layout')
        self.modelopt = self.weight_layout in MODELOPT_LAYOUTS
        # The dense MLPs as one-expert NVFP4 (ModelOpt's own layout); BF16 dense MLPs take the packed dense path.
        self.dense_nvfp4 = self.modelopt and self.weight_layout != MODELOPT_BF16_DENSE_LAYOUT
        self._experts = {}
        self._packet_experts = {}
        self._packet_capabilities = {}
        self._expert_views = {}                    # prepared packed owners, also used by explicit dataflow experiments
        self._quant_scales = {}
        if self.dense_nvfp4:
            self._dense = self._dense_nvfp4
        self.probe = None                            # probe(block, layer, out) after every block, for judges
        # A producer writes its bound C1 consumer's input pack (KDA o_proj from the output norm). False is the
        # same-build control for component probes; serving binds it before capture.
        self.producer_packs = True
        self.mhc_input_packs = True

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
            if not F.is_moe(L) and not self.dense_nvfp4:
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
                self._expert_views[L] = prepare(p[n + "w13"], p[n + "w13_sf"], p[n + "w2"], p[n + "w2_sf"],
                                               F.topk_experts if F.is_moe(L) else 1, F.swiglu_limit, **kw)
            self._experts[L] = partial(self.lanes.moe, w13=p[n+'w13'], w13_sf=p[n+'w13_sf'],
                w2=p[n+'w2'], w2_sf=p[n+'w2_sf'], limit=F.swiglu_limit, **kw)
            if F.is_moe(L) and self.lanes.moe_packets is not None and self.lanes.moe_packets_supported is not None:
                args = dict(w13=p[n+'w13'], w13_sf=p[n+'w13_sf'], w2=p[n+'w2'],
                            w2_sf=p[n+'w2_sf'], limit=F.swiglu_limit, **kw)
                self._packet_experts[L] = partial(self.lanes.moe_packets, **args)
                self._packet_capabilities[L] = partial(self.lanes.moe_packets_supported, **args)

    def warmup_decode_experts(self, rows, device):
        """Load every bound expert variant without tensor-parallel collectives.

        Packed-scale admission is per layer and rank. A rank with a raw-scale
        fallback may load a different CUDA module partway through the first
        decode forward while its peers are already waiting in one-shot. Module
        loading can synchronize the context, so prepare these local calls before
        any rank enters that forward. Use the bound calls themselves to cover
        both packed and fallback weights, including dense NVFP4 layers.
        """
        rows = tuple(rows)
        if any(type(n) is not int or n <= 0 for n in rows):
            raise ValueError("decode expert warmup requires positive row counts")
        for count in rows:
            x = torch.zeros(count, self.F.hidden, device=device, dtype=BF16)
            routes = {}
            for layer, expert in self._experts.items():
                topk = self.F.topk_experts if self.F.is_moe(layer) else 1
                if topk not in routes:
                    ids = torch.arange(topk, device=device, dtype=torch.int32).repeat(count, 1)
                    weights = torch.full((count, topk), 1. / topk, device=device, dtype=F32)
                    routes[topk] = ids, weights
                expert(x, *routes[topk])

    def router_nbytes(self):
        """Replicated FP32 gates, read by every native decode and prefill router."""
        from engine.base.arena import ALIGN
        bias_bytes = ((self.F.experts * 4 + ALIGN - 1) // ALIGN * ALIGN
                      if getattr(self, 'fused_decode_router', False) else 0)
        return sum(self.F.experts * self.F.hidden * 4 + bias_bytes
                   for L in self.layers if self.F.is_moe(L))

    def prepare_routers(self, arena):
        """Convert checkpoint gates once into the declared FP32 arena region."""
        if self._router_layers is not None:
            raise RuntimeError('router bindings were already prepared')
        layers = {layer for layer in self.layers if self.F.is_moe(layer)}
        for layer in layers:
            weight = self.p[f'L{layer}.moe.gate']
            if weight.dtype != torch.bfloat16 or weight.shape != (self.F.experts, self.F.hidden):
                raise ValueError('native router requires BF16 checkpoint weights [experts, hidden]')
        for layer in sorted(layers):
            weight = self.p[f'L{layer}.moe.gate']
            resident = arena.carve(weight.numel() * 4, f'router/{layer}').view(F32).view_as(weight)
            resident.copy_(weight)
            self._router_weights[layer] = resident
        if self.fused_decode_router:
            from engine.base.arena import ALIGN
            if (self.F.hidden, self.F.experts, self.F.topk_experts, self.F.spec_k) != (4096, 288, 8, 7):
                raise ValueError('fused decode router requires the GLM53 K7 profile')
            size = (self.F.experts * 4 + ALIGN - 1) // ALIGN * ALIGN
            for layer in sorted(layers):
                bias = arena.carve(size, f'router-bias/{layer}').view(F32)[:self.F.experts]
                bias.copy_(self.p[f'L{layer}.moe.bias'])
                self._router_fused_bias[layer] = bias
        self._router_layers = layers

    def decode_projection_nbytes(self):
        """Joined indexer matrices; KDA pairs retain their original weight views."""
        return sum(2 * self.F.idx_dim * self.F.hidden * 2 for L in self.layers if self.F.is_dsa(L))

    def prepare_decode_projections(self, arena, *, capture_rows=None):
        """Copy pairs after smoothing, before capture, into the declared arena."""
        if self._decode_pairs or not self.dense:
            raise RuntimeError('prepare decode pairs once, after prepare_dense/input smoothing')
        from engine.kernels.decode_projection import KdaPair, IndexerPair, declared_rows, DECODE_ROWS
        rows = declared_rows(capture_rows)
        if rows is not None:
            if self.F.spec_k != 7 or rows != tuple(8 * n for n in range(1, len(rows) + 1)):
                raise ValueError('bound decode fastpaths require K=7 and contiguous C=1..4 capture widths')
            self.decode_fastpath_rows = rows
            for dense in self.dense.values():
                if hasattr(dense, 'packs'):
                    dense.decode_input_rows = rows
        pair_rows = tuple(sorted(set(DECODE_ROWS).union(rows))) if rows is not None else None
        for L in self.layers:
            if self.F.is_dsa(L):
                n = f'L{L}.idx.'
                wk, gate = self.p[n + 'wk'], self.p[n + 'gate']
                storage = arena.carve((wk.numel() + gate.numel()) * 2, f'indexer-pair/{L}')
                self._decode_pairs[L] = IndexerPair(wk, gate, rows=pair_rows, storage=storage.view(BF16).view(2 * self.F.idx_dim, self.F.hidden))
            else:
                n = f'L{L}.kda.'
                self._decode_pairs[L] = KdaPair(self.p[n + 'f_b'], self.p[n + 'g_b'], rows=pair_rows)

    def _decode_pair(self, L, step, rows):
        # Explicit owner cells: other graph widths and prefill keep their
        # existing projection form. No exception-driven kernel fallback.
        if not self._decode_pairs or not getattr(step, 'captured', False):
            return None
        from engine.kernels.decode_projection import DECODE_ROWS
        pair = self._decode_pairs.get(L)
        allowed = getattr(pair, 'rows', None) or DECODE_ROWS
        if rows not in allowed:
            return None
        if getattr(self, 'decode_fastpath_rows', ()):
            self.decode_pairs_executed.add((L, rows))
        return pair

    def prepare_decode_dsa_inputs(self, rows):
        """Bind existing smoothed W4 readers; no weight copy or new arena region."""
        if (self._query_pairs or self.F.spec_k != 7 or self.lanes.latent_norm_write is None
                or self.lanes.decode_rows is None or self.lanes.decode_rows.update is None
                or self.lanes.decode_rows.compress is None):
            raise ValueError('DSA inputs require one K=7 native preparation with fused latent and pool-cache lanes')
        from engine.kernels.dense.query_pair import QueryPair
        pairs = {L: QueryPair(self.dense[f'L{L}.mla.q_b'], self.dense[f'L{L}.idx.wq_b'], rows=rows)
                 for L in self.layers if self.F.is_dsa(L)}
        if not pairs:
            raise ValueError('DSA inputs require at least one DSA layer')
        self._query_pairs = pairs
        self.decode_dsa_rows = tuple(rows)

    def prepare_decode_indexer_gate(self, rows):
        """Bind the FP32 owners after smoothing and the paired boundary preparation."""
        rows = tuple(rows)
        if (self._indexer_head_gates or self.F.spec_k != 7 or not rows
                or rows != self.decode_fastpath_rows):
            raise ValueError('head gates require one K=7 preparation with matching decode fastpaths')
        from engine.kernels.indexer_gate import IndexerHeadGate
        layers = [L for L in self.layers if self.F.is_dsa(L)]
        if not layers or any(L not in self._decode_pairs for L in layers):
            raise ValueError('head gates require every paired indexer boundary')
        self._indexer_head_gates = {L: IndexerHeadGate(self.p[f'L{L}.idx.w_heads'], rows=rows) for L in layers}
        self.decode_indexer_gate_rows = rows

    def prepare_decode_absorb(self, rows):
        """Retain kv_b views for both captured contractions; no weight allocation."""
        if (self._decode_absorb or self.F.spec_k != 7
                or (self.Hl, self.F.qk_nope, self.F.v_dim, self.F.kv_lora) != (16, 256, 256, 512)):
            raise ValueError('decode absorb requires one K=7 preparation of the GLM MLA geometry')
        from engine.kernels.mla.decode_absorb import DecodeAbsorb
        rows = tuple(rows)
        owners = {}
        for L in self.layers:
            if self.F.is_dsa(L):
                w = self.p[f'L{L}.mla.kv_b'].view(16, 512, 512)
                owners[L] = DecodeAbsorb(w[:, :256], w[:, 256:], rows=rows)
        if not owners:
            raise ValueError('decode absorb needs at least one DSA layer')
        self._decode_absorb, self.decode_absorb_rows = owners, rows

    def _indexer_head_gate(self, L, x, step):
        if (getattr(step, 'captured', False) and not self.probe
                and x.shape[0] in getattr(self, 'decode_indexer_gate_rows', ())):
            return self._indexer_head_gates[L](x), 16
        return x.float() @ self.p[f'L{L}.idx.w_heads'].T, 1

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
        """(norm key, dense consumer keys, resident consumer keys) of every norm output a dense weight reads: the channel
        smoothing of kernels/dense/smoothing divides the norm's weight and multiplies EVERY reader's columns -- a
        reader left out would see an input it was not prepared for. KDA: in_norm -> in_proj. MLA: in_norm -> qkv_a
        and the indexer's wk / gate (bf16) / w_heads (fp32); q_a_norm -> q_b and the indexer's wq_b. Dense MLP: post_norm -> gate_up.
        A MoE layer's post_norm feeds the routed experts and the router as well, which cannot take the factor: none."""
        F = self.F
        groups = []
        for L in self.layers:
            n = f"L{L}."
            if F.is_dsa(L):
                groups.append((n + "in_norm", [n + "mla.qkv_a"], [n + "idx.wk", n + "idx.gate", n + "idx.w_heads"]))
                groups.append((n + "mla.q_a_norm", [n + "mla.q_b", n + "idx.wq_b"], []))
            else:
                groups.append((n + "in_norm", [n + "kda.in_proj"], []))
            if not F.is_moe(L):
                groups.append((n + "post_norm", [n + "mlp.gate_up"], []))
        return groups

    def smooth_inputs(self, amax_of, *, shared_comm=None) -> dict:
        """Fold the calibration's channel smoothing into the norms (kernels/dense/smoothing): `amax_of(store name)`
        gives a norm output's channel peaks or None. Returns {dense key: (smoothed weight, s_eff)} for the packs;
        the resident readers retain their dtype and are rescaled in place, the norms divided in place.
        Native token-sharded prefill exchanges in_norm/post_norm outputs: their factors must agree across TP."""
        from engine.kernels.dense.smoothing import fold, scales, shared_scales, smooth_weight
        names = self.dense_weight_names(self.p)
        smoothed = {}
        for norm_key, dense_keys, resident_keys in self.smoothing_groups():
            if any(self.p.get(k) is None for k in dense_keys + resident_keys + [norm_key]) or dense_keys[0] not in names:
                continue                                                    # a layer subset (a local boot), or a reader already retired
            amax = amax_of(names[dense_keys[0]], width=self.p[dense_keys[0]].shape[1])
            readers = [self.p[k] for k in dense_keys + resident_keys]
            if shared_comm is not None and norm_key.endswith((".in_norm", ".post_norm")):
                factor = shared_scales(amax, readers, shared_comm.all_reduce_max)
            else:
                factor = None if amax is None else scales(amax, readers)
            if factor is None:
                continue
            s_eff = fold(self.p[norm_key], factor)
            for k in dense_keys:
                smoothed[k] = (smooth_weight(self.p[k], s_eff), s_eff)
            for k in resident_keys:
                self.p[k].copy_(smooth_weight(self.p[k], s_eff))
        return smoothed

    def prepare_dense(self, store=None, *, consume_weights=False):
        """Declare and prepare the same dense families as the fleet's MK path."""
        from engine.kernels.dense import DenseLinear
        self.dense = {}
        smoothed = self.smooth_inputs(store.amax, shared_comm=self.comm) if store is not None else {}
        for key, name in self.dense_weight_names(self.p).items():
            weight = self.p[key]
            packed, smooth = smoothed.get(key, (weight, None))
            self.dense[key] = DenseLinear(packed, store=store, name=name, smooth=smooth)
            if key.endswith('.kda.in_proj') and tuple(packed.shape) == (6416, 4096):
                self.dense[key].prepare_cta_layout()
            if consume_weights:
                self.dense[key].consume_weight(weight)
                self.p[key]=None
        from engine.kernels.dense import FP8Linear
        # the vocabulary head stays FP8 (45차: W4 there was folded); its fp8 rounding is GPTQ'd from its own calibration
        head_fp8 = store.pack_fp8(self.p["head"], HEAD_NAME) if (store is not None and store.calibrated(HEAD_NAME)) else None
        # decode rows (<= 16: MAX_SEQS 2 x (SPEC_K + 1), and a draft pass's 7 a request) read the BF16 rows against
        # the FP8 head in one launch: GB10 glm53-head-0919a, 731-743 us against the cuBLASLt reader's 850-929, and
        # a third less error against the BF16 product. The reader keeps larger batches (glm53/cublas.qualify_head).
        self.dense["head"] = FP8Linear(self.p["head"], quantized=head_fp8, name=HEAD_NAME, decode_rows="w8a16")
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
    def linear(self, x, name, *, out=None, producer_pack=None):
        layer = self.dense.get(name)
        if producer_pack is not None:
            if out is not None or layer is None:
                raise ValueError("producer input pack requires its bound dense consumer")
            return layer(x, producer_pack=producer_pack)
        if out is not None:
            return layer(x, out=out) if layer is not None else torch.mm(x, self.p[name].T, out=out)
        return layer(x) if layer is not None else Fn.linear(x, self.p[name])

    def prefill_project(self, transport, x, name):
        project = lambda value: self.linear(value, name)
        layer = self.dense.get(name)
        packets = getattr(layer, "packet_projector", lambda: None)()
        if packets is None:
            return transport.gather_project(x.contiguous(), project)
        return transport.gather_project(x.contiguous(), project, packet_project=packets)

    # -- embed / head -------------------------------------------------------------
    @operation("embed")
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        from engine.modules.token_embedding import lookup
        h = lookup(ids, self.p["embed"], self.rank * self.vp)
        return self.comm.all_reduce(h)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.comm.all_gather(self.head_local(h), dim=-1)

    def head_buffer(self, rows: int, device) -> torch.Tensor:
        """Stable GEMM destination, including the FP8 head's padded columns."""
        head = self.dense.get("head")
        width = head.weight[0].shape[0] if head is not None else self.vp
        return torch.empty(rows, width, device=device, dtype=torch.bfloat16)

    @operation("head_local")
    def head_local(self, h: torch.Tensor, *, out=None, producer_pack=None) -> torch.Tensor:
        if producer_pack is not None:
            head = self.dense.get("head")
            if head is None:
                raise RuntimeError('head producer requires the prepared FP8 head')
            return head.project_mx(h, *producer_pack, out=out)
        if out is None:
            return self.linear(h, "head")
        return self.linear(h, "head", out=out)

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
        if self.mhc is not None and 64 < x.shape[0] <= 32768:
            n, F, p = f"L{L}.", self.F, self.p
            result = self.mhc.prefill(n+f"hc.{side}_fn",x,res,post,comb,p[n+f"hc.{side}_scale"],
                                     p[n+f"hc.{side}_base"],p[n+("in_norm" if side=="attn" else "post_norm")],
                                     F.rms_eps,F.hc_eps,F.post_mult,F.sinkhorn)
            if result is not None:
                return result
        if self.mhc is None or x.shape[0] > 64:
            res = self.lanes.mhc_post(x, res, post, comb)
            post, comb, x = self._hc_pre(L, res, side)
            return res, post, comb, x
        n, F, p = f"L{L}.", self.F, self.p
        return self.mhc(n+f"hc.{side}_fn",x,res,post,comb,p[n+f"hc.{side}_scale"],
                        p[n+f"hc.{side}_base"],p[n+("in_norm" if side=="attn" else "post_norm")],
                        F.rms_eps,F.hc_eps,F.post_mult,F.sinkhorn)

    @operation("kda", layer_arg=1)
    def _kda(self, L: int, x: torch.Tensor, step: Step, caches: Caches, reduce=None, *, projection=None, project=None, input_pack=None) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.kda."
        proj = self.linear(x, n + "in_proj", **({"producer_pack": input_pack} if input_pack is not None else {})) if projection is None else projection
        N = proj.shape[0]; Hl, D, K = self.Hk, F.kda_dim, F.conv
        qkv_all, b_all, f_a, g_a = proj.split([3 * Hl * D, Hl, D, D], dim=-1)
        pair = self._decode_pair(L, step, N)
        if pair is not None:
            g_raw_all, g_out = pair(f_a, g_a)
            g_raw_all, g_out = g_raw_all.view(N, Hl, D), g_out.view(N, Hl, D)
        else:
            g_raw_all = self.linear(f_a, n + "f_b").view(N, Hl, D)
            g_out = self.linear(g_a, n + "g_b").view(N, Hl, D)
        beta_all = b_all                                                             # raw logits: each lane sigmoids as its kernel wants
        wc, wr = self.conv_ring, self.rec_ring
        captured = getattr(step, "captured", False)
        single_chunk = not captured and len(step.segments) == 1 and N > wr
        rows = self._ring_rows(step, wc, wr)
        if rows:
            # a captured decode step: every row's conv and recurrence in one launch each, and the output lands in
            # step order without a copy per row (45차, the C=4 question: four rows were four times the launches --
            # 34 layers x 4 rows x ~7 kernels -- and a copy each; the kernels read each row's slot and context)
            conv_ring, rec_ring, slots = caches.kda_rings_rows(L)
            y = self.lanes.conv_ring_rows(qkv_all, p[n + "conv"], conv_ring, slots, step.contexts)
            q, k, v = (t.reshape(1, N, Hl, D) for t in y.split(Hl * D, dim=-1))
            if getattr(caches, "deferred_state", None) is not None:
                o = caches.verify_kda(L, q, k, v, g_raw_all[None], beta_all[None], p[n + "A_log"], p[n + "dt_bias"],
                                      step.contexts, F.lower_bound)
            else:
                o = self.lanes.kda_recurrent_ring_rows(q, k, v, g_raw_all[None], beta_all[None], p[n + "A_log"], p[n + "dt_bias"],
                                                       rec_ring, slots, step.contexts, F.lower_bound)
            core = o[0]
        else:
            core = None if single_chunk else torch.empty(N, Hl, D, dtype=x.dtype, device=x.device)
        for s in (() if rows else step.segments):
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
            if single_chunk:
                core = o[0].contiguous()
            else:
                core[sl] = o[0]
        pack = self._o_proj_pack(n + "o_proj", core, project)
        if pack is not None:
            # The norm's program per (token, head) is one 128-column block of o_proj's input: it writes the
            # bound cell's pack beside its output (C1's layout at 8 rows, the sixteen-row CTA's wide pack at 16),
            # and the cell reads it instead of launching its own.
            out = self.lanes.kda_output_norm(core, g_out, p[n + "o_norm"], O_NORM_EPS, pack=pack)
            return (reduce or self.comm.all_reduce)(project(out.reshape(N, Hl * D), n + "o_proj", pack=pack))
        out = self.lanes.kda_output_norm(core, g_out, p[n + "o_norm"], O_NORM_EPS)
        return (reduce or self.comm.all_reduce)((project or self.linear)(out.reshape(N, Hl * D), n + "o_proj"))

    def _o_proj_pack(self, name, core, project):
        """Storage for o_proj's input pack when its producer can write it: a direct projector whose bound writer
        reads producer packs at this step's rows, an [8 or 16, heads, 128] step, and a served norm that writes it."""
        rows = core.shape[0] if core.ndim == 3 else 0
        if (not self.producer_packs or project is None or rows not in (8, 16)
                or core.shape[2] != 128 or not getattr(self.lanes.kda_output_norm, "producer_pack", False)
                or not getattr(project, "pack_rows", lambda name, rows: False)(name, rows)):
            return None
        from engine.kernels.dense import producer_pack_nbytes
        return torch.empty(producer_pack_nbytes(rows, core.shape[1] * 128), dtype=torch.uint8, device=core.device)

    def _ring_rows(self, step, wc: int, wr: int) -> int:
        """How many rows a captured step folds into one ring launch per kernel: all of them when both row lanes
        are bound and every segment is a decode block the direct rings accept (the same conditions the per-segment
        loop's `direct_conv` checks); 0 keeps the loop -- eager steps, prefill chunks, a reference table."""
        if (not getattr(step, "captured", False) or getattr(step, "contexts", None) is None
                or self.lanes.kda_recurrent_ring_rows is None or self.lanes.conv_ring_rows is None):
            return 0
        t = step.segments[0].length
        if t > wr or t > min(8, wc) or any(s.length != t for s in step.segments):
            return 0
        return len(step.segments)

    # -- sparse MLA + kpool indexer ------------------------------------------------------
    @operation("indexer", layer_arg=1)
    def _indexer(self, L: int, x: torch.Tensor, qr: torch.Tensor, step: Step, caches: Caches, *, query=None):
        """kpool indexer: per segment, complete this step's pools (pooling the
        tail ring's earlier tokens with the new ones), keep the new tail, then
        select for every query the top-k complete pools before it plus the
        in-progress tail, as latent slots (valid prefix first) and counts."""
        F, p, n = self.F, self.p, f"L{L}.idx."
        N = x.shape[0]; kp, nh, d = F.kpool, F.idx_heads, F.idx_dim
        prefix = self._mla_prefix(N, step)
        shard = None
        if (self.prefill_indexer_shards and N >= 128 and len(step.segments) == 1
                and not getattr(step, "captured", False) and not self.probe):
            from engine.modules.prefill_indexer import QueryShard
            shard = QueryShard(N, step.segments[0].ctx, self.rank, self.comm.world_size, F.topk // kp, kp)
        if shard is not None:
            query_x = shard.project_input(x) if shard.score_rows else None
            query_qr = shard.project_input(qr) if shard.score_rows else None
        elif prefix:
            from engine.modules.prefill_indexer import project_query_rows
            query_x = project_query_rows(x, prefix, N) if prefix < N else None
            query_qr = project_query_rows(qr, prefix, N) if prefix < N else None
        else:
            query_x, query_qr = x, qr
        if query is not None and (not getattr(step, 'captured', False) or prefix or shard is not None
                                  or query.shape != (N, nh * d)):
            raise ValueError('a shared indexer query must cover the entire captured decode step')
        q = (query if query is not None else self.linear(query_qr, n + "wq_b")).view(-1, nh, d) if query_qr is not None else None
        pair = self._decode_pair(L, step, N)
        w, head_splits = self._indexer_head_gate(L, query_x, step) if query_x is not None else (None, 1)
        if head_splits != 1 and (pair is None or prefix or shard is not None):
            raise RuntimeError('bound head-gate partials require the full captured paired boundary')
        if pair is not None:
            from engine.kernels.decode_projection import indexer_boundary
            k, gate = pair(x)
            q8, k, w_eff = indexer_boundary(q, k, w, p[n + "k_norm_w"], p[n + "k_norm_b"], F.idx_scale,
                                            rows=getattr(pair, 'rows', None), head_splits=head_splits)
        else:
            k = self.linear(x, n + "wk")
            if self.lanes.layernorm is None:
                k = Fn.layer_norm(k.float(), (d,), p[n + "k_norm_w"], p[n + "k_norm_b"], K_NORM_EPS).to(x.dtype)
            else:
                k = self.lanes.layernorm(k, p[n + "k_norm_w"], p[n + "k_norm_b"], K_NORM_EPS)
            gate = self.linear(x, n + "gate")
            q8 = w_eff = None
            if q is not None:
                q8, qs = self.lanes.indexer_quant(q.reshape(-1, d))
                q8 = q8.view(q.shape[0], nh, d)
                w_eff = self.lanes.head_gate(w, qs.view(q.shape[0], nh), F.idx_scale)
                if shard is not None or prefix:
                    query_rows = shard.score_rows if shard is not None else N-prefix
                    q8, w_eff = q8[:query_rows], w_eff[:query_rows]
        width = F.topk + kp - 1
        slots_out = torch.empty((N, width), dtype=torch.int32, device=x.device)
        valid_out = torch.empty(N, dtype=torch.int32, device=x.device)
        keys, scales = caches.pool_keys(L), caches.pool_scales(L)
        captured = getattr(step, "captured", False)
        index = iota if captured else fresh                 # see _dsa: kept only for the bounded set
        tail_width = kp - 1 + F.spec_k
        if captured:
            from engine.profiles.glm53.decode_graphs import complete_pools   # the profile's captured writer
            mapped = (self._indexer_rows(step, caches) and step.tokens == 8
                      and N in getattr(self, 'decode_dsa_rows', ()) and not self.probe)
            tails = caches.tail_field(L) if mapped else caches.tails(L)
            if tails.shape[1] != tail_width:
                raise ValueError(f"indexer tail needs {tail_width} positions to support draft rollback")
            # Every segment's pools are completed before any selection runs. A segment
            # selects only from its own sequence's blocks and a step may not carry a
            # sequence twice, so the order the pools are written in changes nothing.
            rows = len(step.segments)
            width = step.tokens
            pooled = complete_pools(self, L, step.contexts, width, tails,
                                    k.view(rows, width, d), gate.view(rows, width, d), caches, mapped=bool(mapped))
            if self._indexer_rows(step, caches):
                # every row's selection with the per-row launches folded (45차, the C=4 question)
                self._select_rows(L, q8, w_eff, keys, scales, pooled, step.contexts, width, caches, slots_out, valid_out)
                return slots_out, valid_out
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
            if shard is not None:
                selected = None
                if shard.score_rows:
                    cand = caches.pool_slots(L, s.seq, index(n_cand, x.device)).long()
                    selected = self._select_pools(q8, w_eff, keys[cand], scales[cand],
                        seq_lens[shard.score_begin:shard.end] // kp, n_cand, F.topk // kp, layer=L,
                        seq_lens=seq_lens[shard.score_begin:shard.end], pool=kp)
                pool_ids = shard.collect(selected, seq_lens // kp, self.comm)
                self.prefill_indexer_executed.add(L)
            elif prefix:
                from engine.modules.prefill_indexer import covered_pool_ids
                pool_ids = torch.empty((s.length, F.topk // kp), dtype=torch.int32, device=x.device)
                covered_pool_ids(seq_lens[:prefix] // kp, F.topk // kp, out=pool_ids[:prefix])
                if prefix < s.length:
                    cand = caches.pool_slots(L, s.seq, index(n_cand, x.device)).long()
                    self._select_pools(q8, w_eff, keys[cand], scales[cand], seq_lens[prefix:] // kp,
                                       n_cand, F.topk // kp, out=pool_ids[prefix:], layer=L,
                                       seq_lens=seq_lens[prefix:], pool=kp)
            elif n_cand:
                cand = caches.pool_slots(L, s.seq, index(n_cand, x.device)).long()
                pool_ids = self._select_pools(q8[sl], w_eff[sl], keys[cand], scales[cand], seq_lens // kp, n_cand, F.topk // kp,
                                          layer=L, seq_lens=seq_lens, pool=kp)
            else:
                pool_ids = torch.full((s.length, F.topk // kp), -1, dtype=torch.int32, device=x.device)
            self.lanes.pool_slots(pool_ids, seq_lens, kp, *caches.token_map(L, s.seq),
                                  slots_out[sl], valid_out[sl])
        if prefix:
            self.prefill_covered_queries_executed.add(L)
        return slots_out.contiguous(), valid_out

    def _indexer_rows(self, step, caches) -> int:
        """How many rows a captured step's DSA layer handles in one folded pass (the latent write and the
        indexer's selection): all of them when the glue lanes are bound, the caches are a graph's (gathered block
        table: `token_maps`, `pool_maps`) and every segment is the step's width; 0 keeps the per-segment loops --
        eager steps, prefill chunks, the eager caches, a table without the glue."""
        if (not getattr(step, "captured", False) or getattr(step, "contexts", None) is None
                or self.lanes.decode_rows is None or not hasattr(caches, "pool_maps") or not hasattr(caches, "token_maps")):
            return 0
        t = getattr(step, "tokens", None)
        if t is None or any(s.length != t for s in step.segments):
            return 0
        return len(step.segments)

    def _select_rows(self, L: int, q8, w_eff, keys, scales, n_cand: int, contexts, t: int, caches, slots_out, valid_out,
                     *, joined: bool = True, native: bool = True) -> None:
        """The segment loop's selection for every row of a captured step, with the per-row launches folded.

        Per row the loop gathers the row's candidate keys and scales, scores them, masks past the row's horizon,
        takes the top-k, pads the misses with -1 and finalizes against the row's block row -- some twenty
        launches a row a layer. Here the lengths are shared across this batch's layers, the candidate keys
        and scales one gather (`lanes.decode_rows`), the horizon a mask written in place and the finalize
        one launch over the rows' block rows. One row keeps the logits kernel and the top-k as the loop runs
        them, on the row's own tensors. Joined rows (C=2) run each once for the step: every query is scored over
        its own row's window of the candidates laid end to end (`lengths` with the candidate width), and column
        c of a row's logits is that row's candidate c, as in the row's own launch. DeepGEMM scores queries four a
        block from the block's first window start rounded down to four, so with t and n_cand multiples of four a
        block holds one row and starts where that row's window starts -- where the row's launch starts, at 0.
        The mask and the top-k then take the [rows*t, n_cand] block whole: its slices keep their size, and up to
        JOINED_SLICES slices torch selects the way it does for a row's t. `joined=False` is the probe's same-build
        control, and so is `native=False`, which keeps the horizon mask and torch.topk where
        `engine.kernels.decode_topk` now selects -- the same set either way, so the finalize writes the same
        slots and counts. The loop's -1 for a winner past the horizon is left to the finalize, which masks
        `id >= length // pool` itself (kernel and oracle alike) -- the slots and counts it writes are the same.
        `contexts` [rows], `t` tokens a row."""
        from engine.base.constants import zeros
        F = self.F
        kp, k = F.kpool, F.topk // F.kpool
        rows = contexts.shape[0]
        dev = q8.device
        if n_cand < k:
            raise ValueError(f"a captured step's candidate capacity ({n_cand} pools) is below the selection width ({k})")
        glue = self.lanes.decode_rows
        joined = joined and 1 < rows and rows * t <= JOINED_SLICES and t % 4 == 0 and n_cand % 4 == 0
        # [rows*t] i32 each, read-only across this forward's DSA layers: lengths, horizons and, joined, the query windows
        lengths = caches.row_lengths(contexts, t, kp, glue.lengths, width=n_cand if joined else 0)
        seq_lens, ke = lengths[:2]
        # The same `index_kpool_always_select_tail` pin the row loop applies: this is the
        # captured decode path, so without it here the guarantee would hold in prefill and
        # eager decode and vanish in the served one.
        pin = tail_pin_pools(seq_lens, kp)
        keys_all, scales_all = glue.candidates(keys, scales, *caches.pool_maps(L), n_cand)   # [rows, n_cand, d], [rows, n_cand]
        # `select_native` returns None whenever it does not admit the shape -- a reference lane's
        # CPU logits included -- and the horizon mask plus torch.topk stay as they were.
        # `native=False` is the probe's same-build control: the Torch path on the same graph.
        pick = select_native if native else (lambda *a: None)
        winners = None
        if joined:
            starts, ends = lengths[2:]
            logits = self.lanes.indexer_logits(q8, keys_all.view(rows * n_cand, -1), scales_all.view(-1), w_eff, ends,
                                               ks=starts, width=n_cand)
            pin_pools_in_logits(logits, pin, k=k)
            winners = pick(logits, ke, k)                     # horizon + top-k, one launch, logits untouched
            if winners is None:
                glue.horizon(logits, ke)                                                  # -inf past each query's pools, in place
                winners = _torch_topk(logits, k)
        else:
            ks = zeros(t, dev)
            for r in range(rows):
                sl = slice(r * t, (r + 1) * t)
                logits = self.lanes.indexer_logits(q8[sl], keys_all[r], scales_all[r], w_eff[sl], ke[sl], ks=ks)[:, :n_cand].float()
                pin_pools_in_logits(logits, pin[sl], k=k)
                got = pick(logits, ke[sl], k)
                if got is None:
                    glue.horizon(logits, ke[sl])                                          # -inf past each query's pools, in place
                    got = _torch_topk(logits, k)
                if winners is None:
                    winners = torch.empty((rows * t, k), dtype=got.dtype, device=dev)
                winners[sl] = got
        self.lanes.pool_slots(winners, seq_lens, kp, *caches.token_maps(L), slots_out, valid_out, tokens=t)


    def _select_pools(self, q8, w_eff, keys, scales, ke, n_cand: int, k: int, *, out=None,
                      layer: "int | None" = None, seq_lens=None, pool: "int | None" = None) -> torch.Tensor:
        """Top-k complete pools per query, in passes of SELECT_ROWS rows: every row's
        selection is independent, so the passes are exact and the transient is bounded.

        `seq_lens` and `pool` carry the raw horizon (`ke` is its pool count): with them
        the selection also honors `index_kpool_always_select_tail` -- a row whose
        sequence length is a whole number of pools has an empty appended tail, so the
        pool that just completed is pinned above that row's maximum instead of being
        left to the top-k (see `tail_pin_pools`)."""
        rows = q8.shape[0]
        pin = tail_pin_pools(seq_lens, pool) if (seq_lens is not None and pool) else None
        if out is not None and (out.shape != (rows, k) or out.dtype != torch.int32
                or out.device != q8.device or not out.is_contiguous()):
            raise ValueError('pool selection destination must match contiguous int32 query rows')
        def select(logits, lengths):
            values = logits[:, :n_cand].float()
            if values.is_cuda:
                # Wide passes keep the prefill selector (one block a row, 1024 threads, the row
                # read twice); a decode-width pass takes the fused one, which is the same answer
                # with the horizon in registers and one pass over the row.
                if rows > 64:
                    from engine.kernels.prefill_topk import select as prefill_select
                    selected = prefill_select(values, lengths, k)
                else:
                    selected = select_native(values, lengths, k)
                if selected is not None:
                    return selected
            return topk_positions(values, k, valid=lengths, inplace=True)
        if rows <= SELECT_ROWS:
            logits = self.lanes.indexer_logits(q8, keys, scales, w_eff, ke)
            pin_pools_in_logits(logits, pin, k=k)
            selected = select(logits, ke)
            result = selected if out is None else out.copy_(selected)
            from_net(self, layer, q8=q8, w_eff=w_eff, keys=keys, scales=scales, ke=ke, seq=seq_lens, pool=pool,
                     n_cand=n_cand, k=k, selected=result, prefill=rows > 64)
            return result
        if out is None:
            out = torch.empty((rows, k), dtype=torch.int32, device=q8.device)
        for r0 in range(0, rows, SELECT_ROWS):
            r1 = min(rows, r0 + SELECT_ROWS)
            logits = self.lanes.indexer_logits(q8[r0:r1], keys, scales, w_eff[r0:r1], ke[r0:r1])
            pin_pools_in_logits(logits, None if pin is None else pin[r0:r1], k=k)
            out[r0:r1] = select(logits, ke[r0:r1])
        from_net(self, layer, q8=q8, w_eff=w_eff, keys=keys, scales=scales, ke=ke, seq=seq_lens, pool=pool,
                 n_cand=n_cand, k=k, selected=out, prefill=rows > 64)
        return out

    @operation("dsa", layer_arg=1)
    def _dsa(self, L: int, x: torch.Tensor, step: Step, caches: Caches, reduce=None, *, project=None) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.mla."
        N = x.shape[0]; Hl = self.Hl
        q_a, kv_c = self.linear(x, n + "qkv_a").split([F.q_lora, F.kv_lora], dim=-1)
        qr = self._norm(q_a, p[n + "q_a_norm"], F.rms_eps)
        folded = self._indexer_rows(step, caches)
        shared = (folded and getattr(step, 'tokens', None) == 8
                  and N in getattr(self, 'decode_dsa_rows', ()) and not self.probe)
        query = None
        if shared:
            q, query = self._query_pairs[L](qr)
        else:
            q = self.linear(qr, n + "q_b")
        q = q.view(N, Hl, F.qk_nope)
        latent = caches.latent(L)
        # A captured step asks for the same few lengths forever, so they come from the kept
        # constants; an eager prefill's follow the request and would grow that cache unbounded.
        captured = getattr(step, "captured", False)
        index = iota if captured else fresh
        if shared:
            self.lanes.latent_norm_write(kv_c, p[n + 'kv_a_norm'], latent, *caches.token_maps(L),
                                         step.contexts, step.tokens, F.rms_eps)
            self.decode_latents_executed.add((L, N))
        else:
            kv_n = self._norm(kv_c, p[n + "kv_a_norm"], F.rms_eps)
        if folded and not shared:
            # every row's new latents in one launch: the slot arithmetic over the gathered block table and the
            # scatter (45차, the C=4 question: three launches a row a layer became one a layer)
            self.lanes.decode_rows.latent_write(kv_n.to(E4M3), latent, *caches.token_maps(L), step.contexts, step.tokens)
        for s in (() if folded else step.segments):                                # fp8 KV, scale 1 (no kv scales in the checkpoint)
            sl = slice(s.start, s.start + s.length)
            latent[caches.token_slots(L, s.seq, (s.ctx + index(s.length, x.device))).long()] = kv_n[sl].to(E4M3)
        slots, valid = (self._indexer(L, x, qr, step, caches, query=query) if shared
                        else self._indexer(L, x, qr, step, caches))
        kv_b = p[n + "kv_b"].view(Hl, F.qk_nope + F.v_dim, F.kv_lora)
        w_uk, w_uv = kv_b[:, : F.qk_nope, :], kv_b[:, F.qk_nope:, :]
        q_abs = self._mla_absorb(L, q, w_uk, step)  # absorb W_UK: MQA over the latent
        ctx_lat = self._mla_context(L, q_abs.contiguous(), latent, slots, valid, step, caches)
        o = self._mla_absorb(L, ctx_lat, w_uv, step, transpose=True)  # un-absorb W_UV
        return (reduce or self.comm.all_reduce)((project or self.linear)(o.reshape(N, Hl * F.v_dim), n + "o_proj"))

    def _mla_absorb(self, L, x, weight, step, *, transpose=False):
        if (getattr(step, 'captured', False) and not self.probe
                and len(x) in getattr(self, 'decode_absorb_rows', ())):
            return self._decode_absorb[L](x, transpose=transpose)
        if (self.prefill_absorb_tiles and self.lanes.mla_absorb is not None
                and not self.probe and not getattr(step, "captured", False)
                and len(step.segments) == 1 and 128 <= len(x) <= 32768):
            out = self.lanes.mla_absorb(x, weight, transpose=transpose)
            self.prefill_absorb_tiles_executed.add((L, "output" if transpose else "query"))
            return out
        return torch.einsum("thc,hvc->thv" if transpose else "thd,hdc->thc", x, weight)

    def _mla_prefix(self, rows, step):
        if (self.prefill_dense_prefix and self.lanes.mla_dense_prefix is not None
                and 128 <= rows <= 32768 and len(step.segments) == 1
                and not getattr(step, 'captured', False) and not self.probe):
            from engine.modules.prefill_attention import covered_prefix
            prefix = covered_prefix(rows, step.segments[0].ctx, self.F.topk, self.F.kpool)
            return prefix if prefix >= 128 else 0
        return 0

    def _mla_context(self, L, q, latent, slots, valid, step, caches):
        prefix = self._mla_prefix(len(q), step)
        if not prefix:
            return self.lanes.mla_sparse(q, latent, slots, valid, self.F.mla_scale, 1.0)
        s = step.segments[0]
        out = torch.empty_like(q)
        self.lanes.mla_dense_prefix(q[:prefix], latent, *caches.token_map(L, s.seq),
                                   s.ctx, self.F.mla_scale, 1.0, out=out[:prefix])
        self.prefill_dense_prefix_executed.add(L)
        if prefix < len(q):
            self.lanes.mla_sparse(q[prefix:], latent, slots[prefix:], valid[prefix:], self.F.mla_scale, 1.0,
                                  out=out[prefix:])
        return out

    # -- MLPs -----------------------------------------------------------------------------
    @operation("dense", layer_arg=1)
    def _dense(self, L: int, x: torch.Tensor, reduce=None, *, project=None) -> torch.Tensor:
        p, n = self.p, f"L{L}.mlp."
        g, u = self.linear(x, n + "gate_up").chunk(2, dim=-1)
        return (reduce or self.comm.all_reduce)((project or self.linear)(self._activation(g, u, self.F.swiglu_limit), n + "down"))

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
        raw scores renormalised, times routed_scaling_factor.

        Decode and every prefill width use IEEE FP32 operands, accumulation
        and logits. Native execution reads the resident gate directly; the
        unprepared reference converts its checkpoint gate at the call site."""
        if (getattr(self, 'fused_decode_router', False) and self._router_layers is not None
                and getattr(self, 'route_skip', None) is None
                and x.shape[0] in (8, 16) and x.shape[0] in self.decode_fastpath_rows):
            from engine.kernels.router_fused import route
            result = route(x, self._router_weights[L], self._router_fused_bias[L],
                           self.F.topk_experts, self.F.routed_scale)
            self._router_fp32.add(L)
            self._router_fused_executed.add((L, x.shape[0]))
            return result
        if self._router_layers is not None:
            from engine.kernels.glm_pointwise import router_logits
            logits = router_logits(x, self._router_weights[L])
            self._router_fp32.add(L)
        else:
            gate = self.p[f'L{L}.moe.gate'].float()
            if x.is_cuda:
                from engine.kernels.router_fp32 import router_logits
                logits = router_logits(x, gate)
            else:
                logits = x.float() @ gate.T
        return self._select_routes(L, logits)

    def _select_routes(self, L, logits):
        F, p, n = self.F, self.p, f"L{L}.moe."
        if self.lanes.route_weights is not None:
            sel, w = self.lanes.route_weights(logits, p[n + "bias"], F.topk_experts, F.routed_scale)
        else:
            s = torch.sigmoid(logits)
            sel = (s + p[n + "bias"]).topk(F.topk_experts, dim=-1).indices
            w = s.gather(-1, sel)
            sel, w = sel.to(torch.int32), w / (w.sum(-1, keepdim=True) + 1e-20) * F.routed_scale
        skip = getattr(self, "route_skip", None)       # absent on lightweight stand-ins that borrow this method
        if skip is not None:
            tau = skip.get(L) if isinstance(skip, dict) else skip
            if tau is not None:
                w = skip_route_weights(w, float(tau), F.routed_scale)
        return sel, w


    def _packet_ffn_layers(self, rows):
        from engine.modules.prefill_packets import agreed_layers, ffn_packet_rows
        if not getattr(self, 'prefill_ffn_packets', False) or not ffn_packet_rows(rows):
            return frozenset()
        supported = set()
        for L, capable in self._packet_capabilities.items():
            gate = self.p[f'L{L}.moe.gate']
            shared = self.dense.get(f'L{L}.moe.sh_gate_up')
            projector = getattr(shared, 'packet_projector', lambda: None)()
            if (projector is not None and gate.is_cuda and gate.is_contiguous()
                    and gate.dtype == torch.bfloat16 and tuple(gate.shape) == (288,4096)
                    and capable(rows)):
                supported.add(L)
        agreed = agreed_layers(self.comm, self.layers, supported)
        self.prefill_packet_planned.update(agreed)
        return agreed


    @operation('moe', layer_arg=1)
    def _moe_packets(self, L, batch, shards):
        from engine.kernels.prefill_collectives.routes import packet_routes
        n = f'L{L}.moe.'
        ids, weights = packet_routes(batch)
        out = self._packet_experts[L](batch, ids, weights)
        project = self.dense[n+'sh_gate_up'].packet_projector()
        if project is None:
            raise RuntimeError('packet FFN reader changed after the agreed plan')
        g, u = project(batch.received, batch.geometry.local_rows, real_rows=shards.rows, routed=True).chunk(2, dim=-1)
        shared = self.linear(self._activation(g, u, self.F.swiglu_limit), n+'sh_down')
        self.prefill_packet_executed.add(L)
        self.prefill_packet_peak_bytes = max(self.prefill_packet_peak_bytes, batch.geometry.nbytes)
        return (shards.reduce_scatter_pair(out, shared) if shards.fuse_sum else
                shards.reduce_scatter(out + shared))

    def _sender_routes(self, L, roundtrip):
        from engine.kernels.prefill_router import router_shard_logits
        gate = (self._router_weights[L] if self._router_layers is not None
                else self.p[f'L{L}.moe.gate'].float())
        logits = router_shard_logits(roundtrip, gate)
        if self._router_layers is not None:
            self._router_fp32.add(L)
        return self._select_routes(L, logits)


    @operation("moe", layer_arg=1)
    def _moe(self, L: int, x: torch.Tensor, reduce=None, *, reduce_pair=None, finalize=None, route_observer=None) -> torch.Tensor:
        F, p, n = self.F, self.p, f"L{L}.moe."
        # GPU component gate: C=1 wins; C=4 with reused routes regresses.
        # Keep the established shared chain for wider captured batches.
        if self.shared_overlap is not None and x.shape[0] <= F.spec_k + 1:
            def routed(consume=None):
                sel, w = self.route(L, x)
                if route_observer is not None:
                    route_observer(L, sel)
                return (self._experts[L](x, sel, w) if consume is None else
                        self._experts[L](x, sel, w, finalize=consume))
            joined = (self.shared_overlap(self.shared_mlp[L], x, routed) if finalize is None else
                      self.shared_overlap(self.shared_mlp[L], x, routed, finish=finalize))
            return joined if finalize is not None else (reduce or self.comm.all_reduce)(joined)
        sel, w = self.route(L, x)
        if route_observer is not None:
            route_observer(L, sel)
        if finalize is not None:
            def consume(acc):
                g, u = self.linear(x, n + "sh_gate_up").chunk(2, dim=-1)
                shared = self.linear(self._activation(g, u, F.swiglu_limit), n + "sh_down")
                return finalize(acc, shared)
            return self._experts[L](x, sel, w, finalize=consume)
        out = self._experts[L](x, sel, w)
        g, u = self.linear(x, n + "sh_gate_up").chunk(2, dim=-1)
        shared = self.linear(self._activation(g, u, F.swiglu_limit), n + "sh_down")
        # Both lanes return BF16. Its add accumulates in FP32 and rounds once,
        # just like the former float() + float() followed by to(BF16).
        if reduce_pair is not None:
            return reduce_pair(out, shared)
        return (reduce or self.comm.all_reduce)(out + shared)

    # -- the step ---------------------------------------------------------------------------
    def forward(self, step: Step, caches: Caches, finish: bool = True, aux_layers=None, aux_ready=None,
                *, last_hidden_only=False, contract=None):
        """One step: every segment's tokens through the chain. Returns the final
        hidden states [N, hidden] (post final norm) when `finish`, else the raw
        mHC carry (res, post, comb, x) for inspection. With `aux_layers`, also
        the contracted residual after each of those layers, concatenated
        [N, len * hidden] -- what the drafter reads (the served model's
        aux_hidden_states: hc_post then hc_contract after layer idx). Prefill may
        request only the final hidden row, which supplies its first sampled token."""
        F = self.F
        if contract is not None and aux_layers and any(L not in self.layers for L in aux_layers):
            raise ValueError("terminal features must name layers in this target")
        N = step.ids.shape[0]
        sp = self.prefill_transport if (finish and not self.probe and len(step.segments) == 1
                                       and N >= 128
                                       and not getattr(step, "captured", False)) else None
        if sp is not None:
            from engine.modules.token_shards import TokenShards
            sp = TokenShards(sp, N, self.rank)
        packet_layers = self._packet_ffn_layers(sp.rows) if sp is not None else frozenset()
        reduce = sp.reduce_scatter if sp else self.comm.all_reduce
        x = self.embed(step.ids)
        for pos, rows in step.patches:                                               # image rows in place of their placeholders
            x.index_copy_(0, pos, rows.to(x.dtype))
        if sp:
            x = sp.shard(x)
            N = x.shape[0]
        res = x[:, None, :].expand(N, F.hc, F.hidden).contiguous()                   # hc_expand
        post = comb = None
        aux = {}
        features = (torch.empty((N, len(aux_layers) * F.hidden), device=x.device, dtype=x.dtype)
                    if contract is not None and aux_layers else None)
        for L in self.layers:
            if post is not None:
                res, post, comb, x = self._hc_post_pre(L, x, res, post, comb, "attn")
            else:
                post, comb, x = self._hc_pre(L, res, "attn")
            projection = None
            if sp and not F.is_dsa(L) and sp.project_tiles:
                projection = self.prefill_project(sp, x, f"L{L}.kda.in_proj")
            elif sp:
                x = sp.all_gather(x.contiguous())
            x = self._dsa(L, x, step, caches, reduce) if F.is_dsa(L) else self._kda(
                L, x, step, caches, reduce, projection=projection)
            if self.probe:
                self.probe("dsa" if F.is_dsa(L) else "kda", L, x)
            res, post, comb, x = self._hc_post_pre(L, x, res, post, comb, "ffn")
            if L in packet_layers:
                packets = sp.all_gather_packets(x.contiguous(), route=lambda local: self._sender_routes(L, local))
                x = self._moe_packets(L, packets, sp)
                del packets  # all readers used this stream; the next FFN owns a new packet
            else:
                if sp:
                    x = sp.all_gather(x.contiguous())
                if F.is_moe(L) and sp and sp.fuse_sum:
                    x = self._moe(L, x, reduce, reduce_pair=sp.reduce_scatter_pair)
                else:
                    x = self._moe(L, x, reduce) if F.is_moe(L) else self._dense(L, x, reduce)
            if self.probe:
                self.probe("moe" if F.is_moe(L) else "dense", L, x)
            if aux_layers and L in aux_layers:
                if contract is None:
                    aux[L] = self.lanes.mhc_post(x, res, post, comb).float().mean(1).to(x.dtype)
                else:
                    for i, requested in enumerate(aux_layers):
                        if requested == L:
                            contract(x, res, post, comb, out=features[:, i * F.hidden:(i + 1) * F.hidden])
                if aux_ready is not None and L == max(aux_layers):
                    if sp is not None:
                        raise ValueError("early draft observation belongs to decode, not SP prefill")
                    if features is None:
                        features = torch.cat([aux[l] for l in aux_layers], dim=-1)
                    aux_ready(features)
        if not finish:
            return res, post, comb, x
        if last_hidden_only:
            last = slice(sp.last_local, sp.last_local + 1) if sp else slice(-1, None)
            x, res, post, comb = x[last], res[last], post[last], comb[last]
        if contract is None:
            res = self.lanes.mhc_post(x, res, post, comb)
            hidden = res.float().mean(1).to(x.dtype)
        else:
            hidden = contract(x, res, post, comb)
        h = self._norm(hidden, self.p["norm"], F.rms_eps)
        if sp:
            h = self.comm.all_gather(h, dim=0) if last_hidden_only else sp.gather_result(h)
        if last_hidden_only:
            h = h[-1:]  # the last SP rank owns the global last token
        if aux_layers:
            if features is None:
                features = torch.cat([aux[L] for L in aux_layers], dim=-1)
            return h, sp.gather_result(features) if sp else features
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
