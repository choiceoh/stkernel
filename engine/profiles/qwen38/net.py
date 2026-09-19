"""Qwen3.8-Flash-Next as the engine serves it at TP=4 (profile): the served composition.

48 layers under the gated residual streams (four copies of hidden 2560), each a GatedDeltaNet (36) or a gated GQA with
QSA selection (12, every fourth), then the MoE (512 NVFP4 experts top-10, EP: 128 a rank, plus a sigmoid-gated shared
expert, TP); PLE's hashed n-gram table injected before layer 1; a closing mixer and a vocab-parallel head. The math is
engine/modules' (engine/profiles/qwen38/composition.py assembles the reference from them); this file is its served form:
plain functions over the rank file's views (specs.py) and the caches (caches.py), kernels from a lane table (lanes.py).

The launches a layer issues are the point of the file (the "cuts" of the Qwen3.8 estimate, 2026-09-15):

    hyper-connection site     5 launches: the previous leave joined to the stream norm, down+inject in one GEMM,
                              the gates, up, the stream mean (engine/kernels/gated_residual)
    GatedDeltaNet             one GEMM for q|k|v|z|b|a (merged at preshard), the conv and the delta rule on their ring
                              kernels (the delta rule's launch computes the decay and beta from the projection's
                              columns; a prefill chunk's gates are one launch before its chunk kernel), the output norm
                              in one launch, out_proj
    attention                 one GEMM for query+gate|k|v|index (merged at preshard), one norm+partial-rope launch for
                              the query heads, one for the key head, one for the index queries; the QSA ops (a host
                              step the budget covers attends every position it sees: no scores, no top-k, no ids,
                              one dense causal launch whose runs of rows share their K/V tiles -- `_covers`; past it each rank scores a quarter of a long prefill
                              step's index queries and the ids are gathered -- `_sharded_blocks`, unless the boot
                              declined `query_shards`)
    MoE                      the router and the shared gate in one GEMM (merged at preshard), top-k, the rank's experts
                              in one dispatcher launch (another rank's routes skip in the micro kernel on a captured
                              step), the shared expert's two GEMMs and activation, ONE all-reduce for routed and shared

TP=4 is the shape of the code (comm must be one rank of four). Collectives: the embedding and the PLE table lookup
(vocab-parallel rows summed), each mixer's output projection, each MoE's sum; the head gathers or takes a vocab argmax.

A step is segments -- (seq, slot, ctx, start, length) -- over a flat token array. Every per-sequence value is addressed
by position (caches.py), so a rejected draft is overwritten, not rolled back, and prefill and decode share this code:
the lanes differ (chunk vs ring kernels), the composition does not.

A captured decode step (`DeviceStep`, decode_graphs.py) is the same forward with its per-row values on the device: the
rows' contexts, slots and sequences are the graph's static inputs, the page table is gathered at the context bucket's
width, and the three places a host step loops over its segments -- the GDN rings, the PLE rings and the PLE table --
run over every row at once (the row ring kernels; the PLE rows gathered off the SSD table on the host BEFORE the
replay into the graph's static staging buffer, ple_table.py). Nothing in the replay reads a device value on the host.

The PLE table is not in memory (the operator's decision of 2026-09-18): each rank's vocabulary range lives in
`ple-r{r}of4.weight` beside its rank file and rows are read by id when a step needs them -- an eager step (prefill)
hashes its tokens on the host and gathers into a fresh tensor (`_ple_embed`), a captured step's rows are staged by
`stage_ple` before its replay. The scale, the gate, the norm and the conv are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from engine.base.constants import iota
from engine.profiles.qwen38 import specs
from engine.profiles.qwen38.facts import TP, Facts
from engine.profiles.qwen38.lanes import Lanes
from engine.profiles.qwen38.ple_table import PLEStaging, local_rows

BF16, F32 = torch.bfloat16, torch.float32
HEAD_NAME = "Qwen4ExpForCausalLM/lm_head"          # the pack store's calibration name of the head's FP8 GPTQ
HC_NAME = "Qwen4ExpForCausalLM/hyper_connection"    # the FP8 mixer lanes' names (hc_fp8; no calibration yet)
MTP_PRECISIONS = ("bf16", "fp8", "w4")                 # the MTP head's dense projections (Qwen38Net mtp_precision)
FIRST_BUCKET = 4096                                # tokens of the smallest context bucket; each next one doubles


def bucket_blocks(block: int, needed: int, top: int) -> int:
    """The page-table width a step addresses: the smallest rung of 4,096 * 2^i tokens, in whole blocks, that covers
    `needed` blocks, never past `top`. The paged QSA kernels compile their table width in (a constexpr), so prefill
    and the captured decode buckets share these rungs: one compile a rung, not one a prompt length."""
    if block <= 0 or top <= 0 or not 0 < needed <= top:
        raise ValueError(f"a step needs 1..{top} blocks of {block} tokens, not {needed}")
    reach = FIRST_BUCKET
    while True:
        blocks = min(top, -(-reach // block))
        if blocks >= needed:
            return blocks
        reach *= 2


@dataclass(frozen=True)
class Segment:
    seq: int
    slot: int
    ctx: int                    # tokens already computed for this sequence: this segment's positions are ctx..
    start: int                  # first token of the segment in the step's flat arrays
    length: int


@dataclass(frozen=True)
class Step:
    ids: torch.Tensor           # [N] int64
    segments: "tuple[Segment, ...]"
    marks: tuple = ()           # ((position into ids, snapshot) ...): block boundaries inside a prefill segment

    def __post_init__(self):
        if self.ids.ndim != 1 or self.ids.dtype != torch.int64 or not self.segments:
            raise ValueError("a step needs a flat int64 token vector and nonempty segments")
        if self.marks:
            positions = [p for p, _ in self.marks]
            if (len(self.segments) != 1 or positions != sorted(set(positions)) or positions[0] <= 0
                    or positions[-1] >= self.ids.numel()):
                raise ValueError("marks are increasing positions strictly inside a single prefill segment")
        end, seqs, slots = 0, set(), set()
        for s in self.segments:
            if s.start != end or s.length <= 0 or s.ctx < 0 or s.seq < 0 or s.slot <= 0:
                raise ValueError("segments must cover tokens contiguously with valid contexts and state slots")
            if s.seq in seqs or s.slot in slots:
                raise ValueError("a sequence and its state slot may appear only once per step")
            seqs.add(s.seq)
            slots.add(s.slot)
            end += s.length
        if end != self.ids.numel():
            raise ValueError("segment lengths must cover every token exactly once")

    @property
    def positions(self) -> torch.Tensor:
        return torch.cat([torch.arange(s.ctx, s.ctx + s.length, device=self.ids.device) for s in self.segments])

    @staticmethod
    def prefill(ids: torch.Tensor, ctx: int, seq: int, slot: int, marks: tuple = ()) -> "Step":
        return Step(ids, (Segment(seq, slot, ctx, 0, ids.shape[0]),), marks)

    @staticmethod
    def decode(chunks) -> "Step":
        """chunks: (ids, ctx, seq, slot) per sequence, 1 + K draft tokens each."""
        segs, start = [], 0
        for ids, ctx, seq, slot in chunks:
            segs.append(Segment(seq, slot, ctx, start, ids.shape[0]))
            start += ids.shape[0]
        return Step(torch.cat([c[0] for c in chunks]), tuple(segs))


@dataclass
class DeviceStep:
    """A captured decode step: `rows` rows of `tokens` ids each, back to back, with each row's context, state slot and
    sequence on the device -- the graph's static inputs, rewritten in place before a replay (decode_graphs.py).
    `blocks` is the context bucket's page-table width: every row's positions lie below blocks * F.block."""
    ids: torch.Tensor           # [rows * tokens] int64
    contexts: torch.Tensor      # [rows] int64
    slots: torch.Tensor         # [rows] int64
    seqs: torch.Tensor          # [rows] int64: block-table rows
    tokens: int
    blocks: int
    captured = True
    marks = ()

    @property
    def rows(self) -> int:
        return self.contexts.numel()


@dataclass
class StepMeta:
    """The step's addressing, built once on the device and read by every QSA layer and the PLE injection."""
    positions: torch.Tensor     # [N] int64
    positions32: torch.Tensor   # [N] int32
    rows_req: torch.Tensor      # [N] int32: the segment index of each row
    page_table: torch.Tensor    # [n, blocks] int32: each segment's physical blocks
    lengths: torch.Tensor       # [n] int32: ctx + length
    starts: torch.Tensor        # [n + 1] int32: row offsets of the segments
    slot_table: torch.Tensor    # [n, 1] int32: each segment's state slot (the key ring's "page")
    kv_slots: torch.Tensor      # [N] int32: page * block + pos % block
    key_slots: torch.Tensor     # [N] int32: page * (block/4) + (pos/4) % (block/4) where pos closes a group, else -1
    ring_slots: torch.Tensor    # [N] int32: slot * ring + pos % ring
    covered_blocks: "torch.Tensor | None" = None   # [N, index_blocks] int32, set by the step's first QSA layer when no
                                                   # row sees more groups than the budget holds (Qwen38Net._covered_blocks)
    groups_seen: "torch.Tensor | None" = None      # [N] int32: the complete groups each row sees, set by the first QSA
                                                   # layer that splits the step's index queries (Qwen38Net._sharded_blocks)


class Qwen38Net:
    def __init__(self, F: Facts, comm, lanes: Lanes, layers=None, *, mtp: bool = True, hc_fp8: bool = False,
                 query_shards: bool = True, mtp_precision: str = "bf16", mtp_experts: str = "nvfp4"):
        """`hc_fp8`: the hyper-connection mixers' two matmuls a site on block-scaled FP8 (engine/kernels/dense
        FP8Linear) instead of BF16 -- half the bytes every step reads from the largest weights it reads. The mixer's
        numbers change (round-to-nearest FP8 weights and activations), so it is a declared choice a boot makes and a
        quality bracket judges, not a default.

        `query_shards`: a long prefill step's index queries scored a quarter a rank and the chosen ids gathered
        (`_sharded_blocks`, carry Q11). The selection is the unsplit step's, row for row; what it trades is three
        quarters of each QSA layer's scoring for one all-gather of ids. On by the operator's decision of 2026-09-18
        with the fleet unmeasured (CHARTER D17): until an onepass record says which side of that trade a fleet lands
        on, a boot can decline it (fleet.py --no-query-shards, the launcher's ST_QUERY_SHARDS=0) and every rank
        scores every row as before.

        `mtp_precision`: the MTP head's dense projections (its attention's two, its shared expert's two), which the
        checkpoint keeps in BF16. "bf16" (the default, the operator's decision of 2026-09-19): the checkpoint's weights
        through torch's matmul, no quantisation -- a K=3 draft graph 4.39 ms against W4A8's 3.97 (q38mtp-0919a). "fp8":
        block-scaled FP8, decode rows on dense/fp8_rows, 4.10 ms. "w4": the target layers' W4A8 at decode rows, as
        before. A drafter's numbers change how many tokens a step yields, never which (verification picks them).

        `mtp_experts`: the MTP head's routed experts as the rank file keeps them ("nvfp4": re-encoded from the export's
        FP8, on b12x) or in the export's own FP8 ("fp8": the side file mtp_fp8.py writes, on kernels/moe_fp8_rows --
        `side_specs` names what a boot loads from it)."""
        if comm.world_size != TP:
            raise ValueError(f"qwen38 is written for TP={TP}; comm has world {comm.world_size}")
        if type(query_shards) is not bool:
            raise ValueError("query_shards is a declared boolean")
        self.F, self.comm, self.lanes, self.mtp, self.hc_fp8 = F, comm, lanes, mtp, hc_fp8
        self.query_shards = query_shards
        if mtp_precision not in MTP_PRECISIONS:
            raise ValueError(f"mtp_precision {mtp_precision!r}: one of {MTP_PRECISIONS}")
        self.mtp_precision = mtp_precision
        if mtp_experts not in ("nvfp4", "fp8"):
            raise ValueError(f"mtp_experts {mtp_experts!r}: nvfp4 or fp8")
        self.mtp_experts = mtp_experts if mtp else "nvfp4"
        self._hc_projections = {}
        self.rank = comm.rank
        self.layers = list(range(F.layers)) if layers is None else list(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(not 0 <= L < F.layers for L in self.layers):
            raise ValueError("model layers must be nonempty, unique and inside the profile")
        self.first_expert = F.expert_range(self.rank)[0]
        self.vp = F.vocab_local
        self.conv_ring = F.conv - 1 + F.spec_k         # GDN conv inputs kept per slot: the window plus K drafts
        self.rec_ring = F.spec_k + 1                   # GDN states kept per slot: one per verify position
        self.p = None
        self.dense = {}
        self.draft_index = None                         # dense/ivf_head over the head's rows (prepare_draft_head)
        self.draft_tap = None                           # kernels/common/row_tap: what draft_tokens saw (fleet --tap-draft-queries)
        self._experts = {}
        self._ple = self._ple_hash = self._ple_scale = None
        self.ple_table = self.ple_stage = None          # attach_ple: the rank's SSD table and the staged rows

    # -- binding ---------------------------------------------------------------------------------------------------
    def specs(self):
        """Every tensor the net binds: the rank file's, and with FP8 MTP experts the side file's in place of the rank
        file's NVFP4 ones (`side_specs`)."""
        out = specs.all_specs(self.F, self.layers, mtp=self.mtp)
        if getattr(self, "mtp_experts", "nvfp4") == "fp8":
            out = [s for s in out if s.name not in specs.MTP_NVFP4] + specs.mtp_fp8_specs(self.F)
        return out

    def side_specs(self):
        """The tensors a boot loads from the MTP side file (mtp_fp8.path), not the rank file."""
        return specs.mtp_fp8_specs(self.F) if getattr(self, "mtp_experts", "nvfp4") == "fp8" else []

    def bind(self, views: dict) -> None:
        from engine.base.params import bind
        from engine.modules.modelopt_scales import ModelOptScales
        self.p = bind(self.specs(), views)
        F, p = self.F, self.p
        for prefix in [f"L{L}." for L in self.layers] + (["mtp.L0."] if self.mtp else []):
            n = prefix + "moe."
            if prefix == "mtp.L0." and self.mtp_experts == "fp8":
                self._experts[prefix] = partial(self._moe_fp8, w13=p[n + "fp8.w13"], s13=p[n + "fp8.s13"],
                                                w2=p[n + "fp8.w2"], s2=p[n + "fp8.s2"])
                continue
            scales = ModelOptScales.bind(*(p[n + s] for s in ("w13_alpha", "a13_scale", "w2_alpha", "a2_scale")),
                                         experts=p[n + "w13"].shape[0], device=p[n + "w13"].device)
            if self.lanes.moe_prepare is not None:
                self.lanes.moe_prepare(p[n + "w13"], p[n + "w13_sf"], p[n + "w2"], p[n + "w2_sf"], F.topk_experts,
                                       scales=scales)
            self._experts[prefix] = partial(self.lanes.moe, w13=p[n + "w13"], w13_sf=p[n + "w13_sf"], w2=p[n + "w2"],
                                            w2_sf=p[n + "w2_sf"], scales=scales, first_expert=self.first_expert)
        ple = [L for L in self.layers if L in F.ple_layers]
        if ple:
            L = ple[0]
            self._ple = self._ple_feature(L)
            # the hash on the host (its tensors are the derived ones, equal to the checkpoint's buffers, checked
            # above): a step's rows are gathered off the SSD table before it runs (ple_table.py)
            self._ple_hash = self._ple.hashes(L)
            self._ple_scale = p[f"L{L}.ple.scale"].float().reshape(())

    @staticmethod
    def dense_names(keys):
        """The dense projections the W4A8/FP8 lanes serve, with the calibration name of each (the pack store's). PLE's
        two projections are not among them: the injection's gate is engine/modules' form over BF16 weights, once a step."""
        names = {"gdn.in_proj": "linear_attn.in_proj_qkvzba", "gdn.out_proj": "linear_attn.out_proj",
                 "attn.in_proj": "self_attn.in_proj_qgkvi", "attn.o_proj": "self_attn.o_proj",
                 "moe.sh_gate_up": "mlp.shared_expert.gate_up_proj", "moe.sh_down": "mlp.shared_expert.down_proj"}
        out = {}
        for key in keys:
            head, _, suffix = key.partition(".")
            if head == "mtp":
                layer, _, rest = suffix.partition(".")
                if rest in names:
                    out[key] = f"Qwen4ExpForCausalLM/mtp.layers.0.{names[rest]}"
            elif head.startswith("L") and suffix in names:
                out[key] = f"Qwen4ExpForCausalLM/model.language_model.layers.{head[1:]}.{names[suffix]}"
        return out

    def prepare_dense(self, store=None, *, consume_weights=False):
        """The dense lanes (engine/kernels/dense): W4A8 at decode rows, FP8 above; the shared expert's 160-column down
        projection through PaddedDenseLinear. No channel smoothing: its fold divides a plain norm weight, and every
        norm this model has is unit-offset. The hyper-connection mixers are BF16 matmuls inside their lane (10,240
        wide, W4 packs do not tile) unless `hc_fp8` put them on FP8 (`_prepare_hc_fp8`); the router stays a BF16 GEMM.

        `consume_weights` moves a lane's W4 and FP8 packs into its BF16 source's arena region and drops the source, where
        the packs fit it (engine/kernels/dense.packed_nbytes, the resident bound, against the source's bytes): every
        projection but the shared expert's down projection, whose 160 columns pack at 256 -- 1,034,496 bytes of packs
        against 819,200 of source -- until the preshard reserves its padded region."""
        from engine.kernels.dense import DenseLinear, FP8Linear, PaddedDenseLinear, packed_nbytes, padded_columns
        self.dense = {}
        self.retained_sources = []
        for key, name in self.dense_names(self.p).items():
            weight = self.p[key]
            aligned = weight.shape[1] % 128 == 0
            mtp = key.startswith("mtp.")
            if mtp and getattr(self, "mtp_precision", "bf16") == "bf16":
                continue                                             # self.linear: torch's BF16 matmul over p[key]
            precision = "fp8" if mtp and getattr(self, "mtp_precision", "bf16") == "fp8" else "w4"
            options = dict(decode_precision="fp8", fp8_decode_rows=True) if precision == "fp8" else {}
            lane = DenseLinear(weight, store=store, name=name, **options) if aligned else \
                PaddedDenseLinear(weight, prefill=True, store=store, name=name, smooth=None, **options)
            self.dense[key] = lane
            if consume_weights and hasattr(lane, "consume_weight"):
                cols = weight.shape[1] if aligned else padded_columns(weight.shape[1])
                if packed_nbytes(weight.shape[0], cols, decode_w4=precision == "w4") <= weight.numel() * weight.element_size():
                    lane.consume_weight(weight)
                    self.p[key] = None
                else:
                    self.retained_sources.append(key)
        head_fp8 = store.pack_fp8(self.p["head"], HEAD_NAME) if (store is not None and store.calibrated(HEAD_NAME)) else None
        # a decode step's head rows on one launch over deep_gemm's own FP8 inputs (dense/fp8_rows: 688-709 against
        # 873-893 us a read of the rank's 159 MB, q38head-0919c)
        self.dense["head"] = FP8Linear(self.p["head"], quantized=head_fp8, name=HEAD_NAME, decode_rows=True)
        self._hc_projections = self._prepare_hc_fp8() if self.hc_fp8 else {}

    def _hc_sites(self):
        """Every mixer the served step runs, as its weight-name prefix and down name: two a layer, the closing mixer,
        and the MTP head's two and its close."""
        sites = [(f"L{L}.hc.{side}.", "down_inject") for L in self.layers for side in ("attn", "mlp")]
        sites.append(("close.", "down"))
        if self.mtp:
            sites += [("mtp.L0.hc.attn.", "down_inject"), ("mtp.L0.hc.mlp.", "down_inject"), ("mtp.close.", "down")]
        return sites

    def _prepare_hc_fp8(self) -> dict:
        """prefix -> (down projection, up projection) on FP8Linear. The down projection reads the 10,240-wide streams
        (128-aligned); the up projection's input is the mixer rank (320), so its weight gains zero columns to 384 and
        the gates are padded with zeros to match -- a zero column adds nothing to a row."""
        from engine.kernels.dense import FP8Linear
        out = {}
        for prefix, down_name in self._hc_sites():
            down, up = self.p[prefix + down_name], self.p[prefix + "up"]
            pad = (-up.shape[1]) % 128
            up_lane = FP8Linear(torch.nn.functional.pad(up, (0, pad)), name=f"{HC_NAME}/{prefix}up")

            def project_up(gates, lane=up_lane, pad=pad):
                return lane(torch.nn.functional.pad(gates, (0, pad)) if pad else gates.contiguous())

            out[prefix] = (FP8Linear(down, name=f"{HC_NAME}/{prefix}{down_name}"), project_up)
        return out

    def _mix(self, prefix: str, normed, down_name: str, *, inject: bool):
        lanes, F, p = self.lanes, self.F, self.p
        proj = self._hc_projections.get(prefix)
        if proj is None:
            return lanes.hc_mix(normed, p[prefix + down_name], p[prefix + "up"], F.hc, inject=inject)
        return lanes.hc_mix(normed, p[prefix + down_name], p[prefix + "up"], F.hc, inject=inject,
                            project_down=proj[0], project_up=proj[1])

    def linear(self, x, name):
        """x @ W.T through the weight's dense lane, or -- a BF16 weight with no lane (the MTP head's at its default
        precision) -- the lanes' rows_linear: the skinny GEMV for a decode step's rows where it has a tile, else
        torch's matmul."""
        lane = self.dense.get(name)
        if lane is not None:
            return lane(x)
        return self._bf16(x, self.p[name])

    def _bf16(self, x, w):
        rows_linear = getattr(self.lanes, "rows_linear", None)
        if rows_linear is None:
            return torch.nn.functional.linear(x, w)
        return rows_linear(x.reshape(-1, x.shape[-1]), w).reshape(*x.shape[:-1], w.shape[0])

    # -- embed / head -----------------------------------------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        from engine.modules.token_embedding import lookup
        return self.comm.all_reduce(lookup(ids, self.p["embed"], self.rank * self.vp))

    def head_local(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h, "head")

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.comm.all_gather(self.head_local(h)[:, :self.vp], dim=-1)

    def head_tokens(self, h: torch.Tensor, decodable=None) -> torch.Tensor:
        from engine.modules.vocab import argmax
        return argmax(self.head_local(h)[:, :self.vp], self.comm, self.rank * self.vp, decodable)

    def prepare_draft_head(self, clusters: int, probes: int) -> dict:
        """After prepare_dense: the MTP head's argmax from an inverted-file index over this rank's head rows
        (dense/ivf_head) instead of the whole head -- a few MB a draft instead of 159. The verify step's head is
        untouched: a draft the index gets wrong is rejected, never emitted. -> the index's shape, for the boot's gauges."""
        from engine.kernels.dense import ivf_head
        self.draft_index = ivf_head.build(self.dense["head"].weight, clusters=clusters, probes=probes, rows=self.vp)
        return {"clusters": clusters, "probes": probes, "cap": self.draft_index.cap,
                "read_MB": round(self.draft_index.read_bytes() / 1e6, 2)}

    def draft_tokens(self, h: torch.Tensor) -> torch.Tensor:
        """The drafter's greedy picks: `head_tokens` over the whole head, or the index's argmax where one is prepared
        (the same key, all-reduced the same way)."""
        index = self.draft_index
        if index is None or not h.is_cuda or not 1 <= h.shape[0] <= 16:
            picks = self.head_tokens(h)
        else:
            from engine.kernels.dense import ivf_head
            key = ivf_head.argmax_key(index, h, self.rank * self.vp, self.vp)
            key = self.comm.all_reduce_max(key)
            picks = 0xffffffff - (key & 0xffffffff)
        tap = getattr(self, "draft_tap", None)
        if tap is not None:                             # kernels/common/row_tap: the draft queries and their picks
            tap(h, picks)
        return picks

    # -- the step's addressing -----------------------------------------------------------------------------------------
    def step_meta(self, step, caches) -> StepMeta:
        """The step's addressing. A host step's page table is cut to the bucket rung its longest segment reaches (the
        QSA scores are as wide as the table; `bucket_blocks`); a captured step's is its bucket's width, gathered on the
        device, with the unreserved entries (-1) read as page 0 -- they lie past every row's length, so no kernel reads
        them. The raw index-key ring takes only each segment's last QSA_KEY_RING positions: a longer prefill would write
        cells more than once in one launch, and which write lands is not defined."""
        F = self.F
        dev = step.ids.device
        from engine.profiles.qwen38.caches import QSA_KEY_RING
        if getattr(step, "captured", False) and step.ids.is_cuda:
            # one launch for what the composition below spells in about forty (engine/kernels/step_addresses); the
            # composition stays the CPU's form and the reference the kernel is held to
            from engine.kernels import step_addresses
            return StepMeta(*step_addresses.captured(step.contexts, step.slots, step.seqs, caches.block_table,
                                                     tokens=step.tokens, blocks=step.blocks, block=F.block,
                                                     ratio=F.idx_ratio, ring=QSA_KEY_RING))
        if getattr(step, "captured", False):
            n, t = step.rows, step.tokens
            positions = (step.contexts[:, None] + iota(t, dev)).reshape(-1)
            # one row reshapes its expand into a stride-0 view, and the QSA kernels load this at `ptr + row`
            rows_req = iota(n, dev, torch.int32)[:, None].expand(n, t).reshape(-1).contiguous()
            page_table = caches.block_table[:, :step.blocks].index_select(0, step.seqs).clamp_min_(0)
            starts = iota(n + 1, dev, torch.int32) * t
            slot_table = step.slots.to(torch.int32)[:, None]
            lengths = (step.contexts + t).to(torch.int32)
        else:
            n = len(step.segments)
            positions = step.positions
            counts = [s.length for s in step.segments]
            rows_req = torch.repeat_interleave(torch.arange(n, device=dev, dtype=torch.int32),
                                               torch.tensor(counts, device=dev))
            seqs = torch.tensor([s.seq for s in step.segments], device=dev, dtype=torch.long)
            blocks = bucket_blocks(F.block, -(-max(s.ctx + s.length for s in step.segments) // F.block),
                                   caches.block_table.shape[1])
            page_table = caches.block_table[:, :blocks].index_select(0, seqs)
            starts = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), device=dev, dtype=torch.int32)
            slot_table = torch.tensor([[s.slot] for s in step.segments], device=dev, dtype=torch.int32)
            lengths = torch.tensor([s.ctx + s.length for s in step.segments], device=dev, dtype=torch.int32)
        per_group = F.block // F.idx_ratio
        rr = rows_req.long()
        pages = page_table[rr, positions // F.block]
        kv_slots = (pages.long() * F.block + positions % F.block).to(torch.int32)
        closes = (positions + 1) % F.idx_ratio == 0
        group = positions // F.idx_ratio
        key_pages = page_table[rr, group // per_group]
        key_slots = torch.where(closes, key_pages.long() * per_group + group % per_group,
                                torch.full_like(positions, -1)).to(torch.int32)
        ring_slots = torch.where(positions >= lengths[rr] - QSA_KEY_RING,
                                 slot_table[rr, 0].long() * QSA_KEY_RING + positions % QSA_KEY_RING,
                                 torch.full_like(positions, -1)).to(torch.int32)
        return StepMeta(positions, positions.to(torch.int32), rows_req, page_table, lengths, starts, slot_table,
                        kv_slots, key_slots, ring_slots)

    # -- the forward ---------------------------------------------------------------------------------------------------
    def takes_mark(self, offset: int) -> bool:
        """Whether `forward` snapshots a prefix boundary `offset` tokens into a prefill segment on its way
        (Step.marks): `_gdn` is handed the state at the chunk kernel's own 64-token chunks, counted from the segment's
        start, and `_ple_inject` reads the conv's taps before the boundary out of the segment."""
        span = (self.F.ple_conv - 1) * self.F.ngram_size if any(L in self.F.ple_layers for L in self.layers) else 0
        return offset > 0 and offset % 64 == 0 and offset >= span

    def forward(self, step: Step, caches, *, last_hidden_only: bool = False, streams: bool = False):
        """One step -> the closing mixer's hidden [N, H] (or the segments' last rows with `last_hidden_only`), and with
        `streams` also the residual streams before the close for every row (what the MTP head fuses)."""
        F, p, lanes = self.F, self.p, self.lanes
        meta = self.step_meta(step, caches)
        rows = getattr(step, "captured", False)
        h = self.embed(step.ids).repeat(1, F.hc)
        out = inject = None
        for L in self.layers:
            n = f"L{L}."
            if L in F.ple_layers:
                if out is not None:
                    h = lanes.hc_leave(h, out, inject, F.hc)
                    out = None
                h = h + (self._ple_inject_rows(L, h, step, caches) if rows else self._ple_inject(L, h, step, meta, caches))
            x, inject, h = self._site(n + "hc.attn.", h, out, inject)
            if F.is_qsa(L):
                out = self._qsa(L, x, step, meta, caches)
            else:
                out = self._gdn_rows(L, x, step, caches) if rows else self._gdn(L, x, step, caches)
            x, inject, h = self._site(n + "hc.mlp.", h, out, inject)
            out = self._moe(n, x, compact=not rows)
        h, normed = lanes.hc_leave_norm(h, out, inject, p["close.norm"], F.rms_eps, F.hc)
        hidden, _ = self._mix("close.", normed, "down", inject=False)
        if last_hidden_only:
            if rows:
                raise ValueError("a captured step keeps every row; its caller selects them")
            last = torch.tensor([s.start + s.length - 1 for s in step.segments], device=hidden.device)
            hidden = hidden.index_select(0, last)
        return (hidden, h) if streams else hidden

    def _site(self, prefix, h, out, inject):
        """Enter a sublayer: the previous one's output leaves into the streams and they are normalised in one pass."""
        F, p, lanes = self.F, self.p, self.lanes
        if out is None:
            normed = lanes.hc_norm(h, p[prefix + "norm"], F.rms_eps, F.hc)
        else:
            h, normed = lanes.hc_leave_norm(h, out, inject, p[prefix + "norm"], F.rms_eps, F.hc)
        x, injection = self._mix(prefix, normed, "down_inject", inject=True)
        return x, injection, h

    # -- GatedDeltaNet -------------------------------------------------------------------------------------------------
    def _gdn(self, L: int, x: torch.Tensor, step: Step, caches) -> torch.Tensor:
        F, p, lanes, n = self.F, self.p, self.lanes, f"L{L}.gdn."
        N = x.shape[0]
        Hk, Hv, D = F.k_heads_local, F.v_heads_local, F.k_dim
        proj = self.linear(x, n + "in_proj")
        qkv, z, b, a = proj.split([F.qkv_local, Hv * D, Hv, Hv], dim=-1)
        wc, wr = self.conv_ring, self.rec_ring
        # the ring lane computes the decay and beta from a and b in its own launch; the chunk lane takes them computed
        decay, beta = ((None, None) if all(s.length <= wr for s in step.segments) else
                       lanes.gdn_gates(a, b, p[n + "A_log"], p[n + "dt_bias"], sigmoid_beta=True))
        core = torch.empty(N, Hv, D, dtype=x.dtype, device=x.device)
        for s in step.segments:
            sl = slice(s.start, s.start + s.length)
            conv, rec = caches.gdn(L, s.slot)
            if s.length <= wr:
                y = lanes.conv_ring(qkv[sl], p[n + "conv"], conv[None], 0, s.ctx)
                q, k, v = self._heads(y, s.length)
                o = lanes.gdn_ring(q, k, v, a[sl][None], b[sl][None], p[n + "A_log"], p[n + "dt_bias"], rec[None], 0,
                                   s.ctx)
            else:
                hist_pos = s.ctx + torch.arange(-(F.conv - 1), 0, device=x.device)
                hist = conv[:, hist_pos.clamp_min(0) % wc].masked_fill((hist_pos < 0)[None, :], 0) if s.ctx else None
                y, _ = lanes.conv_prefill(qkv[sl], p[n + "conv"], hist)
                keep = min(s.length, wc)
                pos = s.ctx + torch.arange(s.length - keep, s.length, device=x.device)
                conv[:, pos % wc] = qkv[sl][-keep:].T
                q, k, v = self._heads(y, s.length)
                state0 = rec[(s.ctx - 1) % wr][None] if s.ctx > 0 else None
                marks = [(m, snap) for m, snap in step.marks if 0 < m < s.length] if step.marks else []
                if marks:
                    if any(m % 64 for m, _ in marks):
                        raise ValueError("a mark inside a prefill chunk sits on a 64-token kernel chunk")
                    o, state, states = lanes.gdn_chunk(q, k, v, decay[sl][None], beta[sl][None], state0,
                                                       states_at=[m // 64 for m, _ in marks])
                    for (m, snap), st in zip(marks, states.unbind(0)):
                        caches.mark_gdn(L, snap, st, qkv[sl][m - (F.conv - 1):m])
                else:
                    o, state = lanes.gdn_chunk(q, k, v, decay[sl][None], beta[sl][None], state0)
                rec[(s.ctx + s.length - 1) % wr] = state[0]
            core[sl] = o[0]
        out = lanes.gdn_norm(core, z.view(N, Hv, D), p[n + "norm"], F.rms_eps)
        return self.comm.all_reduce(self.linear(out, n + "out_proj"))

    def _gdn_rows(self, L: int, x: torch.Tensor, step: DeviceStep, caches) -> torch.Tensor:
        """`_gdn` for a captured decode step: the conv and the recurrence over every row in one launch each, each row's
        program reading its own slot and context from the step's device vectors (the ring writes are those of the
        per-row launches; engine/kernels/causal_conv_ring, engine/kernels/kda/ring)."""
        F, p, lanes, n = self.F, self.p, self.lanes, f"L{L}.gdn."
        N = x.shape[0]
        Hv, D = F.v_heads_local, F.k_dim
        if step.tokens > min(self.rec_ring, self.conv_ring, 8):
            raise ValueError(f"a captured GDN step holds at most {min(self.rec_ring, self.conv_ring, 8)} tokens a row")
        proj = self.linear(x, n + "in_proj")
        qkv, z, b, a = proj.split([F.qkv_local, Hv * D, Hv, Hv], dim=-1)
        conv, rec = caches.gdn_fields(L)
        y = lanes.conv_ring_rows(qkv, p[n + "conv"], conv, step.slots, step.contexts)
        q, k, v = self._heads(y, N)
        # the decay and beta come from a and b, read through their strides, in the delta rule's own launch
        o = lanes.gdn_ring_rows(q, k, v, a[None], b[None], p[n + "A_log"], p[n + "dt_bias"], rec, step.slots,
                                step.contexts)
        out = lanes.gdn_norm(o[0], z.view(N, Hv, D), p[n + "norm"], F.rms_eps)
        return self.comm.all_reduce(self.linear(out, n + "out_proj"))

    def _heads(self, y, t):
        F = self.F
        qk = F.k_heads_local * F.k_dim
        q, k, v = y.split([qk, qk, F.v_heads_local * F.v_dim], dim=-1)
        return (q.reshape(1, t, F.k_heads_local, F.k_dim), k.reshape(1, t, F.k_heads_local, F.k_dim),
                v.reshape(1, t, F.v_heads_local, F.v_dim))

    # -- gated GQA with QSA selection ----------------------------------------------------------------------------------
    @staticmethod
    def _covers(F, step) -> bool:
        """Whether the QSA budget covers a host step: its longest segment ends inside (index_blocks + 1) * ratio - 1
        positions -- 2,051 at the model's widths -- so every row attends every position up to its own. Host
        arithmetic. A captured step is never covered: its contexts are the device's, and its launches are one sequence
        for every context."""
        if getattr(step, "captured", False):
            return False
        return max(s.ctx + s.length for s in step.segments) // F.idx_ratio <= F.index_blocks

    def _covered_blocks(self, step, meta: StepMeta):
        """Every row's chosen blocks when no score can decide them (carry Q11, the covered half; GLM's covered queries,
        engine/modules/prefill_indexer): a row that sees no more complete groups than the budget holds attends all of
        them, whatever they score, so a host step whose longest segment ends inside the budget's reach -- 2,051
        positions at the model's widths -- needs neither the scores nor the top-k of any QSA layer. The ids are the
        positions' alone: built by the step's first QSA layer, read by the rest and by the MTP head's. They come
        ascending with -1 after, the order the block attention sorts any selection into before it reads it (carry Q6),
        so its bytes are the scored selection's. None where a score decides something (`_covers`). The served lanes
        do not come here any more: a covered step's attention reads no ids at all (carry Q10, `_qsa`); this is what a
        lane table without that launch still attends."""
        F = self.F
        if not Qwen38Net._covers(F, step):
            return None
        if meta.covered_blocks is None:
            from engine.modules.prefill_indexer import covered_pool_ids
            meta.covered_blocks = covered_pool_ids((meta.positions32 + 1) // F.idx_ratio, F.index_blocks)
        return meta.covered_blocks

    @staticmethod
    def _score_runs(step) -> int:
        """The rows of one request the index scoring may take a program (carry Q8; engine/kernels/qsa
        `_qsa_mqa_paged_group_kernel`, 1..4): a run reads each key tile once for all its rows, and the logits are the
        row launch's bytes. A captured step lays `tokens` rows a sequence back to back, so a run is the largest divisor
        of that within four -- a K=1 verify step's two rows, K=3's four; a host step of one segment is one request
        throughout; several segments are rows of several requests, and a run may not straddle two."""
        if getattr(step, "captured", False):
            return max(g for g in (1, 2, 3, 4) if step.tokens % g == 0)
        return 4 if len(step.segments) == 1 else 1

    def _sharded_blocks(self, iq: torch.Tensor, step, meta: StepMeta, keys: torch.Tensor):
        """A prefill step's chosen blocks with each rank scoring its quarter of the rows (carry Q11, the rank half;
        GLM's #881, engine/modules/prefill_indexer.QueryShard): the index queries and the index keys are replicated,
        so four ranks scored the same [rows, columns] logits and took the same top-k four times over. A rank scores the
        rows it owns that a score decides -- its covered rows are their positions' ids, as `_covered_blocks`' are -- and
        one all-gather of the ids (two uint16 a word where the groups fit) makes every rank's selection whole. The ids
        are the unsplit step's, row for row, where the selection lane says its split calls select alike
        (`lanes.qsa_select_alike`: every chunk on both sides takes the radix select); the ranks ask it of the same
        numbers, so they split together or not at all. None where the step is not split: the boot did not declare
        `query_shards` (or a boot declined it), a captured step, several segments, fewer rows than ranks, or a lane that
        will not say."""
        if not getattr(self, "query_shards", False) or getattr(step, "captured", False) or len(step.segments) != 1:
            return None
        F, lanes, N = self.F, self.lanes, iq.shape[0]
        alike = getattr(lanes, "qsa_select_alike", None)
        if alike is None or N < TP:
            return None
        from engine.modules.prefill_indexer import QueryShard
        ctx = int(step.segments[0].ctx)
        shards = [QueryShard(N, ctx, rank, TP, F.index_blocks, F.idx_ratio) for rank in range(TP)]
        runs = self._score_runs(step)
        if not alike(N, [shard.score_rows for shard in shards], meta.page_table.shape[1] * keys.shape[1],
                     F.index_blocks, runs):
            return None
        if meta.groups_seen is None:
            meta.groups_seen = (meta.positions32 + 1) // F.idx_ratio
        mine = shards[self.rank]
        scored = None
        if mine.score_rows:
            rows = slice(mine.score_begin, mine.end)
            scored = lanes.qsa_select(iq[rows], keys, meta.page_table, meta.rows_req[rows], meta.positions32[rows],
                                      meta.lengths, F.idx_budget, F.idx_ratio, group=runs)
        return mine.collect(scored, meta.groups_seen, self.comm)

    def _qsa(self, L: int, x: torch.Tensor, step: Step, meta: StepMeta, caches, *, prefix=None, cache_layer=None):
        F, p, lanes = self.F, self.p, self.lanes
        n = prefix or f"L{L}.attn."
        cache_layer = L if cache_layer is None else cache_layer
        N = x.shape[0]
        Hq, D, Hkv = F.heads_local, F.head_dim, F.kv_heads_local
        idx_q = F.idx_heads * F.idx_dim
        proj = self.linear(x, n + "in_proj")
        qg, k, v, idx = proj.split([Hq * 2 * D, Hkv * D, Hkv * D, idx_q + F.idx_dim], dim=-1)
        qg = qg.view(N, Hq, 2 * D)
        gate = qg[..., D:]
        if Hkv != 1:
            raise ValueError("the QSA stores write one KV head a rank (2 KV heads replicated over TP=4)")
        K, V = caches.kv(cache_layer)
        ring = caches.key_ring(cache_layer)
        ik = idx[:, idx_q:]
        # the indexer's keys first: each group this step closes pooled from the raw-key ring (members before the step)
        # and this step's rows, normalised and rotated at its first position and stored -- one launch, which must read
        # the ring before this step's raw keys overwrite it
        lanes.qsa_index_keys(ik, ring, meta.slot_table, meta.rows_req, meta.starts, meta.positions, meta.key_slots,
                             F.idx_ratio, p[n + "idx_k_norm"], F.rms_eps, F.rope_theta, F.rotary_dim,
                             caches.index_keys(cache_layer))
        # then one launch for the rest, all read through their strides: the query and index query heads normalised and
        # rotated at their positions, the key head the same straight into K, the value rows into V and the raw keys
        # into the ring by position
        q, iq = lanes.qsa_inputs(qg[..., :D], k.view(N, Hkv, D), v.view(N, Hkv, D),
                                 idx[:, :idx_q].view(N, F.idx_heads, F.idx_dim), ik, meta.positions, p[n + "q_norm"],
                                 p[n + "k_norm"], p[n + "idx_q_norm"], F.rms_eps, F.rope_theta, F.rotary_dim, K, V,
                                 meta.kv_slots, ring, meta.ring_slots)
        attend_covered = getattr(lanes, "qsa_attend_covered", None)
        if attend_covered is not None and Qwen38Net._covers(F, step):
            # a step the budget covers chooses nothing: one dense causal launch, a run of rows sharing each K/V tile,
            # the sparse launch's bytes (carry Q10)
            attended = attend_covered(q, K, V, meta.positions32, meta.lengths, F.idx_ratio, F.idx_budget,
                                      meta.page_table, meta.rows_req, gate=gate, group=self._score_runs(step))
        else:
            # the chosen blocks, expanded to positions inside the attention's own tiles (no expanded buffer); a lane
            # table without the covered launch attends a covered step's unscored ids
            blocks = self._covered_blocks(step, meta)
            if blocks is None:
                blocks = self._sharded_blocks(iq, step, meta, caches.index_keys(cache_layer))
            if blocks is None:
                blocks = lanes.qsa_select(iq, caches.index_keys(cache_layer), meta.page_table, meta.rows_req,
                                          meta.positions32, meta.lengths, F.idx_budget, F.idx_ratio,
                                          group=self._score_runs(step))
            # the output gate in the attention's final store: BF16(attention * sigmoid(gate)) with no fp32 temporaries
            attended = lanes.qsa_attend(q, K, V, blocks, meta.positions32, meta.lengths, F.idx_ratio,
                                        F.idx_budget, meta.page_table, meta.rows_req, gate=gate)
        out = attended.reshape(N, Hq * D)
        return self.comm.all_reduce(self.linear(out, n + "o_proj"))

    # -- MoE ----------------------------------------------------------------------------------------------------------
    def _moe(self, prefix: str, x: torch.Tensor, *, compact: bool) -> torch.Tensor:
        """`compact`: an eager step, whose experts see only this rank's routed pairs (the lane reads their count on the
        host); a captured step keeps every route, another rank's on local expert 0 at weight 0 (lanes.local_routes)."""
        F, p, lanes = self.F, self.p, self.lanes
        n = prefix + "moe."
        gates = p[n + "gates"]                                       # [experts + 1, H]: the router, then the shared gate
        scores = lanes.rows_linear(x, gates) if lanes.rows_linear is not None else torch.mm(x, gates.t())
        if compact or lanes.route_local is None:
            ids, weights = lanes.route(scores[:, :F.experts], F.topk_experts)
            routed = self._experts[prefix](x, ids, weights, compact=compact)
        else:
            # a captured step's router and EP remap are one launch over the scores row (kernels/moe_route)
            w13 = p[n + "fp8.w13"] if n + "fp8.w13" in p else p[n + "w13"]      # the route's shape: E, I
            ids, weights = lanes.route_local(scores, F.topk_experts, experts=F.experts, first_expert=self.first_expert,
                                             w13=w13, hidden=x.shape[1])
            routed = self._experts[prefix](x, ids, weights, compact=False, local=True)
        # the down projection's 160 columns pad to 256 (PaddedDenseLinear): the activation's launch writes the zeros
        down = getattr(self, "dense", {}).get(n + "sh_down")
        pad_to = down.input_cols + down.pad if getattr(down, "pad", 0) else None
        shared = self.linear(lanes.swiglu(self.linear(x, n + "sh_gate_up"), pad_to=pad_to), n + "sh_down")
        # torch's sigmoid, not the router launch's: the gate is consumed in FP32 and Triton's exp is not torch's
        gate = torch.sigmoid(scores[:, F.experts:].float())
        return self.comm.all_reduce(lanes.moe_finish(routed, shared, gate))

    def _moe_fp8(self, x, ids, weights, *, w13, s13, w2, s2, compact=False, local=False):
        """The MTP head's experts on their FP8 side-file weights (lanes.moe_fp8): global routes (an eager step) are
        made this rank's first -- another rank's at weight 0 -- and more rows than the kernel takes go 16 at a time
        (rows are independent; the MTP head runs past its attention only the rows a caller reads)."""
        from engine.profiles.qwen38.lanes import local_routes
        if not local:
            ids, weights = local_routes(ids, weights, self.first_expert, w13.shape[0])
        run = self.lanes.moe_fp8
        if x.shape[0] <= 16:
            return run(x, ids, weights, w13, s13, w2, s2)
        return torch.cat([run(x[a:a + 16], ids[a:a + 16], weights[a:a + 16], w13, s13, w2, s2)
                          for a in range(0, x.shape[0], 16)])

    # -- PLE -----------------------------------------------------------------------------------------------------------
    def _ple_feature(self, L: int):
        """engine/modules/ngram_embedding.NGramInjection over the served weights: its hash, gate and conv are the
        reference's torch forms (integer and elementwise work on a handful of rows); the table gather is served."""
        from engine.modules.ngram_embedding import VARIANTS, NGramHash, NGramInjection
        F, p = self.F, self.p
        names = {"kv": f"L{L}.ple.kv_proj", "k_norm": f"L{L}.ple.norm_key",
                 "q_norm": f"L{L}.ple.norm_query", "conv_norm": f"L{L}.ple.norm_conv", "conv": f"L{L}.ple.conv"}

        def weights(layer, name):
            return p[names[name]]

        def hashed(layer):
            return NGramHash.splitmix(ngram_size=F.ngram_size, heads=F.heads_per_ngram, unigram_vocab=F.vocab,
                                      base=F.ngram_base, table_index=F.ple_layers.index(layer), seed=F.seed, eos=F.eos)

        feature = NGramInjection(hidden=F.hidden, hc=F.hc, ngram_size=F.ngram_size, conv=F.ple_conv, eps=F.rms_eps,
                                 **VARIANTS["ple"], hash=hashed, weights=weights, table=None, dtype="bfloat16")
        made = feature.hashes(L)
        if not (torch.equal(made.multipliers.cpu(), p[f"L{L}.ple.layer_multipliers"].cpu())
                and torch.equal(made.offsets.cpu(), p[f"L{L}.ple.heads_offsets"].cpu())
                and torch.equal(made.sizes.cpu(), p[f"L{L}.ple.heads_vocab"].cpu())):
            raise ValueError("the PLE hash the engine derives differs from the checkpoint's buffers (D3)")
        return feature

    def attach_ple(self, table, *, max_rows: int) -> None:
        """The rank's PLE table on the SSD (ple_table.PLETable) and the staging rows a captured step reads: `max_rows`
        is the widest captured step (rows x tokens). Before the graphs are captured, after the rank file is bound."""
        F = self.F
        if self._ple is None:
            raise ValueError("this net serves no PLE layer to attach a table to")
        if (table.rows, table.width) != (F.ple_rows_per_rank, F.ple_head_dim):
            raise ValueError(f"the table holds {table.rows} rows of {table.width}, the profile expects "
                             f"{F.ple_rows_per_rank} of {F.ple_head_dim}")
        scale = float(self._ple_scale)
        if abs(table.scale - scale) > 1e-6 * max(abs(scale), 1e-30):
            raise ValueError(f"the table's scale {table.scale} differs from the rank file's {scale}")
        self.ple_table = table
        self.ple_stage = PLEStaging(max_rows, F.ple_heads, F.ple_head_dim, self._ple_scale.device)

    def _ple_values(self, raw: torch.Tensor) -> torch.Tensor:
        """Gathered rows [N, heads, width] uint8 (e4m3 bytes; zero for rows of other ranks) -> embeddings
        [N, heads * width] BF16: the scalar scale in fp32, rounded once."""
        return (raw.view(torch.float8_e4m3fn).float() * self._ple_scale).to(BF16).flatten(-2)

    def _ple_embed(self, rows: torch.Tensor) -> torch.Tensor:
        """Table rows [N, heads] int64 on the host -> embeddings [N, heads * width] BF16 on the device: this rank's
        rows read off its SSD table, other ranks' rows zero -- the caller's all-reduce sums the one rank that holds
        each row."""
        F = self.F
        if self.ple_table is None:
            raise RuntimeError("the PLE table is not attached (fleet.build attaches it after the rank file is bound)")
        local, mine = local_rows(rows.numpy(), self.rank, F.ple_rows_per_rank)
        raw = np.zeros((rows.shape[0], F.ple_heads, F.ple_head_dim), dtype=np.uint8)
        if mine.any():
            raw[mine] = self.ple_table.gather(local[mine])
        return self._ple_values(torch.from_numpy(raw).to(self._ple_scale.device))

    def stage_ple(self, slots, contexts, ids, t: int, caches, carried=None) -> None:
        """Before a captured step replays: the PLE rows its n x t tokens read, gathered into the staging buffer --
        each row's carried ids (DEAD before the sequence), the hash on the host, this rank's rows read by id, other
        ranks' rows zero (the graph's all-reduce sums them). `ids` are the step's n * t tokens. `carried` [n][ngram_size
        - 1]: each row's tokens before its context, from a host that holds them (adapter.ServedModel: the ids ring
        holds what was fed at those positions, which is the row's own history); without it they are read off the
        slots' rings, a launch sequence and a device read a step."""
        from engine.modules.ngram_embedding import DEAD
        F = self.F
        n = len(slots)
        context = F.ngram_size - 1
        if carried is None:
            ids_ring, _ = caches.ple_fields()
            r_ids, dev = ids_ring.shape[1], ids_ring.device
            slot = torch.tensor(list(slots), dtype=torch.int64, device=dev)[:, None]
            ctx = torch.tensor(list(contexts), dtype=torch.int64, device=dev)[:, None]
            prev = ctx - context + iota(context, dev)                    # [n, context]
            carried = torch.where(prev < 0, torch.full_like(prev, DEAD), ids_ring[slot, prev.clamp_min(0) % r_ids]).cpu()
        else:
            carried = torch.tensor(carried, dtype=torch.int64).view(n, context)
        history = torch.cat([carried, torch.tensor(list(ids), dtype=torch.int64).view(n, t)], dim=1)
        rows = self._ple_hash.rows_batched(history, t).reshape(n * t, -1).numpy()
        local, mine = local_rows(rows, self.rank, F.ple_rows_per_rank)
        self.ple_stage.fill(self.ple_table, local, mine)

    def _ple_inject(self, L: int, h: torch.Tensor, step: Step, meta: StepMeta, caches) -> torch.Tensor:
        from engine.modules.causal_conv import causal_conv1d
        from engine.modules.ngram_embedding import DEAD
        F, p, feature = self.F, self.p, self._ple
        made = self._ple_hash
        context = F.ngram_size - 1
        span = (F.ple_conv - 1) * F.ngram_size
        out = torch.empty_like(h)
        w = lambda name: feature.weights(L, name)
        for s in step.segments:
            sl = slice(s.start, s.start + s.length)
            ids_ring, conv_ring = caches.ple(s.slot)
            r_ids, r_conv = ids_ring.shape[0], conv_ring.shape[1]
            ids = step.ids[sl]
            prev = torch.arange(s.ctx - context, s.ctx, device=ids.device)
            carried = torch.where(prev < 0, torch.full_like(prev, DEAD), ids_ring[prev.clamp_min(0) % r_ids])
            history = torch.cat([carried, ids])
            rows = made.rows(history.cpu(), ids.numel())                 # hashed on the host: the table is read there
            embeddings = self.comm.all_reduce(self._ple_embed(rows))
            gated = feature._gated(h[sl], embeddings, w).flatten(-2)
            normed = feature._norm(gated, w("conv_norm"))
            taps = torch.arange(s.ctx - span, s.ctx, device=ids.device)
            held = torch.where((taps < 0)[None, :], torch.zeros((), dtype=conv_ring.dtype, device=ids.device),
                               conv_ring[:, taps.clamp_min(0) % r_conv])
            local, _ = causal_conv1d(normed, w("conv"), None, held if s.ctx else None, "silu", dilation=F.ngram_size)
            out[sl] = gated + local
            keep = min(s.length, r_conv)
            written = s.ctx + torch.arange(s.length - keep, s.length, device=ids.device)
            conv_ring[:, written % r_conv] = normed[-keep:].T.to(conv_ring.dtype)
            keep_ids = min(s.length, r_ids)
            written_ids = s.ctx + torch.arange(s.length - keep_ids, s.length, device=ids.device)
            ids_ring[written_ids % r_ids] = ids[-keep_ids:]
            for m, snap in step.marks:
                if 0 < m < s.length:
                    if m < span:
                        raise ValueError("a PLE mark sits at a block boundary, past the conv's span")
                    caches.mark_ple(snap, history[m:m + context], normed[m - span:m])
        return out

    def _ple_inject_rows(self, L: int, h: torch.Tensor, step: DeviceStep, caches) -> torch.Tensor:
        """`_ple_inject` for a captured decode step, every row at once: each row's carried ids and conv taps gathered
        from its slot's rings at its own positions (DEAD and zero before the sequence), the n-gram rows hashed
        together, the table gathered by address, the gate and norm over all rows, the dilated conv over each row's
        taps and new inputs (engine/modules/causal_conv.causal_conv1d_rows), and the rings written by position."""
        from engine.modules.causal_conv import causal_conv1d_rows
        F, feature = self.F, self._ple
        n, t, dev = step.rows, step.tokens, h.device
        ids_ring, conv_ring = caches.ple_fields()                       # [slots, R_ids] i64, [slots, C, R_conv] bf16
        r_ids, r_conv = ids_ring.shape[1], conv_ring.shape[2]
        context, span, width = F.ngram_size - 1, (F.ple_conv - 1) * F.ngram_size, F.hc * F.hidden
        if context + t > r_ids or span + t > r_conv:
            raise ValueError("a captured PLE step writes more tokens than its rings keep beside their history")
        if self.ple_stage is None:
            raise RuntimeError("a captured PLE step needs the staged table rows: attach_ple before capture")
        slot = step.slots[:, None]
        ctx = step.contexts[:, None]
        ids = step.ids.view(n, t)
        # the rows were gathered off the SSD table before the replay (stage_ple) into the graph's static staging rows
        embeddings = self.comm.all_reduce(self._ple_values(self.ple_stage.device[:n * t]))
        w = lambda name: feature.weights(L, name)
        gated = feature._gated(h, embeddings, w).flatten(-2)             # [n*t, hc*H]
        normed = feature._norm(gated, w("conv_norm"))
        taps = ctx - span + iota(span, dev)                              # [n, span]
        held = conv_ring[slot, :, taps.clamp_min(0) % r_conv]            # [n, span, C]
        held = torch.where((taps < 0)[:, :, None], torch.zeros((), dtype=held.dtype, device=dev), held)
        local = causal_conv1d_rows(normed.view(n, t, width), w("conv"), held.transpose(1, 2), "silu",
                                   dilation=F.ngram_size)
        cells = ctx + iota(t, dev)
        conv_ring[slot, :, cells % r_conv] = normed.view(n, t, width).to(conv_ring.dtype)
        ids_ring[slot, cells % r_ids] = ids
        return gated + local.reshape(n * t, width)

    # -- the MTP head --------------------------------------------------------------------------------------------------
    def mtp_forward(self, step: Step, given: torch.Tensor, caches, *, last_hidden_only: bool = True, rows=None):
        """The MTP head over a step whose tokens are the target's next tokens and `given` the target's streams at the
        positions before them [N, hc*H]: fuse, one QSA + MoE layer (model layer F.layers, its rows in the target's
        blocks), the head's closing mixer -> (hidden, its streams for chaining), for the rows a caller reads: `rows`
        [R] (device int64 indices), each segment's last with `last_hidden_only`, else every row.

        The attention runs over every row -- it stores their keys and values, which the next steps attend -- and all
        that follows it is row by row, so only the rows read go on: a draft observation's MoE runs its rows' last
        position, not the K+1 it observed, and a prompt's the segments' last rows, not the prompt."""
        F, p, lanes = self.F, self.p, self.lanes
        meta = self.step_meta(step, caches)
        e = lanes.hc_norm(self.embed(step.ids), p["mtp.pre_fc_norm_embedding"], F.rms_eps, 1)
        e = self._bf16(e, p["mtp.fc_embedding"])
        g = lanes.hc_norm(given, p["mtp.pre_fc_norm_hidden"], F.rms_eps, 1).view(-1, F.hc, F.hidden)
        h = (self._bf16(g, p["mtp.fc_hidden"]) + e[:, None, :]).reshape(-1, F.hc * F.hidden)
        x, inject, h = self._site("mtp.L0.hc.attn.", h, None, None)
        out = self._qsa(F.layers, x, step, meta, caches, prefix="mtp.L0.attn.", cache_layer=F.layers)
        if last_hidden_only and rows is None:
            rows = torch.tensor([s.start + s.length - 1 for s in step.segments], device=out.device)
        if rows is not None:
            h, out, inject = h.index_select(0, rows), out.index_select(0, rows), inject.index_select(0, rows)
        x, inject, h = self._site("mtp.L0.hc.mlp.", h, out, inject)
        out = self._moe("mtp.L0.", x, compact=not getattr(step, "captured", False))
        streams, normed = lanes.hc_leave_norm(h, out, inject, p["mtp.close.norm"], F.rms_eps, F.hc)
        hidden, _ = self._mix("mtp.close.", normed, "down", inject=False)
        return hidden, streams

__all__ = ["Segment", "Step", "StepMeta", "Qwen38Net", "HEAD_NAME", "MTP_PRECISIONS"]
