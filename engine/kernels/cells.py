"""What each lane is compiled for and where it was measured, in one torch-free place -- and what to
do when a shape misses either: the recipes.

A kernel wrapper refuses a bound kernel shape it cannot serve (engine/base/kernel_shape).
The numbers it refuses against live HERE and the wrappers import them, so the shape wizard
can judge a checkpoint before any boot -- `admission(shape)` -- and can never disagree
with the wrappers. A compiled cell changes in the kernel and in this file together (a third
HIDDEN instance of the mHC segment, a wider Hadamard); the tests pin the wrappers to these
names.

A cell is geometry AND math: two models can share 16 x 512 and still differ in the softmax (a sink term),
the key compression, or the hyper-connection form. The shape declares those operation variants
(engine/base/kernel_shape) and a lane is admitted only when they match what it computes.

Three verdicts. `admitted`: inside the compiled cell and inside a measured one. `unmeasured`:
the wrapper serves it, but the lane's measured dispatch choices (split points, tiles, the BF16/FP8
switch) were taken at another cell -- it runs by declaration. `refused`: the wrapper dies by name.
The measured cells below are the widths a measurement record exists for; a width joins its tuple in
the change that lands the record it cites.

A verdict that is not `admitted` carries a `Recipe`: the kind of work, where it lands, the options
cheapest first, what judges it, what "done" is, and the cost class. `plan()` orders those verdicts
cheapest first, refusals before measurements at equal cost -- the work table an agent starts from
for a new model. The recipes name probes and oracles; nothing here claims a number without a
measurement record (D4, D17).

Every layer also names what serves it (`Serve`), fastest first: the lane's own kernel (specialized), a
shape-generic fast kernel that computes the same math (generic, judged or not), or nothing fast (none).
The engine/modules oracles judge both; they are never a serving candidate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

# ---- the compiled cells: a wrapper refuses a shape outside them -------------------------------------------------------
# mla/glm53_megakernel.cu: MLA_H query heads per rank over an MLA_D latent (kv_lora_rank)
MLA_HEADS = 16
MLA_LATENT = 512
MLA_SINK = False              # its online softmax has no sink term (modules/sparse_attention.mla_sparse_mqa)
# kpool.py: the Hadamard-128 rotation and the one-warp pooling lane
INDEXER_HEAD_DIM = 128
INDEXER_KEY_COMPRESS = "kpool"  # per-channel softmax pooling with APE, Hadamard-128, FP8 keys (modules/sparse_indexer)
# dense/kernels.cu: the HIDDEN and HIDDEN_V41 instances of the mHC segment at HC; MHC_MAX_TOK_DEF rows; HCHUNK
MHC_HIDDEN = (4096, 5120)
MHC_HC = 4
MHC_VARIANT = "mhc"           # the MK segment and the TileLang mixes compute GLM-5.3's mhc_pre/mhc_post
MHC_MAX_TOK = 128
MHC_HCHUNK = 256
# oneshot/dsv4_oneshot_ar.cu: NPEER 3 (four ranks), MAXEL elements, the 12-CTA PDL consumer's element bound
ONESHOT_WORLD = 4
ONESHOT_MAX_ELEMENTS = 64 * 4096
ONESHOT_CONSUMER_MAX_ELEMENTS = 8 * 4096
# prefill_collectives/kernels.py: packets are whole blocks of this many elements
PREFILL_BLOCK = 2048
# kda/ring.py and the chunk lane in kda/kda.py: the fused KDA gate is one value per key channel
FUSED_GATE_DECAY = "channel"
# dense/__init__.py: W4 tiles pack K in 128-aligned columns
DENSE_ALIGN = 128

# ---- the measured cells: where each lane's dispatch choices were taken ------------------------------------------------
# one-shot's split points and PDL consumer bound: timed on GLM-5.3's hidden-4096 rows
ONESHOT_MEASURED_HIDDEN = (4096,)
# the BF16/FP8 switch at FP8_MIN_ROWS: probes/engine_performance_comm_check.py exercises hidden 4096 only
PREFILL_MEASURED_HIDDEN = (4096,)
# DenseLinear's "<=32 rows W4A8, above that FP8": measured on the shapes GLM-5.3's rank serves
DENSE_MEASURED_HIDDEN = (4096,)
# H5120 is compiled; its GPU probe has not run (measurements/dsv41_mhc_20260910: "GPU execution remains pending")
MHC_MEASURED_HIDDEN = (4096,)
# (heads, v_heads, k_dim, v_dim): the BV=16 recurrent tile and the long-prefill regime (measurements/st_gb10_kda_state_20260911)
KDA_MEASURED_CELLS = ((16, 16, 128, 128),)
# the DFlash kernels, timed on GLM-5.3-Flash-DFlash2
DRAFT_MEASURED_HEAD = (128,)

ADMITTED, REFUSED, UNMEASURED = "admitted", "refused", "unmeasured"
_KDA_CELLS_TEXT = ", ".join(f"{h}/{hv} x {k} x {v}" for h, hv, k, v in KDA_MEASURED_CELLS)
COSTS = ("minutes", "hours", "days")          # the cost classes, cheapest first
KINDS = ("establish", "measure", "instance", "kernel", "wire", "convert", "rewrite")


@dataclass(frozen=True)
class Recipe:
    """What to do about a verdict, as data an agent can act on."""
    kind: str           # establish | measure | instance | kernel | wire | convert | rewrite
    where: str          # the files, constants and hooks the work lands in
    how: str            # the options, cheapest first
    judge: str          # the probe / oracle / tolerance that decides
    done: str           # the completion criterion
    cost: str           # minutes | hours | days

    def __post_init__(self):
        if self.kind not in KINDS or self.cost not in COSTS or not all((self.where, self.how, self.judge, self.done)):
            raise ValueError(f"a recipe names a kind in {KINDS}, a cost in {COSTS}, and where/how/judge/done: {self}")


SPECIALIZED, GENERIC, NONE = "specialized", "generic", "none"
TIERS = (SPECIALIZED, GENERIC, NONE)


@dataclass(frozen=True)
class Serve:
    """What runs a layer for this shape, fastest first: the lane's own kernel (specialized), a shape-generic fast kernel
    that computes the same math (generic), or nothing fast (none). The engine/modules oracles judge; they never serve.
    `judged` says whether that kernel has been judged against the oracle for this math; `note` says what the judgment
    covered, or what is missing, and where a kernel that lives outside engine/ has to be ported from."""
    tier: str
    kernel: str
    judged: bool
    note: str

    def __post_init__(self):
        if self.tier not in TIERS or type(self.judged) is not bool:
            raise ValueError(f"a serve names a tier in {TIERS} and whether it is judged: {self}")
        if (self.tier == NONE) != (self.kernel == ""):
            raise ValueError(f"only 'none' serves without a kernel, and it names none: {self}")
        if self.tier == NONE and self.judged:
            raise ValueError("nothing cannot be judged")
        if "engine/modules" in self.kernel or self.kernel.startswith("modules/"):
            raise ValueError(f"an oracle judges; it never serves: {self.kernel}")


@dataclass(frozen=True)
class Verdict:
    lane: str
    status: str
    why: str
    recipe: "Recipe | None" = None
    serve: "Serve | None" = None


def from_dict(d: dict) -> Verdict:
    """A verdict back from its record form (`asdict`)."""
    recipe, serve = d.get("recipe"), d.get("serve")
    return Verdict(d["lane"], d["status"], d["why"], Recipe(**recipe) if recipe else None, Serve(**serve) if serve else None)


def to_dicts(verdicts: "list[Verdict]") -> "list[dict]":
    return [asdict(v) for v in verdicts]


# ---- the recipes ---------------------------------------------------------------------------------------------------

_GPU = "a GPU ticket (bench/fleet.sh run --gpu)"
_FLEET = "a fleet ticket (bench/fleet.sh run --gpu --fleet: four ranks)"
_MLA_JUDGE = "probes/mla_check.py vs modules/sparse_attention.mla_sparse_mqa (rel <= 2e-2, the self-test mla.maybe_arm runs)"
_MOE_JUDGE = ("probes/engine_kernel_check.py --lanes moe vs modules/moe.expert_gemm (rel 2%), then the profile's "
              "quality gate (D4)")
_KDA_JUDGE = ("probes/linear_attention_check.py vs modules/linear_attention (max |o-HF| 9.8e-4, |state-HF| 2.3e-3 "
              "at bf16, T=96; chunked == recurrent)")


def _recipe_device(d):
    return Recipe("rewrite", "every native build (engine/kernels/mla, engine/kernels/dense, engine/kernels/oneshot: sm_121a) "
                  "and the CuTe/Triton caches",
                  f"the lanes are built for GB10 sm_121a with {d.sms} SMs and D5 forbids 'other GPUs': another card is its "
                  "own build, its own measurements and its own cells",
                  "every lane's self-test on that card", "not on this fleet", "days")


def _recipe_mla(a):
    if a.kind != "mla":
        return Recipe("kernel", "a new attention lane beside engine/kernels/mla (flashinfer.decode is in the image, "
                      "engine/INVENTORY.md, not judged)",
                      f"{a.kind} attention has no ST lane and engine/modules has no reference for it: write the torch "
                      "reference first, then (a) judge flashinfer's decode against it as the lane, or (b) write the kernel; "
                      "either way a new cell in cells.py and a wrapper that refuses the rest",
                      "the new torch reference, then the profile's quality gate (D4)",
                      "the lane's arm-time self-test passes against the reference and cells.py names the cell", "days")
    if a.head_dim == MLA_LATENT and a.heads % MLA_HEADS == 0:
        return Recipe("wire", "engine/kernels/mla/__init__.py (_check_cell) and engine/profiles/<profile>/lanes.py",
                      f"{a.heads} heads per rank are {a.heads // MLA_HEADS} groups of {MLA_HEADS}: MQA heads are independent, "
                      f"so the lane calls mla_decode per group of {MLA_HEADS} (engine/profiles/glm53/lanes.py already does "
                      "this at world 1) and _check_cell admits multiples of MLA_HEADS",
                      _MLA_JUDGE, "the self-test passes and the wizard admits the shape", "hours")
    return Recipe("instance", "engine/kernels/mla/glm53_megakernel.cu (MLA_H, MLA_D) and cells.MLA_HEADS/MLA_LATENT",
                  f"the kernel is compiled for {MLA_HEADS} x {MLA_LATENT}; asked {a.heads} x {a.head_dim}. MLA_D enters the "
                  "pitches and lane splits (MLA_VD = MLA_D/32, MLA_CP = MLA_D+8, MLA_RP = MLA_D+16) whose shared-memory "
                  "occupancy was measured at 512, so a new latent is a re-derived, re-measured instance; a head count that "
                  "is not a multiple of 16 can instead pad zero-query heads at the lane and drop their outputs",
                  _MLA_JUDGE, "the self-test passes on the new cell and cells.py names it", "days")


def _recipe_establish(fact, where, judge):
    return Recipe("establish", where,
                  f"the model's {fact} is not established: read it off the pinned reference (engine/profiles/qwen38/plan.py "
                  "HF_PIN for Qwen3.8), state it in the profile's kernel_shape, and rerun the wizard -- the lane can be "
                  "chosen only after the math is known",
                  judge, f"the profile declares the {fact} and the wizard judges the lane on it", "hours")


def _recipe_mla_sink():
    return Recipe("kernel", "engine/kernels/mla/glm53_megakernel.cu (the MLA segment's online softmax, the cluster and "
                  "prefill32 paths) and cells.MLA_SINK",
                  "add the per-head sink to the softmax denominator: exp(sink - running max) joins the partial sums in "
                  "the split summation order the self-test holds; the dense decode rows and the prefill32 tiles need it too",
                  "modules/sparse_attention.sparse_attn (the -1 sentinel, the -1e30 seed) at rel <= 2e-2 in the arm-time self-test",
                  "a sink cell in cells.py and the self-test passes with sinks", "days")


def _recipe_indexer_compress(i):
    if i.compress == "qsa":
        how = ("the engine has no QSA module yet (engine/profiles/qwen38/budget.py): write its compression reference "
               "first, then a key-compression lane against it; the MQA logits formula is the shared part")
    else:
        how = (f"{i.compress} compresses keys its own way (engine/profiles/dsv41/shapes.py: Compressor.kv_state, the packed "
               "E2M1 score path): write that compression and score path as a lane; the MQA logits formula is the shared part")
    return Recipe("kernel", "a key-compression lane beside engine/kernels/kpool.py (kpool is GLM-5.3's) and "
                  "cells.INDEXER_KEY_COMPRESS", how,
                  "the model's compression reference, then modules/sparse_indexer.indexer_logits for the shared scoring",
                  "cells.py names the compression and the indexer wrapper admits it", "days")


def _recipe_mhc_variant(shape, lane):
    if lane == "mhc_decode":
        return Recipe("wire", "engine/kernels/dense/kernels.cu (run_mhc_v41) and engine/kernels/dense/mhc.py",
                      "the megakernel's V4.1 contract (run_mhc_v41, 'Experimental HF V4.1 MHC seam') is the candidate for the "
                      "split-sinkhorn form and its GPU probe never ran (measurements/dsv41_mhc_20260910): judge it against "
                      f"hc_split_sinkhorn on {_GPU}, then give engine/kernels/dense/mhc.py an entry for it",
                      "modules/hyper_connection.hc_split_sinkhorn, pooled and worst-token rel <= 1e-3 (probes/mk_mhc_geometry_bench.py)",
                      "cells.py admits split_sinkhorn for decode and the wrapper binds the V4.1 entry", "hours")
    return Recipe("kernel", "engine/kernels/mhc/__init__.py (mhc_pre_tilelang, mhc_post_tilelang)",
                  "the TileLang mixes compute GLM-5.3's mhc_pre/mhc_post; the split-sinkhorn form (sigmoid gates, its RMS "
                  "normalisation placement and pre-mix epsilon) needs its own mixes",
                  "modules/hyper_connection.hc_split_sinkhorn", "the prefill lane binds the new mixes and cells.py names the variant", "days")


def _recipe_indexer(i):
    return Recipe("kernel", "engine/kernels/kpool.py (Hadamard-128; cells.INDEXER_HEAD_DIM)",
                  f"a parametric Hadamard-D: log2(D) butterfly stages instead of the fixed seven; the one-warp lane needs "
                  f"D/32 channels per thread; the reference modules/sparse_indexer.fwht128_quant widens the same way. "
                  f"Asked D={i.head_dim}",
                  "tests/test_engine_kpool_compress.py and probes/engine_indexer_quant_check.py vs modules/sparse_indexer "
                  "(byte-identical keys and scales)",
                  "cells.INDEXER_HEAD_DIM admits D and the wrappers pass it", "days")


def _recipe_mhc(shape):
    if shape.hc != MHC_HC:
        return Recipe("rewrite", "engine/kernels/dense/kernels.cu (HC, NOUT = HC*(2+HC), the pmix strides)",
                      f"HC {MHC_HC} is a compile-time constant across the whole mHC segment; hc {shape.hc} is a segment "
                      "rewrite, or the shape-generic TileLang mixes for decode as well, at their own measured cost",
                      "tests/test_engine_mk_mhc.py vs modules/hyper_connection.mhc_pre/mhc_post (rel < 0.006, captured replay)",
                      "the segment serves the new hc and the D17 probe boots", "days")
    if shape.hidden % MHC_HCHUNK:
        return Recipe("rewrite", "engine/kernels/dense/kernels.cu (HCHUNK, NCHUNK, MHC_EPT)",
                      f"hidden {shape.hidden} is not a multiple of {MHC_HCHUNK}: NCHUNK = hidden/{MHC_HCHUNK} and MHC_EPT = "
                      "hidden/256 threads would not be integral, so the segment would need a tail block",
                      "tests/test_engine_mk_mhc.py vs modules/hyper_connection.mhc_pre/mhc_post (rel < 0.006, captured replay)",
                      "the segment serves the width and the D17 probe boots", "days")
    return Recipe("instance", "engine/kernels/dense/kernels.cu (HIDDEN, HIDDEN_V41, mk_mhc_launch<HID>, the TORCH_CHECK on "
                  "hidden) and cells.MHC_HIDDEN",
                  f"add an instance for hidden {shape.hidden}: NCHUNK = {shape.hidden // MHC_HCHUNK} and MHC_EPT = "
                  f"{shape.hidden // 256} are integral (5120 was added this way, PR #518); engine/kernels/dense/mhc.py then "
                  "admits it through cells.MHC_HIDDEN",
                  f"tests/test_engine_mk_mhc.py vs modules/hyper_connection.mhc_pre/mhc_post (rel < 0.006, captured replay) and "
                  f"probes/mk_mhc_geometry_bench.py on {_GPU}",
                  "cells.MHC_HIDDEN lists the width and the D17 probe boots", "hours")


def _recipe_mhc_measure(shape):
    return Recipe("measure", f"probes/mk_mhc_geometry_bench.py on {_GPU}; cells.MHC_MEASURED_HIDDEN",
                  f"the H{shape.hidden} instance is compiled but its GPU probe never ran (measurements/dsv41_mhc_20260910: "
                  "H4096/H5120 x T1..128 against PyTorch references, input immutability, graph replay): run it and record the receipt",
                  "the probe's own gates: pooled and worst-token rel <= 1e-3, exact same-input replay",
                  f"the receipt lands in the measurement record and cells.MHC_MEASURED_HIDDEN lists {shape.hidden}", "hours")


def _recipe_oneshot(c):
    return Recipe("rewrite", "engine/kernels/oneshot/dsv4_oneshot_ar.cu (NPEER, RING, MAXEL) and the verbs transport",
                  f"NPEER {ONESHOT_WORLD - 1} and RING {ONESHOT_WORLD} are the transport; world {c.world} is a transport "
                  "rewrite, and a row width that is not a multiple of 8 needs padding at the caller",
                  "the boot self-tests (sum == NCCL, rank-ordered cancellation, captured replay)",
                  "the self-tests pass on the new world", "days")


def _recipe_oneshot_measure(c):
    return Recipe("measure", f"probes/engine_performance_comm_check.py on {_FLEET}; cells.ONESHOT_MEASURED_HIDDEN",
                  f"probes/engine_performance_comm_check.py builds 4096-wide rows: take the width from kernel_shape.bound().comm.hidden, bind this shape, "
                  f"and run it at {c.hidden}; the boot self-test (sum == NCCL) stays the correctness gate",
                  "the probe's graph-replay assert_close against dist.all_reduce (rtol 0.02, atol 0.125) and the exact boot self-test",
                  f"cells.ONESHOT_MEASURED_HIDDEN lists {c.hidden} with the run's record", "hours")


def _recipe_prefill_measure(c):
    return Recipe("measure", f"probes/engine_performance_comm_check.py on {_FLEET}; cells.PREFILL_MEASURED_HIDDEN",
                  f"probes/engine_performance_comm_check.py also runs PrefillCollectives at 128..6912 rows, 4096 wide: parameterize the width with the "
                  f"one-shot change and run it at {c.hidden}; the BF16/FP8 switch (FP8_MIN_ROWS 4096 rows) has only run at "
                  "hidden 4096",
                  "the probe's all_gather (exact against the quantized reference) and reduce_scatter (rtol 0.008, atol 0.03125) checks",
                  f"cells.PREFILL_MEASURED_HIDDEN lists {c.hidden} with the run's record", "hours")


def _recipe_dense(shape, m):
    return Recipe("convert", "the preshard (engine/base/preshard.py) and the profile's call site; "
                  "engine/kernels/dense/__init__.py packed_nbytes refuses unaligned columns",
                  f"pad the columns to a multiple of {DENSE_ALIGN}: zero weight columns in the rank files and a zero-extended "
                  f"input at the call (rows are padded inside the pack already; columns are not). Asked hidden {shape.hidden}, "
                  f"dense intermediate {m.dense_inter_local}",
                  f"tests/test_engine_dense.py on {_GPU}", "DenseLinear binds the padded packs", "hours")


def _recipe_dense_measure(shape):
    return Recipe("measure", f"tests/test_engine_decode_seven.py and probes/engine_decode_fusions.py on {_GPU}; "
                  "cells.DENSE_MEASURED_HIDDEN",
                  f"tests/test_engine_decode_seven.py and probes/engine_decode_fusions.py exercise the W4 plans at hidden 4096 "
                  f"(rows 6416/4096/6144): rerun them at hidden {shape.hidden} and "
                  "this model's projection widths; DenseLinear's '<=32 rows W4A8, above that FP8' split was measured on the "
                  "shapes GLM-5.3's rank serves",
                  "the decode-seven numerical gates, then the fusion timings",
                  f"cells.DENSE_MEASURED_HIDDEN lists {shape.hidden} with the run's record", "hours")


def _recipe_draft_measure(d):
    return Recipe("measure", f"tests/test_engine_draft_attention.py on {_GPU}; cells.DRAFT_MEASURED_HEAD",
                  f"the DFlash fixtures are 128 wide: rerun them at head {d.head_dim}; attend_rows cuts its window by SMs and "
                  "KV heads, and its query tile (BQ 32) was chosen at 128",
                  "tests/test_engine_draft_attention.py (the sliced softmaxes combine into the whole one)",
                  f"cells.DRAFT_MEASURED_HEAD lists {d.head_dim}", "hours")


def _recipe_kda_measure(l):
    return Recipe("measure", "engine/kernels/kda/kda.py (fused_recurrent_kda_fwd) and engine/kernels/kda/ring.py (the BV=16 "
                  "rule); cells.KDA_MEASURED_CELLS",
                  f"the BV=16 tile is chosen only at H == HV == 16, K == V == 128, T <= 6 (measurements/st_gb10_kda_state_20260911); "
                  f"this cell ({l.heads}/{l.v_heads} x {l.k_dim} x {l.v_dim}) runs the conservative BV <= 8 tile. On {_GPU}, "
                  "sweep BV at this cell under the exact rollback gate and extend the rule where it wins",
                  "tests/test_engine_kda_state.py (oracle, replay, rejected drafts) and " + _KDA_JUDGE
                  + "; kda/ring.py records that BV=16 at seven tokens changed rollback results, so the gate is exact",
                  "the rule and cells.KDA_MEASURED_CELLS name the cell", "hours")


def _recipe_kda_ring():
    return Recipe("wire", "engine/profiles/<profile>/lanes.py",
                  "wire the recurrent lane instead: fused_recurrent_kda(compute_gate=False) on linear_decay.per_channel(decay), "
                  "ring writes with state.write_ring; kda_recurrent_ring and kda_recurrent_ring_rows stay None (the "
                  "functional lane and the row loop)",
                  _KDA_JUDGE, "the decode step replays byte-identically across rows (the tests/test_engine_kda_ring.py pattern)", "hours")


def _recipe_kda_chunk():
    return Recipe("kernel", "engine/kernels/kda/kda.py (chunk_kda_with_fused_gate) and engine/kernels/kda/chunk_delta_h.py",
                  "start from chunk_delta_h.chunk_gated_delta_rule_fwd_h, which already takes a decay tensor g: a chunk entry "
                  "that widens the per-head decay with linear_decay.per_channel and calls it without KDA's fused gate",
                  _KDA_JUDGE, "the prefill lane binds it and chunked == recurrent on the oracle", "days")


def _recipe_kda_chunk_measure(l):
    return Recipe("measure", "engine/kernels/kda/kda.py (_glm53_kda_prefill_regime_gate); cells.KDA_MEASURED_CELLS",
                  f"the exact-gated long packed prefill regime admits GLM-5.3's exact shape only (16 heads, 128, TP4, 8192 "
                  f"batched tokens); this cell ({l.heads}/{l.v_heads} x {l.k_dim} x {l.v_dim}) runs the stock regime. Measure "
                  f"the regime at this cell on {_GPU} before widening the gate",
                  _KDA_JUDGE, "the gate and cells.KDA_MEASURED_CELLS name the cell", "hours")


def _recipe_moe_quant(m, measured):
    return Recipe("convert", "engine/profiles/<profile>/preshard.py (the NVFP4 group-16 export) or engine/kernels/b12x "
                  "(a scale-layout cell)",
                  f"the served b12x cell reads {measured.quant} group-16 scales: (a) b12x's MXFP4 kernels already read FP4 in "
                  "groups of 32 with E8M0 scales but quantize activations to FP4 -- judge them where the model's weights have "
                  "that layout; (b) dequantize the experts and re-quantize to NVFP4 group-16 in the preshard, D5's base form, "
                  "and rerun the wizard; (c) a b12x cell with the model's own activation precision (CuTe DSL). The quality "
                  "gate decides between them",
                  _MOE_JUDGE, "the wizard says admitted or unmeasured for moe", "days")


def _recipe_moe_measure(m):
    return Recipe("measure", "engine/kernels/b12x/moe_dispatch.py (_DYNAMIC_TILE_M_OVERRIDE, the probe hook) and "
                  "engine/kernels/b12x/moe_reform_sf_pack.py (pack_stage_bytes)",
                  f"probes/engine_kernel_check.py --lanes moe takes --moe-experts 8|288 today: extend it to the bound cell. On "
                  f"{_GPU}, sweep tile_m 16/32/64/128 through the override at this model's rows per expert (top-{m.topk} x "
                  f"rows / {m.experts_local} local experts); run pack_stage_bytes over the expert scale stages (None: a byte "
                  "span above 64 -- SF6 refused, keep t,r); then `wizard --pin moe.dynamic_tile_m=<best> --write`",
                  _MOE_JUDGE + "; bench/onepass.py on the fleet for the speed claim (D17)",
                  "the record carries the pin and a D17 record names the cell", "hours")


# ---- the table -------------------------------------------------------------------------------------------------------

def _serve(tier, kernel, judged, note):
    return Serve(tier, kernel, judged, note)


def _nothing(note):
    return Serve(NONE, "", False, note)


def _serve_attention(a, i):
    """The fastest kernel for a full attention the MLA lane refuses."""
    if a.kind != "mla":
        if i is not None and i.compress == "qsa":
            return _serve(GENERIC, "qsa_sparse_paged_attention in overlay/modules/qwen38_qsa/ops_qsa.py (vLLM's Triton QSA "
                          "sparse paged GQA attention, the kernel that served Qwen3.8 in the vLLM stack; to port)", False,
                          "judge it against a torch GQA reference over the indexer's selected positions"
                          + ("; reading it also settles the unestablished sink" if a.sink is None else ""))
        if a.sink is None:
            return _nothing("the sink decides which kernel computes this attention; establish it first")
        if a.sink:
            return _nothing("no GQA kernel with sinks is named in the repo or in engine/INVENTORY.md")
        return _serve(GENERIC, "flashinfer BatchDecodeWithPagedKVCacheWrapper and BatchPrefillWithPagedKVCacheWrapper (in "
                      "the image, engine/INVENTORY.md)", False, "never judged in this engine")
    if a.sink is None:
        return _nothing("the sink decides which kernel computes this attention; establish it first")
    if a.sink:
        return _serve(GENERIC, "flashinfer trtllm_batch_decode_sparse_mla_dsv4, which takes sinks (the V4-Flash call in "
                      "overlay/modules/dsv4_flashinfer_sparse/flashinfer_sparse.py)", False,
                      "decode only; whether its trtllm-gen kernel runs on sm_121a is part of the judgment")
    if a.head_dim == MLA_LATENT and a.heads % MLA_HEADS == 0:
        return _serve(GENERIC, f"engine/kernels/mla called per group of {MLA_HEADS} heads", True,
                      "exact: MQA heads are independent, and engine/profiles/glm53/lanes.py already groups them at world 1")
    return _serve(GENERIC, "flashinfer BatchDecodeMlaWithPagedKVCacheWrapper with page-size-1 slot indices (in the image, "
                  "engine/INVENTORY.md)", False,
                  "the vLLM-era GLM lane served sparse MLA through the page-size-1 wrapper; not judged at this latent")


def _serve_indexer(i):
    """The fastest kernels for an indexer the kpool lane refuses."""
    if i.compress == "qsa":
        return _serve(GENERIC, "qsa_compress_groups_with_ratio, qsa_mqa_paged and qsa_select_paged_tokens in "
                      "overlay/modules/qwen38_qsa/ops_qsa.py (vLLM's Triton QSA ops, to port)", False,
                      "judge the compression against the model's reference and the scoring against "
                      "modules/sparse_indexer.indexer_logits")
    if i.compress == "ced":
        return _nothing("the CED compressor exists only in torch (overlay/modules/dsv41_model/dsv41_compressor.py); its "
                        "scoring and packed keys have Triton kernels beside it (dsv41_indexer_triton.py, "
                        "dsv41_packed_index_triton.py)")
    return _nothing(f"no Hadamard-{i.head_dim} kernel; the kpool rotation is fixed at {INDEXER_HEAD_DIM}")


def admission(shape) -> "list[Verdict]":
    """One verdict per lane for a kernel shape: admitted, unmeasured or refused (see the module docstring). Every verdict
    that is not admitted carries its recipe, and every layer names what serves it: the lane's own kernel, a shape-generic
    fast kernel for the same math, or nothing fast. The engine/modules oracles judge; they never serve."""
    from engine.base.kernel_shape import MEASURED
    a, l, i, m, c, d = shape.attention, shape.linear, shape.indexer, shape.moe, shape.comm, shape.device
    out = []

    def admit(lane, why, serve):
        out.append(Verdict(lane, ADMITTED, why, None, serve))

    def unmeasured(lane, why, recipe, serve):
        out.append(Verdict(lane, UNMEASURED, why, recipe, serve))

    def refuse(lane, why, recipe, serve):
        out.append(Verdict(lane, REFUSED, why, recipe, serve))

    if (tuple(d.capability), d.sms) == (MEASURED.device.capability, MEASURED.device.sms):
        admit("device", f"GB10 SM{d.capability[0]}{d.capability[1]}, {d.sms} SMs", None)
    else:
        refuse("device", f"every lane is built for GB10 sm_121a with {MEASURED.device.sms} SMs; asked "
                         f"SM{d.capability[0]}{d.capability[1]}/{d.sms}", _recipe_device(d), None)

    if a.kind != "mla" or (a.heads, a.head_dim) != (MLA_HEADS, MLA_LATENT):
        refuse("mla", f"compiled for the {MLA_HEADS} heads x {MLA_LATENT} latent MLA cell; asked {a.kind} {a.heads}x{a.head_dim}"
                      + (" -- a GQA attention has no ST lane yet" if a.kind != "mla" else ""), _recipe_mla(a),
               _serve_attention(a, i))
    elif a.sink is None:
        refuse("mla", "the attention sink is not established; the lane's math depends on it",
               _recipe_establish("attention sink", "the profile's kernel_shape (Attention.sink)",
                                 "modules/sparse_attention: sparse_attn carries the sink, mla_sparse_mqa does not"),
               _serve_attention(a, i))
    elif a.sink != MLA_SINK:
        refuse("mla", f"the {MLA_HEADS} x {MLA_LATENT} geometry matches, but this attention's softmax carries a sink term "
                      "and the compiled MLA softmax has none", _recipe_mla_sink(), _serve_attention(a, i))
    else:
        admit("mla", f"the compiled {MLA_HEADS} heads x {MLA_LATENT} latent cell, no sink",
              _serve(SPECIALIZED, "engine/kernels/mla (the megakernel's sparse MLA)", True,
                     "the arm-time self-test against modules/sparse_attention.mla_sparse_mqa (rel <= 2e-2)"))

    if i is not None:                                  # a model without a sparse indexer has no indexer lane to judge
        if i.compress != INDEXER_KEY_COMPRESS:
            refuse("indexer", f"the indexer lane compresses keys by {INDEXER_KEY_COMPRESS}; this model compresses by "
                              f"{i.compress} (only the MQA scoring formula is shared)", _recipe_indexer_compress(i),
                   _serve_indexer(i))
        elif i.head_dim == INDEXER_HEAD_DIM:
            admit("indexer", f"{INDEXER_KEY_COMPRESS} keys at Hadamard-{INDEXER_HEAD_DIM}, pool {i.pool}, top {i.topk} at launch",
                  _serve(SPECIALIZED, "engine/kernels/kpool.py, engine/kernels/indexer.py and DeepGEMM fp8_fp4_mqa_logits "
                         "(engine/kernels/deep_gemm.py)", True, "probes/indexer_check.py against modules/sparse_indexer"))
        else:
            refuse("indexer", f"the indexer lanes are written for head_dim {INDEXER_HEAD_DIM}; asked {i.head_dim}",
                   _recipe_indexer(i), _serve_indexer(i))

    mk = "engine/kernels/dense/mhc.py (the MK mHC segment, run_mhc)"
    tilelang = "engine/kernels/mhc (TileLang mhc_pre and mhc_post)"
    hc_unknown = _recipe_establish("hyper-connection form", "the profile's kernel_shape (hc_variant)",
                                   "modules/hyper_connection: mhc_pre/mhc_post or hc_split_sinkhorn reproduces the reference")
    hc_nothing = _nothing("the hyper-connection form decides which kernel mixes; establish it first")
    if shape.hc_variant is None:
        refuse("mhc_decode", "the hyper-connection form is not established; the segment's math depends on it", hc_unknown,
               hc_nothing)
    elif shape.hc_variant != MHC_VARIANT:
        refuse("mhc_decode", f"the MK mHC segment computes {MHC_VARIANT}; this model mixes by {shape.hc_variant}",
               _recipe_mhc_variant(shape, "mhc_decode"),
               _serve(GENERIC, "run_mhc_v41 in engine/kernels/dense/kernels.cu (the megakernel's V4.1 contract: it consumes "
                      "the previous sublayer's pre coefficients)", False,
                      "its GPU probe never ran (measurements/dsv41_mhc_20260910)")
               if shape.hidden in MHC_HIDDEN and shape.hc == MHC_HC
               else _nothing(f"the V4.1 contract is compiled for hidden {MHC_HIDDEN} at hc {MHC_HC} only"))
    elif shape.hidden not in MHC_HIDDEN or shape.hc != MHC_HC:
        refuse("mhc_decode", f"MK mHC is compiled for hidden {MHC_HIDDEN} at hc {MHC_HC}; asked hidden {shape.hidden} "
                             f"hc {shape.hc}", _recipe_mhc(shape),
               _serve(GENERIC, tilelang, True, "the same mhc math, judged at hidden 4096 (probes/mhc_check.py); "
                      "shape-generic in hidden and hc, its timing unmeasured here"))
    elif shape.hidden not in MHC_MEASURED_HIDDEN:
        unmeasured("mhc_decode", f"the MK mHC instance for hidden {shape.hidden} is compiled; its GPU probe has not run",
                   _recipe_mhc_measure(shape), _serve(SPECIALIZED, mk, False, f"the H{shape.hidden} instance's GPU probe has not run"))
    else:
        admit("mhc_decode", f"MK mHC instance for hidden {shape.hidden} at hc {shape.hc}",
              _serve(SPECIALIZED, mk, True, "tests/test_engine_mk_mhc.py against modules/hyper_connection (rel < 0.006)"))
    if shape.hc_variant is None:
        refuse("mhc_prefill", "the hyper-connection form is not established; the mixes' math depends on it", hc_unknown,
               hc_nothing)
    elif shape.hc_variant != MHC_VARIANT:
        refuse("mhc_prefill", f"the TileLang mixes compute {MHC_VARIANT}; this model mixes by {shape.hc_variant}",
               _recipe_mhc_variant(shape, "mhc_prefill"),
               _serve(GENERIC, f"run_mhc_v41 in {MHC_MAX_TOK}-token pieces (the mixing is per token)", False,
                      "no prefill kernel for the V4.1 pairing exists; the vLLM stack mixed it in torch "
                      "(overlay/modules/dsv41_vllm/dsv41_mhc.py)")
               if shape.hidden in MHC_HIDDEN and shape.hc == MHC_HC
               else _nothing(f"the V4.1 contract is compiled for hidden {MHC_HIDDEN} at hc {MHC_HC} only"))
    else:
        admit("mhc_prefill", "TileLang mixes take hidden and hc from the tensors",
              _serve(SPECIALIZED, tilelang, True, "probes/mhc_check.py against modules/hyper_connection"))

    oneshot = "engine/kernels/oneshot (the one-shot RDMA all-reduce)"
    if c.world != ONESHOT_WORLD or c.hidden % 8 or c.hidden > ONESHOT_MAX_ELEMENTS:
        refuse("oneshot", f"one-shot is compiled for {ONESHOT_WORLD} ranks and rows of 8-element multiples within "
                          f"{ONESHOT_MAX_ELEMENTS}; asked world {c.world}, hidden {c.hidden}", _recipe_oneshot(c),
               _serve(GENERIC, "NCCL all-reduce through torch.distributed (engine/base/comm.py)", True,
                      "the reference the one-shot boot self-test compares against"))
    else:
        rows = f"rows of {c.hidden} BF16, up to {ONESHOT_MAX_ELEMENTS // c.hidden} per collective"
        selftest = _serve(SPECIALIZED, oneshot, True, "the boot self-test at every boot: sum == NCCL, rank-ordered "
                          "cancellation, captured replay")
        if c.hidden in ONESHOT_MEASURED_HIDDEN:
            admit("oneshot", rows, selftest)
        else:
            unmeasured("oneshot", f"{rows}; timed at hidden {'/'.join(map(str, ONESHOT_MEASURED_HIDDEN))} only",
                       _recipe_oneshot_measure(c), selftest)

    blocks = PREFILL_BLOCK // _gcd(PREFILL_BLOCK, c.hidden)
    packets = (f"FP8 packets over {c.world} ranks; rows must complete {PREFILL_BLOCK}-element blocks "
               f"(every {blocks} row{'s' if blocks > 1 else ''})")
    collectives = "engine/kernels/prefill_collectives (FP8 packets)"
    if c.hidden in PREFILL_MEASURED_HIDDEN:
        admit("prefill_collectives", packets,
              _serve(SPECIALIZED, collectives, True, "probes/engine_performance_comm_check.py"))
    else:
        unmeasured("prefill_collectives", f"{packets}; exercised at hidden {'/'.join(map(str, PREFILL_MEASURED_HIDDEN))} only",
                   _recipe_prefill_measure(c), _serve(SPECIALIZED, collectives, False, "its checks build 4096-wide rows"))

    dense = "engine/kernels/dense (W4A8 decode, FP8 prefill)"
    if shape.hidden % DENSE_ALIGN or m.dense_inter_local % DENSE_ALIGN:
        refuse("dense", f"dense W4 tiles need {DENSE_ALIGN}-aligned columns; asked hidden {shape.hidden}, dense intermediate "
                        f"{m.dense_inter_local}", _recipe_dense(shape, m),
               _serve(GENERIC, "cuBLAS BF16 GEMM (torch.nn.functional.linear) on unpacked weights", True,
                      "exact BF16 arithmetic; no W4 compression, so the weights stay resident in BF16"))
    elif shape.hidden not in DENSE_MEASURED_HIDDEN:
        unmeasured("dense", f"W4A8 decode / FP8 prefill, K {DENSE_ALIGN}-aligned; its dispatch was measured at hidden "
                            f"{'/'.join(map(str, DENSE_MEASURED_HIDDEN))} only", _recipe_dense_measure(shape),
                   _serve(SPECIALIZED, dense, True, "packing and GEMM are judged across widths (tests/test_engine_dense.py); "
                          "the dispatch timing is unmeasured here"))
    else:
        admit("dense", f"W4A8 decode / FP8 prefill, K {DENSE_ALIGN}-aligned",
              _serve(SPECIALIZED, dense, True, "tests/test_engine_dense.py"))

    if shape.drafter is not None:
        draft = "engine/kernels/draft_attention.py and its siblings draft_conv, draft_select, draft_observe"
        if shape.drafter.head_dim in DRAFT_MEASURED_HEAD:
            admit("draft", f"DFlash kernels at head {shape.drafter.head_dim}",
                  _serve(SPECIALIZED, draft, True, "tests/test_engine_draft_attention.py"))
        else:
            unmeasured("draft", f"DFlash kernels take head {shape.drafter.head_dim} as a constexpr; timed at "
                                f"{'/'.join(map(str, DRAFT_MEASURED_HEAD))} only", _recipe_draft_measure(shape.drafter),
                       _serve(SPECIALIZED, draft, False, "the fixtures are 128 wide"))

    if l is not None:                                  # a model without linear attention has no KDA lane to judge
        measured_cell = (l.heads, l.v_heads, l.k_dim, l.v_dim) in KDA_MEASURED_CELLS
        per_head = l.decay != FUSED_GATE_DECAY
        recurrent = "fused_recurrent_kda over [B,T,HV,K] decays" + (
            "; the per-head decay is widened by linear_decay.per_channel (compute_gate=False)" if per_head else "")
        recurrent_serve = _serve(SPECIALIZED, "engine/kernels/kda (fused_recurrent_kda)", not per_head,
                                 "the precomputed-decay path is unjudged for a per-head decay" if per_head
                                 else "tests/test_engine_kda_state.py" + ("" if measured_cell else f", at {_KDA_CELLS_TEXT}"))
        if measured_cell and not per_head:
            admit("kda_recurrent", recurrent, recurrent_serve)
        else:
            unmeasured("kda_recurrent", f"{recurrent}; the BV=16 tile is measured at {_KDA_CELLS_TEXT} only",
                       _recipe_kda_measure(l), recurrent_serve)
        if per_head:
            refuse("kda_ring", "the ring lane fuses KDA's per-channel gate; a head-decay cell cannot run it", _recipe_kda_ring(),
                   _serve(GENERIC, "fused_recurrent_kda(compute_gate=False) on linear_decay.per_channel(decay), with "
                          "state.write_ring (engine/kernels/kda/kda.py, engine/kernels/state.py)", False,
                          "judge it against modules/linear_attention with a per-head decay"))
            refuse("kda_chunk", "the chunk lane fuses KDA's gate; a head-decay prefill needs a chunk entry without it (not written)",
                   _recipe_kda_chunk(),
                   _serve(GENERIC, "chunk_gated_delta_rule_fwd_h in engine/kernels/kda/chunk_delta_h.py (the vendored FLA "
                          "chunk kernel, which takes a decay tensor)", False,
                          "judge it against modules/linear_attention: chunked == recurrent"))
        elif measured_cell:
            admit("kda_ring", "the ring lane's fused per-channel KDA gate",
                  _serve(SPECIALIZED, "engine/kernels/kda/ring.py", True, "tests/test_engine_kda_ring.py"))
            admit("kda_chunk", "chunk_kda_with_fused_gate over the prefill",
                  _serve(SPECIALIZED, "engine/kernels/kda (chunk_kda_with_fused_gate)", True,
                         "judged against the reference in probes (engine/profiles/glm53/lanes.py: KDA chunk 6.3e-3)"))
        else:
            unmeasured("kda_ring", f"the fused per-channel gate serves this cell; the BV=16 tile is measured at "
                                   f"{_KDA_CELLS_TEXT} only", _recipe_kda_measure(l),
                       _serve(SPECIALIZED, "engine/kernels/kda/ring.py", True, f"tests/test_engine_kda_ring.py, at {_KDA_CELLS_TEXT}"))
            unmeasured("kda_chunk", "chunk_kda_with_fused_gate serves this cell in its stock regime; the long-prefill regime "
                                    "admits GLM-5.3's exact shape only", _recipe_kda_chunk_measure(l),
                       _serve(SPECIALIZED, "engine/kernels/kda (chunk_kda_with_fused_gate)", True,
                              "the stock regime, judged at GLM-5.3's cell"))

    measured = replace(MEASURED.moe, dynamic_tile_m=None)
    b12x = "engine/kernels/b12x (the NVFP4 dispatcher)"
    if m.quant != measured.quant:
        refuse("moe", f"the b12x lane is {measured.quant} only (D5); asked {m.quant}", _recipe_moe_quant(m, measured),
               _serve(GENERIC, "b12x's MXFP4 kernels (engine/kernels/b12x/moe_dispatch.py: FP4 in groups of 32 with E8M0 "
                      "scales)", False, "they read MXFP4-layout weights as they are but quantize activations to FP4; the "
                      "quality gate decides") if m.quant.startswith("mxfp4")
               else _nothing(f"no fast kernel reads {m.quant} experts"))
    elif replace(m, dynamic_tile_m=None) == measured:
        admit("moe", "the measured GB10 TP4 cell" + ("" if m.dynamic_tile_m is None else f", tile pinned at {m.dynamic_tile_m}"),
              _serve(SPECIALIZED, b12x, True, "probes/engine_kernel_check.py --lanes moe --moe-experts 288"))
    else:
        unmeasured("moe", f"admitted by declaration ({m.experts} experts, {m.experts_local} local, I{m.inter_local}, "
                          f"top{m.topk}, {m.activation}); its tiles and scale packing were measured at the GLM-5.3 cell",
                   _recipe_moe_measure(m),
                   _serve(SPECIALIZED, b12x, False, "generic NVFP4 shapes are checked at 2% against the original kernel; "
                          "this cell is not"))

    admit("universal", "sampler, block verify, decode commit, vocab candidates, SwiGLU, norm+RoPE, route histogram, build "
                       "cache, calibration: the arguments are the shape",
          _serve(GENERIC, "engine/kernels/common, bound as the engine's default lanes by engine/base/lanes.py", True,
                 "each kernel's tests against its torch form"))
    return out


def plan(verdicts: "list[Verdict]") -> "list[Verdict]":
    """The verdicts that ask for work: cheapest first, refusals before measurements at equal cost."""
    return sorted((v for v in verdicts if v.recipe is not None),
                  key=lambda v: (COSTS.index(v.recipe.cost), v.status != REFUSED, v.lane))


def counts(verdicts: "list[Verdict]") -> dict:
    return {s: sum(v.status == s for v in verdicts) for s in (ADMITTED, UNMEASURED, REFUSED)}


def serving(verdicts: "list[Verdict]") -> dict:
    """How many layers each tier serves, and how many of the generic ones are judged."""
    served = [v.serve for v in verdicts if v.serve is not None]
    out = {t: sum(x.tier == t for x in served) for t in TIERS}
    out["generic_judged"] = sum(x.tier == GENERIC and x.judged for x in served)
    return out


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def table(verdicts: "list[Verdict]") -> str:
    """One line per lane, then the counts."""
    width = max(len(v.lane) for v in verdicts)
    rows = []
    for v in verdicts:
        rows.append(f"  {v.lane:<{width}}  {v.status:<10}  {v.why}")
        if v.serve is not None:
            x = v.serve
            if x.tier == NONE:
                rows.append(f"  {'':<{width}}  {'serves':<10}  nothing fast -- {x.note}")
            else:
                rows.append(f"  {'':<{width}}  {'serves':<10}  {x.tier}, {'judged' if x.judged else 'unjudged'}: {x.kernel} -- {x.note}")
    n, sv = counts(verdicts), serving(verdicts)
    rows.append(f"  {n[ADMITTED]} admitted, {n[UNMEASURED]} unmeasured, {n[REFUSED]} refused")
    rows.append(f"  serving: {sv[SPECIALIZED]} specialized, {sv[GENERIC]} generic ({sv['generic_judged']} judged), "
                f"{sv[NONE]} with nothing fast")
    return "\n".join(rows)


def work_table(verdicts: "list[Verdict]") -> str:
    """The plan as numbered entries: what to do, where, what judges it, what done is -- in the order to do it."""
    work = plan(verdicts)
    if not work:
        return "  work: none -- every lane serves this shape as measured"
    rows = [f"  work ({len(work)}), cheapest first, refusals first at equal cost:"]
    for n, v in enumerate(work, 1):
        r = v.recipe
        rows += [f"  {n}. {v.lane} [{v.status}] {r.kind}, {r.cost}",
                 f"     how:   {r.how}",
                 f"     where: {r.where}",
                 f"     judge: {r.judge}",
                 f"     done:  {r.done}"]
    return "\n".join(rows)
