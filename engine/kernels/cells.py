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

Three verdicts. `admitted`: inside the compiled cell -- directly, or through an exact adapter (glue) that a GPU judged --
and inside a measured one (the operator's decision of 2026-09-17: a glue cell with a GPU judgment and a measurement
record is admitted; engine/QWEN38_CARRY.md Q3). `unmeasured`:
the wrapper serves it, but the lane's measured dispatch choices (split points, tiles, the BF16/FP8
switch) were taken at another cell -- it runs by declaration. `refused`: the wrapper dies by name.
The measured cells below are the widths a measurement record exists for; a width joins its tuple in
the change that lands the record it cites.

A verdict that is not `admitted` carries a `Recipe`: the kind of work, where it lands, the options
cheapest first, what judges it, what "done" is, and the cost class. `plan()` orders those verdicts
cheapest first, refusals before measurements at equal cost -- the work table an agent starts from
for a new model. The recipes name probes and oracles; nothing here claims a number without a
measurement record (D4, D17).

Every layer also names what serves it (`Serve`), fastest first: the lane's own kernel (specialized), the same
compiled kernel reached through an exact adapter that pads, groups, packs, widens or pieces the tensors (glue), a
shape-generic fast kernel that computes the same math (generic), or nothing fast (none) -- each judged or not. The
engine/modules oracles judge them; they are never a serving candidate. The glue rules -- when an adapter can put a
shape on a compiled kernel -- live here beside the cells, and the adapters refuse by them: a layer the table serves by
glue is one its adapter admits, and an operation variant not yet established serves nothing until it is.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from engine.kernels import arch

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
MHC_V41_VARIANT = "split_sinkhorn"   # run_mhc_v41, the megakernel's V4.1 seam, computes DeepSeek-V4.1's form
MHC_MAX_TOK = 128
MHC_HCHUNK = 256
# oneshot/dsv4_oneshot_ar.cu: NPEER 3 (four ranks), MAXEL elements, and the largest sum the PDL consumer serves --
# C=2's 16 verify rows. The engine build compiles no compact 12-CTA form: the consumer launches the ordinary kernel's
# 48-CTA grid, one ticket per CTA, with a stash that covers MAXEL, so this bound is a dispatch choice, not a compiled one.
ONESHOT_WORLD = 4
ONESHOT_MAX_ELEMENTS = 64 * 4096
ONESHOT_CONSUMER_MAX_ELEMENTS = 16 * 4096
# prefill_collectives/kernels.py: packets are whole blocks of this many elements
PREFILL_BLOCK = 2048
# kda/ring.py and the chunk lane in kda/kda.py: the fused KDA gate is one value per key channel
FUSED_GATE_DECAY = "channel"
# dense/__init__.py: W4 tiles pack K in 128-aligned columns, up to the decode kernel's widest K (kernels.cu KBLK_LIMIT)
DENSE_ALIGN = 128
DENSE_KMAX = 20480

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
# glue and second cells with a GPU judgment and a measurement record. Empty until the records land (the single-GPU lane
# tickets of engine/QWEN38_CARRY.md C2-C5); a cell joins its tuple in the change that lands the record it cites.
DENSE_GLUE_MEASURED_COLUMNS = ()    # unaligned column widths PaddedDenseLinear serves, judged and timed
KDA_DECAY_MEASURED_CELLS = ()       # (heads, v_heads, k_dim, v_dim): per-head decay cells on the decay entries
MHC_V41_MEASURED_HIDDEN = ()        # hidden widths of the split-sinkhorn form on MHCV41
MOE_MEASURED_CELLS = ()             # kernel_shape.MoE cells (dynamic_tile_m None) measured beyond MEASURED's

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


SPECIALIZED, GLUE, GENERIC, NONE = "specialized", "glue", "generic", "none"
TIERS = (SPECIALIZED, GLUE, GENERIC, NONE)


@dataclass(frozen=True)
class Serve:
    """What runs a layer for this shape, fastest first: the lane's own kernel (specialized); the same compiled kernel
    reached through an exact adapter that pads, groups, packs, widens or pieces the tensors (glue -- the glue rules
    below, the adapters in engine/kernels); a shape-generic fast kernel that computes the same math (generic); or
    nothing fast (none). The engine/modules oracles judge; they never serve. `judged` says whether that kernel, with
    its adapter, has been judged against the oracle for this math; `note` says what the judgment covered or what is
    missing, what an adapter changes besides the shape (the KV precision), and where a kernel that lives outside
    engine/ has to be ported from."""
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


# ---- the glue rules: when an exact adapter puts another cell's tensors on a compiled kernel --------------------------
# Each says why its adapter cannot serve a shape, or None when it can. The adapters refuse by the same functions, as the
# wrappers refuse by the cells above, so the table and the adapters cannot disagree.

def mla_glue_refusal(a) -> "str | None":
    """engine/kernels/mla/glue.py puts on the compiled MLA cell an MLA attention of any head count over a latent up to
    MLA_LATENT (zero-query heads fill a partial group of MLA_HEADS; zero coordinates widen the latent), or a GQA attention
    whose key and value fit side by side in the latent. The compiled softmax has no sink term."""
    if a.sink is None:
        return "the attention sink is not established"
    if a.sink != MLA_SINK:
        return "the compiled softmax has no sink term"
    if a.kind == "mla" and a.head_dim > MLA_LATENT:
        return f"a {a.head_dim} latent does not fit the compiled {MLA_LATENT}"
    if a.kind != "mla" and 2 * a.head_dim > MLA_LATENT:
        return f"a {a.head_dim}-wide key and value do not fit side by side in the {MLA_LATENT} latent"
    return None


def attention_terms_refusal(a, indexed: bool) -> "str | None":
    """Every attention lane here computes softmax(q.k) -- with or without a sink -- over the positions it is handed. An
    attention whose logits carry more (a learned relative-position bias, a soft cap) is other math; a sliding window is
    a set of positions, which only an indexer's selection hands a sparse lane. None when the lanes compute this one."""
    terms = []
    if a.relative:
        terms.append(f"a relative-position bias ({a.relative})")
    if a.softcap:
        terms.append(f"a {a.softcap:g} logit soft cap")
    if a.window and not indexed:
        terms.append(f"a {a.window}-position sliding window and no indexer to select it")
    if not terms:
        return None
    return "the attention lanes compute softmax(q.k) over the positions they are handed; this one also has " + \
        ", ".join(terms)


def mhc_v41_refusal(shape) -> "str | None":
    """engine/kernels/dense/mhc.MHCV41 mixes the split-sinkhorn form through the megakernel's V4.1 seam (run_mhc_v41),
    compiled at the mHC segment's widths and hc; its prefill runs the seam in MHC_MAX_TOK-token pieces."""
    if shape.hc_variant != MHC_V41_VARIANT:
        return f"the V4.1 seam mixes by {MHC_V41_VARIANT}; this shape mixes by {shape.hc_variant}"
    if shape.hidden not in MHC_HIDDEN or shape.hc != MHC_HC:
        return (f"the V4.1 seam is compiled for hidden {MHC_HIDDEN} at hc {MHC_HC}; asked hidden {shape.hidden} "
                f"hc {shape.hc}")
    return None


def dense_glue_refusal(cols: int) -> "str | None":
    """engine/kernels/dense.PaddedDenseLinear serves a projection whose input width is not DENSE_ALIGN-aligned with zero
    weight columns and a zero-extended input, while the padded width stays within DENSE_KMAX."""
    if type(cols) is not int or cols <= 0:
        return f"a projection has a positive input width, not {cols!r}"
    padded = -(-cols // DENSE_ALIGN) * DENSE_ALIGN
    if padded > DENSE_KMAX:
        return f"{cols} columns pad to {padded}, past the W4 decode kernel's widest K {DENSE_KMAX}"
    return None


# ---- the recipes ---------------------------------------------------------------------------------------------------

_GPU = "a GPU ticket (bench/fleet.sh run --gpu)"
_FLEET = "a fleet ticket (bench/fleet.sh run --gpu --fleet: four ranks)"
_MLA_JUDGE = "the glue test's mla_sparse_mqa twin vs modules/sparse_attention.mla_sparse_mqa (rel <= 2e-2, the self-test mla.maybe_arm runs; the fork-differential probe retired with the overlay stack)"
_MOE_JUDGE = ("probes/engine_kernel_check.py --lanes moe vs modules/moe.expert_gemm (rel 2%), then the profile's "
              "quality gate (D4)")
_KDA_JUDGE = ("probes/linear_attention_check.py vs modules/linear_attention (max |o-HF| 9.8e-4, |state-HF| 2.3e-3 "
              "at bf16, T=96; chunked == recurrent)")
_GLUE_TEST = "tests/test_engine_kernel_glue.py"     # each adapter against its oracle; the GPU cases run the armed kernel
_COMPOSITION_TEST = "tests/test_engine_composition.py"   # the features against transformers' model on the CPU


def _recipe_device(d):
    return Recipe("rewrite", "every native build (engine/kernels/mla, engine/kernels/dense, engine/kernels/oneshot: sm_121a) "
                  "and the CuTe/Triton caches",
                  f"the lanes are built for GB10 sm_121a with {d.sms} SMs and D5 forbids 'other GPUs': another card is its "
                  "own build, its own measurements and its own cells",
                  "every lane's self-test on that card", "not on this fleet", "days")


def _recipe_check_device(d):
    from engine.base.kernel_shape import MEASURED
    MEASURED_SMS = MEASURED.device.sms
    return Recipe("measure", "measurements/ for that card, and its own cells if it ever needs them",
                  f"the lanes build for {arch.name(d.capability)} and their self-tests judge them there; what is "
                  f"missing is a measurement record on this card, not a kernel. Its dispatch choices (split points, "
                  f"tiles, the BF16/FP8 switch) were all taken at GB10 sm_121a with {MEASURED_SMS} SMs, and this card "
                  f"has {d.sms}",
                  "each lane's own numerical self-test on that card, then a measurement record per D17",
                  "a record this repository cites, or the lane stays a check", "days")


def _recipe_mla(a):
    """The work for an attention the compiled MLA cell refuses by kind or geometry."""
    if a.sink is None:
        return _recipe_establish("attention sink", "the profile's kernel_shape (Attention.sink)",
                                 "modules/sparse_attention: sparse_attn carries the sink; mla_sparse_mqa and gqa_sparse do not")
    if mla_glue_refusal(a) is None:
        if a.kind == "mla":
            entry, oracle = "grouped", "modules/sparse_attention.mla_sparse_mqa"
            how = (f"engine/kernels/mla/glue.grouped runs the {a.heads} heads per rank in groups of {MLA_HEADS} -- zero-query "
                   "heads fill a partial group and their outputs are dropped; heads never interact"
                   + ("" if a.head_dim == MLA_LATENT else
                      f" -- over the {a.head_dim} latent zero-extended to {MLA_LATENT}: the cache rows once, written through "
                      f"glue.pad_rows ({MLA_LATENT}/{a.head_dim} of the latent's memory), the queries per call")
                   + ". Arm it with glue.arm() (the kernel's own self-test at its cell) and bind it as the profile's sparse "
                   "MLA lane; a kernel instance for this cell (engine/kernels/mla/glm53_megakernel.cu MLA_H, MLA_D; days) is "
                   "the option when the padded work or memory matters")
        else:
            entry, oracle = "gqa", "modules/sparse_attention.gqa_sparse"
            how = ("engine/kernels/mla/glue.gqa packs each KV head's key and value side by side into a latent row -- "
                   "[k * k_gain ; v * v_gain ; 0] through glue.pack_kv, written where the latent writer writes (row = "
                   f"position x {a.kv_heads} + KV head) -- and zero-extends the query to [q / k_gain ; 0], so the "
                   "megakernel's softmax(q'.c) c carries sum softmax(q.k) v in its value half. The query heads run in "
                   f"groups of {MLA_HEADS} per KV head, zero-query heads filling a partial group. The arithmetic is exact; "
                   "the KV cache becomes the latent's one-scale e4m3, so choose power-of-two gains that fill its range and "
                   "let the quality gate weigh it against a BF16-KV kernel. Arm it with glue.arm()")
        return Recipe("wire", f"engine/profiles/<profile>/lanes.py (bind engine/kernels/mla/glue.{entry})", how,
                      f"{_GLUE_TEST} on {_GPU}: the adapter over the armed kernel against {oracle} (rel <= 2e-2, the "
                      "lane's self-test band), then the profile's quality gate (D4)",
                      "the glue test passes on the GPU and the profile's lanes bind the adapter", "hours")
    if a.kind != "mla":
        return Recipe("kernel", "a new attention lane beside engine/kernels/mla (flashinfer.decode is in the image, "
                      "engine/INVENTORY.md, not judged)",
                      f"the glue cannot carry this {a.kind} attention ({mla_glue_refusal(a)}): modules/sparse_attention."
                      "gqa_sparse is its reference without a sink -- write the sink form beside it if the model has one -- "
                      "then (a) judge flashinfer's paged decode and prefill against it as the lane, or (b) write the kernel; "
                      "either way a new cell in cells.py and a wrapper that refuses the rest",
                      "the torch reference, then the profile's quality gate (D4)",
                      "the lane's arm-time self-test passes against the reference and cells.py names the cell", "days")
    if a.sink:
        return _recipe_mla_sink()
    return Recipe("instance", "engine/kernels/mla/glm53_megakernel.cu (MLA_H, MLA_D) and cells.MLA_HEADS/MLA_LATENT",
                  f"the kernel is compiled for {MLA_HEADS} x {MLA_LATENT}; asked {a.heads} x {a.head_dim}, wider than the "
                  "glue can pad. MLA_D enters the pitches and lane splits (MLA_VD = MLA_D/32, MLA_CP = MLA_D+8, MLA_RP = "
                  "MLA_D+16) whose shared-memory occupancy was measured at 512, so a new latent is a re-derived, re-measured "
                  "instance; the head count can still be grouped by engine/kernels/mla/glue.grouped",
                  _MLA_JUDGE, "the self-test passes on the new cell and cells.py names it", "days")


def _recipe_attention_terms(a):
    """The work for an attention whose logits or positions no lane computes (attention_terms_refusal)."""
    return Recipe("kernel", "a GQA lane with paged KV, window and sink beside engine/kernels/mla (engine/SM121_INTAKE.md "
                  "U7; flashinfer's paged decode takes window_left and logits_soft_cap, not a learned bias)",
                  "engine/modules/attention.feature is the reference for these terms (select=Window, relative=, "
                  "log_scaling=, kv_conv=) and is held to the model's transformers attention in "
                  "tests/test_engine_attention_family.py: judge a lane against it at the model's widths"
                  + ("; a learned relative bias is added per (row, head, distance) inside the softmax -- clamp the row "
                     "and the distance of that gather, a read past the table on GB10's unified memory returns another "
                     "allocation's bytes instead of faulting (vllm#49049)" if a.relative else ""),
                  "the family test's form of this model on the CPU, then the lane against it on a GPU "
                  f"({_GPU}), then the profile's quality gate (D4)",
                  "cells.py names the lane that computes these terms", "days")


def _recipe_kv_conv(a):
    return Recipe("wire", "engine/profiles/<profile>/lanes.py (engine/kernels/causal_conv over the keys and values, the "
                  "per-sequence taps in modules/state_rings)",
                  f"a {a.kv_conv}-tap causal conv on k and v before they are cached, with per-sequence state like the "
                  "linear attention's q/k/v conv; engine/modules/attention's kv_conv is the reference",
                  f"the conv against modules/attention.feature(kv_conv={a.kv_conv}) on the CPU and {_GPU}",
                  "the profile's lanes bind the conv and the family test holds it", "hours")


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
        how = ("engine/kernels/qsa ports vLLM's Triton QSA ops -- compression from the raw-key ring, paged scoring and "
               "selection with the engine's top-k -- and engine/profiles/qwen38/lanes.py binds them; the reference is "
               "engine/modules/sparse_indexer.qsa_select (held to transformers qwen4_exp); judge the ported lane against it")
    else:
        how = (f"{i.compress} compresses keys its own way (engine/profiles/dsv41/shapes.py: Compressor.kv_state, the packed "
               "E2M1 score path): write that compression and score path as a lane; the MQA logits formula is the shared part")
    return Recipe("kernel", "a key-compression lane beside engine/kernels/kpool.py (kpool is GLM-5.3's) and "
                  "cells.INDEXER_KEY_COMPRESS", how,
                  "the model's compression reference, then modules/sparse_indexer.indexer_logits for the shared scoring",
                  "cells.py names the compression and the indexer wrapper admits it", "days")


GATED_RESIDUAL_VARIANT = "gated_residual"   # Qwen3.8's form: its reference is engine/modules/hyper_connection.gated_residual


def _recipe_gated_residual(lane):
    """The work for the gated residual form: no compiled segment computes it, and every piece already has a fast kernel."""
    return Recipe("wire", "engine/profiles/<profile>/lanes.py (the residual form's enter/leave/close)",
                  "engine/kernels/gated_residual computes the form in five launches a site -- the previous leave joined to "
                  "the grouped unit-offset stream norm, the low-rank down and the injection in one BF16 GEMM (the "
                  "checkpoint keeps hyper_connection weights unquantised; 10,240-wide W4 packs do not tile), the gates, up, "
                  "the stream mean -- and engine/profiles/qwen38/lanes.py binds it for the "
                  + ("decode" if lane == "mhc_decode" else "prefill") + " step; the served net calls it",
                  f"{_COMPOSITION_TEST} (engine/modules/hyper_connection.gated_residual against transformers qwen4_exp on the "
                  f"CPU), then the lane against that reference on {_GPU}",
                  "the profile's lanes bind the composed form and it matches the reference", "hours")


def _recipe_mhc_variant(shape, lane):
    """The work for a hyper-connection form the MK segment and the TileLang mixes do not compute."""
    if shape.hc_variant == GATED_RESIDUAL_VARIANT:
        return _recipe_gated_residual(lane)
    if mhc_v41_refusal(shape) is not None:
        return _recipe_mhc(shape, seam="v41")             # the V4.1 seam lacks the width or hc, not the form
    seam = ("engine/kernels/dense/mhc.MHCV41 wraps the megakernel's V4.1 seam (run_mhc_v41): the previous sublayer's "
            "post and comb mixed into the residual, this sublayer's split-sinkhorn mixes projected from it, the layer "
            "input collapsed by the previous sublayer's pre, and this pre carried to the next call")
    judge = (f"{_GLUE_TEST} on {_GPU} against engine/kernels/dense/mhc_reference.py v41_component_reference (pooled and "
             "worst-token rel <= 1e-3), then modules/hyper_connection.hc_split_sinkhorn in the profile's check")
    if lane == "mhc_decode":
        return Recipe("wire", "engine/profiles/<profile>/lanes.py (bind engine/kernels/dense/mhc.MHCV41)",
                      f"{seam}. Its GPU probe never ran (measurements/dsv41_mhc_20260910): run the glue test on the GPU, "
                      "then bind MHCV41 for decode", judge,
                      "the GPU test passes and the profile's lanes bind MHCV41 for decode", "hours")
    return Recipe("wire", "engine/profiles/<profile>/lanes.py (bind engine/kernels/dense/mhc.MHCV41.prefill)",
                  f"{seam}. The seam mixes each token alone, so MHCV41.prefill runs it in {MHC_MAX_TOK}-token pieces, "
                  "exactly, one launch per piece; when those launches matter, split-sinkhorn TileLang mixes beside "
                  "engine/kernels/mhc/__init__.py mhc_pre_tilelang (days) are the prefill kernel", judge,
                  "the GPU test passes and the profile's lanes bind MHCV41.prefill", "hours")


def _recipe_indexer(i):
    return Recipe("kernel", "engine/kernels/kpool.py (Hadamard-128; cells.INDEXER_HEAD_DIM)",
                  f"a parametric Hadamard-D: log2(D) butterfly stages instead of the fixed seven; the one-warp lane needs "
                  f"D/32 channels per thread; the reference modules/sparse_indexer.fwht128_quant widens the same way. "
                  f"Asked D={i.head_dim}",
                  "tests/test_engine_kpool_compress.py and probes/engine_indexer_quant_check.py vs modules/sparse_indexer "
                  "(byte-identical keys and scales)",
                  "cells.INDEXER_HEAD_DIM admits D and the wrappers pass it", "days")


def _recipe_mhc(shape, seam="mhc"):
    """The work for a hyper-connection width or hc the compiled segment lacks: `seam` "mhc" is the MK segment (run_mhc,
    GLM-5.3's form), "v41" the megakernel's V4.1 seam (run_mhc_v41, the split-sinkhorn form the MHCV41 glue serves)."""
    v41 = seam == "v41"
    judge = (f"{_GLUE_TEST} on {_GPU} against engine/kernels/dense/mhc_reference.py v41_component_reference (pooled and "
             "worst-token rel <= 1e-3)" if v41 else
             "tests/test_engine_mk_mhc.py vs modules/hyper_connection.mhc_pre/mhc_post (rel < 0.006, captured replay)")
    served = "MHCV41 serves it" if v41 else "the D17 probe boots"
    if shape.hc != MHC_HC:
        return Recipe("rewrite", "engine/kernels/dense/kernels.cu (HC, NOUT = HC*(2+HC), the pmix strides)",
                      f"HC {MHC_HC} is a compile-time constant across the whole mHC segment, the V4.1 seam included; hc "
                      f"{shape.hc} is a segment rewrite" + ("" if v41 else ", or the shape-generic TileLang mixes for decode "
                                                               "as well, at their own measured cost"),
                      judge, f"the segment serves the new hc and {served}", "days")
    if shape.hidden % MHC_HCHUNK:
        return Recipe("rewrite", "engine/kernels/dense/kernels.cu (HCHUNK, NCHUNK, MHC_EPT)",
                      f"hidden {shape.hidden} is not a multiple of {MHC_HCHUNK}: NCHUNK = hidden/{MHC_HCHUNK} and MHC_EPT = "
                      "hidden/256 threads would not be integral, so the segment would need a tail block",
                      judge, f"the segment serves the width and {served}", "days")
    return Recipe("instance", "engine/kernels/dense/kernels.cu (HIDDEN, HIDDEN_V41, "
                  + ("mk_mhc_v41_launch<HID>, the hidden TORCH_CHECK in mk_run_mhc_v41" if v41 else
                     "mk_mhc_launch<HID>, the TORCH_CHECK on hidden") + ") and cells.MHC_HIDDEN",
                  f"add an instance for hidden {shape.hidden}: NCHUNK = {shape.hidden // MHC_HCHUNK} and MHC_EPT = "
                  f"{shape.hidden // 256} are integral (5120 was added this way, PR #518); "
                  + ("engine/kernels/dense/mhc.MHCV41" if v41 else "engine/kernels/dense/mhc.py") + " then admits it "
                  "through cells.MHC_HIDDEN",
                  judge + ("" if v41 else f" and engine/kernels/dense/mhc_reference.py on {_GPU}"),
                  f"cells.MHC_HIDDEN lists the width and {served}", "hours")


def _recipe_mhc_measure(shape):
    return Recipe("measure", f"engine/kernels/dense/mhc_reference.py on {_GPU}; cells.MHC_MEASURED_HIDDEN",
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
                  f"one-shot change and run it at {c.hidden}; the BF16/FP8 switch (FP8_MIN_ROWS 2048 rows) has only run at "
                  "hidden 4096",
                  "the probe's all_gather (exact against the quantized reference) and reduce_scatter (rtol 0.008, atol 0.03125) checks",
                  f"cells.PREFILL_MEASURED_HIDDEN lists {c.hidden} with the run's record", "hours")


def _dense_widths(shape, m) -> "tuple[list[tuple[str, int]], str]":
    """The input widths the dense lane packs -- the projections at hidden and, when the model has one, the dense or
    shared MLP's down projection -- and the phrase the table asks with."""
    widths = [("hidden", shape.hidden)] + ([("dense intermediate", m.dense_inter_local)] if m.dense_inter_local else [])
    asked = ", ".join(f"{name} {width}" for name, width in widths)
    return widths, asked + ("" if m.dense_inter_local else " (no dense or shared MLP: the projections alone)")


def _recipe_dense(shape, m):
    widths, asked = _dense_widths(shape, m)
    unaligned = [width for _, width in widths if width % DENSE_ALIGN]
    too_wide = [why for why in map(dense_glue_refusal, unaligned) if why]
    if not too_wide:
        return Recipe("wire", "engine/profiles/<profile>/lanes.py (bind engine/kernels/dense.PaddedDenseLinear where "
                      "DenseLinear refuses the columns)",
                      f"PaddedDenseLinear zero-extends the weight to a multiple of {DENSE_ALIGN} columns before packing and "
                      "the input at the call (rows are padded inside the pack already). Exact: a zero column adds nothing, "
                      "and neither the W4 row shift, the real columns' group scales nor the amax activation scales see it. "
                      f"Asked {asked}",
                      f"{_GLUE_TEST} and tests/test_engine_dense.py on {_GPU}",
                      "the profile's lanes bind the padded projections", "hours")
    return Recipe("rewrite", "engine/kernels/dense/kernels.cu (KBLK_LIMIT) and cells.DENSE_KMAX",
                  f"{'; '.join(too_wide)}: a wider decode kernel, or the projection split into K tiles at the caller",
                  f"tests/test_engine_dense.py on {_GPU}", "DenseLinear binds the projection", "days")


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


def _recipe_kda_chunk():
    return Recipe("wire", "engine/profiles/<profile>/lanes.py (kda_chunk)",
                  "bind engine/kernels/kda/chunk_decay.chunk_kda_with_decay: chunk_kda_with_fused_gate's pipeline, "
                  "states_at and out included, with the decay computed outside the kernel -- a per-head decay summed per "
                  "chunk and read one value a head (G_HEAD), fewer key heads read by their group (QG)",
                  f"{_GLUE_TEST} on {_GPU} (it passes on the CPU under TRITON_INTERPRET=1) and " + _KDA_JUDGE,
                  "the prefill lane binds it and chunked == recurrent on the oracle", "hours")


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


def _judged(serve):
    """The same kernel after its GPU judgment and a timed record at this cell: what a measured glue cell serves."""
    note = serve.note.replace("unjudged on a GPU", "judged on a GPU and timed at this cell")
    return replace(serve, judged=True, note=note if note != serve.note else f"{note}; judged on a GPU and timed at this cell")


def _v41_measured(shape) -> bool:
    return (shape.hc_variant == MHC_V41_VARIANT and shape.hidden in MHC_V41_MEASURED_HIDDEN
            and mhc_v41_refusal(shape) is None)


# the retired dsv4 overlay's flashinfer_sparse (git history): the sink-capable DSV4 decode's head size and query heads
DSV4_SINK_HEAD, DSV4_SINK_MAX_HEADS = 512, 128


REC = "measurements/sm121_candidates_20260919"   # the image's attention kernels judged on GB10 (sm121 intake U6-U8)


def _serve_attention(a, i):
    """The fastest kernel for a full attention the MLA lane refuses. An attention whose sink is not established has
    nothing fast: the adapters refuse it (mla_glue_refusal), so the note names the candidate and the establish recipe
    is the work."""
    qsa = i is not None and i.compress == "qsa"
    qsa_op = ("qsa_sparse_paged_attention in engine/kernels/qsa.py (vLLM's Triton QSA sparse paged GQA attention, the "
              "kernel that served Qwen3.8 in the vLLM stack, ported: BF16 KV; its blocks entry expands the chosen blocks "
              "inside its tiles)")
    if a.relative:
        return _nothing(f"no kernel in the repo or the image adds a learned relative-position bias ({a.relative}) to the "
                        "logits; vLLM's Triton relative attention (vllm#55078, SM8x) is the nearest source")
    if a.kind != "mla" and i is None and (a.window or a.softcap) and a.sink is False:
        return _serve(GENERIC, "flashinfer BatchDecodeWithPagedKVCacheWrapper and BatchPrefillWithPagedKVCacheWrapper (in "
                      "the image, engine/INVENTORY.md) with " + " and ".join(
                          t for t in (a.window and f"window_left={a.window - 1}", a.softcap and
                                      f"logits_soft_cap={a.softcap:g}") if t), False,
                      f"the kernels judged on GB10 against an fp32 reference, window and soft cap within 0.2-0.5% ({REC}); "
                      "no engine adapter yet")
    if a.kind != "mla":
        packed = 2 * a.head_dim <= MLA_LATENT
        if a.sink is None:
            candidate = ("engine/kernels/mla/glue.gqa" if packed else qsa_op if qsa else
                         "flashinfer's paged decode and prefill (engine/INVENTORY.md)")
            return _nothing(f"the sink decides which kernel computes this attention; establish it first -- with no sink, "
                            f"{candidate} serves it")
        if a.sink and i is not None:
            return _nothing("no GQA kernel that takes sinks over an indexer's selection is named in the repo or the image "
                            "(xqa and the AttentionSink variant read the whole paged context)")
        if a.sink:
            return _serve(GENERIC, "flashinfer xqa_batch_decode_with_kv_cache or the AttentionSink JIT variant "
                          "(BatchAttentionWithAttentionSinkWrapper), in the image", False,
                          f"both compute sinks on GB10 within 0.2-0.6% of an fp32 reference, windows too ({REC}); the "
                          "fa2 and CUDA-core wrappers accept sinks= and IGNORE it -- never bind those for a sink model")
        if qsa:
            return _serve(GENERIC, qsa_op, False,
                          "judge it against modules/sparse_attention.gqa_sparse over the indexer's selected positions"
                          + ("; engine/kernels/mla/glue.gqa is the one-scale e4m3 latent alternative" if packed else ""))
        if packed:
            return _serve(GLUE, "engine/kernels/mla/glue.gqa (the megakernel's sparse MLA with each KV head's key and "
                          "value packed side by side into its latent)", False,
                          "exact arithmetic, but the KV cache becomes the latent's one-scale e4m3"
                          + (f"; the BF16-KV alternative is {qsa_op}" if qsa else ""))
        if qsa:
            return _serve(GENERIC, qsa_op, False,
                          "judge it against modules/sparse_attention.gqa_sparse over the indexer's selected positions")
        return _serve(GENERIC, "flashinfer BatchDecodeWithPagedKVCacheWrapper and BatchPrefillWithPagedKVCacheWrapper (in "
                      "the image, engine/INVENTORY.md)", False,
                      f"the kernels judged on GB10 against an fp32 reference within 0.2-0.5% ({REC}); no engine adapter yet")
    if a.sink is None:
        return _nothing("the sink decides which kernel computes this attention; establish it first")
    if a.sink:
        if a.head_dim == DSV4_SINK_HEAD and a.heads <= DSV4_SINK_MAX_HEADS:
            return _serve(GENERIC, "flashinfer trtllm_batch_decode_sparse_mla_dsv4, which takes sinks (the V4-Flash call in "
                          "the retired dsv4 overlay's flashinfer_sparse, git history)", False,
                          "decode only; whether its trtllm-gen kernel runs on sm_121a is part of the judgment")
        return _nothing(f"the sink-capable DSV4 decode takes head size {DSV4_SINK_HEAD} and at most {DSV4_SINK_MAX_HEADS} "
                        "query heads (the retired dsv4 overlay's flashinfer_sparse, git history); asked "
                        f"{a.heads} x {a.head_dim}")
    if a.head_dim <= MLA_LATENT:
        return _serve(GLUE, f"engine/kernels/mla/glue.grouped (the megakernel's sparse MLA per group of {MLA_HEADS} heads"
                      + ("" if a.head_dim == MLA_LATENT else f", the {a.head_dim} latent zero-extended to {MLA_LATENT}") + ")",
                      False,
                      ("exact; engine/profiles/glm53/lanes.py groups its heads the same way at world 1"
                       if a.heads % MLA_HEADS == 0 else "exact; zero-query heads fill the last group")
                      + ("" if a.head_dim == MLA_LATENT else
                         f"; the cache rows are written {MLA_LATENT} wide ({MLA_LATENT}/{a.head_dim} of the latent's memory)"))
    return _serve(GENERIC, "flashinfer BatchDecodeMlaWithPagedKVCacheWrapper with page-size-1 slot indices (in the image, "
                  "engine/INVENTORY.md)", False,
                  "the vLLM-era GLM lane served sparse MLA through the page-size-1 wrapper; not judged at this latent")


def _serve_mhc_variant(shape, lane):
    """The fastest kernel for a hyper-connection form the MK segment and the TileLang mixes do not compute."""
    if shape.hc_variant == GATED_RESIDUAL_VARIANT:
        return _serve(SPECIALIZED, "engine/kernels/gated_residual (the gated residual in five launches a site: the previous "
                      "leave joined to the stream norm, down and inject in one BF16 GEMM, the gates, up, the stream mean; "
                      "a decode step's 1-16 rows in three: the leave with the norm, down with the gates, up with the "
                      "mean)",
                      False, "gated_residual.qualify holds it to engine/modules/hyper_connection.gated_residual at the "
                      "model's widths before a boot serves; unjudged on a GPU")
    why = mhc_v41_refusal(shape)
    if why is not None:
        return _nothing(why)
    if lane == "mhc_decode":
        return _serve(GLUE, "engine/kernels/dense/mhc.MHCV41 (run_mhc_v41, the megakernel's V4.1 seam)", False,
                      "its GPU probe never ran (measurements/dsv41_mhc_20260910)")
    return _serve(GLUE, f"engine/kernels/dense/mhc.MHCV41.prefill (run_mhc_v41 in {MHC_MAX_TOK}-token pieces: the seam "
                  "mixes each token alone)", False,
                  "one launch per piece and unjudged on a GPU; no prefill kernel for the V4.1 pairing exists — the vLLM "
                  "stack mixed it in torch, and that overlay module is retired (git history)")


def _serve_indexer(i):
    """The fastest kernels for an indexer the kpool lane refuses."""
    if i.compress == "qsa":
        return _serve(SPECIALIZED, "engine/kernels/qsa (qsa_compress_groups_with_ratio, qsa_mqa_paged and "
                      "qsa_select_paged_blocks: vLLM's Triton QSA ops, ported, with the engine's top-k)", False,
                      "judge the compression against modules/sparse_indexer.qsa_select and the scoring against its relu "
                      "sum over the index heads; unjudged on a GPU")
    if i.compress == "ced":
        return _nothing("the CED compressor exists only in torch (the retired dsv41 overlay module, git history); its "
                        "scoring and packed keys have Triton kernels beside it (dsv41_indexer_triton.py, "
                        "dsv41_packed_index_triton.py)")
    return _nothing(f"no Hadamard-{i.head_dim} kernel; the kpool rotation is fixed at {INDEXER_HEAD_DIM}")


def admission(shape) -> "list[Verdict]":
    """One verdict per lane for a kernel shape: admitted, unmeasured or refused (see the module docstring). Every verdict
    that is not admitted carries its recipe, and every layer names what serves it: the lane's own kernel, that kernel
    through an exact adapter, a shape-generic fast kernel for the same math, or nothing fast. The engine/modules oracles
    judge; they never serve."""
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
    elif arch.target(d.capability) is not None:
        # A card the lanes can be BUILT for, which is not the card they were measured on.
        # It runs by declaration -- the definition of `unmeasured` -- and it can never be
        # `admitted`, because no measurement here was taken anywhere but a GB10 (D5).
        unmeasured("device", f"{arch.name(d.capability)}, {d.sms} SMs: the lanes compile and run here, and every "
                             f"number they give is this card's. GB10 sm_121a with {MEASURED.device.sms} SMs is what "
                             f"they were measured on", _recipe_check_device(d), None)
    else:
        refuse("device", f"every lane is built for GB10 sm_121a with {MEASURED.device.sms} SMs; asked "
                         f"SM{d.capability[0]}{d.capability[1]}/{d.sms}", _recipe_device(d), None)

    terms = attention_terms_refusal(a, indexed=i is not None)
    if terms is not None:
        # before the geometry: an attention whose math no lane computes is refused for that, whatever its cell
        refuse("mla", terms, _recipe_attention_terms(a), _serve_attention(a, i))
    elif a.kind != "mla" or (a.heads, a.head_dim) != (MLA_HEADS, MLA_LATENT):
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

    if a.kv_conv:                                      # like the indexer: a model without the conv has no lane for it
        unmeasured("kv_conv", f"a {a.kv_conv}-tap causal conv on the keys and values before the cache, per sequence",
                   _recipe_kv_conv(a),
                   _serve(GENERIC, "engine/kernels/causal_conv.py (the short causal conv the linear attention runs on "
                          "its q/k/v)", False, "never run on attention keys and values"))

    if i is not None:                                  # a model without a sparse indexer has no indexer lane to judge
        if i.compress == INDEXER_KEY_COMPRESS and a.window:
            refuse("indexer", f"the {INDEXER_KEY_COMPRESS} selection takes the top {i.topk} by score; this model's layers "
                              f"also read a {a.window}-position window the selection does not add",
                   Recipe("kernel", "engine/kernels/indexer.py (_pool_slots) and the selection it feeds",
                          "append each row's last window positions to its selected ids before the attention reads them, "
                          "deduplicated against the top-k", "modules/sparse_attention with the window's ids, on "
                          f"{_GPU}", "the selection carries the window and the glue test holds it", "hours"),
                   _serve_indexer(i))
        elif i.compress != INDEXER_KEY_COMPRESS:
            refuse("indexer", f"the indexer lane compresses keys by {INDEXER_KEY_COMPRESS}; this model compresses by "
                              f"{i.compress} (only the MQA scoring formula is shared)", _recipe_indexer_compress(i),
                   _serve_indexer(i))
        elif i.head_dim == INDEXER_HEAD_DIM:
            admit("indexer", f"{INDEXER_KEY_COMPRESS} keys at Hadamard-{INDEXER_HEAD_DIM}, pool {i.pool}, top {i.topk} at launch",
                  _serve(SPECIALIZED, "engine/kernels/kpool.py, engine/kernels/indexer.py and DeepGEMM fp8_fp4_mqa_logits "
                         "(engine/kernels/deep_gemm.py)", True, "the retired fork-differential probe against modules/sparse_indexer (removed with the overlay stack)"))
        else:
            refuse("indexer", f"the indexer lanes are written for head_dim {INDEXER_HEAD_DIM}; asked {i.head_dim}",
                   _recipe_indexer(i), _serve_indexer(i))

    mk = "engine/kernels/dense/mhc.py (the MK mHC segment, run_mhc)"
    tilelang = "engine/kernels/mhc (TileLang mhc_pre and mhc_post)"
    # hc 1 is a plain residual: no streams to mix, so there is no mHC lane to judge -- the rule the indexer
    # and KDA lanes follow above. `hc_variant` None on a model that HAS streams is the other thing, and the
    # refusal inside says so.
    if shape.hc > 1:
        hc_unknown = _recipe_establish("hyper-connection form", "the profile's kernel_shape (hc_variant)",
                                       "modules/hyper_connection: mhc_pre/mhc_post or hc_split_sinkhorn reproduces the reference")
        hc_nothing = _nothing("the hyper-connection form decides which kernel mixes; establish it first")
        if shape.hc_variant is None:
            refuse("mhc_decode", "the hyper-connection form is not established; the segment's math depends on it", hc_unknown,
                   hc_nothing)
        elif _v41_measured(shape):
            admit("mhc_decode", f"the split-sinkhorn form on MHCV41 at hidden {shape.hidden}, judged and timed",
                  _judged(_serve_mhc_variant(shape, "mhc_decode")))
        elif shape.hc_variant != MHC_VARIANT:
            refuse("mhc_decode", f"the MK mHC segment computes {MHC_VARIANT}; this model mixes by {shape.hc_variant}",
                   _recipe_mhc_variant(shape, "mhc_decode"), _serve_mhc_variant(shape, "mhc_decode"))
        elif shape.hidden not in MHC_HIDDEN or shape.hc != MHC_HC:
            refuse("mhc_decode", f"MK mHC is compiled for hidden {MHC_HIDDEN} at hc {MHC_HC}; asked hidden {shape.hidden} "
                                 f"hc {shape.hc}", _recipe_mhc(shape),
                   _serve(GENERIC, tilelang, True, "the same mhc math, judged at hidden 4096 (the fork-differential probe retired with the overlay stack); "
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
        elif _v41_measured(shape):
            admit("mhc_prefill", f"the split-sinkhorn form on MHCV41.prefill at hidden {shape.hidden}, judged and timed",
                  _judged(_serve_mhc_variant(shape, "mhc_prefill")))
        elif shape.hc_variant != MHC_VARIANT:
            refuse("mhc_prefill", f"the TileLang mixes compute {MHC_VARIANT}; this model mixes by {shape.hc_variant}",
                   _recipe_mhc_variant(shape, "mhc_prefill"), _serve_mhc_variant(shape, "mhc_prefill"))
        else:
            admit("mhc_prefill", "TileLang mixes take hidden and hc from the tensors",
                  _serve(SPECIALIZED, tilelang, True, "the fork-differential probe (retired with the overlay stack) against modules/hyper_connection"))

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
    # the projections pack K = hidden whatever the MLPs are; a model without a dense or shared MLP (dense_inter_local 0,
    # engine/base/kernel_shape.MoE) is judged on that width alone
    widths, asked = _dense_widths(shape, m)
    unaligned = [width for _, width in widths if width % DENSE_ALIGN]
    glue_fits = all(dense_glue_refusal(c) is None for c in unaligned)
    if (unaligned and glue_fits and shape.hidden in DENSE_MEASURED_HIDDEN
            and all(c in DENSE_GLUE_MEASURED_COLUMNS for c in unaligned)):
        admit("dense", f"W4A8 decode / FP8 prefill over zero-padded columns ({asked}), judged and timed",
              _serve(GLUE, "engine/kernels/dense.PaddedDenseLinear (the W4A8/FP8 lane over zero-padded columns)", True,
                     "exact: zero weight columns and a zero-extended input; judged on a GPU with its dispatch timed"))
    elif unaligned:
        refuse("dense", f"dense W4 tiles need {DENSE_ALIGN}-aligned columns; asked {asked}", _recipe_dense(shape, m),
               _serve(GLUE, "engine/kernels/dense.PaddedDenseLinear (the W4A8/FP8 lane over zero-padded columns)", False,
                      "exact: zero weight columns and a zero-extended input ("
                      + ", ".join(f"{c} -> {-(-c // DENSE_ALIGN) * DENSE_ALIGN}" for c in unaligned) + "); unjudged on a GPU")
               if all(dense_glue_refusal(c) is None for c in unaligned) else
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
        recurrent_serve = (_serve(GLUE, "engine/kernels/kda (fused_recurrent_kda(compute_gate=False) over "
                                  "linear_decay.per_channel(decay))", False,
                                  "the per-head decay read through a stride-0 channel axis; held to modules/linear_attention "
                                  "under Triton's CPU interpreter (tests/test_engine_kernel_glue.py), unjudged on a GPU")
                           if per_head else
                           _serve(SPECIALIZED, "engine/kernels/kda (fused_recurrent_kda)", True,
                                  "tests/test_engine_kda_state.py" + ("" if measured_cell else f", at {_KDA_CELLS_TEXT}")))
        decay_measured = per_head and (l.heads, l.v_heads, l.k_dim, l.v_dim) in KDA_DECAY_MEASURED_CELLS
        if decay_measured:
            admit("kda_recurrent", f"{recurrent}; judged and timed at this cell", _judged(recurrent_serve))
        elif measured_cell and not per_head:
            admit("kda_recurrent", recurrent, recurrent_serve)
        else:
            unmeasured("kda_recurrent", f"{recurrent}; the BV=16 tile is measured at {_KDA_CELLS_TEXT} only",
                       _recipe_kda_measure(l), recurrent_serve)
        # a per-head (GatedDeltaNet) cell's ring lane is the ring kernel's own launch with GDN's gate compiled in
        # (HEAD_GATE): no adapter between the model's arithmetic and the kernel
        ring_gdn = _serve(SPECIALIZED, "engine/kernels/kda/ring.recurrent_gdn_ring and recurrent_gdn_ring_rows (the ring "
                          "kernel computing GatedDeltaNet's per-head decay from its projection)", False,
                          "the fused entry's launch and in-kernel ring writes with engine/kernels/gdn.gates' arithmetic in "
                          "place of KDA's gate: byte-identical to that launch followed by the decay entry "
                          "(recurrent_decay_ring) and held to modules/linear_attention under Triton's CPU interpreter, "
                          "unjudged on a GPU")
        chunk_glue = _serve(GLUE, "engine/kernels/kda/chunk_decay.chunk_kda_with_decay (chunk_kda_with_fused_gate's "
                            "pipeline on a precomputed decay)", False,
                            "the per-head decay summed per chunk and read one value a head; held to modules/linear_attention "
                            "(states_at included) under Triton's CPU interpreter, unjudged on a GPU")
        if decay_measured:
            admit("kda_ring", "the ring kernel computing GatedDeltaNet's per-head decay, judged and timed",
                  _judged(ring_gdn))
            admit("kda_chunk", "the chunk pipeline on a precomputed per-head decay, judged and timed", _judged(chunk_glue))
        elif per_head:
            unmeasured("kda_ring", f"the ring kernel computes GatedDeltaNet's per-head decay in its own launch; the BV=16 "
                                   f"tile is measured at {_KDA_CELLS_TEXT} only", _recipe_kda_measure(l), ring_gdn)
            refuse("kda_chunk", "the chunk lane fuses KDA's gate; a head-decay prefill runs the pipeline on a decay computed "
                                "outside it", _recipe_kda_chunk(), chunk_glue)
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
        refuse("moe", f"the b12x lane is compiled for {measured.quant} only (D5's base form, not a model bound); "
               f"asked {m.quant}", _recipe_moe_quant(m, measured),
               _serve(GENERIC, "b12x's MXFP4 kernels (engine/kernels/b12x/moe_dispatch.py: FP4 in groups of 32 with E8M0 "
                      "scales)", False, "they read MXFP4-layout weights as they are but quantize activations to FP4; the "
                      "quality gate decides") if m.quant.startswith("mxfp4")
               else _nothing(f"no fast kernel reads {m.quant} experts"))
    elif replace(m, dynamic_tile_m=None) == measured:
        admit("moe", "the measured GB10 TP4 cell" + ("" if m.dynamic_tile_m is None else f", tile pinned at {m.dynamic_tile_m}"),
              _serve(SPECIALIZED, b12x, True, "probes/engine_kernel_check.py --lanes moe --moe-experts 288"))
    elif replace(m, dynamic_tile_m=None) in MOE_MEASURED_CELLS:
        admit("moe", f"a measured cell ({m.experts} experts, {m.experts_local} local, I{m.inter_local}, top{m.topk}, "
                     f"{m.activation})" + ("" if m.dynamic_tile_m is None else f", tile pinned at {m.dynamic_tile_m}"),
              _serve(SPECIALIZED, b12x, True, "judged against modules/moe on a GPU with its tiles timed at this cell"))
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
    """How many layers each tier serves, and how many of the glue and generic ones are judged."""
    served = [v.serve for v in verdicts if v.serve is not None]
    out = {t: sum(x.tier == t for x in served) for t in TIERS}
    out["glue_judged"] = sum(x.tier == GLUE and x.judged for x in served)
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
    rows.append(f"  serving: {sv[SPECIALIZED]} specialized, {sv[GLUE]} glue ({sv['glue_judged']} judged), "
                f"{sv[GENERIC]} generic ({sv['generic_judged']} judged), {sv[NONE]} with nothing fast")
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
