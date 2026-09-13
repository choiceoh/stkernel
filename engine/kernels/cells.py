"""What each lane is compiled for, in one torch-free place: the cells.

A kernel wrapper refuses a bound kernel shape it cannot serve (engine/base/kernel_shape).
The numbers it refuses against live HERE and the wrappers import them, so the shape wizard
can judge a checkpoint before any boot -- `admission(shape)` -- and can never disagree
with the wrappers. A compiled cell changes in the kernel and in this file together (a third
HIDDEN instance of the mHC segment, a wider Hadamard); the tests pin the wrappers to these
names.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# mla/glm53_megakernel.cu: MLA_H query heads per rank over an MLA_D latent (kv_lora_rank)
MLA_HEADS = 16
MLA_LATENT = 512
# kpool.py: the Hadamard-128 rotation and the one-warp pooling lane
INDEXER_HEAD_DIM = 128
# dense/kernels.cu: the HIDDEN and HIDDEN_V41 instances of the mHC segment at HC; MHC_MAX_TOK_DEF rows; HCHUNK
MHC_HIDDEN = (4096, 5120)
MHC_HC = 4
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

ADMITTED, REFUSED, UNMEASURED = "admitted", "refused", "unmeasured"


@dataclass(frozen=True)
class Verdict:
    lane: str
    status: str
    why: str


def admission(shape) -> "list[Verdict]":
    """One verdict per lane for a kernel shape: admitted (the wrapper will serve it), refused (the
    wrapper will die by name -- what a boot on this shape would hit), or unmeasured (served by
    declaration, but its cell has no measurement yet)."""
    from engine.base.kernel_shape import MEASURED
    a, l, i, m, c, d = shape.attention, shape.linear, shape.indexer, shape.moe, shape.comm, shape.device
    out = []

    def verdict(lane, ok, why_ok, why_not, *, unmeasured=False):
        out.append(Verdict(lane, (UNMEASURED if unmeasured else ADMITTED) if ok else REFUSED, why_ok if ok else why_not))

    verdict("device", (tuple(d.capability), d.sms) == (MEASURED.device.capability, MEASURED.device.sms),
            f"GB10 SM{d.capability[0]}{d.capability[1]}, {d.sms} SMs",
            f"every lane is built for GB10 sm_121a with {MEASURED.device.sms} SMs; asked SM{d.capability[0]}{d.capability[1]}/{d.sms}")
    verdict("mla", (a.kind, a.heads, a.head_dim) == ("mla", MLA_HEADS, MLA_LATENT),
            f"the compiled {MLA_HEADS} heads x {MLA_LATENT} latent cell",
            f"compiled for the {MLA_HEADS} heads x {MLA_LATENT} latent MLA cell; asked {a.kind} {a.heads}x{a.head_dim}"
            + (" -- a GQA attention has no ST lane yet" if a.kind != "mla" else ""))
    if i is not None:                                  # a model without a sparse indexer has no indexer lane to judge
        verdict("indexer", i.head_dim == INDEXER_HEAD_DIM,
                f"Hadamard-{INDEXER_HEAD_DIM} keys, pool {i.pool}, top {i.topk} at launch",
                f"the indexer lanes are written for head_dim {INDEXER_HEAD_DIM}; asked {i.head_dim}")
    verdict("mhc_decode", shape.hidden in MHC_HIDDEN and shape.hc == MHC_HC,
            f"MK mHC instance for hidden {shape.hidden} at hc {shape.hc}",
            f"MK mHC is compiled for hidden {MHC_HIDDEN} at hc {MHC_HC}; asked hidden {shape.hidden} hc {shape.hc} "
            "-- a new width is a third instance in dense/kernels.cu")
    verdict("mhc_prefill", True, "TileLang mixes take hidden and hc from the tensors", "")
    verdict("oneshot", c.world == ONESHOT_WORLD and c.hidden % 8 == 0 and c.hidden <= ONESHOT_MAX_ELEMENTS,
            f"rows of {c.hidden} BF16, up to {ONESHOT_MAX_ELEMENTS // c.hidden} per collective"
            + ("" if c.hidden == MEASURED.hidden else f" (timed at hidden {MEASURED.hidden}; this width is unmeasured)"),
            f"one-shot is compiled for {ONESHOT_WORLD} ranks and rows of 8-element multiples within {ONESHOT_MAX_ELEMENTS}; "
            f"asked world {c.world}, hidden {c.hidden}")
    blocks = PREFILL_BLOCK // _gcd(PREFILL_BLOCK, c.hidden)
    verdict("prefill_collectives", True,
            f"FP8 packets over {c.world} ranks; rows must complete {PREFILL_BLOCK}-element blocks (every {blocks} row{'s' if blocks > 1 else ''})", "")
    verdict("dense", shape.hidden % DENSE_ALIGN == 0 and m.dense_inter_local % DENSE_ALIGN == 0,
            f"W4A8 decode / FP8 prefill, K {DENSE_ALIGN}-aligned",
            f"dense W4 tiles need {DENSE_ALIGN}-aligned widths; asked hidden {shape.hidden}, dense intermediate {m.dense_inter_local}")
    if shape.drafter is not None:
        verdict("draft", True, f"DFlash kernels at head {shape.drafter.head_dim} (a constexpr)", "")
    if l is not None:                                  # a model without linear attention has no KDA lane to judge
        verdict("kda_recurrent", True,
                "fused_recurrent_kda over [B,T,HV,K] decays" + (
                    "" if l.decay == FUSED_GATE_DECAY else "; the per-head decay is widened by linear_decay.per_channel (compute_gate=False)"), "")
        verdict("kda_ring", l.decay == FUSED_GATE_DECAY,
                "the ring lane's fused per-channel KDA gate",
                "the ring lane fuses KDA's per-channel gate; a head-decay cell runs fused_recurrent_kda(compute_gate=False) "
                "and writes its ring with state.write_ring")
        verdict("kda_chunk", l.decay == FUSED_GATE_DECAY,
                "chunk_kda_with_fused_gate over the prefill",
                "the chunk lane fuses KDA's gate; a GDN prefill needs the chunk kernel without it (not written)")
    measured = replace(MEASURED.moe, dynamic_tile_m=None)
    if m.quant != measured.quant:
        verdict("moe", False, "", f"the b12x lane is {measured.quant} only (D5); asked {m.quant}")
    else:
        same = replace(m, dynamic_tile_m=None) == measured
        verdict("moe", True,
                "the measured GB10 TP4 cell" + ("" if m.dynamic_tile_m is None else f", tile pinned at {m.dynamic_tile_m}") if same
                else f"admitted by declaration ({m.experts} experts, {m.experts_local} local, I{m.inter_local}, top{m.topk}, {m.activation}); "
                     "measure the prefill tile (pin moe.dynamic_tile_m) and the SF6 scale span before serving",
                "", unmeasured=not same)
    verdict("universal", True,
            "sampler, block verify, decode commit, vocab candidates, SwiGLU, norm+RoPE, route histogram, build cache, calibration: "
            "the arguments are the shape", "")
    return out


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def table(verdicts: "list[Verdict]") -> str:
    width = max(len(v.lane) for v in verdicts)
    rows = [f"  {v.lane:<{width}}  {v.status:<10}  {v.why}" for v in verdicts]
    counts = {s: sum(v.status == s for v in verdicts) for s in (ADMITTED, UNMEASURED, REFUSED)}
    rows.append(f"  {counts[ADMITTED]} admitted, {counts[UNMEASURED]} unmeasured, {counts[REFUSED]} refused")
    return "\n".join(rows)
