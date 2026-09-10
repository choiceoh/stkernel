"""Which components each of DeepSeek-V4.1's 40 layers carries, from the config.

The model is described as a "Causal Encoder-Decoder (CED)": a 20-layer causal
encoder followed by a 20-layer decoder whose global KV is projected from the
final encoder states rather than derived per decoder layer. That sentence is
from the card; what makes it buildable is that the config states the same thing
three independent ways, and all three agree:

  compress_ratios          length 43 = num_hidden_layers 40 + the 3 MTP layers,
                           and it runs 0,0 | 2 x18 | 1 x20 | 0,0,0 -- so layers
                           2..19 compress by 2 and 20..39 by 1. The step from 2
                           to 1 IS the encoder/decoder boundary.
  candidate_source_layer_id  20 -- the boundary again, named.
  kv_source_layer_ids      [2, 8, 14, 20] -- every one of them at or before 20.
                           Nothing PAST the boundary produces KV, which is the
                           CED claim stated as a weight layout.

The boundary layer is the subtle one and it is worth stating before it bites:
layer 20 is the FIRST decoder layer and it is also a KV source, which reads like
a contradiction of "the decoder produces no KV of its own". It is not. Layer 20
receives the output of layers 0..19 -- the encoder -- so its compressor is
exactly the projection of the final encoder states the card describes. Every
other decoder layer (21..39) sources nothing. So the rule to build to is not
"no decoder layer sources KV" but "exactly one may, and only at the boundary".

`index_source_layer_ids` is the one list that crosses: [2, 8, 14, 20, 24, 28,
32, 36]. The indexer keeps running in the decoder because selecting from KV is
not producing it -- the decoder half indexes into what the encoder built.

Nothing here is inferred from the tensor names. probes/dsv41_layer_plan.py
takes the plan this file derives from config alone and requires it to predict
the checkpoint's 96,085 tensors exactly, in both directions: a layer the plan
gives a compressor must have one, and a layer that has one must have been given
it. A misread of the CED split shows up there as a layer that disagrees, not as
a model that trains to a worse loss six weeks later.
"""

from __future__ import annotations

from dataclasses import dataclass

ENCODER = "encoder"
DECODER = "decoder"


@dataclass(frozen=True)
class LayerPlan:
    """One layer's shape. `index` is the position in the main stack."""

    index: int
    role: str
    compress_ratio: int
    #: carries `attn.compressor.*` and feeds the global KV cache
    kv_source: bool
    #: carries `attn.indexer.*`
    indexer: bool
    #: carries `engram.*`; the table this layer reads, or None
    engram_table: "int | None"
    #: every layer of this model routes; there is no dense prefix
    moe: bool = True

    @property
    def engram(self) -> bool:
        return self.engram_table is not None


def _runs(values):
    out = []
    for v in values:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(v, n) for v, n in out]


def plan_layers(cfg: dict) -> "list[LayerPlan]":
    """The per-layer plan, and the checks that make it a plan rather than a guess.

    Every assertion here is one the checkpoint would violate if the config were
    read wrong, so they are cheap and they fire at load rather than at inference.
    """
    n = int(cfg["num_hidden_layers"])
    n_mtp = int(cfg.get("num_nextn_predict_layers", 0))
    ratios = list(cfg["compress_ratios"])
    if len(ratios) != n + n_mtp:
        raise ValueError(
            f"compress_ratios has {len(ratios)} entries; expected "
            f"{n} layers + {n_mtp} MTP = {n + n_mtp}. The tail belongs to the "
            f"MTP heads, so a mismatch means the two are no longer laid out "
            f"end to end and every index below is off.")
    main = ratios[:n]

    boundary = int(cfg["candidate_source_layer_id"])
    kv_src = set(cfg["kv_source_layer_ids"])
    idx_src = set(cfg["index_source_layer_ids"])
    engram_ids = list(cfg.get("engram_layer_ids", ()))

    beyond = sorted(layer for layer in kv_src if layer > boundary)
    if beyond:
        raise ValueError(
            f"kv_source_layer_ids {sorted(kv_src)} reach past the "
            f"encoder/decoder boundary {boundary} at {beyond}. In a CED stack "
            f"the decoder consumes the encoder's KV; a source beyond the "
            f"boundary means this is not the layout assumed here.")
    # The boundary layer may source -- that source IS the projection of the
    # final encoder states -- but it is the only decoder-side one.
    decoder_sources = sorted(layer for layer in kv_src if layer >= boundary)
    if decoder_sources not in ([], [boundary]):
        raise ValueError(
            f"decoder-side kv sources {decoder_sources}: at most the boundary "
            f"layer {boundary} may source, because that one alone reads the "
            f"encoder's final output.")

    # The boundary read off compress_ratios must be the one the config names.
    shape = _runs(main)
    step = None
    seen = 0
    for value, count in shape:
        if step is None and seen and value != shape[0][0] and seen > 1:
            pass
        seen += count
    encoder_ratio = main[boundary - 1] if boundary else None
    decoder_ratio = main[boundary] if boundary < n else None
    if encoder_ratio == decoder_ratio:
        raise ValueError(
            f"compress_ratios does not step at layer {boundary} "
            f"({encoder_ratio} -> {decoder_ratio}); the boundary the config "
            f"names and the one the ratios show disagree.")

    return [
        LayerPlan(
            index=i,
            role=ENCODER if i < boundary else DECODER,
            compress_ratio=main[i],
            kv_source=i in kv_src,
            indexer=i in idx_src,
            engram_table=engram_ids.index(i) if i in engram_ids else None,
        )
        for i in range(n)
    ]


def describe(plans: "list[LayerPlan]") -> str:
    enc = [p for p in plans if p.role == ENCODER]
    dec = [p for p in plans if p.role == DECODER]
    return (
        f"{len(plans)} layers: encoder {len(enc)} (compress "
        f"{sorted({p.compress_ratio for p in enc})}), decoder {len(dec)} "
        f"(compress {sorted({p.compress_ratio for p in dec})}); "
        f"kv sources {[p.index for p in plans if p.kv_source]}; "
        f"indexers {[p.index for p in plans if p.indexer]}; "
        f"engram {[p.index for p in plans if p.engram]}")


def expert_rank(expert: int, n_routed_experts: int, world_size: int) -> int:
    """Which rank owns routed expert `expert`. CONTIGUOUS blocks.

    This is a contract between two pieces of code that never run together: the
    tool that writes a rank's weights and the `load_weights` that reads them.
    They agree by importing this, not by each implementing it -- a strided
    partition (`expert % world_size`) is an equally valid split, produces files
    that open and tensors that are byte-exact, and routes every token to the
    wrong expert. Nothing downstream can tell.

    Contiguous rather than strided for a second reason: a rank's experts are
    then adjacent in the source checkpoint, so the builder's copy is a
    sequential read of a 269 GiB region rather than a stride over it.
    """
    if n_routed_experts % world_size:
        raise ValueError(
            f"{n_routed_experts} routed experts do not divide over "
            f"{world_size} ranks; expert parallelism moves whole experts, so "
            f"a remainder would leave a rank short.")
    return expert // (n_routed_experts // world_size)
