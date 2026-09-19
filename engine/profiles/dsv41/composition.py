"""DeepSeek-V4.1-Flash's layer plan (profile): which feature each of the 40 layers runs, from the config alone.

This is the first half of a composition (engine/base/composition): the plan. The other half -- the features bound by
name, the residual form, the weight names -- is not here, because the module families do not yet carry every form this
model computes; engine/DSV41_COMPOSITION.md is the list of what is there and what is missing, and `build` lands when
that list closes.

The model is a "Causal Encoder-Decoder" (CED): a 20-layer causal encoder, then a 20-layer decoder whose global KV is
projected from the final encoder states rather than derived per decoder layer. The config states that three
independent ways and all three agree -- the derivation below is the one the retired overlay's `dsv41_layers.py` ran
(git history, PR #1152 removed it), whose checks are kept because each one is a way the config could be misread:

  compress_ratios           43 = 40 layers + 3 MTP heads, running 0,0 | 2 x18 | 1 x20 | 0,0,0. Layers 2..19 compress
                            by 2 and 20..39 by 1; the step from 2 to 1 IS the encoder/decoder boundary.
  candidate_source_layer_id 20 -- the boundary again, named.
  kv_source_layer_ids       [2, 8, 14, 20] -- none past the boundary. The CED claim as a weight layout.

Layer 20 is the first decoder layer AND a KV source, which is not a contradiction: it receives layers 0..19, so its
compressor is the projection of the final encoder states. Every other decoder layer sources nothing.

A ratio of 0 is a third thing the list says: layers 0 and 1 do not compress at all and attend a sliding window, and
they take a DIFFERENT rotary table -- the base `rope_theta` with YaRN off, where every compressing layer takes
`compress_rope_theta` with YaRN on (`LayerPlan.rope`). Two tables in one model, and a layer handed the wrong one
still returns numbers.

`index_source_layer_ids` is the one list that crosses the boundary ([2, 8, 14, 20, 24, 28, 32, 36]): selecting from
KV is not producing it, so the decoder half keeps indexing into what the encoder built.

WHAT IS NOT ESTABLISHED HERE: engram's place inside its layer. The config names the layers that carry a table
(`engram_layer_ids` [1, 14], and modules/ngram_embedding holds the hash and the injection to the vendor's), but
nothing in this tree says where in layer 1 the gated values are written. The composition has one injection site --
before the layer -- so that is where this plan puts it, and the vendor `inference/model.py` (sha-pinned in caches.py,
on srv4) is what settles it. It is the first row of DSV41_COMPOSITION.md's open list.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.base.composition import Layer, Plan

ENCODER, DECODER = "encoder", "decoder"
#: the feature names this plan plays: a ratio-0 layer's sliding window, every other layer's CED-selected attention,
#: the router every layer runs (`intermediate_size` is null: there is no dense prefix), and the engram injection
WINDOW_ATTENTION, SPARSE_ATTENTION, MOE, ENGRAM = "window_attention", "sparse_attention", "moe", "engram"


@dataclass(frozen=True)
class LayerPlan:
    """One layer's shape. `index` is its position in the main stack."""

    index: int
    role: str
    compress_ratio: int
    kv_source: bool          # carries `attn.compressor.*` and feeds the global KV cache
    indexer: bool            # carries `attn.indexer.*`
    engram_table: "int | None"   # the table this layer reads, or None

    @property
    def engram(self) -> bool:
        return self.engram_table is not None

    @property
    def swa_only(self) -> bool:
        """No compression at all: a sliding window, and the other rotary table (`rope`)."""
        return self.compress_ratio == 0

    def rope(self, cfg: dict) -> "tuple[int, float]":
        """(original_seq_len, theta) for this layer's rotary table; original 0 means YaRN off."""
        scaling = cfg.get("rope_scaling") or {}
        if self.swa_only:
            return 0, float(cfg["rope_theta"])
        return int(scaling.get("original_max_position_embeddings", 0)), float(cfg["compress_rope_theta"])

    def layer(self) -> Layer:
        """This layer as the composition's: the mixer its compression chooses, the router every layer runs, and the
        engram injection where the config puts a table (see the module docstring on that placement)."""
        return Layer(WINDOW_ATTENTION if self.swa_only else SPARSE_ATTENTION, MOE,
                     (ENGRAM,) if self.engram else ())


def layer_plans(cfg: dict) -> "list[LayerPlan]":
    """The per-layer plan, and the checks that make it a plan rather than a guess: each one is a way the config could
    be read wrong, and each fires at load rather than at inference."""
    n = int(cfg["num_hidden_layers"])
    n_mtp = int(cfg.get("num_nextn_predict_layers", 0))
    ratios = list(cfg["compress_ratios"])
    if len(ratios) != n + n_mtp:
        raise ValueError(f"compress_ratios has {len(ratios)} entries; expected {n} layers + {n_mtp} MTP = {n + n_mtp}. "
                         "The tail belongs to the MTP heads, so a mismatch means the two are no longer laid out end to "
                         "end and every index below is off.")
    main = ratios[:n]
    boundary = int(cfg["candidate_source_layer_id"])
    kv_src, idx_src = set(cfg["kv_source_layer_ids"]), set(cfg["index_source_layer_ids"])
    engram_ids = list(cfg.get("engram_layer_ids") or ())

    # one rule, not two: nothing past the boundary sources, so the boundary layer is the only decoder-side source
    # there can be -- and that source IS the projection of the final encoder states
    beyond = sorted(layer for layer in kv_src if layer > boundary)
    if beyond:
        raise ValueError(f"kv_source_layer_ids {sorted(kv_src)} reach past the encoder/decoder boundary {boundary} at "
                         f"{beyond}. In a CED stack the decoder consumes the encoder's KV, and at most the boundary "
                         "layer may source because that one alone reads the encoder's final output; a source beyond it "
                         "means this is not the layout assumed here.")
    encoder_ratio = main[boundary - 1] if boundary else None
    decoder_ratio = main[boundary] if boundary < n else None
    if encoder_ratio == decoder_ratio:
        raise ValueError(f"compress_ratios does not step at layer {boundary} ({encoder_ratio} -> {decoder_ratio}); the "
                         "boundary the config names and the one the ratios show disagree.")
    if any(i >= n for i in engram_ids):
        raise ValueError(f"engram_layer_ids {engram_ids} name a layer outside the {n}-layer stack")

    return [LayerPlan(index=i, role=ENCODER if i < boundary else DECODER, compress_ratio=main[i],
                      kv_source=i in kv_src, indexer=i in idx_src,
                      engram_table=engram_ids.index(i) if i in engram_ids else None)
            for i in range(n)]


def plan(cfg: dict) -> Plan:
    """The composition's plan for text config `cfg`."""
    return Plan(tuple(p.layer() for p in layer_plans(cfg)))


def describe(plans: "list[LayerPlan]") -> str:
    enc = [p for p in plans if p.role == ENCODER]
    dec = [p for p in plans if p.role == DECODER]
    return (f"{len(plans)} layers: encoder {len(enc)} (compress {sorted({p.compress_ratio for p in enc})}), "
            f"decoder {len(dec)} (compress {sorted({p.compress_ratio for p in dec})}); "
            f"kv sources {[p.index for p in plans if p.kv_source]}; "
            f"indexers {[p.index for p in plans if p.indexer]}; "
            f"engram {[p.index for p in plans if p.engram]}")


__all__ = ["ENCODER", "DECODER", "WINDOW_ATTENTION", "SPARSE_ATTENTION", "MOE", "ENGRAM",
           "LayerPlan", "layer_plans", "plan", "describe"]
