"""DeepSeek-V4.1-Flash as a vLLM architecture, derived from this image's V4.

V4.1 is not a new family. Of the 19 config attributes `deepseek_v4/nvidia/
model.py` reads, V4.1's `text_config` supplies 16 unchanged; the rest of the
diff against V4 is sizes (hidden 4096 -> 5120, 256 -> 384 experts, 43 -> 40
layers, q_lora 1024 -> 1280) that the V4 model already reads from config. So
this registers a new architecture and adapts, rather than reimplementing.

What V4.1 actually adds, and where each is handled:

  engram          two [384,006,168 x 256] lookup tables at layers 1 and 14,
                  188.8 GiB. They do not live in GPU memory at all -- see
                  dsv41_engram, which reads rows off an SSD. Nothing in the V4
                  model knows about them, so they are the one genuinely new
                  structure.
  CED sources     V4 derives which layers source KV and run the indexer from
                  `compress_ratios`; V4.1 states them outright in
                  `kv_source_layer_ids` and `index_source_layer_ids`. Stated
                  wins where both exist -- a derivation that happens to agree
                  on this config is still a derivation.
  candidate_*     block-sparse candidates (2048 blocks of 8) at the CED
                  boundary layer 20. dsv41_packed_index holds that arithmetic.
  DSpark          3 stages instead of V4's 1, under the same `mtp.*` names.

Three attributes the V4 model reads are absent from V4.1's config:

  expert_dtype     present, but in the OUTER `quantization_config` rather than
                   `text_config`. The V4 model getattr-defaults it to "fp4",
                   which is right here, but defaulting to the right answer by
                   accident is not the same as reading it, so it is copied
                   across explicitly and a disagreement aborts.
  num_hash_layers  V4 hashed its first `num_hash_layers` MoE layers. V4.1 has
                   no hash MoE -- engram replaced it -- so 0 is the value, and
                   it is set rather than left to AttributeError at layer 0.
  virtual_tp       never in either config; a runtime knob, read through
                   getattr helpers.

Not yet handled here, and each fails loudly rather than silently: the engram
layers, the DSpark stages, and the vision tower.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The V4 model's own gate. V4.1's hidden 5120 is not in the image's
# b12x MHC_SUPPORTED_HIDDEN_SIZES (4096, 7168), so the b12x mHC path must stay
# off until that kernel is generalized -- which is a kernel change, not a model
# one. VLLM_USE_B12X_MHC=0 is the boot-time expression of that.
V41_UNSUPPORTED_BY_B12X_MHC = 5120


def adapt_text_config(hf_config):
    """Give V4.1's text config the surface the V4 model reads. Idempotent."""
    text = getattr(hf_config, "text_config", None) or hf_config

    if not hasattr(text, "num_hash_layers"):
        # `is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers`
        # is a bare attribute read, so its absence is an AttributeError at the
        # first decoder layer rather than a fallback.
        text.num_hash_layers = 0

    quant = getattr(hf_config, "quantization_config", None) or {}
    stated = quant.get("expert_dtype") if isinstance(quant, dict) else None
    if stated is not None:
        current = getattr(text, "expert_dtype", None)
        if current is not None and current != stated:
            raise ValueError(
                f"expert_dtype disagrees: text_config says {current!r} and "
                f"quantization_config says {stated!r}. One of them decides "
                f"which MoE kernel runs and which weight layout is expected; "
                f"picking either silently is how a model loads and computes "
                f"garbage.")
        text.expert_dtype = stated

    return text


def layer_roles(text_config):
    """Which layers source KV and which run the indexer, from the config.

    V4.1 states both outright. Returning frozensets rather than lists is so a
    membership test cannot be written as an accidental `in` over a list of
    ratios, which is how the V4 derivation reads.
    """
    kv = frozenset(getattr(text_config, "kv_source_layer_ids", ()) or ())
    idx = frozenset(getattr(text_config, "index_source_layer_ids", ()) or ())
    if not kv or not idx:
        raise ValueError(
            "V4.1 states kv_source_layer_ids and index_source_layer_ids in "
            "its config; an empty one means the config is not V4.1's and the "
            "CED split would be guessed from compress_ratios instead.")
    return kv, idx


def engram_layers(text_config):
    """The layer ids carrying an engram table, and their row counts."""
    ids = tuple(getattr(text_config, "engram_layer_ids", ()) or ())
    rows = tuple(getattr(text_config, "engram_num_embeddings", ()) or ())
    if len(ids) != len(rows):
        raise ValueError(
            f"engram_layer_ids {ids} and engram_num_embeddings {rows} must "
            f"line up: each table's row count belongs to exactly one layer.")
    return dict(zip(ids, rows))
