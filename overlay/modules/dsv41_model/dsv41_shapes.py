"""Every tensor DeepSeek-V4.1-Flash carries, predicted from config.

dsv41_layers says which layers carry which components. This says what those
components ARE: 96,085 tensors, each with a dtype and a shape, derived from
config.json and nothing else. probes/dsv41_shape_plan.py holds it to the
checkpoint's index, name for name and shape for shape.

Predicting is the point. A shape read off the checkpoint cannot be wrong and
cannot be checked either; a shape derived from `n_heads * head_dim` is a claim
about what the projection DOES, and getting it wrong is how a model loads and
computes nonsense.

## Three quantization granularities, not one

  fp8 dense      F8_E4M3 weight [out, in], F8_E8M0 scale [out/32, in/32] --
                 a [32, 32] block, which is `weight_block_size` in the config.
  fp4 routed     I8 weight [out, in/2] (two e2m1 per byte) with F8_E8M0 scale
                 [out, in/32] -- blocked along K ONLY. The shared expert sits
                 beside it in the same FFN and is fp8, so the two differ in
                 both dtype and scale rank; a loader that treats them alike
                 reads the shared expert's scale as if it had a row per output.
  bf16           norms, gates, the mHC coefficients, the vision tower.

## Three asymmetries the layer list alone does not show

  the indexer splits. Its QUERY side (`wq_b`, `weights_proj`) is on all eight
    index_source_layer_ids; its KEY side (`wk`, `k_norm`) only on the four
    kv_source_layer_ids, because the keys come from the KV a compressor
    produced and only those four produce any.
  the compressor's gate follows compress_ratio, not the boundary. `wgate` is
    on [2, 8, 14] and not on 20, and the rule is `compress_ratio > 1`: a
    ratio-1 compressor pools nothing and needs no gate. Layer 20 is ratio 1
    because the decoder half is, which is the CED split again -- but "not the
    boundary" would predict the same tensors here for a reason that is not the
    reason, and would be wrong on a config where the ratios stepped elsewhere.
    The reference also promotes wkv and wgate to fp32 above ratio 1 while the
    checkpoint stores bf16, so that promotion is a load-time step, not a
    storage dtype.
  the DSpark block is a block, not three layers. mtp.0 carries the entry
    (`main_norm`, `main_proj` [5120, 15360] = three hidden states wide, one per
    dspark_target_layer_id) and mtp.2 the exit (`norm`, `markov_head`,
    `confidence_head` [1, 5376] = hidden + markov head dim). The three layers
    between them are one draft block, which is what "semi-autoregressive" means
    here.
"""

from __future__ import annotations

FP8_BLOCK = 32


def _blk(n: int) -> int:
    return (n + FP8_BLOCK - 1) // FP8_BLOCK


def _fp8(out: int, inn: int, prefix: str, into: dict) -> None:
    """A dense fp8 linear: [32, 32]-blocked scale."""
    into[prefix + ".weight"] = ("F8_E4M3", [out, inn])
    into[prefix + ".scale"] = ("F8_E8M0", [_blk(out), _blk(inn)])


def _fp4(out: int, inn: int, prefix: str, into: dict) -> None:
    """A routed expert: two e2m1 per byte, scale blocked along K only."""
    into[prefix + ".weight"] = ("I8", [out, inn // 2])
    into[prefix + ".scale"] = ("F8_E8M0", [out, _blk(inn)])


def _bf16(shape, prefix: str, into: dict) -> None:
    into[prefix] = ("BF16", list(shape))


def _attention(cfg, prefix: str, into: dict) -> None:
    h = cfg["hidden_size"]
    heads, hd = cfg["num_attention_heads"], cfg["head_dim"]
    q_lora, o_lora, o_groups = cfg["q_lora_rank"], cfg["o_lora_rank"], cfg["o_groups"]
    kv = cfg["num_key_value_heads"] * hd
    _fp8(q_lora, h, f"{prefix}.wq_a", into)
    _fp8(heads * hd, q_lora, f"{prefix}.wq_b", into)
    _fp8(kv, h, f"{prefix}.wkv", into)
    # the o-projection is grouped: each of o_groups slices of the head output
    # goes through its own low-rank pair
    _fp8(o_lora * o_groups, heads * hd // o_groups, f"{prefix}.wo_a", into)
    _fp8(h, o_lora * o_groups, f"{prefix}.wo_b", into)
    _bf16([q_lora], f"{prefix}.q_norm.weight", into)
    _bf16([kv], f"{prefix}.kv_norm.weight", into)
    into[f"{prefix}.attn_sink"] = ("F32", [heads])


def _mhc(cfg, prefix: str, into: dict) -> None:
    hc = cfg["hc_mult"]
    nout = hc * (2 + hc)
    for which in ("attn", "ffn"):
        into[f"{prefix}.hc_{which}_fn"] = ("F32", [nout, hc * cfg["hidden_size"]])
        into[f"{prefix}.hc_{which}_base"] = ("F32", [nout])
        into[f"{prefix}.hc_{which}_scale"] = ("F32", [3])


def _ffn(cfg, prefix: str, n_experts: int, into: dict) -> None:
    h, mi = cfg["hidden_size"], cfg["moe_intermediate_size"]
    _bf16([n_experts, h], f"{prefix}.gate.weight", into)
    into[f"{prefix}.gate.bias"] = ("F32", [n_experts])
    into[f"{prefix}.gate.bias_vl"] = ("F32", [n_experts])
    for e in range(n_experts):
        _fp4(mi, h, f"{prefix}.experts.{e}.w1", into)
        _fp4(h, mi, f"{prefix}.experts.{e}.w2", into)
        _fp4(mi, h, f"{prefix}.experts.{e}.w3", into)
    # the shared expert is fp8, not fp4, and its scale is [32, 32]-blocked
    _fp8(mi, h, f"{prefix}.shared_experts.w1", into)
    _fp8(h, mi, f"{prefix}.shared_experts.w2", into)
    _fp8(mi, h, f"{prefix}.shared_experts.w3", into)


def tensor_plan(cfg: dict, vision: "dict | None" = None) -> dict:
    """name -> (dtype, shape) for the whole checkpoint."""
    from dsv41_layers import plan_layers

    out: dict = {}
    h = cfg["hidden_size"]
    boundary = int(cfg["candidate_source_layer_id"])
    kv_src = set(cfg["kv_source_layer_ids"])
    idx_n, idx_d = cfg["index_n_heads"], cfg["index_head_dim"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]

    _bf16([cfg["vocab_size"], h], "embed.weight", out)
    _bf16([cfg["vocab_size"], h], "head.weight", out)
    _bf16([h], "norm.weight", out)
    for tok in ("image_start", "image_end", "image_newline"):
        _bf16([h], tok, out)

    for p in plan_layers(cfg):
        pre = f"layers.{p.index}"
        _attention(cfg, f"{pre}.attn", out)
        _mhc(cfg, pre, out)
        _ffn(cfg, f"{pre}.ffn", cfg["n_routed_experts"], out)
        _bf16([h], f"{pre}.attn_norm.weight", out)
        _bf16([h], f"{pre}.ffn_norm.weight", out)
        if p.indexer:
            # query side: every index source
            _fp8(idx_n * idx_d, cfg["q_lora_rank"], f"{pre}.attn.indexer.wq_b", out)
            _bf16([idx_n, h], f"{pre}.attn.indexer.weights_proj.weight", out)
        if p.kv_source:
            # key side: only where a compressor made KV to key against
            _bf16([idx_d, kv_dim], f"{pre}.attn.indexer.wk.weight", out)
            _bf16([idx_d], f"{pre}.attn.indexer.k_norm.weight", out)
            _bf16([kv_dim, h], f"{pre}.attn.compressor.wkv.weight", out)
            _bf16([kv_dim], f"{pre}.attn.compressor.norm.weight", out)
            if p.compress_ratio > 1:
                # The gate exists because there is something to pool: the
                # reference's Compressor builds wgate only above ratio 1, and
                # a ratio-1 compressor is a plain projection. On this config
                # that is layer 20 alone and "not the boundary" predicts the
                # same tensors -- for the wrong reason, and only by accident,
                # since the boundary layer is ratio 1 rather than ratio 1
                # BECAUSE it is the boundary.
                _bf16([kv_dim, h], f"{pre}.attn.compressor.wgate.weight", out)
        if p.engram:
            rows = cfg["engram_num_embeddings"][p.engram_table]
            cols = (cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"]
            hd_e = cfg["engram_head_dim"]
            out[f"{pre}.engram.embed.weight"] = ("F8_E4M3", [rows, hd_e])
            out[f"{pre}.engram.embed.scale"] = ("F8_E8M0", [rows, _blk(hd_e)])
            _bf16([cfg["hc_mult"], h], f"{pre}.engram.q_weight", out)
            _bf16([cfg["hc_mult"], h], f"{pre}.engram.k_weight", out)
            _fp8(h * (cfg["hc_mult"] + 1), cols * hd_e, f"{pre}.engram.wkv", out)

    n_mtp = int(cfg.get("num_nextn_predict_layers", 0))
    if n_mtp:
        mk = cfg["dspark_markov_rank"]
        for i in range(n_mtp):
            pre = f"mtp.{i}"
            _attention(cfg, f"{pre}.attn", out)
            _mhc(cfg, pre, out)
            _ffn(cfg, f"{pre}.ffn", cfg["dspark_n_routed_experts"], out)
            _bf16([h], f"{pre}.attn_norm.weight", out)
            _bf16([h], f"{pre}.ffn_norm.weight", out)
        # entry on the first, exit on the last: the three layers are one block
        _bf16([h], "mtp.0.main_norm.weight", out)
        _fp8(h, h * len(cfg["dspark_target_layer_ids"]), "mtp.0.main_proj", out)
        last = f"mtp.{n_mtp - 1}"
        _bf16([h], f"{last}.norm.weight", out)
        _bf16([cfg["vocab_size"], mk], f"{last}.markov_head.embed.weight", out)
        _bf16([cfg["vocab_size"], mk], f"{last}.markov_head.head.weight", out)
        _bf16([1, h + mk], f"{last}.confidence_head.proj.weight", out)

    if vision:
        vh, vi = vision["hidden_size"], vision["intermediate_size"]
        patch = vision["patch_size"]
        _bf16([vh, 3 * patch * patch], "vision.patch_embed.proj.weight", out)
        _bf16([vh], "vision.patch_embed.proj.bias", out)
        for b in range(vision["num_hidden_layers"]):
            p = f"vision.blocks.{b}"
            _bf16([3 * vh, vh], f"{p}.attn.wqkv.weight", out)
            _bf16([3 * vh], f"{p}.attn.wqkv.bias", out)
            _bf16([vh, vh], f"{p}.attn.wo.weight", out)
            _bf16([vh], f"{p}.attn.wo.bias", out)
            _bf16([2 * vi, vh], f"{p}.mlp.w1.weight", out)   # gated
            _bf16([vh, vi], f"{p}.mlp.w2.weight", out)
            _bf16([vh], f"{p}.norm1.weight", out)
            _bf16([vh], f"{p}.norm2.weight", out)
        _bf16([vh], "vision.norm.weight", out)
        # the aligner takes the downsampled patch grid into the text hidden
        _bf16([h, vh * vision["downsample_ratio"] ** 2], "aligner.w1.weight", out)
        _bf16([h], "aligner.w1.bias", out)
        _bf16([h, h], "aligner.w2.weight", out)
        _bf16([h], "aligner.w2.bias", out)
    return out
