"""Qwen3.8-Flash-Next served at TP=4, stated once (profile): every constant the served path leans on, read from the
checkpoint's config.json and CHECKED at load -- a checkpoint that differs in any of them kills the boot (D3).

The served layout is TEP=4 (profiles/qwen38.env, the vLLM stack that answered on this fleet): attention and GDN
tensor-parallel by heads, routed experts EXPERT-parallel (a rank holds 128 whole experts of 512, so the NVFP4
intermediate stays 640 and tiles; split four ways it would be 160, which the FP4 lanes cannot tile), the shared
expert TP-sharded, the PLE table vocabulary-parallel, hyper-connection weights replicated.

engine/profiles/qwen38/composition.py assembles the same model from engine/modules for the reference lane; this file
is what the served lane (specs, caches, net) sizes itself from. The kernel shape the wizard judges is derived by
shapes.kernel_shape from the same config (tests/test_engine_kernel_shape pins it), so `kernel_shape()` delegates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from engine.base.box import BOX, check_box  # noqa: F401 -- the box is the engine's (D5); boot calls facts.check_box

CKPT = Path("/home/choiceoh/models/qwen38-flash-next-nvfp4")
RANKS = Path("/home/choiceoh/models/st-qwen38-tep4")               # preshard output: rank{r}of4.safetensors + metadata
TP = 4                                                             # four Sparks: the only world this profile has
CHUNK_ALIGN = 2304                                                 # the prefill chunk's alignment: whole blocks (D9's 2,304)
BLOCK = 768                                                        # paged KV / prefix block: 64-aligned for the GDN kernel's
                                                                   # chunks, whole QSA compression groups of 4
SPEC_K = 1                                                         # the checkpoint's one MTP layer drafts one token a step
                                                                   # (fleet --spec-k K chains it: decode_graphs.draft_chain)
KV_DTYPE = "bf16"                                                  # QSA K/V rows as the checkpoint computes them (no latent fp8)
GDN_STATE_DTYPE = "fp32"                                           # the delta rule's state, as the reference keeps it
EXPERTS = "ep"                                                     # 128 whole experts a rank (profiles/qwen38.env)
WEIGHT_LAYOUT = "st-qwen38-tep4-modelopt-v3"                       # rank files this profile reads (preshard writes it);
                                                                   # v3: the PLE table left the rank file for its SSD file
PLE_FILE = "ple-r{rank}of{world}.weight"                           # the rank's PLE table beside its rank file (raw e4m3 rows)
PLE_SIDECAR = "ple-r{rank}of{world}.json"                          # what that file is: rows, width, shards, scale, sha256


def ple_file(rank: int, world: int = TP) -> str:
    return PLE_FILE.format(rank=rank, world=world)


def ple_sidecar(rank: int, world: int = TP) -> str:
    return PLE_SIDECAR.format(rank=rank, world=world)


@dataclass(frozen=True)
class Facts:
    hidden: int
    layers: int
    kinds: tuple                    # per layer: "gdn" | "qsa"
    vocab: int
    rms_eps: float
    max_position: int
    eos: int
    # gated residual streams
    hc: int
    hc_rank: int
    # GatedDeltaNet
    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv: int
    # gated GQA attention with QSA selection
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    idx_heads: int
    idx_dim: int
    idx_budget: int
    idx_ratio: int
    # MoE
    experts: int
    topk_experts: int
    moe_inter: int
    shared_inter: int
    # PLE (hashed n-gram table), zero-indexed layers it is injected before
    ple_layers: tuple
    ngram_size: int
    heads_per_ngram: int
    ple_dim: int
    ple_conv: int
    ngram_base: int
    ngram_parts: int
    seed: int
    # MTP
    mtp_layers: int
    # what the export keeps (load() reads them off its quantization config; the preshard checks the files)
    mtp_experts: str = "bf16"       # "bf16": fused [E, 2I, H] / [E, H, I] | "fp8_block": per expert e4m3 + weight_scale_inv
    mtp_block: int = 0              # the fp8 block (128 in NVIDIA's export); 0 with bf16 experts
    ngram_divisible: int = 128      # the table's rows are padded to a multiple (make_ngram_vocab_size_divisible_by)
    # serving
    block: int = BLOCK
    chunk_align: int = CHUNK_ALIGN
    spec_k: int = SPEC_K
    kv_dtype: str = KV_DTYPE
    gdn_state_dtype: str = GDN_STATE_DTYPE
    weight_layout: str = WEIGHT_LAYOUT
    config: dict = field(default=None, compare=False, hash=False, repr=False)

    # -- what one of the four ranks holds -----------------------------------------------------------------------------
    @property
    def k_heads_local(self) -> int:
        return self.k_heads // TP

    @property
    def v_heads_local(self) -> int:
        return self.v_heads // TP

    @property
    def heads_local(self) -> int:
        return self.heads // TP

    @property
    def kv_heads_local(self) -> int:
        return max(1, self.kv_heads // TP)

    def kv_head_of(self, rank: int) -> int:
        """The KV head rank `rank` keeps: fewer KV heads than ranks are replicated, query heads 6r..6r+5 read head r//2."""
        return rank * self.kv_heads // TP

    @property
    def experts_local(self) -> int:
        return self.experts // TP

    def expert_range(self, rank: int) -> "tuple[int, int]":
        n = self.experts_local
        return rank * n, (rank + 1) * n

    @property
    def shared_inter_local(self) -> int:
        return self.shared_inter // TP

    @property
    def vocab_local(self) -> int:
        return self.vocab // TP

    @property
    def qkv_width(self) -> int:
        """GDN's fused q|k|v rows on the whole model: 2 key halves and the value half."""
        return 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim

    @property
    def qkv_local(self) -> int:
        return self.qkv_width // TP

    @property
    def gdn_layers(self) -> "list[int]":
        return [i for i, k in enumerate(self.kinds) if k == "gdn"]

    @property
    def qsa_layers(self) -> "list[int]":
        return [i for i, k in enumerate(self.kinds) if k == "qsa"]

    def is_qsa(self, i: int) -> bool:
        return self.kinds[i] == "qsa"

    @property
    def ple_head_dim(self) -> int:
        return self.ple_dim // self.ple_heads

    @property
    def ple_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ple_rows_total(self) -> int:
        """Rows of the whole hashed n-gram table: the heads' bucket counts (consecutive primes from ngram_base, the
        hash's own head_tables) padded to the checkpoint's multiple -- 320,001,536 for the served checkpoint."""
        from engine.modules.ngram_embedding import head_tables
        _, _, total = head_tables(self.ngram_size, self.heads_per_ngram, self.ngram_base, 0)
        return -(-total // self.ngram_divisible) * self.ngram_divisible

    @property
    def ple_rows_per_shard(self) -> int:
        """Rows of one of the checkpoint's `ngram_embedding.shard_N` tensors (2,500,012 for the served checkpoint)."""
        rows = self.ple_rows_total
        if rows % self.ngram_parts:
            raise ValueError(f"the table's {rows} rows do not split into {self.ngram_parts} shards")
        return rows // self.ngram_parts

    @property
    def ple_rows_per_rank(self) -> int:
        """Rows of one rank's vocabulary range: its ngram_parts // TP shards (the table is vocabulary-parallel)."""
        return self.ple_rows_per_shard * (self.ngram_parts // TP)

    @property
    def attention_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def index_scale(self) -> float:
        return self.idx_dim ** -0.5

    @property
    def index_blocks(self) -> int:
        """Compressed blocks a query may attend: the budget in whole groups of `idx_ratio` positions."""
        return self.idx_budget // self.idx_ratio

    def kernel_shape(self) -> "KernelShape":
        """This checkpoint's kernel shape per rank at TP=4, as the wizard derives it (engine/profiles/qwen38/shapes)."""
        from engine.profiles.qwen38 import shapes
        return shapes.kernel_shape(self.config, TP, self.spec_k)


def kernel_shape_of(ckpt: "str | Path" = CKPT) -> "KernelShape":
    return load(ckpt).kernel_shape()


def text_config(ckpt: "str | Path" = CKPT) -> dict:
    return json.loads((Path(ckpt) / "config.json").read_text())["text_config"]


def load(ckpt: "str | Path" = CKPT) -> Facts:
    c = json.loads((Path(ckpt) / "config.json").read_text())
    q = c["quantization_config"]
    if q.get("quant_method") != "modelopt":
        raise ValueError("this profile serves NVIDIA's ModelOpt NVFP4 checkpoint")
    g = q["config_groups"]["group_0"]
    for side in ("weights", "input_activations"):
        scheme = g[side]
        if scheme["num_bits"] != 4 or scheme["type"] != "float" or scheme["group_size"] != 16:
            raise ValueError("NVFP4 group 16 for weights and activations")
    return architecture(c, **quantised(c, q))


def quantised(c: dict, q: dict) -> dict:
    """What the export quantised, as the Facts fields it fixes. NVFP4 must be exactly the routed experts, and two
    exports say so two ways: the older copy (`quant_algo` NVFP4) by the glob patterns it ignores; NVIDIA's Hub
    revision fc694b54 (MIXED_PRECISION) by naming every quantised module -- the 48 layers' experts NVFP4 group 16, the
    PLE table FP8 (the e4m3 rows under one scalar scale the loader reads either way) and the MTP head's experts FP8
    with block scales (specs.py dequantises them before their NVFP4 encoding)."""
    algo = q.get("quant_algo")
    t = c.get("text_config", c)
    if algo == "NVFP4":
        excluded = set(q.get("ignore") or q.get("exclude_modules") or ())
        for pattern in ("*.self_attn.*", "*.linear_attn.*", "*.mlp.gate*", "*.mlp.shared_expert.*", "*hyper_connection*",
                        "*.ple.*", "lm_head"):
            if pattern not in excluded:
                raise ValueError(f"NVFP4 is exactly the routed experts: the config quantises {pattern}")
        return dict(mtp_experts="bf16", mtp_block=0)
    if algo != "MIXED_PRECISION":
        raise ValueError(f"quant_algo {algo!r}: this profile reads the NVFP4 export or NVIDIA's MIXED_PRECISION one")
    layers = dict(q.get("quantized_layers") or {})
    for L in range(len(t["layer_types"])):
        name = f"model.language_model.layers.{L}.mlp.experts"
        spec = layers.pop(name, None)
        if spec is None or spec.get("quant_algo") != "NVFP4" or spec.get("group_size") != 16:
            raise ValueError(f"{name}: NVFP4 group 16 expected, the config says {spec}")
    for i in t.get("ple_layer_ids") or ():
        name = f"model.language_model.layers.{i - 1}.ple.ple_embedding.ngram_embedding"
        spec = layers.pop(name, None)
        if spec is None or spec.get("quant_algo") != "FP8":
            raise ValueError(f"{name}: the PLE table is FP8 under one scale, the config says {spec}")
    mtp = layers.pop("mtp.layers.0.mlp.experts", None)
    # the same tensors under two names: config.json says FP8_PB_WO (per-block, weight-only), hf_quant_config.json
    # FP8_BLOCK_SCALES -- e4m3 [out, in] with a BF16 [out/128, in/128] weight_scale_inv either way
    if mtp is None or mtp.get("quant_algo") not in ("FP8_BLOCK_SCALES", "FP8_PB_WO") or not mtp.get("group_size"):
        raise ValueError(f"mtp.layers.0.mlp.experts: FP8 block scales expected, the config says {mtp}")
    if layers:
        raise ValueError(f"the config quantises modules this profile does not read: {sorted(layers)[:4]}")
    ignored = set(q.get("ignore") or ())
    for name in ("lm_head", "model.language_model.embed_tokens"):
        if name not in ignored:
            raise ValueError(f"{name} must be left unquantised")
    return dict(mtp_experts="fp8_block", mtp_block=int(mtp["group_size"]))


def architecture(c: dict, *, mtp_experts: str = "bf16", mtp_block: int = 0) -> Facts:
    t = c.get("text_config", c)
    types = t["layer_types"]
    kinds = tuple("qsa" if k == "full_attention" else "gdn" for k in types)
    rope = t.get("rope_parameters") or {}
    rotary = int(t["head_dim"] * rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0)))
    eos = t.get("eos_token_id")
    config = dict(t)
    f = Facts(
        hidden=t["hidden_size"], layers=len(types), kinds=kinds, vocab=t["vocab_size"], rms_eps=t["rms_norm_eps"],
        max_position=int(t["max_position_embeddings"]), eos=int(eos[0] if isinstance(eos, list) else eos),
        hc=t["hc_count"], hc_rank=t["hc_lowrank"],
        k_heads=t["linear_num_key_heads"], v_heads=t["linear_num_value_heads"], k_dim=t["linear_key_head_dim"],
        v_dim=t["linear_value_head_dim"], conv=t["linear_conv_kernel_dim"],
        heads=t["num_attention_heads"], kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"], rotary_dim=rotary,
        rope_theta=float(rope.get("rope_theta", t.get("rope_theta"))),
        idx_heads=t["indexer_n_heads"], idx_dim=t["indexer_head_dim"], idx_budget=t["indexer_budget"],
        idx_ratio=t["indexer_compress_ratio"],
        experts=t["num_experts"], topk_experts=t["num_experts_per_tok"], moe_inter=t["moe_intermediate_size"],
        shared_inter=t["shared_expert_intermediate_size"],
        ple_layers=tuple(i - 1 for i in (t.get("ple_layer_ids") or ())), ngram_size=t["ngram_size"],
        heads_per_ngram=t["heads_per_ngram"], ple_dim=t["ple_embed_dim"], ple_conv=t["ple_conv_kernel_size"],
        ngram_base=t["ngram_vocab_size_base"], ngram_parts=t["split_ngram_parts"], seed=int(t.get("seed", 1234)),
        mtp_layers=int(t.get("mtp_num_hidden_layers") or 0), mtp_experts=mtp_experts, mtp_block=int(mtp_block),
        ngram_divisible=int(t.get("make_ngram_vocab_size_divisible_by") or 128), config=config,
    )
    if mtp_experts not in ("bf16", "fp8_block") or (mtp_experts == "fp8_block") != (f.mtp_block > 0):
        raise ValueError(f"MTP experts {mtp_experts!r} with block {f.mtp_block}")
    if f.mtp_block and (f.hidden % f.mtp_block or f.moe_inter % f.mtp_block):
        raise ValueError(f"the MTP experts' fp8 block {f.mtp_block} does not tile hidden {f.hidden} x inter {f.moe_inter}")
    # -- what the served code assumes, checked against the checkpoint (D3) --------------------------------------------
    assert t["model_type"] == "qwen4_exp_text", t["model_type"]
    assert set(types) == {"linear_attention", "full_attention"}
    assert all((k == "full_attention") == (i % t["full_attention_interval"] == t["full_attention_interval"] - 1)
               for i, k in enumerate(types)), "a full attention layer closes every interval"
    assert f.hc == 4 and t["hidden_act"] == "silu" and t["output_gate_type"] == "sigmoid" and t.get("norm_topk_prob", True)
    assert rope.get("mrope_interleaved") and sum(rope["mrope_section"]) * 2 == f.rotary_dim, \
        "text positions make the interleaved mrope an ordinary rotary over the first rotary_dim channels"
    assert t["indexer_kv_heads"] == 1 and f.idx_budget % f.idx_ratio == 0 and 2 * (f.idx_dim // 2) == f.idx_dim
    assert f.ple_layers == (1,) and f.ple_dim == f.hidden and f.ple_dim % f.ple_heads == 0, "PLE before layer 1"
    assert f.mtp_layers == 1 and not t.get("mtp_use_dedicated_embeddings") and not t["tie_word_embeddings"]
    assert f.shared_inter == f.moe_inter and f.moe_inter % 128 == 0, "EP keeps the whole 640 intermediate: tiles of 128"
    assert f.v_heads % f.k_heads == 0 and f.k_dim == f.v_dim, "value heads repeat key heads; square delta state"
    assert (f.k_heads % TP == 0 and f.v_heads % TP == 0 and f.heads % TP == 0 and f.experts % TP == 0
            and f.vocab % TP == 0 and TP % f.kv_heads == 0 and f.shared_inter % TP == 0), "TEP=4 splits"
    assert f.block % 64 == 0 and f.block % f.idx_ratio == 0 and f.chunk_align % f.block == 0, \
        "a block is whole GDN kernel chunks and whole QSA groups; a prefill chunk is whole blocks"
    return f


def _selfcheck() -> None:
    f = load()
    assert f.layers == 48 and len(f.qsa_layers) == 12 and f.qsa_layers[:2] == [3, 7] and len(f.gdn_layers) == 36
    assert (f.k_heads_local, f.v_heads_local, f.heads_local, f.kv_heads_local) == (4, 12, 6, 1)
    assert (f.experts_local, f.expert_range(3), f.shared_inter_local, f.vocab_local) == (128, (384, 512), 160, 62080)
    assert (f.qkv_width, f.qkv_local, f.rotary_dim, f.index_blocks) == (10240, 2560, 64, 512)
    assert [f.kv_head_of(r) for r in range(TP)] == [0, 0, 1, 1]
    assert (f.ple_rows_per_shard, f.ple_rows_per_rank) == (2_500_012, 80_000_384), "the checkpoint's shard header"
    print(f"  facts: qwen38 {f.layers} layers ({len(f.gdn_layers)} gdn + {len(f.qsa_layers)} qsa), hc {f.hc} rank "
          f"{f.hc_rank}, {f.experts}x top-{f.topk_experts} NVFP4 experts ({EXPERTS}, {f.experts_local} a rank), "
          f"PLE before {f.ple_layers} ({f.ple_rows_per_rank:,} rows a rank on the SSD), MTP experts {f.mtp_experts}, "
          f"block {f.block}, spec {f.spec_k} OK")


if __name__ == "__main__":
    _selfcheck()
