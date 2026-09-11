"""GLM-5.3-Flash, stated once (profile): every constant the engine's GLM code
leans on, read from the checkpoint's config.json and the launcher, and
CHECKED at load -- a checkpoint that differs in any of them kills the boot
(D3) instead of running a slightly different model.

The served path derives these through vLLM's Glm5NextTextConfig (mhc
post_mult 2.0, sinkhorn 20, mla_nope, is_kda_layer ...); the values below
were read off that object in the judge image (44th ledger) and are asserted
here against the raw config so the two cannot drift apart silently.

Launcher facts (start-glm53-nvfp4-tp4.sh, fleet public defaults): TP=4,
routed experts TP-sharded (ENABLE_EP=0: "the TP-sharded path is the measured
one"; EP is EXP-1, an experiment), block 2304, KV fp8_e4m3, DFlash2 drafter
with SPEC_K=5 -- so the checkpoint's MTP block (layer 45, fp8 experts) is
NOT served and is not part of this profile.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CKPT = Path("/home/choiceoh/models/glm53-redhat-nvfp4")
RANKS = Path("/home/choiceoh/models/glm53-redhat-nvfp4-tp4")          # preshard output, one file per rank
BLOCK = 2304                                                          # launcher --block-size (shapes.py's 6,912 law)
SPEC_K = 5                                                            # DFlash2 draft slots per decode step
KV_DTYPE = "fp8_e4m3"                                                 # launcher KV_DTYPE
EXPERTS = "tp"                                                        # launcher ENABLE_EP=0


@dataclass(frozen=True)
class Facts:
    hidden: int
    layers: int
    kinds: tuple                    # per layer: "kda" | "dsa"
    dense: tuple                    # layer ids with a dense MLP (first_k_dense_replace)
    vocab: int
    rms_eps: float
    # KDA (linear attention)
    kda_heads: int
    kda_dim: int
    conv: int
    lower_bound: float
    # DSA (sparse MLA, nope)
    heads: int
    qk_nope: int
    v_dim: int
    q_lora: int
    kv_lora: int
    idx_heads: int
    idx_dim: int
    topk: int
    kpool: int
    # MoE
    experts: int
    topk_experts: int
    moe_inter: int
    dense_inter: int
    routed_scale: float
    swiglu_limit: float
    # mHC
    hc: int
    hc_eps: float
    sinkhorn: int
    post_mult: float
    # serving
    block: int = BLOCK
    spec_k: int = SPEC_K

    @property
    def dsa_layers(self) -> "list[int]":
        return [i for i, k in enumerate(self.kinds) if k == "dsa"]

    @property
    def kda_layers(self) -> "list[int]":
        return [i for i, k in enumerate(self.kinds) if k == "kda"]

    def is_dsa(self, i: int) -> bool:
        return self.kinds[i] == "dsa"

    def is_moe(self, i: int) -> bool:
        return i not in self.dense

    @property
    def mla_scale(self) -> float:
        return self.qk_nope ** -0.5            # rope 0 and no yarn: no mscale

    @property
    def idx_scale(self) -> float:
        return self.idx_dim ** -0.5 * self.idx_heads ** -0.5   # softmax_scale * n_head**-0.5, folded once


def load(ckpt: "str | Path" = CKPT) -> Facts:
    c = json.loads((Path(ckpt) / "config.json").read_text())
    t = c.get("text_config", c)
    la = t["linear_attn_config"]
    n = t["num_hidden_layers"]
    kinds = tuple("dsa" if i in set(la["full_attn_layers"]) else "kda" for i in range(n))
    f = Facts(
        hidden=t["hidden_size"], layers=n, kinds=kinds,
        dense=tuple(range(t["first_k_dense_replace"])), vocab=t["vocab_size"], rms_eps=t["rms_norm_eps"],
        kda_heads=la["num_heads"], kda_dim=la["head_dim"], conv=la["short_conv_kernel_size"],
        lower_bound=t["linear_lower_bound"],
        heads=t["num_attention_heads"], qk_nope=t["qk_nope_head_dim"], v_dim=t["v_head_dim"],
        q_lora=t["q_lora_rank"], kv_lora=t["kv_lora_rank"],
        idx_heads=t["index_n_heads"], idx_dim=t["index_head_dim"], topk=t["index_topk"], kpool=t["index_kpool"],
        experts=t["n_routed_experts"], topk_experts=t["num_experts_per_tok"], moe_inter=t["moe_intermediate_size"],
        dense_inter=t["intermediate_size"], routed_scale=t["routed_scaling_factor"], swiglu_limit=t["swiglu_limit"],
        hc=t["hc_mult"], hc_eps=t["hc_eps"], sinkhorn=t["hc_sinkhorn_iters"], post_mult=2.0,
    )
    # -- what the code assumes, checked against the checkpoint (D3) ----------
    assert t["model_type"] == "glm5_next_text", t["model_type"]
    assert t["qk_rope_head_dim"] == 0 and t["mla_use_nope"] and t.get("rope_parameters") is None, "this profile is nope-only: no rotary anywhere"
    assert t["head_dim"] == 0 and t["num_key_value_heads"] == f.heads
    assert sorted(la["kda_layers"] + la["full_attn_layers"]) == list(range(n))
    assert t["mhc"] and f.hc == 4 and la.get("gate_lower_bound", f.lower_bound) == f.lower_bound
    assert t["topk_method"] == "noaux_tc" and t["scoring_func"] == "sigmoid" and t["norm_topk_prob"]
    assert t["n_group"] == 1 and t["topk_group"] == 1 and t["moe_router_dtype"] == "float32"
    assert t["n_shared_experts"] == 1 and t["hidden_act"] == "silu"
    assert t["index_kpool_compress"] and t["index_kpool_always_select_tail"] and t["indexer_rope_interleave"]
    assert f.topk % f.kpool == 0 and f.block % f.kpool == 0 and f.idx_dim == 128, "kpool pools of 4 tile the block; FWHT is 128-wide"
    assert not t["tie_word_embeddings"]
    q = c["quantization_config"]["config_groups"]["group_0"]
    assert q["format"] == "nvfp4-pack-quantized" and q["weights"]["group_size"] == 16 and q["input_activations"]["group_size"] == 16
    assert q["targets"] == ["re:.*\\.layers\\.(?:[3-9]|[1-3][0-9]|4[0-4])\\.mlp\\.experts\\..*(gate|up|down)_proj$"], "NVFP4 is exactly the routed experts of layers 3-44"
    assert f.kda_heads % 4 == 0 and f.heads % 4 == 0 and f.moe_inter % (4 * 16) == 0 and f.dense_inter % 4 == 0, "TP=4 splits"
    return f


def _selfcheck() -> None:
    f = load()
    assert f.layers == 45 and len(f.dsa_layers) == 11 and f.dsa_layers[:3] == [3, 7, 11] and len(f.kda_layers) == 34
    assert f.dense == (0, 1, 2) and f.is_moe(3) and not f.is_moe(2) and f.is_dsa(3) and not f.is_dsa(4)
    assert (f.hidden, f.heads, f.kv_lora, f.q_lora, f.experts, f.topk_experts) == (4096, 64, 512, 1536, 288, 8)
    assert f.mla_scale == 256 ** -0.5 and abs(f.idx_scale - 128 ** -0.5 * 32 ** -0.5) < 1e-12
    print(f"  facts: glm53 {f.layers} layers ({len(f.kda_layers)} kda + {len(f.dsa_layers)} dsa), dense {f.dense}, "
          f"{f.experts}x top-{f.topk_experts} NVFP4 experts ({EXPERTS}), hc {f.hc} post_mult {f.post_mult}, "
          f"topk {f.topk}/kpool {f.kpool}, block {f.block}, spec {f.spec_k} OK")


if __name__ == "__main__":
    _selfcheck()
