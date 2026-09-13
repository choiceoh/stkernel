"""Qwen3.8-Flash-Next's shape constraints (profile), from config and the pinned files."""
from __future__ import annotations

from engine.base.shapes import Constraint, chunk_for as base_chunk_for
from engine.profiles.qwen38.plan import text_config


def constraints() -> "list[Constraint]":
    c = text_config()
    ratio = c["indexer_compress_ratio"]
    return [
        Constraint("indexer compress group", ratio, "config.indexer_compress_ratio",
                   "QSA compresses keys in groups of this many tokens; a chunk that ends "
                   "mid-group leaves a partial group the next chunk cannot finish "
                   "(common/qsa_cache.py keeps the tail)"),
        Constraint("chunk alignment", ratio, "the compress group",
                   "the raw-token multiple a prefill chunk must end on"),
        Constraint("conv history", c["linear_conv_kernel_dim"] - 1, "config.linear_conv_kernel_dim",
                   "GDN's causal conv reads this many previous tokens: carried in the conv "
                   "state slot across chunks, never re-read from tokens"),
        Constraint("n-gram history", c["ngram_size"] - 1, "config.ngram_size",
                   "PLE hashes this many previous tokens with the current one; the first "
                   "tokens of a chunk need the previous chunk's tail (model_state.py "
                   "_prepare_ngram_context)"),
        Constraint("indexer budget", c["indexer_budget"], "config.indexer_budget",
                   "compressed positions a query may attend to; the QSA cache row width"),
        Constraint("gdn heads per rank", c["linear_num_value_heads"] // 4, "num_v_heads / TP",
                   "the state slot's leading dim; 48 v-heads split 4 ways, 16 k-heads likewise"),
        Constraint("nvfp4 group", 16, "hf_quant_config.group_size",
                   "the expert GEMM's scale granularity along K; a K tile must be a multiple"),
        Constraint("hyper-connection width", c["hc_count"], "config.hc_count",
                   "the residual stream is hc_count copies of hidden; every kernel that touches "
                   "it sees [T, 4, 2560], not [T, 2560]"),
    ]


def chunk_for(token_budget: int, draft_slots: int = 0) -> int:
    align = next(k.value for k in constraints() if k.name == "chunk alignment")
    return base_chunk_for(align, token_budget, draft_slots)


def kernel_shape(c: "dict | None" = None, tp: int = 4, spec_k: int = 1) -> "KernelShape":
    """Qwen3.8-Flash-Next's kernel shape (engine/base/kernel_shape) from its text config.

    Per rank at TP=4 the way plan.py places it: query heads and GDN heads split by heads, KV heads
    replicated when fewer than tp, routed experts EXPERT-parallel (a rank holds `experts // tp`
    whole experts, so `inter_local` is the model's `inter`), the shared expert TP-sharded. GDN's
    decay is per head; the checkpoint is NVFP4 (D5) with a plain gated SiLU, so the MoE lane is
    admitted for `silu` without a clamp. `spec_k` is the MTP head's one draft. A width the config
    lacks (the shared expert's) counts as 0: that lane is not served for this model.
    """
    from engine.base.kernel_shape import Attention, Comm, Indexer, KernelShape, LinearAttention, MoE
    c = text_config() if c is None else c
    hidden = c["hidden_size"]
    return KernelShape(
        comm=Comm(world=tp, hidden=hidden), hidden=hidden, hc=c["hc_count"], tp=tp,
        attention=Attention(kind="gqa", heads=c["num_attention_heads"] // tp, head_dim=c["head_dim"],
                            kv_heads=max(1, c["num_key_value_heads"] // tp)),
        linear=LinearAttention(heads=c["linear_num_key_heads"] // tp, v_heads=c["linear_num_value_heads"] // tp,
                               k_dim=c["linear_key_head_dim"], v_dim=c["linear_value_head_dim"],
                               conv=c["linear_conv_kernel_dim"], decay="head"),
        indexer=Indexer(heads=c["indexer_kv_heads"], head_dim=c["indexer_head_dim"],
                        pool=c["indexer_compress_ratio"], topk=c["indexer_budget"]),
        moe=MoE(experts=c["num_experts"], experts_local=c["num_experts"] // tp, hidden=hidden,
                inter=c["moe_intermediate_size"], inter_local=c["moe_intermediate_size"],
                topk=c["num_experts_per_tok"], quant="nvfp4", activation=c.get("hidden_act", "silu"),
                swiglu_limit=None, dense_inter_local=c.get("shared_expert_intermediate_size", 0) // tp),
        spec_k=spec_k)


if __name__ == "__main__":
    cs = constraints(); w = max(len(k.name) for k in cs)
    for k in cs:
        print(f"  {k.name:<{w}}  {str(k.value):>6}   [{k.source}]"); print(f"  {'':<{w}}          {k.bites}")
    print(f"\n  legal prefill chunks: " + ", ".join(f"{b}->{chunk_for(b, 1):,}" for b in (2048, 4096, 8192, 16384)))
