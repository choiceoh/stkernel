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


if __name__ == "__main__":
    cs = constraints(); w = max(len(k.name) for k in cs)
    for k in cs:
        print(f"  {k.name:<{w}}  {str(k.value):>6}   [{k.source}]"); print(f"  {'':<{w}}          {k.bites}")
    print(f"\n  legal prefill chunks: " + ", ".join(f"{b}->{chunk_for(b, 1):,}" for b in (2048, 4096, 8192, 16384)))
