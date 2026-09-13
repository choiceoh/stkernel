"""GLM-5.3-Flash's shape constraints (profile) -- including the one that cost holds.

The 40th campaign spent several fleet holds learning why a prefill chunk was
6,912: `floor((MAX_BATCHED - draft_slots) / 2304) * 2304` with MAX_BATCHED
8192 and six draft slots, where 2304 is the Mamba/KDA cache block size in
"align" mode (--block-size 2304, VLLM_GLM53 APC). MAX_BATCHED=9216 did not
move it -- six tokens short. The self-check below IS that finding: if it
ever stops reproducing, the chunk law changed.

ST serves a budget of 10,240, so a chunk of 9,216 (45차 §23 조사 19차): the
routed expert lane is one fused kernel that is half empty at 6,912 rows, and
four blocks instead of three take 39% off its cost a token.
"""
from __future__ import annotations

from engine.base.shapes import Constraint, chunk_for as base_chunk_for
from engine.profiles.glm53.plan import text_config

BLOCK = 2304          # launcher --block-size; the prefill chunk's alignment (facts.CHUNK_ALIGN). The paged KV / prefix block
                      # is facts.BLOCK = 768, a divisor: the 6,912 law is unchanged, its boundaries are three times as many
SPEC_K = 6            # profile SPEC_K=6: DFlash2 draft slots taken out of the token budget


def constraints() -> "list[Constraint]":
    c = text_config(); la = c["linear_attn_config"]
    return [
        Constraint("kv/state block", BLOCK, "launcher --block-size 2304",
                   "paged MLA KV and the KDA state checkpoints share one block size; a prefill "
                   "chunk that ends mid-block leaves a KDA state the next chunk cannot resume "
                   "from (mamba 'align' mode: state is saved at block boundaries only)"),
        Constraint("chunk alignment", BLOCK, "the block above",
                   "the raw-token multiple every prefill chunk must end on -- the 6,912 law"),
        Constraint("draft slots", SPEC_K, "profile SPEC_K",
                   "DFlash2 verifies K draft tokens per decode step; they come out of the "
                   "same token budget, which is why 9,216 misses by six"),
        Constraint("conv history", la["short_conv_kernel_size"] - 1, "linear_attn_config.short_conv_kernel_size",
                   "KDA's q/k/v conv reads this many previous tokens; carried in the slot"),
        Constraint("kda heads per rank", la["num_heads"] // 4, "num_heads / TP",
                   "the KDA state slot's leading dim (64 heads / 4)"),
        Constraint("mla latent", c["kv_lora_rank"], "config.kv_lora_rank",
                   "the paged KV row width: one latent per token, nope-only (rope dim 0)"),
        Constraint("indexer topk", c["index_topk"], "config.index_topk",
                   "positions a query keeps; the sparse attention's gather width"),
        Constraint("nvfp4 group", 16, "quantization_config group_size",
                   "expert GEMM scale granularity along K (compressed-tensors nvfp4-pack)"),
        Constraint("hyper-connection width", c["hc_mult"], "config.hc_mult",
                   "residual stream is hc_mult copies of hidden (mHC)"),
    ]


def chunk_for(token_budget: int, draft_slots: int = SPEC_K) -> int:
    return base_chunk_for(BLOCK, token_budget, draft_slots)


def _selfcheck() -> None:
    assert chunk_for(8192) == 6912, chunk_for(8192)          # the 40th campaign, exactly
    assert chunk_for(9216) == 6912                           # six tokens short: still 6,912
    assert chunk_for(9222) == 9216                           # N x 2304 + SPEC_K is the only way up
    assert chunk_for(10240) == 9216                          # what ST serves: four blocks (조사 19차)
    assert chunk_for(32768) == 32256                         # (32768-6)//2304 = 14 blocks
    assert chunk_for(16384) == 16128                         # (16384-6)//2304 = 7 blocks
    print("  glm53 shapes: 8192->6,912, 9216->6,912, 9222->9,216, 10240->9,216 -- the chunk law reproduces OK")


if __name__ == "__main__":
    _selfcheck()
    cs = constraints(); w = max(len(k.name) for k in cs)
    for k in cs:
        print(f"  {k.name:<{w}}  {str(k.value):>6}   [{k.source}]")
