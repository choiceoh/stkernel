# glm53_kpool_tail_select

GLM-5.3's indexer pools keys four at a time (`index_kpool = 4`) and selects
pools, not tokens, so the pool-level top-k decides what the model may look at.
Two changes to that path.

## Honor `index_kpool_always_select_tail`

The model config sets this True. Nothing reads it — across the whole install the
name appears in the config dataclass, in the assignment that stores it, and in
one docstring on `append_tail_to_topk` claiming it "keeps the (incomplete)
trailing pool so the most recent tokens are always attended to".

That is what the code does, and the incomplete trailing pool is
`[pool_len*4, seq_len)`, which is **empty whenever `seq_len % 4 == 0`**. At those
steps the four most recent tokens are visible only if their just-completed pool
outranks the entire history, and the top-k ranks by relevance, not recency. The
declared guarantee has never held.

The fix raises that pool's logit before the top-k rather than editing its output:
neither `top_k_per_row_decode` nor `persistent_topk` documents the order it
writes indices in, so evicting "the weakest" from the result would be a guess,
while biasing the input lets each kernel drop its own weakest. Written with
gather/scatter rather than `nonzero`, so it adds no host sync and keeps static
shapes under CUDA graph capture. Applied only on the token-granular path — the
`positions is None` fallback reads `decode_metadata.seq_lens`, which is already
pool-granular, and dividing it again would aim the bias at the wrong pool.

**Inert below the top-k budget.** With `index_topk = 2048` and `kpool = 4`, every
pool is selected until ~2048 tokens, so short contexts are unaffected. This is a
long-context correctness fix and does **not** address the Korean-token corruption
tracked in MEASUREMENTS.md, which reproduces at ~1000 tokens.

Not yet exercised by a boot: static checks only (AST, scope of the hoisted
`dec_seq`, no undefined names).

## Route small-SM parts away from `persistent_topk`

Carried forward from the bring-up patch. `persistent_topk` oversubscribes past
~24K context on a 48-SM part and its FilteredTopK fallback wants 128 KB of shared
memory against GB10's 99 KB. The stock guard excludes capability family 120,
which already covers GB10; keying on SM count instead states the actual
constraint and keeps larger family-120 parts on the fast kernel.

Base contract from `glm53:v13-b12x`.
