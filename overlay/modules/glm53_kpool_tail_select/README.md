# glm53_kpool_tail_select

GLM-5.3's indexer pools keys four at a time (`index_kpool = 4`) and selects
pools, not tokens, so the pool-level top-k decides what the model may look at.

## Skip dead short-prefill scoring under dense MLA

The stock sparse indexer has a request-level bypass for fresh short prefills
that the selected MLA backend will execute as dense MHA. The kpool fork
predates that bypass, so even when `glm53_sm121_mla_prefill` admitted dense MHA
it still filled and causally masked a `[tokens, 2048]` top-k buffer in every
sparse layer. Dense MLA never reads that buffer.

The kpool path now derives the sibling MLA metadata key from the exact
`*.indexer.k_cache -> *.attn` GLM module layout. After it writes the pooled
index-K cache and the persistent tail cache, it returns before the unused
top-k buffer work when the MLA metadata says `use_dense_mha=True`. Mixed
prefill/decode batches, CUDA graph capture, cached-context or long MQA
prefills, and any module-name drift retain the old path. This makes the change
inert unless the separate SM121 dense-prefill arm is admitted.

The remaining changes to the kpool selection path are below.

## Keep prefill pool windows as views

Every prefill cache write compresses four consecutive raw K and gate rows.
The old caller built an `[tokens, 4]` index grid and advanced-indexed both
inputs, materializing two overlapping `[tokens, 4, 128]` tensors before the
compression kernel. The kernel ABI already accepts independent row and
pool-slot strides, but its public wrapper immediately called `contiguous()`
and discarded that capability.

The kpool prefill caller now forms `[window, 4, 128]` with
`unfold(...).movedim(...)`, which is a zero-copy view with strides
`[128, 128, 1]`, and invokes the pinned stock Triton kernel with those strides.
Destination slots are shifted to each window's completion token, preserving
the previous mask and cache layout. This removes the index grid and both K/gate
gathers for every prefill size; the last head dimension remains contiguous as
required by the kernel.

## Make short MQA prefill a one-pass index write

When a fresh context is within `index_topk`, the MQA fallback attends to every
causal token and does not run sparse logits. The old fast path nevertheless
performed three full-buffer passes: initialize `[tokens, 2176]` to `-1`,
broadcast an arange over it, then build/apply a boolean causal mask. It also
requested the sparse logits gather workspace before iterating an intentionally
empty chunk list.

A 2-D Triton kernel now writes `column` or `-1` directly from each row's token
position. Pure short-prefill batches skip the overwritten sentinel fill;
mixed batches retain it for their decode/padded rows. The shared logits
workspace is requested only in the non-short branch that actually consumes
it. This path remains the fallback when the separate dense-MLA arm is off;
when dense MLA is admitted, the earlier no-consumer return skips the top-k
buffer completely.

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

## Reuse inactive top-k storage for packed pool ids

The model owns one persistent `int32` top-k buffer shared by all 11 sparse-MLA
layers. With `index_topk=2048` and `index_kpool=4`, each layer previously made
a fresh packed `[rows, 512]` `int32` buffer and then widened it to `int64`
before expansion. The image's CUDA top-k kernels fully write all 512 outputs,
and the fused expand kernel accepts any integer input dtype, so neither the
fresh `-1` fill nor the widening is needed on the GLM CUDA path.

A column slice of the output buffer is deliberately not used: its row stride
is 2176 while the CUDA top-k ABI assumes packed stride 512. Instead, the code
uses a packed view over persistent storage after the active output rows. A
capacity/contiguity/dtype guard falls back to the original temporary buffer.
Because scratch is wholly outside the active prefix, padded decode rows cannot
overwrite prefill results in mixed batches, and the rounded output columns keep
their required `-1` mask.

Base contract from `glm53:v13-b12x`.
