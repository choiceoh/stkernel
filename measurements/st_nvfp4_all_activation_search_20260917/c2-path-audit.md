# FC activation-search change and C=2 quality: source and receipt audit

The later [logical/native-PTX audit](c2-logical-audit.md) found a missing
generic-global -> TMA async-proxy publication fence and adds its correction.
It supersedes the original no-defect finding below; the quality receipts and
configuration comparisons in this file remain unchanged.

At the original receipt-review stage, no definite C=2-only defect was found. Two concrete execution
differences can interact with the new search: concurrent prefill uses smaller
chunks, and batched decode uses its own input fanout and output scatter paths.
Neither mechanism has been isolated as the cause of the observed score loss.

This audit reads the completed ss1/as1 campaign and the source at f9067277.
It does not launch GPU work, change defaults, or regrade responses.

## What changed

The matched consumer sources are b2faf816 (ss1) and 1fd46c9e (as1). They differ
only in the default recipe and associated comments/tests. The activation-search
implementation itself is present in both. Compared with ss1, as1 enables:

- Static routed **FC1** activation search.
- Dynamic-prefill **FC1 and FC2** activation search.
- The separate ModelOpt dense NVFP4 adapter; this is not the consumer
  checkpoint's BF16 dense path.

Static decode **FC2 still uses radius 1 in both arms**, at both C=1 and C=2.
`moe_dispatch.py:2697` supplies the all-activation radius or the old FC2 radius.
CPU execution of the real selectors confirms that the other per-width options
are unchanged between ss1/as1 (`c2-path-config.json`). This is an interaction
investigation, not evidence that the change introduced the separate C=2 paths.

## Concurrent prefill is a real confound, not a theoretical possibility

`scheduler.py:91` chooses a smaller budget while any decoder is live;
`glm53/boot.py:893` sets that budget to one 2,304-token aligned chunk plus draft
reservation. Rank 0/1/2/3 server receipts agree on the following actual chunks
in both arms (`c2-path-server.json`, with source-file SHA-256 receipts):

| Workload | C=1 and first admitted C=2 request | Second admitted C=2 request |
|---|---|---|
| 2K portfolio, 2,641 prompt tokens | 2,641 | 2,304 + 337 |
| 32K combined, 33,955 prompt tokens | 32,256 + 1,699 | 2,304 x 14 + 1,699 |

The 32,256-row and 2,304-row MoE calls select different long-prefill / short-Q0
bodies (`moe_dispatch.py:4524`, `:4537`, `:4840`). Packet input is eligible
only above 8,192 rows; eligibility alone does not prove a particular call used
packet transport. The new search is added to both bodies. Chunk boundaries
also change the attention/state computation preceding each MoE call, so the
activation distribution reaching the search need not match.

This gives a concrete mechanism for different effects under concurrency.
It does **not** prove the short path is wrong, or that the search necessarily
makes it worse. The malformed 2K JSON occurred on the earlier-first-token
client, whose prefill was not the smaller-chunk sequence. Both 32K clients
have failed checks. A second-request-only prefill defect cannot explain every
failure on its own. Client number is not a stable admission-order identifier.

## Decode has two independent boundaries to check

1. **FC1 packing:** C=1 M8 uses `input_reuse=3` (per-token cache), C=2 M16
   uses `input_reuse=4` (register fanout). Both call the same search helper
   on one token's 16 values and its expert's global scale. The unequal-scale
   fallback also calls the helper. The index formulas write the same logical
   expert/token/scale locations. The two resident-grid publication barriers
   remain, and the change does not modify route metadata or scratch sizes.
   No cross-request maximum, omitted search branch, stale cache selector,
   or changed search radius was found in this review.

   The production folded-scale path supplies unit global scales
   (`glm53/lanes.py:464`, `moe_dispatch.py:6295`); the sampled layer-3 GPU
   receipt confirms folded scales. Wrong expert-specific reciprocal handling
   is consequently not supported as an explanation for that tested path.

2. **Output accumulation:** C=1 stages BF16 outputs through shared memory and
   uses v4 FP32 reductions; C=2 directly scatters register pairs with v2 FP32
   reductions (`moe_static_kernel_v4.py:2551`, `:2602`). The helper contracts
   preserve the same per-element BF16 contribution rounding, but explicitly
   do not promise deterministic accumulation order
   (`moe_micro_kernel.py:364`). New quantized values and execution timing can
   interact with this existing boundary. Small differences might change
   subsequent routing or token choices; this is a hypothesis, not a measured
   logits divergence or a demonstrated quality regression mechanism.

The GPU receipt does not show a conspicuously larger as1 perturbation at C=2:
single-layer relative output differences from ss1 are 16.59-17.25% at M8 and
16.81-17.30% at M16. These are differences from another quantized arm, **not
ground-truth error**, and the M8/M16 fixtures are not identical-input paired
tests. One repeated ss1 M16 case differs by 1 FP32 ULP at two elements with
zero BF16 differences. That supports the existence of small nondeterminism,
not a causal link to the answer errors.

## What the grades do and do not establish

The matched 2K/32K contexts retain the opposite movement even after removing
the C=1-only 128K coverage:

| Original strict checks, 2K/32K | ss1 | as1 |
|---|---:|---:|
| C=1, two runs | 71/76 | 76/76 |
| C=2, two clients in one wave per prompt | 72/76 | 62/76 |

These denominators contain correlated checks, not 76 independent questions;
two C=1 runs and two concurrent C=2 clients are not equal replication.
All four C=2 prompt pairs have identical workload hashes within their pair
but different output hashes, in **both** ss1 and as1. The harness uses
temperature zero. Output divergence therefore predates the FC1 extension;
this does not exclude a new interaction that changes its quality impact.

The C=2 loss is not uniform: 2K logic improves from 12/14 to 14/14, while the
2K portfolio drops from 12/12 to 5/12. Five of its seven lost points come from
one missing closing JSON brace; two from a real conflict-constraint omission.
All reported primary decisions remain correct under the prior manual audit.
The original parsed/certificate grades remain unchanged.

## Remaining discriminating tests

The existing input-reuse raw-byte gate ran without as1; the as1 whole-MoE
gate checked finiteness, zero-route behavior and differences from ss1, not
C=1/C=2 batch invariance. The block quantizer's SSE gate does not close this
integration gap. A focused future GPU investigation should:

1. Hold BF16 input, top-8 IDs and weights fixed; compare M8 with the same
   eight rows in each M16 slot, under both ss1 and as1. First compare packed
   FC1 bytes and scale bytes, then per-expert FC2 contributions and the sum.
   At M16, use explicit reuse=0/3/4 and disable direct-scatter together with
   its dependent reuse/prefetch switches to isolate the boundaries.
2. Separately hold the prompt and continuation tokens fixed; compare normal
   prefill chunks with 2,304-token chunks while retaining the same decode
   path. Isolate added FC1 decode search from added prefill FC1/FC2 search.
3. Inspect the first hidden-state/logit difference before allowing free
   generation. Whole-answer reruns alone cannot locate the defect.

No new GPU test was run for this audit. The 19 existing CPU tests covering
activation-search policy, scatter configuration and the search oracle passed.
`audit_c2_paths.py` reproduces the server summaries, original grade breakdown
and selector configurations; all three modes were executed successfully.
