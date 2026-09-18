# Completed-pool pin: mixed quality effects, retain production policy

Removing the completed-pool pin did not recover the exact Telemachus request.
It made the temperature-1 replay substantially worse. The earlier reconstructed
Ithaca request became more readable, but retained lexical/factual errors.
These observations justify retaining the existing production policy while
investigating the remaining corruption; they do not establish a general
quality benefit for pinning.

| Prompt / policy | Temperature | Output tokens | Manual reading |
| --- | ---: | ---: | --- |
| Exact 50,005 IDs, current pin | 1 | 477 | Malformed prose |
| Exact 50,005 IDs, no completed-pool pin | 1 | 2,048 (cap) | Severe multilingual collapse and repeated failed restarts |
| Reconstructed 47,173-ID Ithaca, current pin | 1 | 859 | Malformed ending |
| Reconstructed 47,173-ID Ithaca, no pin | 1 | 572 | More readable structure; lexical/factual errors remain |
| Exact 50,005 IDs, current pin | 0 | 298 | Coherent prose in this run |
| Exact 50,005 IDs, no pin | 0 | — | Incomplete: isolated control aborted when a second request entered |

The final row is **not a quality result**. The five completed requests each
had zero cached tokens and exactly one served-count increment. Seed 7,
`top_p=1`, `top_k=-1`, maximum 2,048, one target token per step, no speculative
proposals, and one immutable build were held fixed. Temperature-0 coherence
here does not reverse earlier failing greedy observations on other builds.
The original incident was unseeded; these are seeded reproductions.

## What changed

Transformers' GLM5Next indexer and current upstream vLLM select complete
four-token pools by score, then append the incomplete tail (zero to three
tokens). ST's added pin forces the newest completed pool into the selection
when the sequence length is divisible by four and its score would lose.
The pin changes at most one selected pool on such a row; token count alone
does not bound its attention weight or downstream effect.

Upstream source identities checked on 2026-09-18:

- [Transformers modeling_glm5_next.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py), Git blob `8efbef9839129b5db0164653c1d9e978ab215666`.
- [vLLM sparse_indexer.py](https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/nvidia/sparse_indexer.py), Git blob `1fc42c9e4114bf9ff997dcf49cd88015127b5b3e`.

The private control is source `c30bae949a9d5968389e50c58b83f6f4ee108b64`:
mode 20 preserves pinning; mode 25 suppresses only that mutation in eager
prefill and target decode. Both use the same underlying seed. Eleven source
files were verified in all four ranks before the first request; see
[runtime-identity.json](runtime-identity.json). The first 32 logit rows of
each completed request were retained privately with prefix hashes, draw
addresses and uniforms.

The production restoration candidate `b9eac949d32f1cdda74641087c0068e0bd8c5ef4`
also removes pinning from captured and tree paths and adjusts the selection
analyzer. Its CPU gate ran 85 tests: 68 passed, 17 GPU/interpreter cases
skipped. It is **not merged**, because source fidelity alone is insufficient
to justify the observed quality deterioration.

## Limits and cleanup

Pinning was added in #1158/#1159 after the original incident, so it cannot
explain that first occurrence. The unpinned Ithaca prompt is reconstructed,
not the exact retained per-turn request. No broad benchmark or throughput
win is claimed. Short answers and low unusual-character counts alone are
not quality gates.

The last control stopped at step 4297 with the private single-request guard;
no completed output was retained for it. The fleet stop path removed all four
tiers and released the lease at 10:19:28 KST. Private prompts, output text,
token IDs and logit arrays remain outside the repository. Only counters,
hashes and runtime identity are included in [evidence.json](evidence.json).
