# glm53_draft_noise — the draft's Gumbel stream, disjoint from the target's (vLLM #54282)

Seven files replace their `0.1.dev20051+g487ecf187` originals (base shas in
`manifest.tsv`). The ninth file the upstream PR touches,
`v1/worker/gpu/sample/sampler.py`, is already `glm53_runtime`'s, so its one line
(`is_drafting=False`) lives there.

## The defect

During target verification the Gumbel noise used to re-sample a rejected draft
token is drawn from the same Philox offset as the noise that produced that
draft token — the same `(seed, pos)`, so byte-identical noise. Conditioned on
the proposal having won the argmax, the other tokens' Gumbels are truncated
below that max, most tightly for the tokens the draft ranked highest. The
residual therefore under-weights exactly those tokens and **the output
distribution is no longer the target's**.

Upstream is explicit that this affects `draft_sample_method="probabilistic"`
**only**: under the default greedy, `draft_logits` is None, the draft never
calls `gumbel_sample`, and there is no shared noise vector. This profile serves
`DRAFT_SAMPLE=probabilistic`, so it is the affected configuration.

## The change

A salt of `1 << 30` on the draft's Philox offset. Positions are int64 and never
approach 2**30, so the two streams cannot collide. `IS_DRAFTING` is threaded as
a `tl.constexpr` so the salt costs nothing on the target's path.

Ported rather than applied: this image's `gumbel.py` predates the upstream
refactor that factored `gumbel_noised_argmax` out of `gumbel_block_argmax`, so
the salt goes into `gumbel_block_argmax` here, and into the **local**
`gumbel_noised_argmax` that `dflash2/speculator.py` carries as an SM121 port of
the same helper. A blind `patch` mis-placed the hunk into an unrelated function
body; the hunks below were applied by hand against this image's structure.

| file | change |
|---|---|
| `gumbel.py` | `_DRAFT_NOISE_SALT`; `IS_DRAFTING` through `gumbel_block_argmax` -> `_gumbel_sample_kernel` -> `gumbel_sample(is_drafting=...)` |
| `dflash2_speculator.py` | the local `gumbel_noised_argmax` takes and applies the salt; the selector walk passes `IS_DRAFTING=True` |
| `spec_speculator.py` | `sample_draft` takes `sample_src_positions` from the caller instead of doing `positions + 1` itself, and passes `is_drafting=True` |
| `spec_rejection_sampler_utils.py` | the resample is the target's draw: `IS_DRAFTING=False` |
| `autoregressive_speculator.py`, `mtp_speculator.py`, `dspark_speculator.py` | the callers moved to the new `sample_src_positions` contract |

The PR's eighth source file, `dflash/speculator.py` (`sample_pos - 2` -> `- 1`),
is deliberately **not** here: `tests/test_logic.py` pins that this repo does not
overlay it -- a 721-line verbatim copy re-synced against every image bump for a
branch this hardware cannot take -- and the hunk is dead for us anyway, because
`DFlash2Speculator` overrides `_generate_draft` and drafts through its own
selector walk.

Not gated. This is a distribution-correctness fix on the path this profile
already serves.
