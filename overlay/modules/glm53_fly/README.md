# glm53_fly — entropy-gated deferred verification (vLLM #53987)

`fly.py` is new (`absent`); the other two replace their
`0.1.dev20051+g487ecf187` originals. Two more of the PR's source files are
owned by modules that already had them: `config/speculative.py`
(`glm53_dynamic_k`) and `v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
(`glm53_draft_noise`).

## What it is

Standard rejection sampling stops at the first rejected draft token, so one
ambiguous position throws away the rest of a window the target had already
verified. FLy (arXiv 2511.22972, ICLR 2026) observes that when the *following*
acceptance decisions all still agree, an isolated rejection at a genuinely
ambiguous position is usually not worth a full truncation, and defers to the
draft token there instead. Upstream reports it faster than standard speculative
decoding in all 40 model–dataset configurations they measured.

This is the same family as `REJECT_METHOD=block`, which this profile now runs:
block verification accepts *at least* as many tokens for the same
distributions, FLy accepts more by giving up exactness at the ambiguous
positions. **So FLy is an approximation and quality is a gate, not an
assumption** — onepass's retrieval 9/9 and the corruption scan are what decide
it, not tokens/step alone.

## Arming

`REJECT_METHOD=fly`, plus optional `FLY_WINDOW` (default `min(6, K-1)`) and
`FLY_ENTROPY` (default 0.3) and `VLLM_FLY_ENTROPY_TOP_K` (default 3). Unset,
none of this is reached: the profile stays on `block`.

## Deviations from the upstream diff

Three of the PR's eight source files are not here, each for a stated reason:

| file | why not |
|---|---|
| `vllm/envs.py` | one key, 2,321 lines. Overlaying it would put every other env default under this repo's re-sync burden, and a stale copy after an image bump would revert them silently. `fly.py` reads `VLLM_FLY_ENTROPY_TOP_K` with `os.getenv` instead — same name, same default |
| `vllm/v1/sample/rejection_sampler.py` | 953 lines, the V1 sampler. Nothing under `v1/worker/gpu/` imports it; this profile verifies in the V2 kernel |
| `vllm/v1/worker/gpu_model_runner.py` | 8,033 lines for one predicate in the V1 runner's `_dummy_sampler_run`. The V2 runner has no counterpart site (`grep` for the predicate returns nothing there), so there is nothing to port |

`llm_base_proposer.py` *is* here: `v1/spec_decode/dflash.py` imports it, and the
one-line predicate decides whether draft probabilities are cached at all under
`fly` — getting that wrong degrades silently, which is the failure class this
whole branch exists to remove.
