# Low-cost DFlash acceptance review

> 그날의 조사 — **2026-09-13 의 조사다.** 그날 참이었던 것이고 유지되지 않는다 — 이후 무엇이 바뀌었는지는 `MEASUREMENTS.md` 가 안다.

Code review on 2026-09-13, against the implementation merged in #863
(`c99e2e56`). No fleet reservation, GPU execution or live acceptance measurement
was used. The accompanying default change enables the existing three controls;
the selector, request-policy and quantization-fitting candidates below are
findings, not additional enabled algorithms. The signed-fold defect is fixed
alongside the defaults, with a CPU product-preservation regression test.

## Priorities

| Candidate | Added serving work | Where it can help | Evidence and limit |
|---|---|---|---|
| Correct signed RMSNorm folding (implemented) | None; preparation only | Any smoothed norm containing negative weights | CPU counterexample and signed/zero-channel regression. Actual checkpoint exposure is unverified. |
| Fit the selector's edge strength by draft position | One scalar multiply per candidate inside the existing walk; six FP32 constants for K=6 | Greedy `selector_miss`, especially the first drafted token | Code uses a fixed coefficient of one. Better coefficients need held-out target labels. |
| Give the drafter known target restrictions | A few token masks/forced-token choices; fuse into existing selection where possible | `min_tokens`, reasoning-budget boundaries, biased or constrained requests | Target processing exists; the proposal API currently receives none of these options. Full grammar synchronization is a larger change. |
| Fit draft-only smoothing/packing choices offline | None after preparation; same packed formats and resident shapes | `candidate_miss` caused by draft quantization error | Existing smoothing fixes alpha at 0.5 and GPTQ starts with 1% damping. Lower reconstruction loss alone is not acceptance proof. |

For ordinary unconstrained greedy generation, selector fitting is the most
direct next experiment. Restrictions matter only when the request actually
uses them. The folding defect is repaired as a correctness issue; it
cannot explain the current acceptance rate without checkpoint evidence.

## Confirmed conditional defect: signed norm folding

Before this fix, [`fold`](../engine/kernels/dense/smoothing.py) computed the undo factor from
`old / new.float().clamp_min(float32.tiny)`. A negative `new` is replaced by a
tiny **positive** denominator, losing its sign and scale. The product-preserving
original test in `tests/test_engine_dense_smoothing.py` generated positive norm
weights, so this case was absent from its coverage.

CPU reproduction with BF16 `old=[-1,2]`, requested factors `[2,2]`,
`x=[[1,1]]` and `W=[[1,1]]`:

- Folded norm: `[-0.5,1]`.
- Expected undo factor: `[2,2]`.
- Actual undo factor: `[-8.507059e37,2]`.
- Original linear result: `1`; folded result: `4.253530e37`.

The fix preserves nonzero denominators, with a safe denominator only where
`new == 0`. Both the target and draft use this helper. The regression checks
negative, positive and signed-zero BF16/FP32 norm channels, both sides of the
power-of-two scaling range, multiple readers, and the corresponding Hessian
undo. No serving operation or resident weight format is added. No acceptance
increase is claimed from this synthetic input.

## Selector calibration without another model pass

[`_walk_scores`](../engine/kernels/draft_select.py) currently selects
`argmax(unary + edge)`, with
`edge = sum(successor * predecessor(previous_token) * projected_hidden)`.
The candidate count is 16 and the selector rank is 256. Both terms already
exist in the fused kernel. Replacing this by `unary + alpha[position] * edge`
needs no extra head projection, weight reader, collective or kernel launch.
Kernel register/scheduling effects still need measurement; small arithmetic
does not prove zero wall-time cost.

Start with alpha in `{0, 0.5, 0.75, 1, 1.25}` on held-out, teacher-forced
positions, prioritizing position zero. An early mismatch cuts off the whole
remaining prefix, whereas a late improvement cannot recover earlier rejected
tokens. This is a prioritization argument, not a guarantee about the best alpha.

Current first-rejection records retain reason and prefix length, not unary/edge
scores or target token IDs. They can choose between selector work and candidate
coverage work, but cannot fit these coefficients. A bounded sampled score trace
would be needed. Only rows through the first target mismatch are valid labels
for that actual prefix. If a counterfactual walk changes a predecessor, later
target logits in the old trace no longer establish live acceptance. Retain a
matched live comparison for the final decision.

## Match known request policy

[`process_logits`](../engine/base/sampler.py) and
[`_row_logits` / `_reasoning_over`](../engine/profiles/glm53/adapter.py) apply
penalties, bias, minimum-token EOS exclusions and reasoning-budget forcing to
the target. `Drafter.propose_rows` receives temperature and uniforms, while the
synchronous proposal calls likewise receive no request policy. Therefore a
draft can spend a position on a token the target is already forbidden to emit.
Onepass uses a reasoning budget and its fixed-length phase also uses
`min_tokens`, so this is relevant to specific benchmark boundaries too.

Start with known sparse restrictions and forced tokens. Apply exclusions
before top-k so replacement candidates are available; forcing may need to
insert a token absent from the original top-16. A forced predecessor must feed
the following selector edge. Preserve the target's existing precedence when
minimum-token restrictions or grammar conflict with a reasoning end.

For stochastic rows, the reported q must be the normalized distribution
actually sampled after adjustment. At a fixed shared prefix, conditioning q
onto a set containing all nonzero target probability cannot reduce
`sum(min(p,q))`: removed tokens have p=0 and each surviving q increases.
This is a per-position bound, not a claim about a changed multi-step trajectory.
Soft penalties and draft-derived top-p truncation have no such guarantee.
Per-position CPU grammar calls are outside this low-cost proposal.

## Preparation-only quantization fitting

[`Drafter.smoothing_plan`](../engine/profiles/glm53/drafter.py) already folds
channel factors into the draft norm and all its readers.
[`scales`](../engine/kernels/dense/smoothing.py) fixes alpha at 0.5 and rounds
factors to powers of two; [`gptq_factor`](../engine/kernels/dense/packing.py)
starts damping at 0.01. Fit a small grid for the draft's attention/MLP groups,
using existing calibration and separate validation activations. Preserve the
unsmoothed context-KV readers, every consumer of a folded norm, and pack cache
provenance. Restrict the fit to draft weights so target behavior is unchanged.

Keep W4/FP8 layouts and kernels fixed. Preparation and cache rebuilds cost
time; steady decoding gains no operations or weight bytes. Judge candidate
coverage/top-1 agreement as well as Hessian-weighted reconstruction loss.

## Changes that do not meet this objective

- Lowering K can raise the reported acceptance fraction while accepting fewer
  useful tokens per step. Keep K fixed for the acceptance comparison.
- Draft-temperature fitting affects sampled requests, not temperature-zero
  onepass, and must retain the real proposal distribution in verification.
- Increasing top-16 to top-32 is not free: `vocab_candidates` repeats reductions
  K times for its local top-k, then the selector reads more codebook rows. It
  needs timing evidence before being called negligible.
- Block verification, TP draft agreement and orphan-vocabulary masking already
  exist. Reintroducing them is not a new acceptance improvement.

Final promotion requires the same target build, workload, K and precision,
per-question acceptance and tok/s, and preserved target quality/output. The
default bootstrap collector must be complete before a timing comparison.
