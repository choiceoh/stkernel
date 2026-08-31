# glm53_v2_sampler_guards

Keeps V2 sampler-side request guards on the logits-processing path.

`Sampler.apply_sampling_params()` returns the input logits unchanged when
`_requires_logits_processing()` says that the active requests need no
processing. The predicate covered logit bias, penalties, bad words,
temperature, min-p, top-k, and top-p, but omitted `ThinkingBudgetState`.
Consequently, a request whose only active processor was a thinking-token
budget skipped `ThinkingBudgetState.apply()` entirely and did not force the
configured reasoning end marker when its budget was reached.

The guard now treats any active per-request thinking budget as requiring logits
processing. The `enabled` check is required because `ThinkingBudgetState`
intentionally does not allocate `use_thinking_budget` when the model has no
complete reasoning-token configuration.

Base contract from `glm53:v13-b12x`. This is a whole-file replacement for the
V2 model runner sampler, not the older `vllm/v1/sample/rejection_sampler.py`
path, so the manifest pin must be refreshed on an image bump.
