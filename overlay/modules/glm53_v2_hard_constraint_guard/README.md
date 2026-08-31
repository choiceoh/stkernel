# glm53_v2_hard_constraint_guard

Disables speculation for requests whose target distribution has a hard mask
the DFlash2 drafter does not apply. `Scheduler.update_draft_token_ids` clears
the received draft on the normal path; the overlapping/async output path marks
the already-reserved speculative slots invalid with its existing `-1` padding
contract. Both act when any of these conditions is active:

- `allowed_token_ids` is set;
- `logit_bias` or `bad_words` is non-empty;
- `thinking_token_budget` is set; or
- `min_tokens` is still active for the request.

Structured-output grammar keeps the existing prefix validation path. General
`top_k`, `top_p`, and `min_p` sampling are deliberately unchanged because they
do not describe a static hard support that the scheduler can reproduce safely.

This is a whole-file overlay against the `glm53:v13-b12x` V2 scheduler. Its
manifest preimage contract must match before deployment.
