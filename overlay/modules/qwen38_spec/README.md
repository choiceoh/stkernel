# qwen38_spec — MTP speculative decoding, ordering and length

The checkpoint ships its own MTP head (`mtp_num_hidden_layers` 1, hybrid
`full_attention`), so `--speculative-config '{"method":"mtp"}'` resolves without
a draft model.

## `gpu_model_runner.py` — the n-gram ordering fix

The deferred spec-decode correction runs **before** the PLE n-gram context is
read, unconditionally. Without it, async scheduling plus MTP feeds the n-gram
table garbage — silently, because the ids it reads are still valid ids. The
launcher still exports `DENEB_NGRAM_FIX`, but nothing in this module reads it.

## `llm_base_proposer.py` — adaptive K

`DENEB_ADAPTIVE_SPEC=1` picks K per step from an acceptance EMA instead of a
fixed number. Boot with `SPEC_TOKENS` as the CAP; the scheduler reserves slots
from that maximum.

Both are overrides and carry the image's preimage SHA. `DENEB_ADAPTIVE_SPEC=0`
leaves the proposer's added branches dead (identical to upstream); the
`gpu_model_runner.py` n-gram correction above is unconditional.
