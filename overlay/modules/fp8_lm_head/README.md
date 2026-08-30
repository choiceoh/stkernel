# fp8_lm_head

Block-quantized fp8 (W8A16) vocabulary head, for either end of speculative
decoding. The deepgemm variant adopted on this fleet -- not the rowwise
`_scaled_mm` one that was measured and rejected.

`Fp8HeadLogitsProcessor` takes the env name that arms it, because the risk
differs: a bad draft head costs acceptance only, while the target's logits
decide the sampled token and the accept/reject.

- `VLLM_SPEC_FP8_LM_HEAD` -- draft head
- `VLLM_TARGET_LM_HEAD_FP8` -- target head (needs a divergence gate)

New file, so portable across images. DSV4 still carries its own copy inside
`dspark_drafter/dspark_v2.py`; the block here was taken from it verbatim and
verified byte-identical.
