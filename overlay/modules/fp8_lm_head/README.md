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

That fork is why the UE8M0 hardening (#119/#123/#129/#131) never reached DSV4:
this lane fixed its copy six times and the original never moved. DSV4 has the
guard now (2026-09-01 back-import review), with one deliberate difference —
`_describe_ue8m0_scales` here rejects a **zero** scale, and the kernel does
not: `0x00000000` passes its mask, and an all-zero 128x128 block (vocab
padding, a dead expert row) has no other scale to have. Nothing has tripped it
on this profile, so it is left as-is, but it is a stricter contract than the
assert it stands in for, and the failure it would produce is an aborted boot
that reads like a real defect. If a head with 128 consecutive zero rows ever
shows up here, that is the line to look at first.
