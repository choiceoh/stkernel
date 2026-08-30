# spec_fp8_lm_head

Block-quantized fp8 (W8A16) vocabulary head for a speculative drafter --
the deepgemm variant adopted on this fleet, not the rowwise `_scaled_mm`
one that was measured and rejected.

Armed by `VLLM_SPEC_FP8_LM_HEAD=1`. New file, so portable across images.

DSV4 still carries its own copy inside `dspark_drafter/dspark_v2.py`; the block
here was taken from it verbatim rather than rewritten, and verified byte-identical.
Converging the two means editing a deployed overlay, so it is left for when that
file is next touched for its own reasons.
