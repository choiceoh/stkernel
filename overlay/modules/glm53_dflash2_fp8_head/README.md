# glm53_dflash2_fp8_head

Builds DFlash2's candidate logits through `Fp8HeadLogitsProcessor`, so the
single head GEMM can use the fp8 copy. Everything else in
`get_top_k_tokens` is vLLM's.

The candidate selector also keeps its final rank-256 bilinear edge scores in
FP32. Its codebooks remain BF16, but casting the gathered rows before the
modulation and reduction prevents distinct path scores from being rounded to
the same BF16 value before `_selector_walk` ranks them. The score tensor is
only `B x 7 x 16 x 16` for this profile, and the walk already consumes FP32.

Base contract from `glm53:v13-b12x`.
