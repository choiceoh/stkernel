# glm53_dflash2_fp8_head

Builds DFlash2's candidate logits through `Fp8HeadLogitsProcessor`, so the
single head GEMM can use the fp8 copy. Everything else in
`get_top_k_tokens` is vLLM's.

Base contract from `glm53:v13-b12x`.
