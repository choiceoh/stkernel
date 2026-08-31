# glm53_fp8_dense

Block-fp8 (W8A8, ue8m0, 128x128) copies of the bf16 dense projections the
RedHat nvfp4 checkpoint leaves unquantized: attention projections (including
the KDA merged `in_proj_qkvbfg_a`, zero-padded to the block grid in the copy
only), the shared expert, and the first-3 dense MLPs — 15.44 GB checkpoint-wide,
~3.86 GB/rank read every forward, ~17 ms of the decode step at the measured
bandwidth. DSV4 serves its dense path in exactly this scheme on this fleet
(9/9 retrieval, pos-1 acceptance 78.5%), so this is convergence to the proven
lane config, not a precision experiment.

Quantized once after `load_weights` (the call site lives in `glm53_model_wiring`)
and before compile/capture, by swapping each Linear's `quant_method`; the bf16
originals stay for fallback. Armed by `VLLM_GLM53_FP8_DENSE=1` (default off).
The kernel pair is the one `fp8_lm_head` already runs under capture here.

Boot-log fingerprint: `[fp8-dense] N linears quantized (X GB), M kept bf16`.
Gates for adoption: 9/9 retrieval, 0/16 Korean corruption, pos-1 acceptance
within 2 pct of the same-boot control, C=1/2/4 bracket. Rollback = env only.
