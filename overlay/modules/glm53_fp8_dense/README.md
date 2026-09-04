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

## b-projection arm — `VLLM_GLM53_FP8_DENSE_BPROJ` (default off)

STEP_KERNEL_MAP #108 §2: after W8A8, 145 `cutlass_80_wmma` bf16 GEMMs/step
remain — the rear halves of the low-rank projections. This arm extends the
pattern set to `self_attn.q_b_proj` `[4096,1536]`, `self_attn.kv_b_proj`
`[4096,512]` and `self_attn.indexer.wq_b` `[4096,1536]` (replicated — full
read on every rank). ~160 MB/step fewer bytes at C=1 ≈ 0.9% step ceiling at
the 273 GB/s floor; honest expectation is under that (the indexer GEMMs are
aux-stream contention-stretched, not bandwidth-bound, per the 08-10
decomposition).

Deliberately out: `f_b_proj`/`g_b_proj` (per-rank `[2048,128]` — under the
`min(shape) >= 512` guard, ~17 MB/step win at most), and `wk_weights_proj`
(the loader upcasts it to bf16 to keep the wk+weights_proj fusion; quantizing
it would break that contract). Requires `VLLM_GLM53_FP8_DENSE=1` to have any
effect; the existing per-layer guards, stale-copy check and fallbacks apply
unchanged. Rollback = the env alone.

Boot-log fingerprint: `[fp8-dense] N linears quantized (X GB), M kept bf16`.
Gates for adoption: 9/9 retrieval, 0/16 Korean corruption, pos-1 acceptance
within 2 pct of the same-boot control, C=1/2/4 bracket. Rollback = env only.

## DFlash2 drafter arm — `VLLM_DFLASH2_FP8_DENSE` (rewired 2026-09-04)

`glm53_dflash_loader_fp8` runs this pass on the drafter under its own knob.
Until 2026-09-04 the base pattern set was all it had, and that set lists
`q/k/v_proj` and the target's fused names — never the drafter's MERGED
`qkv_proj`, its aux-hidden `fc` (`[4096, 5 x 4096]`, ReplicatedLinear, read
whole on every rank) or the two conv `kernel_projection`s per layer. So the
knob covered `o_proj`/`gate_up`/`down` and left 43% of the drafter's bytes
bf16, the fc alone 23%. `_DRAFTER_INCLUDE` closes that under the drafter knob
only; the target's set is untouched.

Why it matters: the armed 09-03 trace ends every decode step in a 7.5-8.5 ms
tail (target head, drafter, draft head) the GPU is 95% busy through, and the
drafter's bf16 GEMMs are 3.2-3.6 ms of it (STEP_KERNEL_MAP supplement 4).
Offline at the fleet shapes, TP=4 sharding, DRAM-cold (`probes/
drafter_fc_check.py`, srv2): bf16 3.23 ms/step → fp8 pair 2.34 → MK W4 1.25.

Two things are different about the drafter:

- **Its forward is torch.compiled** (`@support_torch_compile`). The Python
  MK-or-fp8 choice `Fp8DenseMethod.apply` makes for the (eager) target is not
  traceable — the lane's eligibility test guards on the token count and the
  extension call is a pybind function — so drafter methods are marked
  `_opaque` and route through ONE custom op, `glm53_fp8_dense::gemm_mk_or_fp8`
  (`_mk_or_fp8_dense_gemm`), which arms, tries the lane and falls back to the
  verified fp8 pair at run time, exactly as eager does.
- **The fc is wider than the lane's K** (20480 vs `MK_GEMM_KMAX` 4096). It
  carries one pack per K-chunk (`build_mk_weight_w4_kchunks`) and `gemm_w4a8`
  runs five launches summed in fp32 — 301 µs against 682 bf16 / 489 fp8.

Values: `1` = fp8 pair + MK W4 packs (W4 numerics when MK-GEMM is armed);
`w8` = fp8 pair only, never a pack — one numerics axis per boot. Offline gate:
`probes/drafter_dense_path_check.py` (build → lane serves bitwise → fullgraph
and dynamic compile → CUDA-graph capture → w8). Adoption gate: acceptance
(pos-1 within 2 pct of the same-boot control) with prefill measured in the
same boot; rollback = the env. The drafter's bf16 sources are NOT released
(+~0.57 GB/rank over today): `_build_fused_kv_buffers` copies the k/v halves
of `qkv_proj.weight` at load, so a release looks safe, but it is a separate
change with its own boot.
