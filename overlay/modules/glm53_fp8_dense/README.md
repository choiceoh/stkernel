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

**Default on since 2026-09-04 (28차).** The bracket (same image, MK-MLA off
on both arms, W4 drafter actually served -- see below): C=1 step/s 15.95 →
16.235 (+1.8%, six reps each), pos-1 acceptance 64.5% vs 61.6%, quality 9/9,
Korean 0/16, prefill unchanged (2,300 tok/s at 23K both arms). The fp8-only
`w8` arm is gone: the operator's rule is that a proven improvement becomes
the default and the other side is removed, not kept as a second setting.
Offline gate: `probes/drafter_dense_path_check.py` (build → lane serves
bitwise → fullgraph and dynamic compile → CUDA-graph capture). Rollback =
`VLLM_DFLASH2_FP8_DENSE=0`.

**Armed is not served -- the compile cache.** vLLM keys its torch.compile
cache, and under `VLLM_USE_AOT_COMPILE=1` the whole AOT artifact, on the env
vars registered in `vllm.envs`, the vllm config and the forward's source,
then loads it with guard checks off. A `quant_method` swapped in after load
is no part of that key, so every boot with the knob on served the drafter
from the artifact of the first boot that ever compiled it (09-03: bf16
`F.linear` on all 30 layer projections) while the fingerprint reported 31
linears armed; only the eager fc reached the lane, and the first bracket
measured exactly nothing. Two pieces close that: the knob registers itself
into `vllm.envs.environment_variables` (`_register_compile_factor`, so each
value is its own artifact), and `install_drafter_serving_check` counts
opaque-op calls over the drafter's first forwards and writes the verdict
into the boot log -- `[fp8-dense] drafter lane serving: 30 of 31 opaque GEMM
calls per forward` (the fc runs outside the compiled forward) or
`drafter lane NOT SERVING`. A CUDA-graph replay runs no Python and is not
judged; a forward under stream capture is definitive.

The drafter's bf16 sources are released at load under the target's knob
(`VLLM_GLM53_FP8_DENSE_FREE_BF16=1`, 28차): the checkpoint walk is finished
when the pass runs, `_build_fused_kv_buffers` has already `torch.cat`'ed the
k/v halves of `qkv_proj.weight` into its own buffer, the compiled forward
reads the fp8 copies and W4 packs through the opaque op, and no drafter
linear carries a bias. `[fp8-dense] drafter: VLLM_GLM53_FP8_DENSE_FREE_BF16=1:
released 0.73 GB of bf16 sources` is the line; the bytes come back before KV
sizing.
