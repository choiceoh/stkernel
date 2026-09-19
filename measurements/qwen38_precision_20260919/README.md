# Qwen3.8: model arithmetic and engine precision fixes

Two defects were reproduced independently of weight/activation quantisation and fixed. The changed GDN recurrence
and the existing Qwen cells passed **77 tests on NVIDIA GB10**, with no skips. This is kernel/cell evidence, not a
48-layer TP4 logit-equivalence, output-quality or throughput claim. No production deployment was performed.

Base: `65b6e477c70ccafb176e42aa1a258c49193c3eeb`. The GPU ran that base with the candidate source files recorded in
[`gpu-source.sha256`](gpu-source.sha256). The final PR is rebased onto `011f5b76` (main): upstream changed the
hyper-connection leave/prefetch path and draft sampling in `net.py`/`lanes.py`; the GDN kernel and its reference-ring
correction are unchanged. The image-forward fixture now supplies the close mixer's weight for that upstream
prefetch interface. The GDN test introduction, source provenance manifest and documentation were also updated
after the GPU run. The hashes record the original GPU snapshot, not the rebased integration tree.

## Changes and negative controls

1. **GDN beta disagreed between prefill and decode/verify.** The model's `GatedDeltaNet` and `gdn.gates` prefill
   evaluate `sigmoid(b)` in the projection's dtype before the recurrence consumes FP32 values. Both the served
   ring and the reference ring instead widened the BF16 logits before sigmoid. With identical BF16 inputs, two
   tokens already produced different FP32 saved states. In the CPU interpreter negative control, the largest
   absolute state difference was `1.036226749420166e-4` (relative `7.368726655840874e-4`); the reference ring versus
   chunk difference was `7.769465446472168e-5`. The new `ROUND_BETA` specialization and reference-ring correction
   make the two GDN entries follow the model. State storage and recurrence accumulation remain FP32. Per-channel
   KDA and the generic functional entry retain their existing sigmoid arithmetic.

   The regression checks every saved state against `modules.linear_attention.gated_delta_rule` with the model's
   BF16 beta, at `rtol=2e-5, atol=2e-7`, and requires byte-equal BF16 outputs. It tests both the fused GDN gate and
   precomputed-decay entry. On GB10 it uses the actual rank cell: 4 key heads, 12 value heads, 128x128 FP32 state.
   The CPU reference chunk/ring comparison is exact. Existing functional-lane comparisons now receive beta already
   sigmoided in the model's dtype; on GPU, output and ring storage equality remain byte-exact. The first GPU run
   exposed that test's old FP32-sigmoid expectation; the final run passes with the model-aligned expectation.

2. **Image patches overwrote the eager/captured flag in `Qwen38Net.forward`.** The loop variable `rows` replaced
   the boolean flag with an image embedding tensor. PLE dispatch then raised
   `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`. Naming the patch tensor
   `patch_rows` preserves dispatch. A regression runs the actual forward layer loop, PLE branch, host MoE dispatch
   and final hidden/stream extraction, comparing image patches with pre-replaced embeddings exactly. Arithmetic
   sublayers use small test doubles; this is not a vision-model quality evaluation.

[`before.log`](before.log) runs the new three regression tests with the original engine from the base commit.
It records two numerical failures and the image-forward exception. The same checks pass after the fixes.

## Long-context check

Added `NormRopeTests` to the GPU cell probe. A new FP32 test compares standalone norm/RoPE and the fused served
Q/K/index-query path against independent module operations at positions
`0, 1, 31, 4095, 32767, 32768, 65535, 131071, 131072, 262143`.
Both maximum error divided by maximum reference magnitude and RMS error divided by reference RMS must be `<=2e-5`.
It passed on GB10. There was no demonstrated RoPE defect, so the RoPE kernel was left unchanged.

## Verification

| Run | Result | Evidence |
| --- | --- | --- |
| GDN ring and image-forward CPU/interpreter suites | 18 passed | [fixes.log](fixes.log) |
| KDA decay, native GDN chunk, prefill marks, draft ahead/sampling/chain | 71 tests: 63 passed, 8 GPU-only skips | [regression.log](regression.log) |
| Updated KDA functional comparison, FP32 RoPE, probe wiring | 13 passed | [rope-and-probe.log](rope-and-probe.log) |
| Kernel package/provenance and source contracts | 12 passed | [source-checks.log](source-checks.log) |
| GB10 Qwen boot qualification plus cell tests | 77 passed, no skips | [gpu-cells.log](gpu-cells.log) |
| Integration on main `011f5b76`: GDN, pictures, KDA decay, RoPE, leave/prefetch, draft sampling, provenance, probe wiring | 73 passed, no skips | [integration.log](integration.log) |

CPU: isolated copy in `stk-test`, PyTorch `2.14.0+cpu`, Triton `3.8.0`, `OMP_NUM_THREADS=1`,
`TRITON_INTERPRET=1` for kernel suites. The source-contract run omits interpreter mode. macOS resource-fork sidecars
from the initial copy were removed before the final source-contract check; they are not repository sources.

GPU: `srv4`, NVIDIA GB10, PyTorch `2.13.0+cu132`, CUDA `13.2`, image `st-engine:glm53`, 4 GiB single-GPU budget.
Fleet session `qwen38-precision-4436b`, ticket `17898122452629709`, payload exit 0; tests took 54.874 s.
The queue ran beside production without taking the fleet lease.

From the controller, with the candidate checkout as the working directory:

```sh
ST_PROBE_GIB=4 OMP_NUM_THREADS=1 bash bench/fleet.sh run --gpu --detach qwen38-precision-check 5 \
  'Qwen model beta parity and FP32 long-context RoPE' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_cells \
  --output /cache/qwen38-precision-check.json
```

Whole-model TP4 logits with identical quantised weights/activations, full consumer quality/acceptance and performance
remain unmeasured here. Ordinary floating-point reduction/rounding differences are not claimed to be zero.
