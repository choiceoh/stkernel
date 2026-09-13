# Default selector FP32 and automatic fitted FC correction

Base: `4a2cf527da70fef78044af21cd58d1ea86310975` (main including #898).
User-requested adoption of the two features introduced by #894 / #896.

- Selector projection now keeps FP32 output by default, including omitted
  profile fields and all greedy/sampled, synchronous/batched proposal paths.
  `selector_projection_fp32: false` preserves the old BF16-output path.
- Native boot automatically looks for `/cache/draft-fc-bias.json`. It loads
  only a complete per-rank fitter artifact with held-out FC and RMSNorm error
  improvements, agrees file contents before packing, then binds each correction
  to its executed FP8 reader before source retirement and graph capture.
- Missing, malformed, divergent, or stale automatic artifacts leave correction
  absent on every rank and record why. Discovery itself is enabled; no fitted
  vector ships with this change. An explicit vector still takes precedence and
  fails on a mismatch. `fc_bias_auto: false` plus an empty map disables discovery.
- Boot and lane metadata distinguish the policy from actual correction use.
  No automatic pair collector or teacher GEMM was added. Without a candidate,
  boot does no reader hashing or vector allocation; the existing 16 KiB arena
  reservation per rank is unchanged. KDA remains FP32.

## CPU verification

`cpu.log`: 82 tests discovered, 77 passed, 5 CUDA cases skipped. Modules:

```
tests.test_engine_draft_precision
tests.test_engine_draft_tuning
tests.test_engine_draft_tuning_integration
tests.test_engine_drafter_storage
tests.test_engine_draft_acceptance
tests.test_engine_early_observe
tests.test_engine_drafter
```

Tests include the real CPU fitter CLI to automatic-loader path, source/pack
binding and peer failure, corrupt/oversized/unsupported artifacts, explicit
overrides, projection default and opt-out in all three proposal paths, and
preservation of legacy BF16 selector trace interpretation.

macOS Python 3.12 / torch 2.14.0. Local CPU contract imports use the same
import-only Triton decorator shim as the original precision change; the shim
cannot execute GPU kernels. The normal engine-check CI imports real Triton.
Python compilation and `git diff --check` pass.

No GPU, fleet queue, SSH, boot, deployment, or onepass run was used. These are
default-setting and CPU-contract results, not measured acceptance, quality,
latency, or CUDA graph evidence. See the [tuning guide](../../docs/GLM53_DRAFT_TUNING.md)
for artifact placement and OFF settings.
