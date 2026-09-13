# Optional selector FP32 output and decode FC bias

Scope: implement the first two follow-up candidates and research the third.
Base: `9f0b72f1` (`origin/main` when the branch was created).

- Selector projection: an explicit `selector_projection_fp32` profile field
  selects BF16 inputs/weights with FP32 output. Greedy, sampled and batched
  paths share it; selector trace/fitting preserves the precision mode.
- Decode FC: explicit collection of committed native/reference output pairs,
  CPU mean-residual fitting, disjoint request-family validation, per-rank pack
  identity and preparation vote, then FP32 addition fused into RMSNorm. The
  optional vector occupies a declared 16 KiB region per production rank.
- Existing defaults stay unchanged: projection flag false, bias map empty,
  FP8 FC and FP32 KDA. No checkpoint-fitted vector is provided.
- The separate [rounding study](../../docs/GLM53_DRAFT_ROUNDING_RESEARCH.md)
  reviews AdaRound, GPTQ and FP4DiT against the actual ST pack arithmetic.
  It does not modify a pack or claim an acceptance gain.

## Local verification

`cpu.log`: 60 tests discovered, 57 passed, 3 CUDA cases skipped. Covered modules:

```
tests.test_engine_draft_precision
tests.test_engine_draft_tuning
tests.test_engine_draft_tuning_integration
tests.test_engine_drafter_storage
tests.test_engine_draft_acceptance
```

macOS Python 3.12, torch 2.14.0. Triton is unavailable on this host; the local
runner supplied import-only decorators for CPU serving-contract tests. A GPU
kernel call through that shim cannot execute. Numerical fitting, the standalone
CPU CLI and projection references use real CPU torch. This is not Triton
compilation, CUDA graph replay, a real-weight result, or performance evidence.
The normal GitHub engine-check runs the complete CPU suite with real Triton
installed and reports its separate verdict on the PR.

Python compilation and `git diff --check` passed. The new CUDA numerical/graph
cases are retained for a future authorized hardware run, alongside the existing
selector CUDA case. No GPU, fleet queue, SSH, boot, deployment or onepass run
was used. Collection adds a reference GEMM/readback only when explicitly called;
its output-error metrics must not be described as measured acceptance.

See the [tuning guide](../../docs/GLM53_DRAFT_TUNING.md) for exact collection,
fitting, profile loading and memory contracts.
