# Native ST decode at seven verification rows

Target: GLM53 TP4, C=1, SPEC_K=6 (seven target rows), 22 step/s with usable
answers and actual output throughput. This folder records component evidence;
the candidate-only two-run consumer gate is still pending. The operator asked
not to measure a new consumer baseline: **no baseline on this build**.

The preceding stopped consumer used `7b6873019af67087eebdc86eaa6b142a3e738e1e`
and harness 42. Its 2K request-interior median was 18.903 step/s, with 51.818%
acceptance. It was not a completed onepass, and its 800-token reasoning cap
truncated answers. Harness 43 raises individual completion/reasoning budgets
to 8192/4096 and combined budgets to 24576/12288. Longer reasoning remains a
quality hypothesis until the new consumer actually answers.

## Changes under validation

- Native shared-expert GU/activation/down fusion and overlap with routed MoE,
  restricted to C=1 after C=4 regressions. The same native W4 packs are reused.
- W4 row strides remove copies of narrow KDA projection inputs.
- Input quantization reuse and CTA-local split folding now cover M=7. The
  former M=6 dispatch silently excluded the actual C=1 verification width.
  Each shape retains its ordinary K-slice boundaries, FP32 fold order and BF16
  rounding; M=7/N=6144 needs a two-slice variant, not M=6's three-slice variant.
- The 42 router projections use original BF16 checkpoint values with tensor
  cores and FP32 accumulation/output. Wider batches retain their FP32 path.
  Actual router weights, selected expert IDs and weights are checked separately.
- Native execution proofs require all 42 shared and router paths to run.
- The diagnostic profiler explicitly selects native execution, budgets complete
  state storage, disables calibration observers and distinguishes elapsed time
  from overlapping CUPTI activity. KDA ring tests now include seven rows.

## Completed evidence

All GPU component runs use `st-engine:main-ff728f43` on the reserved fleet.
Raw logs retain admission, execution and lease-release records.

| Evidence | Scope and result |
| --- | --- |
| `shared-mlp-10413016.log` | Five numerical/replay tests pass. M=7 shared fusion: 22.6 to 20.5 us. Routed+shared M=7: about 1% faster. M=28 reused routes regress about 3%, so wider batches retain the prior chain. |
| `profile-8336027d.log` | 45 layers, synthetic hidden states, zero KV, rank 0 with collectives replaced by identities. Elapsed 51.599 ms; overlapping activity 55.936 ms. Routed MoE 31.858 ms, ordinary W4 GEMM 9.542 ms, FP32 SGEMM 3.344 ms. No input-reuse W4 kernels ran at M=7. These are diagnostic kernel times, not consumer speed. |
| `kda7-8f67f39a-rejected.log` | BV=16 at seven rows fails output bit equality in six tests. Performance change reverted; seven-row tests retained. No speed result was recorded. |
| `cpu-8f67f39a.log` | 76 focused tests: 32 pass and 44 GPU skips. The preceding wrong module invocation is retained separately. |
| `cpu-07f98a06.log` | 68 focused tests: 33 pass and 35 GPU skips. |
| `decode7-07f98a06-plan-failure.log` | Synthetic router selection and graph tests pass. The W4 test stops at an incorrect dispatch assumption: M=7/N=6144 retains two K slices and cannot use the three-slice CTA. `c80c59cb` adds the matching two-slice implementation. |

No component result establishes 22 step/s, answer quality, speculative
acceptance or a same-build consumer speedup.
