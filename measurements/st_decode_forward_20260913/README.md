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
- Sampled requests now allocate draft candidate probabilities from the drafter's facts. The target facts have no `sel_top_k`; reading it there crashed the first sampled request on every rank.
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
| `kda7-8f67f39a-rejected.log` | BV=16 at seven rows fails output bit equality; the six-row cases pass. Performance change reverted; seven-row tests retained. No speed result was recorded. |
| `cpu-8f67f39a.log` | 76 focused tests: 32 pass and 44 GPU skips. The preceding wrong module invocation is retained separately. |
| `cpu-07f98a06.log` | 68 focused tests: 33 pass and 35 GPU skips. |
| `decode7-07f98a06-plan-failure.log` | Synthetic router selection and graph tests pass. The W4 test stops at an incorrect dispatch assumption: M=7/N=6144 retains two K slices and cannot use the three-slice CTA. `c80c59cb` adds the matching two-slice implementation. |
| `cpu-suite-c80c59cb.log` | 1046 engine tests, 201 GPU skips; one package-boundary check rejects the new bare relative symbol import. Replaced with the explicit native package import. |

| `decode7-c80c59cb.log` | Two dense/router CUDA tests and seven KDA ring tests pass. Seven-row W4 same-pack captured time falls 1.87–2.58% across three shapes. The 42 complete router pipelines fall from 3.173 to 1.550 ms (51.16%). The then-default rank pack yields exactly matching expert IDs for 2352 rows. |
| `cpu-import-3b40dddc.log` | 24 tests: 17 pass and seven GPU skips after the package import repair. |
| `production-sampling-failure.json`, `cpu-sampling-04f6dba4.log` | Preserved four-rank production tracebacks identify `Facts.sel_top_k`. The focused CPU gate includes a real target Facts fixture and a first sampled request: 34 pass, seven GPU skips. Production was not stopped by this task. |
| `router-weight-identity.json` | All 42 gate matrices match between the diagnostic and consumer rank files, but all 42 correction biases differ. Router selection must also pass with the exact consumer pack; the earlier result is not relabelled as that proof. |
| `moe-compact-e6dc7ee0-scatter-failure.log` | The probe-only compact tile uses 44032 bytes of shared memory, 96 registers/thread, zero local bytes and supports two blocks/SM in the actual cubin. Its first execution fails with illegal memory access: four 64-column scatter warps retained the old N256 width after the tile became N128. No speed result and no production dispatch. |
| `moe-scatter-layout-6369964e.log` | CPU layout audit passes after deriving scatter width from the tile and proving complete, unique, bounded output coverage. The GPU retry remains required. |

The MoE compact experiment is not selected by serving. It retains the M16/FC1
N128/K256 arithmetic and the FC2 K128 rounding boundary, reduces both pipeline
stages to one and halves FC2 output width. Its 96-CTA cooperative launch is
refused unless the CUDA occupancy API proves full residency for the exact fresh
compiled cubin. The probe checks changed graph inputs, zero routing weights,
SF6 and original scale storage before timing B/C48/C96/C96/C48/B.

Consumer preparation uses private cache copies on all four nodes, excluding
`mkcalib` (see `cache-preparation.json` and `prepare_consumer.py`). Planned shape:
TP4, SPEC_K=6, C=1/C=4, KV=6 GiB, FP32 KDA state, the up/gate full rank pack,
and the canonical harness 43 budgets. Candidate-only `st_bracket.sh chain` will
run two complete onepasses and release its own boot through the official stop.

No component result establishes 22 step/s, answer quality, speculative
acceptance or a same-build consumer speedup.
