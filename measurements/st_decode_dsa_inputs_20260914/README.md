# K=7 DSA input and FP32 head-gate work, 2026-09-14

Status: implemented, **default off**. The pinned Linux CPU checks and full native
compile pass. GPU numerical/replay checks and timing are pending. No decode
step/s, tokens/s, acceptance or answer-quality improvement is claimed.

Implementation/CPU base: `32fb2892ed7a17c8eda8f16e845bc0724ad3fd94`
(#909–#912, including HY defaults). Queue/oracle base now includes
`0d3d6d5944bccf0739c5ea5ec8cea04347a64a1e` (#913). Its MoE changes do not alter
the three compiled DSA sources or the focused CPU test modules. Their existing
validation is reused; the source oracle was refreshed against this newer base.

## Changes

- Both DSA queries (`mla.q_b` and `idx.wq_b`) read the same smoothed normalized
  input. Pack it once into invocation-owned FP8/scales and reuse it. Each existing
  W4 owner keeps its own weights, N, split-K choice, MMA and reduction. There is
  no concatenated weight or change to GPTQ calibration.
- Fuse KV RMS normalization, BF16-to-FP8 conversion and mapped latent scatter.
  Keep both BF16 rounding boundaries, the four-warp 512-element reduction, and
  read context/block tables on every graph replay. This removes the BF16 and
  FP8 intermediate allocations and two kernel launches per DSA layer.
- Bind only captured K=7 rows 8/16/24/32. Eager prefill and other widths retain
  their existing path. Boot evidence requires both changes at every DSA layer
  and every declared capture width.

There are 11 DSA layers. C=1 query projection gains a separate pack launch, so
the combined candidate removes **11 launches per target forward** at C=1;
with both C=4 queries already using input packing, it removes **33**. These are
source operation counts, not a measured latency benefit. The shared input costs
50,688 transient bytes per invocation. Persistent weights, KV capacity and
recurrent precision are unchanged; CUDA graph pool bytes remain unmeasured.

Select the candidate in experimental boot with `STK_decode_dsa_inputs=1`.
Its rollback is `STK_decode_dsa_inputs=0`, and the knob expires on 2026-09-30.
Production stays off pending paired GPU evidence. This change is separate from
PR895 compact KDA and the concurrent MoE/precision work.

## Additional FP32 head gate

`STK_decode_indexer_gate=1` independently selects a fixed K=7 owner for the
replicated indexer projection. It requires `decode_fastpaths=1`; production
defaults to zero, rollback is `STK_decode_indexer_gate=0`, expiry 2026-09-30.
The owner retains the original FP32 `[32,4096]` weight after preparation.
There is no weight transpose, resident allocation or recurrent-state change.

One kernel splits K into 16 tiles, sharing each four-head weight tile across
eight input rows. It loads BF16 inputs directly and performs FP32 products and
reductions. The existing indexer boundary reduces those partials and applies
the original query scale and softmax scale in their original order. This
replaces the activation `.float()` allocation and the small cuBLAS projection;
it saves **one launch per DSA layer**. Combined with the input changes above,
source launch counts decrease by **22 per target forward at C=1, 44 at C=4**.
These counts are not a measured timing or consumer-throughput benefit.

Transient partial storage is 16/32/48/64 KiB per invocation at C=1/2/3/4.
Graph-pool retention is still unmeasured. This reduction tree differs from
cuBLAS, so preserved FP32 precision does **not** establish bitwise agreement
or unchanged acceptance. The separate switch keeps that numerical axis
independently selectable from the query-pack and latent-write changes.

The historical overlay split-K result is motivation only. That implementation
used transposed weights, a separate reduction launch and admitted M<=16; the
new native path uses the original weight layout, fused reduction, and all four
K=7 widths. Its performance cannot be inferred from the old overlay result.

`head-gate-cpu.log` records 70 checks: 55 passed, 15 require a reserved GPU.
This includes kernel-package CI, config dependencies, ownership, every capture
width, and the actual model handoff through the boundary to pool selection.
`head-gate-compile.json` records the cached full dense extension and four new
SM121/PTXAS cells: contiguous/strided partial inputs and full/partial boundary
reduction. CUDA remained hidden in the pinned runtime. The first CPU run
caught the missing entry in the explicit knob test allowlist; it was corrected
before this passing run.

`head-gate-interpreter.json` runs the actual partial kernel in Triton's CPU
interpreter: 24 cases across all four widths, contiguous/padded input strides,
zero inputs and changed signed inputs. Every partial was written, surrounding
guards were untouched, and the maximum row-relative FP64-reference error was
`2.18e-7`. This is address/formula proof; compiled GPU rounding remains pending.
Reproduce with the pinned image, GPUs hidden and `TRITON_INTERPRET=1`, running
`python probes/engine_decode_head_gate_cpu.py`.

The short GPU gate adds all 11 real FP32 head-gate weights, C=1–4, changed
strided inputs, poisoned partial/output buffers, both replay orders, independent
FP64 projection comparisons and 50 deterministic replays. It checks query/key
bytes separately from gate error, then synthetic pool-set sensitivity at
32K/128K. Sensitivity results do not substitute for live acceptance. Timings
include the boundary in both arms and retain B/A/A/B warm/evicted brackets.
Each component records its own failure so one failed gate does not waste the
reservation by skipping the other independent components; the overall exit
still fails if any gate fails.

## Completed evidence

- CI follow-up: `test_relative_imports_resolve_inside_the_vendored_package`
  interpreted `from . import bound_input_cell, extension` as submodule imports.
  `query_pair.py` now names `engine.kernels.dense` explicitly, matching the other
  dense helpers. `ci-import-proof.json` confirms identical imported callables,
  unchanged `QueryPair` method code and unchanged compiled kernels. The GPU
  reservation stays frozen at `0bba87ed`; this import-only fix needs no extra
  GPU run. `ci-fix-tests.log` records the kernel-package and DSA CPU checks.
- `cpu-tests.log`: 77 tests, 64 passed and 13 explicitly skipped CUDA checks.
  Eight focused modules, including new ownership/routing/proof gates and existing
  execution, indexer, K=7 and prefill integration tests. An initial test-copy
  failure (missing launcher) was corrected before this passing run.
- `compile.json`: complete production-flag `kernels.cu` extension build/load plus
  fused latent Triton/PTXAS specializations, with CUDA hidden throughout.
  Runtime image `sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`,
  Torch 2.13.0+cu130, Triton 3.7.1, SM121. CPU containers used `--runtime=runc`,
  no network, 1–2 CPUs and 2–4 GiB, with `NVIDIA_VISIBLE_DEVICES=void` and
  `CUDA_VISIBLE_DEVICES=`. No GPU or model boot was used.
- `oracle.json`: PR875 tool commit
  `95c27e6340159413c44a38eef9e54a84b13f9fc5`, source snapshot comparison with the
  candidate enabled, K=7, C=1/C=4, contexts 32,000/128,000. Layout deltas are
  zero; changed kernel components are **unpriced**, so total speed deltas are
  `null`. `paired-profile-template.json` is deliberately unmeasured.

## Pending short GPU gate

Canonical entry:

```sh
bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
  --lanes dsa_inputs --ranks /home/choiceoh/models/st-glm53-hybrid-gptq-v1
```

The probe covers all 11 real rank query matrices with identical RTN W4 packs
between arms (not a consumer GPTQ/acceptance test), K=7 C=1–4, contiguous and
strided inputs, changed values, poisoned outputs and both replay orders.
The latent gate uses 768-token pages, 32K/128K addresses, a boundary crossing,
rebound tables, all 11 layer offsets and poisoned record padding. Captured
component timing uses B/A/A/B in warm and evicted conditions, excluding eviction
from event intervals. It does not boot a model.

Even a passing component result does not establish 22 step/s. Final adoption
still needs actual consumer decoding and acceptance in the next batched campaign;
the user requested C=1 twice, C=4 once and no new baseline model boot.
