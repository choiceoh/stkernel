# K=7 decode step recovery candidate

The serving K=7 verification widths are 8/16/24/32 rows. The original recipe
selected several previously qualified fastpaths only at K=6 widths. Commit
`c4059295` bound the missing widths to the model before capture. On 2026-09-14
the operator explicitly requested ON: both production and experimental serving
now default to `decode_fastpaths=1`. The experimental rollback is
`STK_decode_fastpaths=0` (expires 2026-09-30); production has a fixed default
without an expiry. Bare `ExecutionPlan()` remains neutral like its other lanes.
Base: `613dfa0fa01f` (PR #908). No performance gain has been measured for
this candidate. The unchanged FP32 KDA state, K=7 and enabled prefill
settings remain part of the candidate.

## Recorded speed, before this candidate

The short-window step/s median is not the pooled GPU iteration rate.
`analyze_latency.py` reads rank 0 device-body/TP4-stop timestamps from the
retained profiler-OFF measurement phase. These include varying C1 contexts
and natural response lengths; this table is observational, not a matched
kernel comparison. The builds also differ in acceptance and quality.

| Run | Build | Iterations | Mean device iteration | Pooled device step/s |
|---|---|---:|---:|---:|
| LATESTFIX, K=7 | 8dd9fd0f | 5,346 | 53.220 ms | 18.790 |
| K6 | e11fe12c | 6,336 | 49.344 ms | 20.266 |
| HY, K=7 | 640859a9 | 5,023 | 52.822 ms | 18.932 |

The roughly 15.9 step/s previously reported for K=7 was a short-window
median. It is not replaced or relabeled as a GPU timing result. Neither
GPU metric includes all client/tokenizer/HTTP overhead or is output tok/s.
The underlying run IDs, source paths, SHA256 and stage means/tails are in
`recorded-device-stages.json`. The measurement source is retained on srv2
under `/home/choiceoh/expert-capture/onepass-runs/`.

LATESTFIX averages 48.473 ms forward, 3.087 ms proposal and 1.425 ms
observation per iteration. A separate profiled 2K diagnostic has 42 MoE
calls totaling 26.918 ms and 184 ordinary small GEMMs totaling 9.844 ms.
`diagnostic-kernels.json` preserves that trace's hash and complete kernel
aggregation. These overlapping diagnostic durations cannot be added to
estimate whole-step time. MoE remains the largest visible kernel family;
this change does not optimize MoE or establish a large total speedup.

## Candidate mechanism

- Explicit immutable row declarations are bound to each model's KDA and
  indexer owners. They preserve existing 1/6/7/14/21/28 cells. Preparation
  cannot change another model's global projection allowlist.
- W4 input quantization is shared by output tiles at the bounded M8
  4096-input cells, using the reduced-CTA path. Selected 16/24/32-row
  projection cells use the existing wide packed-input algorithm.
- An explicit native entry accepts ordinary outputs, private partial/counter
  workspaces and replay-time TX descriptors. Wide direct-output kernels
  now retain the descriptor when using packed input. No process-wide probe
  setter is needed to select the candidate.
- Native execution reporting requires every declared layer/width pair and
  each eligible W4 layer/width to execute before opening serving.

W4 packing, row scaling, split geometry, FP32 reduction and output rounding
are retained. Long prefill stays on its existing FP8 path; an eligible tiny
8/16/24/32-row W4 call can also select the bound input route. The drafter's
precision, speculative depth, recurrence and acceptance logic are unchanged.

## Validation completed without a GPU

Pinned existing image:
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`
(Torch 2.13.0+cu130, CUDA toolkit 13.0). Checks used runc, no network, CUDA
hidden, 2 CPUs, 4 GiB RAM, one build worker and explicit Python entrypoint.
No image build, model boot, GPU context, service restart or queue submission
was issued.

- Focused CPU gate: 38 tests, 35 passed and 3 GPU-only skips (`cpu-tests.log`).
  The first copy omitted a launcher needed by an existing test; copying that
  file repaired the test environment before this passing run.
- Added a joined-indexer owner check, then reran the affected module: five
  CPU tests passed and one GPU test skipped (`cpu-owner-tests.log`). This
  verifies actual strided CPU GEMM output, row ownership and copied weights;
  CUDA admission alone is mocked for that check.
- Full native CUDA/Torch extension compiled and loaded without initializing
  CUDA (`native-compile.json`). The source hash matches the committed source.
  New packed-input direct-output specializations use 116/118 registers and
  zero local/stack bytes (`native-bound-resources.json`).
- All eleven SM121 Triton projection/reduction variants compiled with CUDA
  uninitialized (`triton-compile.json`).
- `git diff --check` passed. No GPU numerical, replay or consumer speed gate
  has run for this candidate.

The new `BoundDecodeGpuTests` is prepared for exact same-pack output checks
at every selected shape, mixed strides, zero/changed inputs, shrinking and
growing rows, shared/private scratch, and changed TX destinations with guard
sentinels. It was deliberately not enqueued or run. Existing real-weight
projection probes are also needed before consumer qualification.

## Upgraded Oracle

Tool checkout `st-oracle-pr875`, head
`e2bfbb9afdcc6e8fe1e5fe47ddddd23180b278b3`, compared base `613dfa0f` to
candidate `c4059295` with the retained actual checkpoint config and
`--set decode_fastpaths=1`. `oracle-c1.json` covers 2K/32K/128K C1;
`oracle-c4.json` covers only 32K C4. `--acc 0` is an explicit timing-only
scenario; its output tok/s is not an acceptance forecast.

The Oracle observes the enabled candidate and unchanged cache/state layout.
Total decode delta remains **null** because the modified kernels have no
matched GPU timings. This is an unknown benefit, not 0% or an assumed gain.
The source/config-bound paired templates are intentionally empty. Allocated
state bytes and unpriced model coefficients are not evidence of step speed.

Reproduce the CPU gates in the pinned image with CUDA hidden:

```sh
python3 -m unittest -v tests.test_engine_decode_fastpaths \
  tests.test_engine_decode_k7 tests.test_engine_native_execution \
  tests.test_engine_execution_plans tests.test_engine_knobs \
  tests.test_engine_decode_projection
python3 probes/engine_decode_native_compile.py --output /out/native-compile.json --build-root /out/build
python3 probes/engine_decode_projection_compile.py --k7 --output /out/triton-compile.json
```

## CI fixture repair

The first full PR check reached 1,645 tests and found two stale test doubles
outside the initial focused slice. The pair-owner fixture replaced the whole
projection module and omitted the new row declaration helper; it now mocks
only the CUDA owner and keeps real row validation/dispatch. The drafter W4
fixture now accepts the explicit `bound_input` argument and asserts it is
false, preserving its target/drafter isolation check.

Both failures were reproduced in the same CPU-only image. The repaired pair,
drafter acceptance and bound-fastpath modules then ran 21 tests: 20 passed,
one GPU-only skip, no failures (`cpu-ci-fix.log`). Engine/kernel source and
serving defaults did not change, so the existing compiler and Oracle source
receipts remain applicable.

## Operator-enabled default

The 2026-09-14 ON request sets the shared serving default to 1 in both boot
modes. Experimental `STK_decode_fastpaths=0` remains an explicit rollback;
production refuses environment overrides and remains restartable after the
experimental expiry. No kernel or model execution code changed in this step.

`cpu-default-on.log` records 45 tests across defaults, ownership, drafter
acceptance and execution: 44 passed, one GPU-only skip. The production expiry,
experimental rollback and neutral bare-plan checks passed.
`default-on/oracle-{c1,c4}.json` and their new source-bound templates read the
enabled default without `--set`; contexts remain C1 2K/32K/128K and C4 32K.
They leave the timing delta unknown. The earlier Oracle records above remain
historical records of `c4059295`, not receipts for the changed boot source.
This default change did not restart an engine or submit GPU work.
