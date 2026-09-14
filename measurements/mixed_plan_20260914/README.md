# M2 preparation: immutable packed route tables

> Historical evidence: PR #895 now retains only packet FFNs. This experiment
> and its current-tree probes were removed. Reproduce using the frozen
> revision named in this record, not moving HEAD. See the
> [scope decision](../../bench/ST_GB10_PACKET_ONLY_20260914.md).

Fresh route planning and admission no longer expand hundreds of thousands of
routes into Python tuples, then rebuild them as JSON and GPU metadata. The
same-build, one-GB10 comparison reduces warm mixed-FFN preparation/admission
medians by **86.8–90.8%** at hot quota 128. This is a component result; the mixed
path still has no serving selector and the PR remains draft.

The [next preparation follow-up](../mixed_prepare_20260914/README.md) measures
joined hot/cold planning, a fused value check and one aligned metadata upload
against this packed preparation's components in a newer single build. Its
sources and timings are separate from the historical comparison below.

## Implementation and exact comparison

- Source: `c33370f9008f13c880f488e50a55f83f05091b37`, including main
  `bec3b6dd` (#918 FC2 word expansion), #919 decode absorb and #917.
  The merge retains compact KDA and packet FFNs beside the new decode defaults.
- `RouteTable` owns immutable little-endian int32 bytes. Its NumPy views cannot
  be made writable; tensor uploads copy their storage. The input arrays can be
  released or changed without changing an already prepared host descriptor.
- Validated expert IDs are always in 0..287. Stable grouping uses uint16 keys,
  retaining the original token/slot order while avoiding the wider sort. Stored
  source/task descriptors and the kernel ABI remain int32. Decode/hot order,
  M16/M32 admission, whole M128 tails, padding and task windows remain identical
  to the retained scalar planner. Every invocation plans fresh routes.
- Binary agreement includes identity, quotas, every source map, counts, tile
  bases and windows, with version and row/column framing. Equal scalar and packed
  plans have the same digest; equal histograms with different token/slot routes
  do not. Cold mapping changes are also detected before dispatch.
- The probe's **legacy** arm selects the retained scalar planners and the old
  JSON digest. The **packed** arm uses the new planner, contiguous tensor copy
  and complete binary digest. Only these host-preparation choices differ; both
  arms use the same current kernels, router, source tensors and shared packs.
  These arms are in one build, not timings from two historical PR heads.

## Real-weight GPU gate

Canonical reservation `st-mixed-plan0914v1`, ticket `17893528621572580`, succeeded
in **84.0 seconds**. The immutable image is
`sha256:8190d08e822e1f9d18dda5a127a5d9a9e8c53ef4b5136d1154f9ea4b7727f1ed`:
Torch 2.13.0+cu130 / CUDA 13.0 on srv4, one GB10, real L3 rank 3-of-4 weights.
`Comm(world_size=1)` is the identity collective; this is not a TP4 NCCL run.
Inputs are synthetic normalized FFN activations routed by the actual profile.
Shared W4/FP8 readers use identical RTN packs from the checkpoint in every arm;
no GPTQ calibration store is loaded. The native FFN also supplies numerical
references and repeat-error measurements, but is not the timing baseline.

Four D=8/32 × P=9240/32768 cells × two hot quotas (0/128) × two planning arms ×
four samples yield **64 complete measurements**. Each sample creates fresh
routes, plans, GPU storage and admission. Arm and quota order alternate. The
table uses hot quota 128 and three warm samples per arm/cell, excluding each
arm's conservatively marked first sample. Full data for quota 0 is also retained.

| D rows | P rows | Prepare/admit old → packed | Reduction | Decode ready old → packed | Full FFN completion old → packed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 9240 | 124.54 → 15.99 ms | 87.2% | 126.08 → 17.47 ms | 146.85 → 38.32 ms |
| 32 | 9240 | 129.36 → 17.08 ms | 86.8% | 133.15 → 20.13 ms | 156.09 → 40.88 ms |
| 8 | 32768 | 407.12 → 37.38 ms | 90.8% | 408.50 → 38.75 ms | 471.90 → 102.83 ms |
| 32 | 32768 | 377.48 → 39.94 ms | 89.4% | 380.52 → 43.32 ms | 443.90 → 110.35 ms |

All times start before the actual router and fresh preparation. Prepare/admit
is host wall time and can leave padding/copies queued on the stream; decode
ready and full completion include those writes and synchronize their output
events. Planning substages are included in the raw samples. First-use builds,
source copies, finite checks, metadata allocation, padding and admission are
not amortized over repeated use of a prepared owner. Stage synchronizations
are part of this component measurement, which ran beside production.

Every measured decode relative maximum and RMS error is **zero**. Worst packed
prefill relative maximum/RMS error is **0.976% / 0.236%**, within the unchanged
2% / 0.4% component gate. Native BF16 scatter repeat error is also recorded.
Cancellation at queued/decode/cold phases, foreign-stream consumers, source
and shared-pack immutability, and final retirement passed. Peak Torch
allocation is **3.874 GiB** within the 8 GiB budget; the raw legacy field
`scratch_peak_bytes` includes weights and outputs, not only scratch.

The 16–40 ms remaining preparation cost and first cold packing still matter.
This does not establish full-model TTFT, output tok/s, acceptance or generation
quality. M3 still needs actual arrivals, layer continuation, cache/graph
ownership and a matched 32K/128K C=1/C=4 serving comparison. Earlier S/P/M1
reservations retain their own sources and proof scope; this run replaces none
of their distinct state/route/transport gates.

## CPU and compiler evidence

The implementation passed [CI](https://github.com/choiceoh/stkernel/actions/runs/34799225228/job/103838316286).
The pinned Linux gate passed **344 tests / 26 CUDA skips across 370 discovered
tests in 46 isolated modules**. `cpu.json` retains each module's result and
source manifest; `cpu_runner.py` reproduces the gate. Six new packed-plan tests
include 168 width/tail/quota combinations, all four long shuffled-route cells,
immutable host/upload storage, endianness, invalid IDs and descriptor changes.
The existing real four-process Gloo test now also agrees mixed host
representations and refuses a single-rank cold mapping change. S/P execution,
current decode integration, ownership and canonical admission gates also pass.

`cpu_benchmark.json` measures the same fresh scalar/packed CPU pipeline on srv2:
4 samples per arm, one unmeasured warmup, alternating order, one CPU, 3 GiB,
GPU hidden. Input generation and GC are outside timing; plan construction,
descriptor hashing and the cold-source CPU tensor copy are inside. Whole cold
upload bytes match between arms. For hot quota 128, medians change from
**110.6–111.7 → 7.05–7.33 ms** at P=9240 and
**438.4–441.2 → 24.35–24.40 ms** at P=32768. This CPU fixture is independent of
the real-router GPU measurement above and is not a decode throughput result.

`compile.json` passes all **eight actual SM121 CuTe/PTXAS/TVM-FFI builds**:
hot/cold producers and ordinary/prepared M16, M32, M128 bodies. The two prefill
lengths reuse the same dynamic compiled handle. The ordinary kernel AST pin
was independently recomputed from reviewed main `bec3b6dd`, retaining #918's
FC2 scale reconstruction in both ordinary and prepared C1 bodies.

CPU and compiler runs use immutable image
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`
with Torch 2.13.0+cu130, NumPy 2.2.6 and CUDA hidden/uninitialized. This image
differs from the available srv4 GPU image; both identities are explicit.

## Reproduction and artifacts

```sh
# CPU only, in the pinned image with GPU hidden and PYTHONPATH=/repo:
python3 probes/engine_mixed_plan_bench.py --samples 4 --output /out/cpu_benchmark.json
python3 probes/engine_mixed_completion_compile.py --output /out/compile.json
python3 /out/cpu_runner.py

# Only through the canonical fleet queue, from the frozen clean checkout:
ST_IMAGE=sha256:8190d08e822e1f9d18dda5a127a5d9a9e8c53ef4b5136d1154f9ea4b7727f1ed \
ST_PROBE_TREE=st-mixed-plan-c33370f9 \
bash bench/fleet.sh run --gpu --detach st-mixed-plan0914v1 20 \
  "PR895 packed planning versus scalar JSON on one frozen build" -- \
  bash probes/run_engine_probe.sh probes/engine_mixed_tickets_check.py \
  --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --ckpt-meta /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --samples 4 --compare-planning --output /cache/st-mixed-plan0914v1.json

# Rebuild the table from the retained raw results without a GPU:
python3 measurements/mixed_plan_20260914/summarize.py
```

Use a new reservation name and probe tree for a new run; the recorded admission
and its remote checkout remain frozen. `gpu.json` is the raw report,
`gpu_summary.json` is the deterministic summary, and `gpu_admission.json` records
the successful canonical reservation. CPU/compiler/GPU source manifests match
the implementation and are retained with their reports.
