# GB10 MLA copy pipeline and split reduction

Baseline: public `main` revision `c10e8e9d` after the TMA/ldmatrix adoption in
PR #634. These changes are adopted as source defaults under the operator's
instruction to proceed without benchmark results. GPU execution, numerical
equivalence, graph replay, performance, and serving rollout remain pending.
The fleet is reserved by the concurrent serving-recovery task; these checks
used only the local x86_64 CPU and CUDA compiler.

## Changes

- **Query copy overlaps the initial KV transfer.** The 16 KiB Q tile uses
  `cp.async.ca.shared.global` directly into its existing shared allocation.
  It joins the first KV async group, without an extra commit or wait. The
  first existing `wait_group<1>` completes that group before the CTA barrier
  and the first matrix read. `.ca` keeps the L1/L2 cache hint for reused Q;
  the scattered KV ring retains `.cg`. Empty splits issue no Q reads.
- **Every CTA participates in a three-block cluster's merge.** Output heads
  use `warp * splits + rank`, assigning 6/5/5 heads instead of 8/8/0. Both
  cluster barriers remain, including the final barrier protecting peer shared
  memory lifetime. The serving cluster domain remains two or three blocks.
- **Global split partials use coalesced addresses.** Consecutive lanes read
  consecutive dimensions with `e * 32 + lane`, as in the existing DSMEM
  reduction. The output stores use the same permutation. Each output still
  accumulates splits in the original order, with the same FP32 FMA and BF16
  rounding; only thread ownership and memory addresses change.

No new precision, workspace allocation, serving knob, or dispatch domain is
introduced. The shared allocation remains 46,976 bytes in both kernel paths.

## CPU and compiler evidence

`compile.json` records NVCC 13.0.88 compilation of the actual extracted kernel
bodies with `-O2 -std=c++17 -arch=sm_121a --cubin`, matching the serving driver
target and optimization level. It does not compile the full Torch binding.

| Compiler property | Baseline | Current |
|---|---:|---:|
| Ordinary registers | 111 | 110 |
| Cluster registers | 64 | 64 |
| Spill loads/stores, both paths | 0 | 0 |
| Ordinary static instruction sites | 1,352 | 1,352 |
| Cluster static instruction sites | 1,544 | 1,544 |
| CTA barrier sites, ordinary / cluster | 8 / 9 | 8 / 9 |

Each path replaces one static `LDG` plus `STS` site with a `LDGSTS` async-copy
site. Loop execution counts differ from static site counts; the totals above
include padding and do not measure speedup, traffic, occupancy, or utilization.

`stream-host.json` runs the **extracted copy statements, commit/wait sequence,
and address expressions** against a delayed-copy CPU model. Copies complete
only when the wait-group rule requires them. Sixty slice cases cover twenty
boundary-focused lengths from 0 to 2,176, starting offsets 0/1/17, changing
query rows, and 256 threads per case. The model verifies copied Q/KV bytes,
no unfinished copies at the end, no Q reads for empty slices, head ownership,
and all 512 reduction dimensions. A negative control with an insufficient
wait is rejected. This models per-thread completion guarantees and validates
the CTA barrier placement; it does not execute GPU threads or prove device
visibility, liveness, or numerical integration.

`fp8-host.json` preserves the earlier CPU check of all 65,536 FP8 pair
conversions and Q/P/PV matrix-fragment addresses. The conversion helpers and
MMA arithmetic are unchanged. This is not a GPU NaN-encoding test.

The nine kernel/source-boundary tests and four MLA driver/workspace contract
tests pass. Python syntax checks and `git diff --check` also pass.

## Reproduction

```sh
git show c10e8e9d:engine/kernels/mla/glm53_megakernel.cu > /tmp/st-stream-baseline.cu
python3 probes/engine_mla_compile_check.py --baseline /tmp/st-stream-baseline.cu --cuda-root /usr/local/cuda --output /tmp/st-stream-compile
python3 probes/engine_mla_stream_host_check.py --output /tmp/st-stream-host
python3 probes/engine_mla_host_check.py --cuda-root /usr/local/cuda --output /tmp/st-stream-fp8
```

When a GPU execution window is available, use the existing comparison probe
with the same baseline and the current engine source. It covers the ordinary
and cluster paths, full/ragged/empty/duplicate slot lists, changed-input graph
replays, and a separate FP32 attention reference:

```sh
python3 probes/engine_mla_hardware_check.py --baseline /tmp/st-stream-baseline.cu --candidate engine/kernels/mla/glm53_megakernel.cu --output /tmp/st-stream-gpu
```

The completion model follows [NVIDIA PTX 9.0 async-group semantics](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)
and the [cache qualifiers for cp.async](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).
Peer lifetime follows [CUDA 13 distributed shared memory](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#distributed-shared-memory).
