# cuBLAS FC split-K improvement — RTX 5050, 2026-09-17

**The prepared FC candidate now beats both the direct cuBLAS path and DeepGEMM
on this RTX 5050.** It is enabled in explicit preparation for BF16 FC shapes
M=8/16, N=4096, K=20480. Serving FP8Linear is unchanged; this is not GB10
admission, engine throughput, or acceptance evidence. No fleet queue or service
restart was used. The user explicitly authorized using the 5050.

## Independent repeated comparison

Each entry averages all four arm samples from two B/A/A/B holdout brackets,
after algorithm preparation. Every paired candidate sample beats its baseline
by more than 2%. Timing includes BF16 quantization, scale writes, GEMM, and final
BF16 output. The split path includes its FP32 partial reduction. Compilation,
weight repacking and algorithm search are outside the warm captured interval.

| FC rows | Baseline | Baseline ms | Split cuBLAS ms | Latency change |
|---|---|---:|---:|---:|
| 8 | Direct cuBLAS | 0.358056 | 0.315628 | −11.85% |
| 8 | DeepGEMM | 0.346873 | 0.315433 | −9.06% |
| 16 | Direct cuBLAS | 0.361208 | 0.322289 | −10.77% |
| 16 | DeepGEMM | 0.352303 | 0.322301 | −8.52% |

[Raw FC receipt](rtx5050/split-fc.json), [summary](rtx5050/summary.json).
The direct baseline is the best full-K finalist from the same preparation;
it is bound afresh and measured again during holdout. DeepGEMM consumes the
same original FP8 weight bytes. Input values are seeded synthetic BF16 and are
changed before holdout; these are not captured live model activations.

The four head shapes remain on direct cuBLAS and pass the same final C++ binding:
M=7/8/14/16 latency is 0.593344 / 0.594336 / 0.598272 / 0.598976 ms, respectively,
12.38–13.09% below their paired DeepGEMM baselines. [Head receipt](rtx5050/split-head.json).
These are selected preparation brackets, not the additional FC holdout brackets.

## Implementation and cost

Split K into five 4096-column matrices. A one-warp Triton producer writes all five
activation partitions and MX scale arrays in one launch. One cuBLASLt strided
batch computes FP32 partials; a final Triton kernel sums in FP32 and casts once
at the BF16 output. FP8 bytes and MX scale bytes must match the existing producer
exactly. The weight is rearranged, never requantized.

The native plan validates batch count, rank, exact dimensions, output dtype,
alignment and buffer overlap. Every binding owns its descriptor, activation
buffers, FP32 partials and workspace; no search or allocation occurs on replay.
The admitted batch algorithm is ID 70 / tile 20 / stages 36 with zero workspace.
Five partitions are a bounded FC candidate, not a claim that five is optimal
on GB10. Preparation compares it against the already selected whole-K path and
requires both pairs in each of two brackets to clear a 2% margin.

**Resident cost:** the repacked FC weight is 80 MiB plus 2.5 MiB of MX scales,
while the original DeepGEMM weight remains resident. Private FP32 partials are
0.625 MiB for M=8 and 1.25 MiB for M=16, in addition to the normal activation
and output buffers. A weak cache shares the 82.5 MiB repack across prepared row
shapes while owners exist; original tensor references and version counters
prevent address reuse or an in-place update from retrieving a stale repack.
Bindings require immutable weight and buffer storage for graph lifetime.

## Validation and limitations

- All six real-pack shapes pass `allclose(rtol=.01, atol=.001)`, two changed-input
  CUDA graph replays, and zero additional Torch allocation bytes after binding.
- Three dedicated native GPU tests pass: per-batch scale strides and dtype/shape
  rejection; query versus execution scale ownership; independent bindings and
  changed-input graphs, alias rejection, and mutated-weight rejection.
- CPU/interpreter tests: **21 passed, 3 GPU-only tests skipped**. Split producer
  checks include exact FP8 bytes, independent MX layout, padding/redzones and an
  FP32 cancellation case that would fail with BF16 partial accumulation.
- GPU-hidden build/dlopen and **80 SM121 Triton variants** pass, including both
  new producers and reductions. The one-warp producers use no shared storage or
  CTA barriers. This is compilation evidence, not SM121 runtime validation.
- Final receipts match all recorded current kernel/probe source hashes.

[CPU tests](rtx5050/split-cpu.log), [GPU tests](rtx5050/split-native-tests.log),
[SM121 compile](rtx5050/split-compile-sm121.json).
One intermediate probe stopped at the older peak-minus-resident allocation
check; that record omitted the signed delta, so it does not identify whether the
allocator observation was allocation or reclamation. The final check uses the
monotonic `allocated_bytes.all.allocated` counter and records zero for every cell.
The failed receipt/log are retained as `allocation-counter-error.*`.

Hardware/runtime, image and DeepGEMM dependency identity, and the two actual pack
hashes are inherited unchanged from the [initial comparison](../st_cublaslt_compare_20260917/README.md).
RTX 5050 / SM120 / 20 SMs, CUDA 13.2, Torch 2.13.0+cu132, cuBLASLt 130400.
The direct host lock excludes cooperating ST probes; it cannot exclude Windows
desktop activity. Before/after GPU utilization was 0%; see the CSV receipts.
There is no GB10 execution, live acceptance or one-pass claim.

## Reproduction and rejected experiments

Run `run_5050.sh` on the owned `ost-97x` scratch after syncing the recorded sources.
It uses the pinned image, host lock, 2 CPUs / 4 GiB and the same two immutable packs.
It runs three native checks, two FC cells and four head cells. No queue or service
control is involved; its cleanup only addresses its own named container.

The experimental scripts and patches preserve the alternatives considered:

- `compare_layouts.py` + `transpose_plan.patch`: swapped operands and a transpose
  epilogue cost about 0.8% more. Exhaustive tile/stage search checked 3618 configs
  versus 447 but still admitted only four. Neither change is retained in the
  runtime implementation. Raw evidence: `rtx5050/fc-layouts.*`.
- `compare_split.py` + `split_plan.patch`: separate-stream FP32 splits for
  P=2/4/5/8/10. Four/five splits helped but are superseded by the single batched
  call; no extra-stream synchronization enters the final bound path.
  Raw evidence: `rtx5050/fc-split.*`.
- `compare_batch.py` + `batch_plan.patch`: prototype five-way strided batch.
  The first M=8 screening sample was **4.438 ms**; later paired samples stabilized
  around 0.328 ms. It is retained, not discarded as a fast-sample selection.
  Final implementation receipts use explicit warm replays and repeated brackets.
  Raw evidence: `rtx5050/fc-batch.*`.

Historical patches apply to `git show d670dcef:engine/kernels/dense/cublaslt.cpp`;
write that base to a temporary file and use `patch -o <name>_plan.cpp <base>
< <name>_plan.patch` in this directory. The scripts load these isolated native
namespaces. They are research artifacts, not runtime imports. The final harness
uses the maintained native binding and `cublaslt_split.py` directly.
