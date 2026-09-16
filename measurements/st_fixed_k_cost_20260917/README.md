# Fixed K7 verification cost

Original base: `da917a4d`; integrated main: `53019ef8`. K remains 7, verification remains 8 rows per request,
and KDA state remains FP32. Three independently controlled changes:

* C1 resident MoE waves choose 48/44/40/36/32 active CTAs only when the
  choice does not add a wave. Every work item keeps one owner. The control
  is `resident_waves=False` in the static configuration; kernel keys differ.
* The direct mHC consumer writes the existing C1 FP8 input pack from its
  rounded BF16 layer input. The next bound KDA input projection consumes
  that invocation's pack. Observed/calibrated or unsupported cells cannot
  consume it. `net.mhc_input_packs=False` disables only this new boundary;
  existing KDA output producer packs remain enabled.
* MLA pairs share the multiset union of two selections. Separate membership
  bits preserve repeated slots and each query's selection. Context shards
  retain GPU parallelism at 8/16 rows and write FP32 partials, followed by
  one BF16 merge. Weak overlap uses the two original lists in the same
  split kernel. `ENABLE_MLA_DECODE_PAIR=False` selects the same-build control.
  Tree attention and large prefill keep their existing paths.

The new defaults are enabled per the ST charter D11. This is an adoption
choice, not a measured speed claim. Temporary controls are for this
comparison and should be retired after the fleet verdict.

Validation entry points (canonical fleet runner):

```
probes/engine_kernel_check.py --lanes fixed_k_compile --output /cache/fixed-k-0917/compile.jsonl
probes/engine_kernel_check.py --lanes fixed_k_cost --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank0of4.safetensors --output /cache/fixed-k-0917/gpu.jsonl
```

The compile lane requires `CUDA_VISIBLE_DEVICES=` and a container without
GPU devices. It builds the complete dense and MLA Torch extensions plus
both actual MoE kernels. The GPU lane checks every real mHC coefficient
set, pack bytes, changed packet descriptors, sparse MLA against its FP32
reference, changed selections/empty rows/duplicates, and real L3 MoE
weights with changed routing. Captured B/A/A/B component intervals include
pack/union preparation and the merge.

Current evidence: the Linux CPU gate ran 81 tests (72 passed, 9 GPU tests
skipped); the separate execution/fastpath gate ran 16 (15 passed, 1 GPU
test skipped). The added serving-proof gate passes locally. Full dense/MLA
and both MoE handles built in Torch 2.13.0+cu132,
CUDA 13.2. Native C1 packet mHC uses 128 registers and 29,232 shared bytes.
The first GPU component verdict is recorded below; the revised candidate
and TP4 consumer 32K/128K quality, acceptance, TTFT and tok/s are pending. C1/C2 is the
current serving shape; any C4 consumer comparison needs a matching declared
four-request capacity in both arms.

The GPU component reservation is `fixedkgpu-0917`, pinned to `18e14162`.
It completed successfully; numerical qualification and performance are separate below.
The subsequent proof-only changes expose and require actual target execution
in `ST_NATIVE_EXECUTION.fixed_k_cost`. The compile artifact records its own
source and native binary hashes.

## First GPU verdict (`18e14162`)

`gpu-v1.jsonl`: all 90 real mHC coefficient sets pass bitwise output and
pack equality at four magnitudes, with and without changing TP4 packet
descriptors. MLA passes all 30 geometry/selection fixtures and repeated
graph replay, with relative L2 error below 0.0021. Real L3 MoE passes
changed routing and zero-output replay with no observed output difference.

Performance is rejected for the first fused mHC/MLA implementation:
packet mHC +2.11% (20.19 -> 20.61 us), C1 MLA identical selection
43.11 -> 142.99 us, C2 59.84 -> 222.99 us. The MoE wave change has no
clear win (U40 evicted 679.47 -> 679.21 us). These are component intervals,
not consumer tok/s. The second candidate removes repeated mHC conversion,
specializes pack-writing consumers, and adopts the stock decode kernel's
shared-Q and matrix-load instructions for MLA. Its qualification is pending.

## Second GPU verdict (`a236b5fa`)

`gpu-v2.jsonl` passes every numerical/replay gate. Revised C1 MLA improves
from 142.99 to 85.49 us for identical selections, but its matched control
is 43.17 us; it is still rejected for speed. C2 is 61.72 -> 107.15 us.
Packet mHC is 19.40 -> 20.50 us (+5.66%). The third candidate assigns one
warp group to each MLA query and transposes completed mHC values for
warp-local packing. The MLA profiler runs only after all unprofiled timings.

## Third GPU verdict (`03cb3652`)

`gpu-v3.jsonl` passes all numerical/replay gates. Warp-local mHC packing
reaches parity without packets (18.57 -> 18.56 us), but packet mode still
regresses (19.63 -> 20.17 us, +2.73%). MLA remains slower: C1 identical
43.07 -> 87.40 us. Diagnostic profiling identifies substantial union
preparation; overlapping PDL kernel durations must not be added as latency.
The fourth candidate removes per-stripe output-counter contention, widens
shared KV once for both queries, and executes weak-overlap rows concurrently.

## Fourth GPU verdict (`8f3fe48a`)

All correctness gates pass. C1 identical MLA is 43.00 -> 84.08 us, disjoint
43.05 -> 74.11 us. Hash preparation remains substantial despite compact
reservations. The next candidate uses bounded radix grouping and prefix
counts for the exact multiset union. mHC moves packing onto the 32 finished
projection CTAs; each row is published and rearmed before graph completion.

## Fifth GPU verdict (`2715eda1`)

All correctness gates pass; both changes are rejected for performance.
Packet mHC rises 19.56 -> 22.65 us, and C1 identical MLA is 43.05 -> 92.33 us.
Ready-row helper fences/counters and radix preparation are removed. The sixth
candidate shares only matching KV rows within the current 16-slot tiles,
retains each original sparse list/split, and merges in the same resident launch.
No union, sorting, separate merge launch or per-replay barrier reset is needed.
Changed tile order and asymmetric lengths join the existing numerical gates.
