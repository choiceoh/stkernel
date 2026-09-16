# Fixed K7 mHC input packing

Original base: `da917a4d`; integrated main: `72b708cd`. K remains 7,
verification remains 8 rows per request, and KDA state remains FP32.
The final candidate retains mHC producer packing only. Resident MoE waves
had no clear gain; seven implementations of adjacent-query MLA sharing
passed numerics but remained slower, so both changes were removed.
Their implementations remain in the recorded commits below.

The direct mHC consumer writes the existing C1 FP8 input pack from its
rounded BF16 layer input, using warp-local 128-column packing. The next
bound KDA input projection consumes that invocation's exact pack.
Observed/calibrated or unsupported cells cannot consume it.
`net.mhc_input_packs=False` disables only this new boundary; existing KDA
output producer packs remain enabled. Ordinary mHC template instantiations
and occupancy entries retain their original arithmetic and register lifetimes.
`ST_NATIVE_EXECUTION.fixed_k_cost.mhc_input_packs` requires actual consumption
by the target projection before serving starts.

All 90 real mHC coefficient sets pass bitwise output and FP8 pack equality
at four magnitudes, with and without changing TP4 packet descriptors.
The final native mHC bytes were qualified in revisions 3, 4, 6 and 7.
Two distinct-coefficient chain comparisons reduce the packet boundary
interval by 19.12% and 19.43%. The graphs contain all 90 coefficient sets
and include the separate pack in the control. Serving only fuses eligible
KDA input boundaries: this is not a whole-engine speed claim.

Final validation entry points:

```
probes/engine_kernel_check.py --lanes fixed_k_compile --output /cache/fixed-k-0917/compile.jsonl
probes/engine_kernel_check.py --lanes fixed_k_cost --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank0of4.safetensors --output /cache/fixed-k-0917/gpu.jsonl
```

The compile lane requires `CUDA_VISIBLE_DEVICES=` and a container without
GPU devices. The narrowed final probe builds dense and checks real mHC
coefficients, pack bytes, changed packet descriptors and captured B/A/A/B
component intervals. Earlier records also contain the removed MLA/MoE
numerical and timing gates. CUDA source/binary hashes and runtime identity
are recorded in each artifact (Torch 2.13.0+cu132, CUDA 13.2, SM121).

Matched TP4 consumer validation is pending. It uses the extended onepass
profile for 2K/32K/128K coverage, C1 twice and the current serving capacity
(C2) once, without 128K C2. C4 is not measured. Compilation and capture
precede the consumer measurements. Quality, acceptance, TTFT, output tok/s
and warm step/s decide adoption; component timings alone do not.

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

## Sixth GPU verdict (`b631febd`)

All 42 MLA fixtures and prior gates pass. MLA C1 identical improves to
42.97 -> 61.52 us, still slower. No local-memory spill was found (64 registers,
0 local bytes). The next iteration removes tile search/ballots and skips empty
query arithmetic. Native mHC remains the warp-local implementation.

The added distinct-coefficient chain materially changes the mHC result:
90 packet boundaries take 1592.05 -> 1287.60 us (-19.12%); nonpacket chain
1522.76 -> 1223.76 us (-19.64%). Each graph contains all 90 real coefficient
sets, no pack clone. These are component throughputs; serving fuses only
eligible KDA input boundaries, so they are not a whole-engine speed claim.

## Seventh GPU verdict (`1f049629`)

All correctness/replay gates pass. Distinct-coefficient packet mHC chain
1601.07 -> 1290.03 us (-19.43%); nonpacket 1520.75 -> 1226.71 us (-19.34%).
Single-coefficient packet interval remains near parity, 19.57 -> 19.72 us.
C1 identical MLA is 43.20 -> 47.50 us (+9.96%); C2 is 59.62 -> 71.18 us
(+19.39%). Even after eliminating union preparation, separate merge,
tile search and empty-query arithmetic, MLA sharing is rejected.
Resident MoE waves remain without a reproducible gain. Final source removes
both unsuccessful experiments and retains the qualified warp-local mHC pack.

## Consumer admission and source-lifetime repair

The first full bracket (`fixedkfull-0917`, control `40272da2`) stopped before
serving: all four ranks raised `draft FC capture needs the retained BF16 source`.
`boot-failure.json` pins the logs. Automatic FC collection inherited from main
was armed after compact drafter storage had retired its reference weight.

Both arms now retain an independent host copy before compaction and move it
back to the reader device only when collecting the shutdown bundle. The live
reader and compact arena are unchanged. A regression test overwrites the original
storage and checks the collected reference, actual output and reader identity.
The focused Linux gate runs 40 tests: 39 passed, one GPU-only skip. The prior
55-test gate (52 passed, three GPU skips) and full CI also passed. One retry
caught a script-mode relative import; commit `69c5420c` uses the absolute import.
No consumer speed claim is taken from either failed boot.

The matched full bracket is `fixedkfull3-0917`: baseline `ffd19b44630b`,
candidate `69c5420c`. The entire tree difference is one boolean assignment,
`net.mhc_input_packs=False/True`. The canonical command is:

```
ST_BRACKET_VALIDATION=full ONEPASS_PROFILE=extended REPO=/home/choiceoh/st-worktrees/fixed-k-cost-0917 bash /home/choiceoh/stkernel/bench/fleet.sh st-chain fixedkfull3-0917 50   "fixed K7 mHC pack same-build B/A with compact source preserved"   -- B=ffd19b44 A=69c5420c
```

Each `NAME=sha` already adds one arm to the order; no extra names are needed
for a single B/A pair. Consumer results follow when the run completes.

The first baseline C1 pass completed with quality 6/9 and no Korean corruption;
its raw measured request rate is 87.30 tok/s (32K: 85.28, 128K: 88.38).
These are observations, not an eligible speed baseline: the harness correctly
clears `decode.windows_med` when quality fails. The baseline ticket retains
its second C1 pass, then returns the failure before the candidate arm. Candidate
`69c5420c` is separately queued as `fixedkcandidate-0917` with the same extended
profile and two runs, so the matched output/acceptance comparison can still be
completed without rerunning the failed baseline a third time.

The same baseline boot is already non-identical across C1 repetitions:
the three 2K outputs contain 2176/3611/4671 tokens in run 1 and
2313/3967/8391 in run 2. Prompt workloads and temperature (0) are unchanged.
Therefore request-rate deltas cannot by themselves establish the small
mHC speed effect. `summarize.py` retains actual request timings and hashes,
and refuses an eligible speed claim when outputs differ or evidence fails.
