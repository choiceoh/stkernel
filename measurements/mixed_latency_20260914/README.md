# M2 latency/adoption gate — 2026-09-14

**Retired on 2026-09-14.** Mixed kernels, planners, tickets, profile bindings and
canonical probe entries have been removed. PR #895 now retains only the ordinary
prefill packet FFN. The measurements below remain unchanged historical evidence.
Reproduce from each `sources.json` revision; the last implementation is
`3ac2b17f34a21eebd9f2947dda61461adca4bad8`.
[The packet-only decision](../../bench/ST_GB10_PACKET_ONLY_20260914.md) supersedes
the former M3 follow-up.

**The 52 ms target is not met. M2 is still slower than the actual ordinary FFN.**
Preparation improvements do not establish an adoption benefit. S/P stay opt-in,
M has no serving selector, and PR #895 stays draft.

The target used here is complete D8/D32 + 32K-prefill FFN wall time, starting
before routing and fresh preparation and ending after output synchronization.
It is one L3 FFN on one GB10 with actual rank 3-of-4 weights and identity
communication, not a whole-model or TP4 latency target.

## Current measured result

Frozen source `3ac2b17f34a21eebd9f2947dda61461adca4bad8`, reservation
`st-mixed-native520914v1`, ticket `1789361160696118`. Eight samples per cell,
with the first sample excluded: seven warm samples per native/candidate arm.
The table uses hot quota 128. Quota 0 and every raw sample are also retained.

| D | P | Native decode ready | Mixed prepare/admit | Mixed decode ready | Native full FFN | Mixed full FFN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 9240 | 3.38 ms | 8.87 ms | 10.34 ms | 47.98 ms | 54.00 ms |
| 32 | 9240 | 7.43 ms | 10.94 ms | 16.75 ms | 52.43 ms | 60.51 ms |
| 8 | 32768 | 3.21 ms | 26.15 ms | 30.06 ms | 137.66 ms | 154.73 ms |
| 32 | 32768 | 7.49 ms | 23.95 ms | 29.64 ms | 142.31 ms | 152.03 ms |

Both arms use the same activations, actual router, weight/scale owners and
DenseLinear RTN W4/FP8 shared packs. Native D completes and synchronizes before
native P, matching the candidate's D publication boundary. Native blocks
alternate before/after mixed blocks. This run uses explicit cold drain and
sequential shared prefill; the optional shared overlap is off.

There are **64 mixed and 32 native measurements**. Decode relative maximum/RMS
error is zero. Worst prefill relative maximum/RMS error is **0.976% / 0.225%**,
inside the unchanged 2% / 0.4% component gate. All 260 value/scale differential
checks and six cold-producer/padding byte comparisons pass. Cancellation,
foreign-stream consumption, immutable source/pack checks and retirement pass.
Peak Torch allocation is **5.319 GiB**, including validation, weights, sources,
workspace and outputs; it is not a resident-memory reduction measurement.

The GPU runs beside another engine. Across separate reservations the ordinary
32K baseline changes from 60–62 ms to 134–142 ms. Do not splice the earlier
baseline into later candidate timings or normalize a noisy result to 52 ms.
Each reservation retains its own native comparison. **Contention does not
excuse the candidate being slower than its same-run native arm.**

## Implemented changes

- A pending GPU value check overlaps independent CPU route planning; its
  readback is drained on both normal and exceptional exits before dispatch.
- `drain()` agrees one explicit full cold invocation across ranks. The ordinary
  `advance()` still submits one bounded task window. A full drain makes no
  interleaved-decode or preemption promise.
- A token CTA reads H4096 once and sends its live routes to an inverse
  token/slot destination table. Consecutive equal expert scales reuse FP4/SFA
  quantization; distinct scales retain their own quantization. Moved hot
  routes are excluded. Old/new FP4, SFA and routing bytes match exactly.
- Only reachable M128 padding is initialized, instead of clearing the full
  roughly 600 MiB cold packed-input plane. Live rows are overwritten by the
  producer. Complete/partial tiles and untouched live bytes are checked.
- The fixed CPU plan is built in one C++ call, with input validation and stable
  per-expert ordering. Its immutable descriptors match the scalar and NumPy
  planners, including all widths, quotas, tails, endian/stride cases and errors.
  The tiny C ABI uses the existing C++ toolchain and source-addressed cache;
  it links neither Torch nor CUDA. No Mojo runtime dependency is introduced.
- An optional shared-prefill fork uses the existing join helper around a full
  drain. It joins on errors, retains its result until finish, and includes the
  policy in rank agreement. It is off by default and has no demonstrated
  latency win in these runs.

On the matched ARM64 CPU benchmark, with eight samples per arm and fresh input
plans, the actual NumPy/native planning medians are:

| D | P | NumPy | Native |
| ---: | ---: | ---: | ---: |
| 8 | 9240 | 2.76 ms | 0.83 ms |
| 32 | 9240 | 3.06 ms | 1.01 ms |
| 8 | 32768 | 10.11 ms | 2.52 ms |
| 32 | 32768 | 10.10 ms | 2.83 ms |

The fresh host-library build takes **200.6 ms** in this image. Runtime row
counts do not compile another library. The GPU byte gate uses the independent
NumPy planner so it does not hide native-planner first build outside the first
measured admission. Warm CPU planning is not an FFN latency result.

## Rejected N128 consumer

`38e275d6`, reservation `st-mixed-n1280914v1`, tried the existing raw-scale N128
input-reuse body against SF6. CPU and actual compilation passed, but GPU full
FFN validation failed with relative maximum error **2.247%**, above the fixed
2% limit (relative RMS 0.258%). The limit was not relaxed. The prepared N128
consumer, scale-expansion selection and its probe flag were removed in
`3ac2b17f`. Existing unrelated main-branch N128 experiments are unchanged.
The failed raw report and admitted source are retained; no speed verdict is
drawn from the incomplete run. Later probes record active sample identity on
failure, which this earlier report did not yet retain.

## Validation and remaining work

Frozen Linux ARM64 CPU gate: **153 passed / 3 skipped across 156 tests in 11
isolated modules**, including four-process Gloo order/refusal, output sums and
retirement. Ten actual SM121 CuTe/PTXAS/TVM-FFI builds, the Triton value checker
and the host C++ library pass. Ordinary/prepared dynamic row extents reuse one
compiled handle. Implementation CI passed on
[3ac2b17f](https://github.com/choiceoh/stkernel/actions/runs/34807058450/job/103860896224).
Local non-Gloo validation passes 72 tests; local Gloo connection initialization
stalled and was stopped. The actual frozen Linux Gloo test passes. See
`local_gloo_limit.txt`; no local multiprocess success is claimed.

The rejected implementation's issue was structural: admission waits for prefill routes,
planning and full descriptor agreement before decode can run. Reducing those
costs does not remove that dependency. A follow-up must move preparation away
from decode arrivals and reduce cold compute without weakening numerics.
These were requirements for a future mixed implementation, which is no longer
planned. The ordinary-path campaign retains matched consumer timing and
32K/128K C1/C4 quality/acceptance gates. This record does not claim TTFT or tok/s gains.

`sources.json` binds each CPU/GPU record to its admitted source. Rebuild the
tables with `python3 measurements/mixed_latency_20260914/summarize.py`; verify
all recorded source hashes with `python3 measurements/mixed_latency_20260914/verify_sources.py`.
Earlier runs retain their original preparation and overlap policies and are
not presented as a single cross-revision A/B comparison.
