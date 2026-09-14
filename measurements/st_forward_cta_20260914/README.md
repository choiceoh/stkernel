# C1 forward: reduce W4 partials inside the CUDA block

Implementation base: `4dbc0713` (#938); final integrated main: `77d80b6c` (#940–945).
Target: K7 C1, 24 consumer step/s.
This changes the forward GEMMs, enabled in the existing default-on decode
fastpath and DSA query owners. It does not change KDA state precision, weights,
MoE kernels or transport protocol. The GPU component gate passes; consumer timing is pending.

The ordinary kernel splits K across three CTAs, writes FP32 partials to device
memory, then uses arrival atomics and fences before folding them. The candidate
extends the existing local-reduction kernel to K=1536/2048/3072. The three warps
for each output tile exchange partials in shared memory and fold in the same
slice order. Input quantization, W4 decoding, MMA K order, FP32 accumulation,
row scales and final BF16 rounding remain in their existing arithmetic order.
K=2048 retains the unequal 5/5/6 groups; K=1536 uses 4/4/4, K=3072 uses 8/8/8.

The real hybrid rank header contains 34 KDA output projections, three dense
MLP down projections, and 22 DSA query projections at these shapes: **59 GEMMs
per target forward**. Each has 8x4096x3 FP32 partials. Removing one write and one
read of those partials eliminates **44.25 MiB of logical device traffic** and
5,664 slice-arrival atomics per forward. These are source work counts, not
measured DRAM bandwidth or latency savings. Output projections now add one
input-pack launch each; C1 query pairs keep one shared pack launch, reducing
their scratch from 50,688 to 12,672 bytes. The existing C4 paths remain selected.

Model-owned capture declarations select the output path before replay.
The native entry rejects a changed split count instead of silently changing
the arithmetic. Direct TX outputs still resolve their descriptor after PDL
and system-fence every final writer. The C1 paired-query control is available
as the final `local_c1=False` argument to the native probe API. Production calls
its true default; no process-global tuning switch was added.

Validation so far:

- `cpu-tests.log`: 30 checks, 21 passed and nine GPU-only skips. Checks include
  the real DenseLinear/slot-writer routing and boot rejection when a newly bound
  output owner has not executed its C1 path.
- `integration-cpu-tests.log`: after including main #940–942, 256 checks ran:
  246 passed and ten required GPUs. This includes the changed verifier, boot
  ownership, serving adapter and chat-template integration.
- `ci-fix-tests.log`: 86 fleet-lease/retention checks pass. The full CI found
  two old source-string assertions requiring the Server call to end immediately
  after `park_min_tokens`; #942 adds `reasoning_opener`. These now inspect the
  fleet Server call's actual keyword bindings, independent of its final argument.
  Runtime code and the queued consumer revision are unchanged by this repair.
- `ci-memory-tests.log`: 11 boot checks pass. A subsequent full CI run exposed
  a live-MemAvailable race between the zero-KV and two-GiB-KV assertions. The
  test now checks exact known values from one fixed meminfo snapshot, including
  a KV budget exceeding available memory; no production guard or tolerance changes.
- `compile.json`: complete production-flag CUDA/Torch extension compile and
  load, with CUDA hidden. Five new native specializations; ordinary variants
  use 76 registers, direct-output variants 78, all with zero stack/local spill.
  Dynamic shared memory remains the existing 3-slice kernel's size.
- The initial compile succeeded but the report reader expected a different
  cuobjdump header. The repaired reader reused the same compiled binary.
  CPU staging initially included macOS AppleDouble sidecars; removing those
  from the task-owned staging directory resolved package text-reader errors.
  No numerical tolerance was changed.

The GPU entry `engine_kernel_check.py --lanes forward_reduce --ranks RANKS`
checks real weights against the same native build: zero and signed
changed inputs, padded strides, poisoned outputs, private scratch, both replay
orders and alternating direct-output addresses with guards. Paired-query
comparisons explicitly retain the previous implementation as the control,
including C4 and a return to C1. Warm/evicted B/A/A/B timings follow exact checks.
This check also joins the existing `dsa_inputs` bundle, avoiding another model
boot. A passing component check does not establish 24 step/s or acceptance.

## Completed GPU gate

`st-forward24-0914c` passed all ten components in the combined probe in 92.8 s,
with no failed components and no model boot (`gpu-receipt.json`, `gpu.jsonl`).
The measured source was `c64e1038`; native CUDA SHA is
`4d64f731dbfa44edd3cef3af0d3975cfd93c23c879471330d7b1f6cdd9c47597`.
The integration commit `d1300ecf` keeps the dense package and new forward probe
identical. New main changes have their own validation and are not gains
attributable to this PR.

Runtime: srv4 NVIDIA GB10, driver 580.159.03, Torch 2.13.0+cu130, CUDA 13.0,
image `sha256:062bb8e5d4c4ef658c8b57987e99c5e84e2b149c73c0ac721ea25263b258c93e`.
Weights: `/home/choiceoh/models/st-glm53-hybrid-gptq-v1/rank3of4.safetensors`.
Each tensor's SHA is in the report. Both arms use identical RTN packs of those
real BF16 weights; this is not a consumer GPTQ-pack or acceptance measurement.
The report's `engine_shape` is the historical measured descriptor (K6); actual
checked rows are 8/16/24/32, corresponding to the requested K7 C1–4 widths.

All eight new output/query exactness groups passed at zero tolerance, including
private workspace, changed inputs, alternating direct TX addresses, guards,
both replay orders, wide query rows and returning to C1. The five new native
specializations were exercised. Arithmetic and graph replay are GPU-proven.

Mean of each arm's two B/A/A/B samples, microseconds per projection or query pair:

| C1 component | Warm B → A | Warm change | Evicted B → A | Evicted change |
|---|---:|---:|---:|---:|
| KDA output, K=2048 | 18.601 → 16.448 | −11.58% | 53.458 → 50.052 | −6.37% |
| Dense MLP output, K=3072 | 21.406 → 20.542 | −4.04% | 71.270 → 68.722 | −3.57% |
| DSA query pair, K=1536 | 24.463 → 22.592 | −7.65% | 75.331 → 71.184 | −5.51% |
| Query pair after C4 | 24.417 → 22.595 | −7.46% | 76.667 → 73.364 | −4.31% |

The input-pack launch is included. Eviction runs outside the event intervals.
These are short component comparisons, not an end-to-end reduction or a
prediction that the remaining gap to 24 step/s is closed. The older query-pack,
latent-write, indexer gate, absorption, pool/ID, copy/length and draft-QK checks
also passed in this run; their separate timings remain in the raw report.

The table above measures ordinary outputs. A serving-path follow-up,
`st-forward24-direct0914` on `7feacf58`, also passed in 13.5 s with the **same
native kernel** (`gpu-direct.jsonl`, `gpu-direct-receipt.json`). It compares
baseline and candidate `_write_slot` graphs with independent destinations,
rebinding both addresses and checking both outputs against the ordinary
reference. Warm direct-output KDA time is 18.709 → 16.657 us (−10.97%);
dense MLP is 22.482 → 21.341 us (−5.07%). The additional pack launch and final
system fences are included; NIC transport is not exercised. This run's evicted
samples vary widely (for example, direct KDA A is 47.98/121.13 us), so its
evicted speed delta is **inconclusive**, not the large mean-derived win.

Two failed attempts are retained: the first named a srv2-only image ID; the
second used a directory whose rank0 file is absent on srv4. The runner now
accepts an explicit resident shard without copying the full rank file.
Neither failed attempt is counted as a forward numerical pass.

PR875's source Oracle comparison and unfilled paired-profile template are
included. It correctly leaves every total speed delta `null`: changed native
components have no complete measured step-cost profile. Memory-layout bytes
and the partial-traffic counts above are not substituted for those costs.

Final consumer validation remains K7, 32K/128K, C1 twice and C4 once. Report
pooled/window step/s, tokens/s, tokens/step and acceptance. Answer grading does
not decide the performance target, per the operator's instruction.

## Reserved consumer observation

Final reservation: `st-forward24-onepass0914c`, ticket `17893695583322090`,
freezes `a0ccfd1a4dab9dcfa16b10699ae3e7b400299dc7` after including main #943/#945.
It officially replaces the reservation below, preserving its enqueue time and
all workload budgets (`onepass-final-queue-receipt.json`). The dense package
and its repaired CPU test sources remain identical to the validated versions.
Other main changes are included in this single boot, with their own evidence;
their speed is not attributed to the local GEMM reduction. The earlier receipt
below records the history, not a second pending model boot.

Accepted session `st-forward24-onepass0914b`, ticket `17893676782779630`, freezes
`d1300ecffbaed6d11ae6dac468ca50707152d7a5`. It uses the canonical ST bracket's
single-arm chain with `ST_BRACKET_VALIDATION=full`: one candidate boot,
`onepass.py` C1 twice and C4 once, exclusive traffic, 2K/32K/128K and K7.
There is no additional baseline boot. The prior admission attempt requested
an older main and was refused before taking GPUs; the integrated candidate
was accepted. `onepass-queue-receipt.json` preserves admission separately
from measurement completion.

This is a bounded performance observation with individual output cap 2,048,
combined cap 6,144 and combined reasoning cap 4,096. The workload is recorded
explicitly in both runs. These are benchmark request budgets; model defaults
are unchanged. Grading remains in raw records but cannot establish or reject
the requested speed/acceptance result. It is not the default 16K/49K quality
fixture or a matched consumer A/B, and must not be advertised as final quality
adoption proof. The queue estimate is 35 minutes of work, not a waiting-time
promise. Admission found another session's expert-requantization lease and
the earlier KDA ticket; neither is stopped by this work.
