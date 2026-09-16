# Fixed K7 follow-up: scoped mHC, MoE and MLA defaults

The candidate is `3c3c3260ba8a26929d0f466a90db9e9589c2f278`. The matched
control is `0e949d11d0421784f2681aa21f5ba7465662a776`; it changes only
three Python booleans. Native sources, packing, weights and build inputs
are identical between arms. The four full consumer runs are complete.
The three scoped paths are adopted as requested; the whole-engine speedup
remains unproven. Component improvements below are not a tok/s claim.

## Selected execution

- mHC: retain the eight-row KDA input-pack fusion. The original BF16
  rounding, FP8 scale and consumer bytes are unchanged. Thirty eligible
  boundaries per rank were proven in the earlier TP4 campaign.
- MoE: at eight/sixteen decode rows, load each BF16x16 activation scale
  group with two aligned 128-bit loads. Keep all sixteen FP32 conversions,
  the maximum reduction, quantizer, expert scale and output accumulation.
  This uses the existing vector-load helper; there is no new weight copy.
- MLA: sixteen rows and selection width at least 128 use 32-slot tiles
  with retained BF16 query fragments and the existing split count (six on
  GB10). Eight-row, narrow-width, tree and cluster cells keep their old
  dispatch. Tile32 changes FP32 grouping; independent FP32 error checks
  pass, but this is not bitwise equality.

## Bounded measurements

All GPU tickets used the canonical fleet reservation on srv2, the ST
image (Torch 2.13.0+cu132, CUDA 13.2), and GB10. Compile-only containers
have no GPU exposure and assert no initialized CUDA context. Hashes and
native binaries are recorded in each JSONL. The rank file is
`/home/choiceoh/models/st-glm53-9391-up-gate-full/rank0of4.safetensors`.

| Candidate | Evidence | Finding |
|---|---|---|
| MLA register queries, tile16 | `mla-v8.jsonl`, commit `9a1890f0` | 42 fixtures bitwise; no useful latency gain; removed |
| MoE byte/nibble/plain bulk and linear shared stages | `moe-v8.jsonl`, `9a1890f0` | Corrected a double-tiling error in the probe. Byte-swizzled z and linear zl pass; zn/zp fail. z is near-neutral; zl is slower at C1. Remove zl. |
| MLA tile32 | `mla-mhc-v10.jsonl`, `705083ea` | 42 independent-reference fixtures pass. C2 normal full selections improve 3.6–6.7%; C1/short rows do not. |
| MLA split plans | `mla-mhc-v12.jsonl`, `ed11409d` | Six splits retain the C2 win (60.03 → 55.67 µs, −7.26%). Other full-selection plans lose. Keep C1 unchanged. Plans requested beyond workspace capacity were clamped by `mla_splits`; they are not larger allocations. |
| MoE vector input | `moe-v11.jsonl`, `36e8f03d` | Three real layers; independent/shared/calibrated routes, poisoned outputs, zero route, forward/reverse replay pass. Frontend falls 13.488 → 12.696 µs at C1 and 18.840 → 18.048 µs at C2. C2 evicted three-layer chain −0.84%; C1 chain +0.06%, no engine claim. |
| mHC C1 fusion | `mla-mhc-v12.jsonl` | 90 real coefficients, four magnitudes, packet/ordinary forms and pack bytes pass bitwise; packet coefficient chain −19.19%. Full mHC + three KDA projections is only −0.36%. |
| mHC C2 extension | `mla-mhc-v12.jsonl` | Same numerical coverage passes; coefficient chain −5.08%, but mHC + KDA projection chain **+3.94% slower**. Removed rather than adopting the isolated mHC win. |

The v9 tile prototype stopped with illegal memory access: its local K-quarter
formula doubled the number of score planes. v10 fixes the geometry and adds a
compile-time warp-coverage assertion. The v10 record has no completion marker:
its final sixteen-row projection correctly refused an unextended inner guard.
v12 extends that guard and completes all numerics/timings; its later measured
C2 regression is why that extension is absent from the final code.

## Consumer gate

The previous mHC-only campaign is documented in
`../st_fixed_k_cost_20260917/README.md`: both arms failed some quality checks,
so its raw request rates are not evidence for adopting this three-way candidate.
The matched full onepass retained 2K/32K/128K, C1 twice per boot, C2 once
per boot (2K/32K), quality, acceptance, actual request tok/s, TTFT and build
identity. Serving capacity was two; C4 is unmeasured.

## Final component and CPU gates

`final-compile.jsonl`, `final-mhc-mla.jsonl` and `final-moe.jsonl` all
complete successfully at measured commit `3c3c3260`. The final MLA reference
error maxima are 0.00179969 (control) and 0.00181095 (candidate). Final MoE
evicted chains change -0.41% at C1 and -0.45% at C2; single-layer results
are +0.26% and +0.25%, so the component evidence is modest. No extra bulk
weight copies are allocated for this gate (peak allocation 5,968,438,272
bytes, versus 13,560,244,736 in the earlier bulk experiment).

The CPU suite ran 239 files / 2,229 tests with 363 skips. The original run
had two failed files: the missing MLA provenance hash and process-reclaim
tests under a container without an init process. After updating metadata,
a focused rerun of both files in a CPU-only container with `--init` passes
all 27 tests. The 36 directly relevant ownership/dispatch/cache tests pass.
The provenance correction after `3c3c3260` changes metadata only; the native
source hashes being measured are unchanged.

GitHub `engine check` run `35141668624` passed at `0b382a91`. The initial
CPU log and the focused correction are retained in `cpu-tests-initial.txt`
and `cpu-tests-retest.txt`.

## Reproduction and interpretation

The two immutable arms were submitted separately with the same controller
and environment, so a failed baseline quality gate cannot prevent collection
of the candidate. This does not waive either quality gate:

```sh
ST_BRACKET_VALIDATION=full ONEPASS_PROFILE=extended \
ONEPASS_FIXED_DECODE_TOKENS=1024 ONEPASS_FIXED_DECODE_REPS=3 \
ONEPASS_JSONL=/home/choiceoh/glm53-logs/fixedk-triple-0917.jsonl \
bash bench/fleet.sh run --gpu --fleet --detach SESSION 35 NOTE -- \
bash bench/st_bracket.sh chain ARM=SHA
```

Use `B=0e949d11d0421784f2681aa21f5ba7465662a776` and
`A=3c3c3260ba8a26929d0f466a90db9e9589c2f278` for `ARM=SHA` above,
with distinct session names. The fixed-token override makes the recorded
workload profile `custom`; both arms have the same workload dictionary.
`control.patch` reproduces the complete candidate-to-control difference;
the control ref was retained on srv2, not published as a separate GitHub PR.

`runtime_identity.py` reads the actual mapped native modules and startup
execution reports on all four nodes, without initializing CUDA or issuing
GPU work. `collect_launches.py` reads the completed onepass diagnostic
artifacts; it never starts profiling, and keeps counts rather than timing.
`summarize.py` reads the retained consumer JSONL. It validates matching
unsalted workloads and shapes, separates normal/fixed/concurrent requests,
and recalculates pooled fixed-window step/s and measured request tok/s.
Hash-matched subsets remain observations, not a substitute for full quality.

The canonical quality tally includes the three 1,024-token fixed-decode
responses. Their bounded output can fail the complete JSON proof rubric;
the report therefore also shows the ordinary 2K/32K/128K quality cases.
Neither tally is rewritten, and the canonical failed gate stays failed.

## Runtime identity

`consumer-runtime-B.json` and `consumer-runtime-A.json` identify the modules
mapped by each worker, not just a flag or an available cache file. On every
node, both arms mapped the same dense and MLA binaries with unchanged hashes
and mtimes. Binary hashes differ across nodes; the equality is **between arms
on each node**. The common CUDA source hashes are:

- dense: `8d3bd8a91b1d81b3de92c00ef9d250e4a0e24915f3ff5602df0ae13ebda71154`
- MLA: `89cb0d45d8c54076d142af1c017586b599073fa4523220c1cd9077ac5a097196`

All four candidate startup execution reports require and record 30 mHC pack
consumers at eight rows. All candidate eight/sixteen-row served MoE handles
include `inputv16`; none of the control handles does. The control has zero
mHC input-pack consumers. Native build/loading precedes the consumer phases.
The first measured runs start at 04:39:02 (B) and 05:17:37 (A), KST.

The separate diagnostic replay confirms the following on **each rank**
over four decode steps (all recorded C1 contexts and C2 2K/32K):

| Executed operation | Control | Candidate |
|---|---:|---:|
| C1 separate input-pack launches | 298 | 178 |
| C1 mHC fused-pack launches | 0 | 120 |
| C1 MLA tile16 launches | 44 | 44 |
| C2 MLA tile16 launches | 44 | 0 |
| C2 MLA tile32 launches | 0 | 44 |
| C2 separate input-pack launches | 10 | 10 |

This proves dispatch and removal of 30 separate input-pack launches per
C1 step per rank. It is not a sum of profiler times or a tok/s estimate.

## Completed consumer result and adoption status

Both canonical reservations completed and released their leases: B at
05:12:25 KST, A at 05:54:17 KST. All four records are complete, with K=7,
FP32 KDA, identical workload dictionaries and serving shapes. All 48 scored
or fixed-decode request records had zero cached tokens and no foreign traffic
issue. All 64 rank/context diagnostic records have the dispatch counts above.

| Arm / run | Normal request decode tok/s | Fixed 1024 decode tok/s | Pooled fixed step/s | Raw acceptance | Normal C1 quality |
|---|---:|---:|---:|---:|---:|
| B / 1 | 85.3135 | 93.8092 | 19.71683 | 51.8831% | 7/9 |
| B / 2 | 87.8596 | 91.3914 | 19.71681 | 54.1088% | 6/9 |
| A / 1 | 85.4295 | 91.9405 | 19.76948 | 52.1515% | 8/9 |
| A / 2 | 86.2512 | 94.7501 | 19.83227 | 53.0411% | 6/9 |

Request rates above are `sum(completion_tokens - 1) / sum(decode_s)`;
fixed step/s is `sum(window steps) / sum(window seconds)`. Neither rate is
estimated by multiplying step/s and acceptance. The normal request changes
are +0.136% / -1.831%; fixed-decode request changes are -1.992% / +3.675%.
The response lengths and output hashes differ: none of the ordinary C1
outputs matches between arms. One fixed-decode request in run 1 matches
(87.8239 -> 88.2812 tok/s, +0.52%), and one separate fixed-concurrency C1
request matches (106.4298 -> 106.3782, -0.05%). These isolated pairs do not
establish an engine win or a regression.

Normal quality totals are **23/30 control, 25/30 candidate**: C1 13/18 ->
14/18, C2 10/12 -> 11/12. This is not evidence of a quality regression, nor
is the small sample sufficient to prove a quality improvement. C1 Korean
corruption is zero in both arms (0/16 responses each, including fixed decode).
The canonical score also includes nine bounded fixed-decode cases per run,
all of which fail the complete-proof rubric. Even after separating them,
ordinary quality is not perfect. `consumer-verdict.txt` / `verdicts.jsonl`
therefore retain **NO EVIDENCE**, with no qualifying warm sample or noise
floor; no failed gate is converted into a pass.

The extra four-client fixed-concurrency comparison has valid candidate
preparation, but the control C2 measurement entered two Triton compile/load
calls on every rank (specializations 61 -> 63). Its observed aggregate rates
114.0778 -> 112.6820 tok/s are **not a valid paired performance result**.
`consumer-preparation.json` retains the before/after counters. The ordinary
C1/C2 measured phases have unchanged preparation counters in both arms.

The requested decision is to retain all three scoped code defaults and merge
them. That decision is separate from the still-unproven whole-engine speedup.
The source after the measured commit changes only provenance metadata;
subsequent files added here are measurement records and offline readers.
