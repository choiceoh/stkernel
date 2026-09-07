# Check5: BF16 passes; FP8 comparison remains inconclusive

Normal GO: 2026-09-08 03:01:43 KST. Probe: 03:02:54–03:04:50.
Exact incoming container recovery and outer exit 1 completed at 03:07:27.
The gate failed; sanitizers and direct TTFT were not run.

The Q0-corrected kernel passes all ten BF16 cases, including all six active
M64 cases, changed input/routes, local MoE, retained outputs and M128 capture.
Local MoE and capture also pass in every FP8-v3 case. The catastrophic Q0
scale-address error from check4 no longer reproduces.

Two FP8-v3 cases fail the unchanged comparison:

- 4096/skew, where M64 is not admitted: independent stock control has one bad
  row on rank 0; the candidate-labelled stock path has none.
- 8192/balanced: candidate eager has eight bad rows on rank 0 and two on rank 3;
  independent stock control has two on rank 0. Changed-input repetitions and
  local MoE pass. A stock-control failure makes the comparison inconclusive;
  ten candidate rows versus two control rows must not be dismissed as noise.

All thresholds remain unchanged. No sanitizer or serving gate is waived.

## Component timing, not full-model TTFT

These are six alternating slowest-rank component samples (MoE, shared expert
and TP transport), with latency reduction against the baseline median:

| Rows | BF16 balanced | BF16 concentrated | FP8 balanced | FP8 concentrated |
| --- | ---: | ---: | ---: | ---: |
| 6144 | 20.40% | -8.95% | 7.55% | -17.28% |
| 6912 | 18.88% | -10.01% | 6.66% | -11.94% |
| 8192 | 12.68% | -15.41% | 1.11%* | -11.46% |

*The 8192 FP8 comparison fails numerics/control; its time is descriptive only.
All timings are synthetic, single-run component evidence, not a 40% claim or
full-model prefill verdict. Concentrated routing regresses materially.

## Next diagnostic, before any further acceptance run

Keep the kernel and numerical thresholds fixed. Add a separately marked FP8
comparison diagnostic for the failed 4096/skew and 8192/balanced cases, plus
6144/balanced as a passing active control. Use a fixed set of three input seeds
and eight alternating candidate/control trials per seed. Each trial uses a
fresh stock baseline and stock repeat, then an independent stock control and
candidate in alternating order. Preserve the original per-row L2 .02 / peak
.04 and three-times same-row repeat comparison; do not change it to make the
candidate pass. Record failing row IDs, original error/threshold/noise values,
control/candidate intersections, finite status and local-before-transport
comparison. Report repeated stock instability and candidate excess separately;
rows from the same seed/trial are not independent experiment repetitions.

This diagnostic must use a new immutable probe source and a normal fleet turn,
with exact incoming recovery. Its completion marker cannot satisfy the serving
GPU gate. Do not rerun completed check5/serving2 unchanged or adjust tolerances
before understanding the independent control behavior. No new job was submitted
as part of this archived result.
