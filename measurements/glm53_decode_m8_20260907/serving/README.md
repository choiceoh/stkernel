# C=1 serving result: inconclusive; candidate retained

**The evidence is insufficient to reject or promote this candidate.** In the
clean 2K-context samples, engine window speed changed **21.432 → 21.689 step/s
(+1.20%)**, while output speed changed **75.074 → 73.153 tok/s (-2.56%)** and
mean output-token latency changed **13.321 → 13.680 ms**. Acceptance varies,
and only five baseline and six candidate 2K windows survived the phase filter.
These observations do not isolate a kernel slowdown or establish a speedup.
Retain the candidate for further measurement; defer default promotion and
leave both options off. The earlier microbenchmark gain is not a measured
whole-decoder gain. The initial emphasis on lower output tok/s was too strong
as a reason to stop pursuing the candidate.

## Matched experiment

- Fleet session `glm53m8serve`, 2026-09-07 05:14:55–05:40:47 KST.
- Four DGX Spark / GB10 nodes, TP=4, DFlash2 speculative K=5, C=1. EP and
  quantization settings were unchanged. Prefix caching stayed at the profile
  default. `PREFILL_WARMUP=0` for every boot.
- Deployed source `f00a4eb2c1329678cbce2fbcc50fb45fad41ef41`, based on main
  `9d43bba`. Overlay stamp `1af1a93f83b5`. The later `47f9e3c` commit changes
  benchmark metadata only; deployed Python/CUDA and request timing are identical.
- CUDA SHA256 `4cabebb9919abbddb778d3c6addec09afb16133c6c10bf3cd43acaee2cb936d9`.
  All four nodes also have identical driver Python. See [runtime identity](runtime-identity.txt).
- Immutable image `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
- Baseline: `VLLM_GLM53_MK_FP8_PACK2=0`, `VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=0`.
  Candidate: `1`, `2`. Separate compiled cache entries contain the intended
  flags; see [compile-flags.txt](compile-flags.txt).
- The intended sequence was baseline B1/B2 → candidate A1/A2 → baseline B3,
  with one boot per group. B3 was contaminated by external traffic and excluded.
  The primary comparison therefore has two repeated samples on one boot per
  arm; it lacks a clean final reversal.
- Each onepass sends three 2K requests and one combined three-question request
  at 32K and 128K. Temperature=0, thinking on, seed=7, max output 400/1200 tokens.
  There is no separate benchmark workload or inferred microkernel-to-serving gain.

## Direct output measurements

Each request records usage-token count, first content/reasoning arrival, end
of the stream and SSE chunk gaps. Standard TPOT is
`(request elapsed - TTFT) / (completion_tokens - 1)`; output speed is its inverse.
This excludes prefill and includes the final stream/usage tail. SSE chunks can
contain multiple speculative tokens, so chunk gaps are not individual-token ITL.
The table takes medians within each run, then medians across the two runs.

| Context | Baseline tok/s | Candidate tok/s | Change | Baseline ms/token | Candidate ms/token | Baseline run spread |
|---|---:|---:|---:|---:|---:|---:|
| 2K | 75.074 | 73.153 | -2.56% | 13.321 | 13.680 | 2.01% |
| 32K | 76.107 | 77.133 | +1.35% | 13.179 | 12.965 | 11.03% |
| 128K | 75.071 | 75.392 | +0.43% | 13.348 | 13.382 | 9.01% |

| Run | 2K tok/s | 32K tok/s | 128K tok/s | Engine step/s | Tokens/step |
|---|---:|---:|---:|---:|---:|
| B1 | 74.319 | 80.304 | 71.690 | 21.935 | 3.429 |
| B2 | 75.830 | 71.911 | 78.452 | 21.426 | 3.396 |
| A1 | 75.167 | 76.706 | 82.467 | 21.681 | 3.541 |
| A2 | 71.140 | 77.560 | 68.316 | 21.442 | 3.305 |

The positive long-context deltas are within observed baseline variability.
Output lengths and acceptance vary even between baseline repeats. Median of
reciprocals need not equal reciprocal of medians, especially with two samples.
## Engine window measurements and decision limits

The 2-second window statistic counts engine steps inside response decode
phases, excluding prefill and phase edges. Below, each run is summarized by
its median, then the two run medians are combined. The two repeats share one
boot per arm; individual windows are not independent boot replications.

| Window population | Baseline step/s | Candidate step/s | Change | Baseline/candidate windows |
|---|---:|---:|---:|---:|
| 2K only | 21.432 | 21.689 | +1.20% | 5 / 6 |
| 2K, 32K and 128K combined | 21.680 | 21.562 | -0.55% | 23 / 21 |

One extra step in a two-second window changes its rate by about 0.5 step/s,
roughly 2.3% at these rates. More windows can improve an aggregate estimate,
but this small sample and the missing clean reverse arm cannot establish a
1% effect. The reported baseline spread is an observed range, not a confidence
interval or an equivalence bound.

The existing engine-step judge also returns **inconclusive**: A1 +1.2% and A2
+0.1% against B2, both within the same-build 2.3% baseline spread. These are
engine steps, not output tokens. Neither the negative output-rate observation
nor the positive 2K step-rate observation is sufficient for a final verdict.
The next adjudication needs an uncontaminated, counterbalanced boot sequence,
fixed input/context and decoding length, more steady-state steps, and separate
step-time and acceptance reporting. No additional GPU run was made as part of
this interpretation correction.

## Correctness and execution evidence

All four clean runs passed retrieval **9/9**, Korean corruption **0/5**:
36/36 retrieval checks and 0/20 corrupted responses total. This is the existing
onepass gate, not a comprehensive model-quality evaluation or exact-output parity.
Both candidate runs passed serving proof **2/2**. Captured M=6 launches select
compact mode for `(N,K)=(4096,2048)` and `(6144,4096)` and regular transposed
mode elsewhere. Both server boots pass GEMM startup numerics; earlier GPU
oracle/replay validation remains in the parent directory.

## Excluded final arm and metadata audit

B3 passes its own retrieval/Korean checks, but its 128K phase overlaps **11
external POST requests**, with `Running: 2 reqs` and queued requests in the
archived server log. Its 128K speed is 35.602 tok/s and global engine windows
fall to 14.456 step/s. Its effective K counter becomes 4.9757 despite the
configured K=5, further showing unrelated traffic in the counters. It cannot
serve as a C=1 baseline. Comparing A1 to it would produce a misleading +50%
engine-step delta. The entire B3 arm is excluded from the primary comparison.
Its earlier 2K result, 77.094 tok/s, is preserved but not used in that aggregate.
The other four logs contain no non-local completion POSTs and never exceed
one running request.

The first two raw records incorrectly label the launcher's `VLLM_GLM53_SPEC_K=5`
alias as an unknown experimental option. The deployed [profile excerpt](deployed-profile.txt),
archived boot arguments and measured counters all prove that it is the default.
Commit `47f9e3c` fixes future fingerprints; tests verify that real overrides and
unknown knobs remain visible. [analyze.py](analyze.py) normalizes only that known
alias and excludes B3, preserving [raw records](records.raw.jsonl) unchanged.
The [correction audit](ledger-corrections.json) also records the exact original
and corrected shared-ledger rows. Only B1/B2 metadata and B3's aggregate validity
were corrected there; B3's raw window median was retained as `raw_windows_med`,
its usable `windows_med` set to null, and the contamination reason attached.
The corrected A1/A2 inconclusive verdicts were appended to the shared ledger.

## Reproduce and final state

Run `python3 analyze.py` in this directory to generate
[normalized records](records.normalized.jsonl) and [summary.json](summary.json).
[runner.sh](runner.sh) contains the fleet-held deployment and chain command.
Per-arm boot logs, the full [runner log](glm53-m8-serving.log), deployment log,
source identities and compile flags are retained alongside the records.

Final live check: health 200, both candidate options 0, SPEC_K=5; fleet released.
The measured source remains mounted with baseline options. No default promotion,
PR, push or merge was performed. Local integration checks passed 6,656 contracts
(including 30 megakernel and 20 fleet regressions); local Torch-dependent checks
were skipped. GPU serving startup tests and onepass supply the runtime evidence.
The request-timing fixture verifies multi-token SSE chunks, usage counts, TPOT
and chunk gaps. The metadata fixture checks default aliases and real overrides.
