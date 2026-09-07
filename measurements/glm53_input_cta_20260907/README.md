# C=1 input reuse: keep split-K partials within one CTA

This follow-up compares against the default input-reuse kernel from PR449,
not the older repeated-input-quantization baseline. The actual serving-source
kernel reduces warm latency 24.51% and read-evicted latency 7.88%, with exact
output bits and clean sanitizers. `VLLM_GLM53_MK_INPUT_CTA` remains 0 pending
serving acceptance; the first baseline boot failed before any requests.
The input-reuse default remains enabled.

For the exact M6/N6416/K4096 route, one CTA owns 16 output columns and its
eight warps own the original eight K slices. Each warp keeps the same W4
expansion, FP8 input bytes, four K groups and accumulation sequence. A single
block barrier makes 3,072 bytes of shared partials visible before summing
slices in their original order. This removes the existing global partial
stores/loads, per-tile atomic arrival counter and device fences. The probe
compares two- and three-stage weight pipelines against the enabled default.

The probe requires bit-identical BF16 outputs and an independent FP32 oracle,
changing inputs, two retained layer graphs, startup checks, fallback shapes,
and racecheck/memcheck. Warm and read-evicted measurements include input
preparation and use 32 alternating samples per mode. The private-source and
actual-source GPU results below are complete. Serving step/output acceptance
remains pending.

The channel recorder's previous zero-window failure is also fixed: its global
transport wrapper attempted to wrap the concurrent sampler's plain metrics
URL as an SSE request. That raised an AttributeError which the sampler caught,
discarding every sample. The wrapper now only handles the matching completion
POST; metrics and unrelated requests keep their original response. A regression
executes a metrics fetch while the wrapper is installed, as the sampler does.

Final CPU gate: 6,687 core checks, 30 megakernel regressions, 92 fleet checks
and 12 driver/transport regressions pass. Native CUDA compilation and GPU
validation are complete. The runners pass Bash syntax validation.

Reproducer: `probes/gemm_input_cta.py`; fleet maintenance runner:
`probes/run_gemm_input_cta.sh`. The runner checks clean/current-main ancestry
before touching service, uses the fixed serving image and restores latest main
through a separate clean checkout on every exit after stopping service.

## Completed prototype on GB10

Fleet `inputcta0907`, source `7989fc4`, fixed serving image, generated CUDA
SHA-256 `2fc4fa19fe960225f611b99673f5700ba16cebc36cd24d548160d889225c356e`.
All 120 numeric rows are finite, bit-identical to the enabled default and
pass the independent FP32 oracle. Both retained layer graphs pass 80 changing
input checks. Both startup gates pass; racecheck reports zero hazards/errors/
warnings and memcheck zero errors.

| Kernel | Warm us | Read-evicted us | Warm change | Read-evicted change |
|---|---:|---:|---:|---:|
| Enabled input reuse | 32.512 | 74.656 | baseline | baseline |
| CTA, two buffers | 28.800 | 69.360 | -11.42% | -7.09% |
| CTA, three buffers | 30.464 | 71.552 | -6.30% | -4.16% |

The two-buffer kernel uses 78 registers, no local spills, 22,528 shared bytes
and three resident blocks per SM. These 32 alternating pairs include input
preparation; they do not measure whole-model step improvement.

[Raw numerical/timing results](prototype/result.json), [racecheck](prototype/racecheck.log),
[memcheck](prototype/memcheck.log).

## Serving integration and next variants

The two-buffer layout is integrated behind an independent gate. Mode 1 keeps
runtime geometry; modes 2/3 specialize M=6, K=4096 and eight slices, unroll the
four K groups, and request three/four resident blocks per SM. The integrated
GPU comparison below selects mode 2. A failed CTA boot
gate disables CTA alone and preserves the established input-reuse route.
Dedicated capture receipts distinguish actual CTA launches from startup tests.

The actual-source runner selects a mode only when both cache regimes improve
at least 1% in median and at least 29/32 pairs are faster, then uses the lowest
warm median. It runs the chosen mode in a fresh B/A/A/B serving bracket, with
three fixed 2048-token C=1 responses per boot, the standard context/quality
ladder, fixed workload/seed/image, separate SSE channels, and four-rank
capture/source receipts before and after traffic. All existing onepass quality,
window-count and exclusivity gates remain active.

Integrated CPU gate: 6,686 core checks, 30 megakernel regressions, four CTA
fallback/capture tests, four input-reuse driver tests and four channel recorder
tests pass. Only the two kernel/occupancy source-count expectations changed
in the audited logic file; its pure math/layout helpers and loaders are unchanged.

After integrating current main `944f65c`, the combined CPU suite passes 6,687
logic checks, 30 megakernel regressions and 92 fleet regressions. The final
plan receipt reports the CTA variant's actual occupancy. Both generated
overlays match source bytes and the focused kernel/12 driver-transport tests pass.

## CPU native compilation and queued serving run

The reviewed `bench/cpu_compile.py` entrypoint compiled the actual translation
unit with host nvcc 13.0.88/GCC 13.3.0, `sm_121a`, fixed-image Torch headers,
C++17 and the serving FP8/M8 definitions. It ran in fleet CPU session
`inputctacpu0907`, without loading a CUDA context or holding the GPU lane.
Modes 1/2/3 use 78/75/64 registers, respectively; all have zero spill loads,
zero spill stores and one barrier. Mode 3 meets its four-block register budget;
the actual GPU measurements below confirm occupancy and timing. Torch headers
emit C++20-extension warnings under C++17; compilation returns 0.
[Compile log](cpu/compile.log) and [source/environment receipt](cpu/receipt.json).

The first prototype runner returned 0 and restored approved main `944f65c`
at 22:13:14 KST. The next immutable source `916adc0` ran as
`inputctaserve0907`, after the startup-key and MoE overlap campaigns. Preflight
passes, including the new knob declared by the candidate profile. Its evidence
is under `/home/choiceoh/glm53-logs/INPUTCTASERVE0907`. No runtime source
or executing runner is changed while it waits or runs.

`analyze.py --root <evidence>` verifies the selected mode, GPU/sanitizer
receipts, pre/post rank identity, common source/image/workload, every request
hash, exclusive traffic and at least 20 fixed decode windows per boot before
computing step/output changes. Independent boots have equal weight. It also
attributes the unchanged Korean scanner's findings to SSE channels without
changing the original combined quality verdict. `--incomplete` preserves an
interrupted bracket separately and does not write a completed summary.

The other common M6/N6144/K4096 and M6/N4096/K4096 plans use three K slices
under their established occupancy rules. They cannot reuse this eight-slice
kernel while preserving the original summation order, so their existing paths
remain selected. Extending coverage would require a separate three-slice layout.

## Actual serving-source GPU gate

Fleet `inputctaserve0907` began at 22:50:28 KST on source `916adc0`, including
main `944f65c`. CUDA SHA-256:
`0fddbd841b0bafd51100d1f0c2b5e990e8ce2493b1af31f790ae82636ac351ae`.
All 160 numerical rows pass exact baseline bits, the independent FP32 oracle
and finite checks. All three candidates pass 120 alternating retained-graph
checks and startup checks. Racecheck reports zero hazards/errors/warnings;
memcheck reports zero errors.

| Mode | Registers / blocks per SM | Warm us | Read-evicted us | Warm reduction | Read-evicted reduction |
|---|---:|---:|---:|---:|---:|
| Enabled input reuse | 80 / 3 | 32.512 | 75.360 | baseline | baseline |
| 1: generic CTA | 78 / 3 | 28.416 | 70.240 | 12.60% | 6.79% |
| **2: fixed geometry** | **75 / 3** | **24.544** | **69.424** | **24.51%** | **7.88%** |
| 3: fixed geometry, four blocks | 64 / 4 | 26.368 | 71.296 | 18.90% | 5.39% |

All modes have zero local spills. Mode 2 wins 31/32 pairs in each cache regime;
its paired median reductions are 24.11% warm and 7.78% read-evicted. Mode 1
wins only 28/32 warm pairs, below the predeclared 29-pair selection threshold.
Mode 2 is selected for serving. Increasing occupancy to four blocks did not
beat the three-block specialized kernel; the CPU register result alone would
have chosen incorrectly. No outliers are removed from the raw evidence.

[Full GPU results](serving/result.json), [paired summary](serving/paired-kernel-summary.json),
[selection receipt](serving/selection.json), [racecheck](serving/racecheck.log),
[memcheck](serving/memcheck.log). These are kernel results.

## First serving attempt: failed baseline initialization

The B/A/A/B chain began at 22:58:10 KST on `916adc0`. `ICTAB1` used CTA=0.
At 23:02:53, srv1's first MHC BF16 differential check raised CUDA error 800,
`operation not permitted`, from `run_mhc`. Later CUDA calls failed and the
worker exited at 23:03:57, exit 1, `OOMKilled=false`. The other three ranks
passed MHC/GEMM/input-reuse startup checks on the same CUDA source. The head
waited for its missing peer; no health, decode window or output measurement
was produced. This is an initialization failure, not a performance result,
and the cause is not established by the logs.

All four logs were preserved before stopping the blocked head at 23:10:53.
The canonical lever detected the failure, and the runner restored approved
main `944f65c`; `/health=200` was observed at 23:20:22 KST and restoration
completed at 23:20:48. The original runner exits 1, preserving the failed test.
[First srv1 error](serving/failed-ICTAB1/srv1-first-error.log)
and [failed baseline preparation](serving/failed-ICTAB1/prepare.log) are retained.
The next immutable runner checks fresh-process MHC and both input modes on
all four nodes before loading the model; it does not edit the running attempt.

Retry `inputctaserve20907` is queued from immutable source `d920bea`, with
unchanged serving CUDA/Python bytes, in a separate clean checkout. It uses
`/home/choiceoh/glm53-logs/INPUTCTASERVE20907` and follows the two earlier
GPU reservations. Preflight passes. No step gain or default promotion is
claimed while the serving bracket is pending.
