# C=1 input reuse: keep split-K partials within one CTA

This follow-up compares against the default input-reuse kernel from PR449,
not the older repeated-input-quantization baseline. The new serving knob `VLLM_GLM53_MK_INPUT_CTA` remains 0 pending actual-source
and serving acceptance; the input-reuse default remains enabled.

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
preparation and use 32 alternating samples per mode. The private-source results below are complete. Actual-source and serving
step/output acceptance remain pending.

The channel recorder's previous zero-window failure is also fixed: its global
transport wrapper attempted to wrap the concurrent sampler's plain metrics
URL as an SSE request. That raised an AttributeError which the sampler caught,
discarding every sample. The wrapper now only handles the matching completion
POST; metrics and unrelated requests keep their original response. A regression
executes a metrics fetch while the wrapper is installed, as the sampler does.

CPU gate: 6,684 core checks, 30 megakernel regressions and four channel recorder
regressions pass. The generated private CUDA source renders successfully and
the runner passes Bash syntax validation. GPU compilation is still pending.

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
four K groups, and request three/four resident blocks per SM. Compilation and
GPU results for these integrated variants remain pending. A failed CTA boot
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
actual runtime occupancy and timing still require the GPU probe. Torch headers
emit C++20-extension warnings under C++17; compilation returns 0.
[Compile log](cpu/compile.log) and [source/environment receipt](cpu/receipt.json).

The first prototype runner returned 0 and restored approved main `944f65c`
at 22:13:14 KST. The next immutable source `916adc0` is queued as
`inputctaserve0907`, after the startup-key and MoE overlap campaigns. Preflight
passes, including the new knob declared by the candidate profile. Its evidence
will be under `/home/choiceoh/glm53-logs/INPUTCTASERVE0907`. No runtime source
or executing runner is changed while it waits or runs.
