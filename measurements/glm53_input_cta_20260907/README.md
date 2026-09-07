# C=1 input reuse: keep split-K partials within one CTA

This follow-up compares against the default input-reuse kernel from PR449,
not the older repeated-input-quantization baseline. Production defaults and
CUDA sources are unchanged while the private probe is evaluated.

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
preparation and use 32 alternating samples per mode. GPU results and serving
step/output acceptance are pending; no speedup is claimed from the code.

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
