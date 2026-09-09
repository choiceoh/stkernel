# EP tile-major decode and prefill

This candidate retains EP4 expert ownership and shares one tile-major weight
allocation between a dedicated static decode kernel and the EP-local prefill
kernel. It is experimental and defaults off. Neither recovered decode speed
nor retained prefill gains have been established for this implementation.

Enable `ENABLE_EP=1 VLLM_GLM53_EP_TILED=1 VLLM_GLM53_TP_SF6_Q0=0` on the GLM
profile. The TP Q0 owner/canary does not apply to EP weights. Keep the old
`VLLM_GLM53_EP_PREFILL_LOCAL`, `VLLM_B12X_EP_ZERO_WEIGHT_MICRO`, and
`VLLM_B12X_EP_WARM_COMPACT` experiments off. The new flag also preserves the
EP-compatible attention/MHC prefill SP configuration. Attention remains TP4.

## Implementation

- Exact SM121, BF16 activations, E72/H4096/I2048/top8, unpadded local experts,
  SwiGLU `(1,0,10)`, and maximum batch capacity 8192..16384.
- Weight loading permutes NVFP4 bytes in place into the TP v5 tile-major
  layout. Both kernels alias those same bytes; no second model weight copy
  or request-time EP-to-TP redistribution is introduced.
- Native M1..32 uses the EP static kernel, retaining the v5 TMA descriptors
  and v4 FC1/activation/FC2 pipeline. Remote IDs and zero route weights are
  discarded before expert indexing. BF16-rounded contributions accumulate
  in FP32 and are converted once to the caller's BF16 output.
- M33..capacity uses EP-local M128 prefill with the existing route/Q0/task
  publication and FP32 scatter. Only its weight descriptors change.
- All M1..32 static compiler keys and the runtime-shaped prefill kernel are
  prepared before inference. One-launch remapping handles both ranges.
- Raw scales are retained for this first experiment. SF6 storage savings are
  not part of its proposed performance mechanism.
- Prepared owners reject changed weights/scales, unsupported geometry,
  overlapping output storage, and conflicting forced backends. They cannot
  fall back to a row-major kernel after relayout.

Rank imbalance, sparse-row padding, and collective costs remain. The proposed
gain comes from the changed weight access and decode implementation; there
is no justified numerical speedup estimate before measurement.

## Validation and decision

The startup canary collects independent stock compact outputs on the first
actual layer before relayout, then checks the new owner on the same inputs.
It retains the existing numerical limits and stock-repeat control checks.
It includes short decode, small/large prefill, concentrated and remote routes,
changed input contents at fixed addresses, and current/side-stream graph
replay. Failure prevents readiness and cannot be cleared by calling weight
finalization again. The canary covers one actual layer per rank, not all model
layers or sanitizer acceptance.

CPU lowering uses `probes/run_glm53_ep_tiled_cpu.py` through normal fleet
`--cpu` admission, an immutable serving image and CUDA bindings capsule, a
no-device/no-network container, 4 GiB memory and 2 CPU limits, and the existing
12 GiB available-memory guard. Its result is compilation evidence only.

The decision run must compare same-source TP and EP-tiled arms with matched
capacity/runtime settings, warm cache classification, the original quality
checks, and direct 2K/32K/128K prefill TTFT/tok/s plus fixed-1024 decode tok/s.
An old TP baseline or a component timing cannot establish non-regression.
Default adoption remains conditional on both retained prefill improvement
and resolved decode regression.
