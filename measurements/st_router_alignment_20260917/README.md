# Fused router tail alignment

The prior fused router changed three numerical details: the FP32 projection
association, selected-column tie order, and weight denominator association.
This change aligns the last two while retaining the one-launch projection.
It does not establish that the earlier combined consumer quality difference
was caused by the router, or that aligning the tail improves answer quality.

## Implementation

- The served runtime is PyTorch 2.13.0+cu132, source
  `cf30153c4c131c8164ee7798e5022d810682e2cb`, and Triton 3.7.1.
- The CUDA 288-expert top-8 path gathers scores above the boundary in expert-id
  order, then boundary ties, then sorts in a 32-entry bitonic network with
  padding. The aligned tail reproduces this network only when selected scores
  tie. Ordinary strictly ordered rows avoid the extra permutation.
- Compiling `glm_pointwise._weights` for SM121 confirms FP32 XOR shuffle steps
  4, 2, 1. The aligned tail uses those same additions instead of eight sequential
  additions. Accurate exp and round-to-nearest division remain unchanged.
- Arrival counters are separated by device and execution/capture stream.
  Graphs captured on one stream still share the counter and require ordered
  replay, as the engine's shared graph pools already do.
- `route_skip` capture experiments fall back to the common routing path so the
  fused early return cannot skip the requested mask/renormalization.

## Validation

Pending: same-build CUDA comparison of served, served self-control, prior fused,
and aligned fused. The probe now requires exact selection order and weight bits
on identical logits, unchanged projection bits between fused variants, finite
tie/boundary/saturation fixtures, and independent-stream graph replay checks.
It measures both aligned-versus-prior-fused cost and aligned-versus-served cost.
Full-engine answer quality is outside this component check.

The consumer selector remains at its existing value during this implementation
comparison. `_align=False` is a private component control, not a serving knob.
