# Qwen MoE precision and smoothing review — 2026-09-19

Port two GLM facilities to Qwen's bound expert-parallel cell: FP32 accumulation
inside the generic dynamic prefill kernel and five-candidate (`as2`) activation
scale search in FC1/FC2. Both full top-10 routes and compact local top-1 pairs
are admitted, including micro/static compact tails. Model weights, global scales,
SiLU and existing BF16 contribution/output boundaries remain unchanged.

The FP32 path reuses GLM's saturated BF16-contribution scatter helper. It widens
the sum, not the contribution. FP32 atomics do not promise a fixed summation
order; their subnormal FTZ behavior is unchanged from the shared GLM helper.
The complete FP32 plane is initialized, retained by the workspace and stream
recorded. Compact pair counts are bounded by the accumulator's byte extent,
not GLM's source-token limit. Precision choices participate in native cache
identity. Qwen declares radius 2 before weight preparation, without selecting
GLM's tile-major layout; GLM's existing recipe is unchanged. Calibration identity
is revised because downstream real-prefill activations change.

## Validation

CPU policy, buffer ownership and neighboring contract checks pass; native GPU
validation is pending. No real-checkpoint TP4 output quality, acceptance or
throughput result is claimed. The probe is
`engine_kernel_check --lanes qwen38_moe_precision`, submitted through the
single-GB10 fleet queue with its 8 GiB budget.

The probe independently runs five FC2 slices, then sums their unchanged native
BF16 contributions in FP64. It compares old BF16 and new FP32 accumulation on
identical activation packing. A separate FP64 quantizer check measures search
SSE and exceptional-input parity. Projection oracles, all generic tile sizes,
poisoned-accumulator graph replay, zero routes and served compact prefill are
checked separately. Weights and inputs are synthetic at the actual Qwen cell.

## Channel smoothing: reviewed, not enabled

`probes/qwen38_smoothing_review.py` reproduces `smoothing-review.json` on CPU.
Blindly folding `((1+w)/s)-1` into Qwen's BF16 unit-offset norm changes the model
before quantization. For `w=0.125,s=64`, the stored new weight is `-0.984375`;
rescaling recovers gain **1.0 rather than 1.125**. On the seeded synthetic
64×2560 input / 320×2560 projection, 2162/2560 norm gains change and projection
relative RMSE is **2.258%**, without any W4/FP8 quantization.

A viable alternative leaves the norm and its BF16 output boundary intact and
scales the already-rounded input by a power of two in the dense input packer;
weight columns and the GPTQ Hessian must receive reciprocal matching factors.
On the finite normal fixture, input round-trip and FP64 projection are exact.
This does not prove invariance for underflow/overflow, native kernel rounding,
repacked weights or real-model quality. Scaling only the selected dense reader
also avoids changing other consumers of the mixed residual stream. Full norm
folding would need to include the hyper-connection gates, PLE, router and
experts that read affected streams; GLM's simple norm-to-projection map is not
Qwen's map.

Qwen target dense readers are W4A8/FP8, whereas GLM's strongest smoothing result
was on NVFP4 activations. Real Qwen channel statistics, repacked-weight error and
cost need evaluation before implementation. This review does not enable a fold.
