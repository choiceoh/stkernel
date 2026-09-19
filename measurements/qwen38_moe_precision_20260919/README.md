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

CPU policy, buffer ownership and neighboring contract checks pass. Native GPU
validation passed twice on GB10; final ticket `q38moeprecision-0919b`
(`17898179582898539`), source `71b1e973`, PyTorch `2.13.0+cu132`, CUDA 13.2.
Peak tensor allocation was **1,246,733,824 bytes**. No real-checkpoint TP4 output
quality, acceptance or throughput result is claimed. The probe is
`engine_kernel_check --lanes qwen38_moe_precision`, submitted through the
single-GB10 fleet queue with its 8 GiB budget.

The probe independently runs five FC2 slices, then sums their unchanged native
BF16 contributions in FP64. It compares old BF16 and new FP32 accumulation on
identical activation packing. A separate FP64 quantizer check measures search
SSE and exceptional-input parity. Projection oracles, all generic tile sizes,
poisoned-accumulator graph replay, zero routes and served compact prefill are
checked separately. Weights and inputs are synthetic at the actual Qwen cell.

| Check | Final result |
|---|---|
| FC2 accumulation, identical five native BF16 contributions | SSE **0.0161063 → 0.00519431 (−67.75%)**; FP32 output byte-equal to rounded independent FP64 sum |
| FP64 quantizer reconstruction | **151,322** finite blocks; no worsened block SSE beyond 2e-5 relative tolerance; radius-0/exception fallback byte parity and graph replay pass |
| FC1/FC2 `as2`, same dequantized weights vs no activation quantization | Full MLP SSE **−21.35%, −13.16%, −20.03%** across three synthetic input batches; both comparison arms use FP32 accumulation |
| Native projection vs independently decoded packing / torch matmuls | **12** cases, compact top-1 and full top-10, micro/static/dynamic; max relative error **0.00625**, below declared 0.02 gate |
| Changed-input graph replay with poisoned FP32 workspace | Tiles **32/64/128**, byte equality; entire workspace zero-fill and zero route weights pass |
| Served compact prefill | **4,096** rows, finite output; all foreign routes give exact zeros |

The SSE changes are local numerical evidence, not model accuracy percentages.
The first ticket passed too (`gpu-initial.log`); it preceded full-route dynamic
and activation-projection ablations. BF16 atomic scheduling changes the old
sum slightly between runs (initial reduction 68.11%, final 67.75%); the new
rounded FP32 result was identical. `gpu.json`, `gpu.log` and
`gpu-source.sha256` retain final records and tested source identity.

The final CPU regression run covers **664 tests in 78 modules: 546 passed,
118 CUDA-only skips**, no failures (`cpu.log`, 110.165 seconds). It includes
Qwen engine/probe tests, MoE dispatch, source contracts, calibration identity,
graph owners and default declarations. An earlier stdin-based invocation
could not spawn the MTP distributed-test children; rerunning the complete
suite through `python -m unittest` passes. No kernel change was needed for it.

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
