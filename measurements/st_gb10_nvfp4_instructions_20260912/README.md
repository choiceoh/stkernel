# GB10 NVFP4 instruction and dependency-tree optimization

2026-09-12, srv4 GB10, CUDA 13.0.88, torch 2.13.0+cu130. Baseline:
`0990b6e0` (PR #572 merge). This change shortens the activation-scale reduction
feeding native NVFP4 MMA in the **current ST micro and stock-static MoE paths**.
Quantization, weights, MMA tiles, accumulation, and TP are unchanged.

**The isolated quantizer is 9.9–20.1% faster. A material whole-MoE speedup is
not established: the eight measured real-weight cases differ by less than
0.7% in paired medians under the final run's shared-GPU conditions.**

## Actual GB10 instructions

PTX acceptance alone does not identify hardware throughput. This run extracts
PTX and cubins from the compiled CuTe functions, then uses NVIDIA `nvdisasm`
13.0.85. The tool was extracted into a task temporary directory; neither the
CUDA image nor the host packages were modified.

| Operation | PTX | SASS on this GB10 / CUDA toolchain | Decision |
|---|---|---|---|
| Native NVFP4 GEMM | `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` | `OMMA` with E2M1 operands | Preserve native Tensor Core path |
| FP4 conversion | `cvt.rn.satfinite.e2m1x2.f32` | `F2FP.SATFINITE.E2M1...` | Preserve existing rounding and packing |
| Three-input absolute max | `max.abs.f32 d,a,b,c` | **Two `FMNMX` instructions** | Use a tree to expose independent work; no claim of half as many SASS instructions |
| Two packed FP32 multiplies | `mul.rn.f32x2` | **Two scalar `FMUL` instructions** | Reject for production; no additional benefit |

The standalone old/new quantizers each contain **17 FMNMX and 19 FMUL**
instructions. Sixteen max operations belong to amax and one to scale
clamping. The old zero-initialized left fold has 16 dependent max operations.
Eight max3 PTX operations in a tree lower to at most six dependent FMNMX
operations; the first five groups can proceed independently. This changes
instruction scheduling, not total floating-point work.

Every audited MoE specialization retains **448 native NVFP4 MMA PTX sites and
448 OMMA SASS sites**, before and after. These are static instruction-site
counts, not dynamic executed counts or FLOPS. Register usage is unchanged:

| Specialization | Registers/thread, old → new | Local / stack bytes |
|---|---:|---:|
| Micro T1 | 239 → 239 | 0 / 0 |
| Micro T4 | 222 → 222 | 0 / 0 |
| Static T6 | 234 → 234 | 0 / 0 |
| Static T12 | 234 → 234 | 0 / 0 |

`cuobjdump` reports 1 KiB of **static** shared memory for these functions.
This excludes launch-time dynamic shared memory and is not their total
shared-memory footprint. Tile geometry and allocation code are unchanged.

NVIDIA documents three-input max and packed FP32 multiply as available from
SM100. The lowering above is measured on SM121a and must not be generalized
to other Blackwell chips or compiler versions.
[PTX 9.0 max](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#floating-point-instructions-max),
[PTX 9.0 mul](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#floating-point-instructions-mul).

DGX Spark's advertised **1 PFLOP FP4 includes sparsity**. Dense expert weights
cannot claim that throughput just because they use FP4 storage. Native E2M1
input, 16-element E4M3 scaling, and FP32 accumulation already match the dense
NVFP4 compute form. This work reduces preparation latency while preserving
it; no weight pruning or change to scale granularity is introduced.
[NVIDIA DGX Spark hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html),
[CUTLASS NVFP4 warp MMA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html#cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op).

## Implementation and dispatch

- `engine/kernels/b12x/fp4_quant.py`: exact 16-value amax tree. The final
  positive zero preserves all-NaN and signed-zero behavior; no FTZ or
  NaN-propagating modifier is added.
- `moe_micro_kernel.py` and `moe_static_kernel.py`: apply it to input and
  intermediate activation quantization when `sf_vec_size=16`. The 32-value
  MXFP4 branch retains its original loop.
- `moe_dispatch.py`: include the helper in persistent CuTe cache identity,
  so existing objects cannot conceal the change.

Current profile boot defaults to `stock`. `MOE_STATIC_PRODUCTION="t,r,sf6"`
records a 2026-09-09 adoption; it is not the current ST default. That
configuration with today's GLM TP FP32-scatter guard raises
`GLM TP FP32 scatter requires the declared row-major static kernel`.
The final probe follows the current default and records actual dispatch:
T1/T4 **micro**, T6/T12 **static**. Optional v4/v5 tiled paths and the software
direct-micro body are not changed.

## Correctness and measurements

`quant.json` tests **135,210 blocks** covering all 65,536 BF16 encodings,
mixed exceptional values, wide-range FP32 values, FP4 rounding midpoints and
their immediate FP32 neighbors. Packed FP4 bytes, E4M3 scale bytes, and FP32
amax bits match the serial baseline under varied positive, +0 and -0 global
scales. The regression suite also uses an independent NaN-ignoring amax oracle.

Timed inputs are BF16. A CTA has 128 threads; each thread handles 16 values,
matching per-thread quantization work in MoE. CUDA Graphs contain 16 calls,
and 16 replays form one sample. Twelve rounds use six AB and six BA orders;
adjacent rounds form six balanced cycles. Every graph must overwrite
poisoned outputs before timing, rejecting empty or wrong-stream captures.

| 16-value blocks | Serial quantizer µs | Tree quantizer µs | Paired reduction |
|---:|---:|---:|---:|
| 256 | 1.673 | 1.509 | 9.91% |
| 1,536 | 1.723 | 1.535 | 10.89% |
| 16,384 | 2.561 | 2.047 | 20.07% |

The paired-multiply alternative performs essentially the same as the tree
alone and has the same FMUL count. It remains in the diagnostic probe only.

`moe.json` loads **only L3's four MoE tensors** from the actual TP=4 rank0
checkpoint (E288/H4096/I512/top8). Kernels share weights, activations, routes,
and the FP32 scatter implementation. T1/T4/T6/T12 use shared and spread-out
expert sets, including a zero-weight route. All eight old/new eager outputs
are byte-identical in this run; all repeated-call and graph errors are zero.
The guard permits 0.1% relative maximum error because FP32 atomic scheduling
can vary. Bitwise equality is measured here, not guaranteed for all schedules.

| T | Routing | Warm paired change | Evicted paired change |
|---:|---|---:|---:|
| 1 | Shared | +0.10% | +0.52% |
| 1 | Spread out | +0.24% | +0.51% |
| 4 | Shared | −0.03% | −0.09% |
| 4 | Spread out | −0.05% | +0.14% |
| 6 | Shared | +0.31% | +0.47% |
| 6 | Spread out | −0.60% | +0.02% |
| 12 | Shared | +0.34% | +0.52% |
| 12 | Spread out | −0.07% | −0.01% |

Positive means lower latency. Values are medians of paired cycle percentages,
not ratios of independent medians. Warm graphs contain eight calls; evicted
graphs contain one call after a 64 MiB write. Warm timings sometimes exceed
evicted timings by several times, so those absolute times cannot isolate a
cache effect. Other work can compete during a graph. This is a **shared-GPU
comparison**, not a dedicated fleet gate. All raw samples are retained. No
end-to-end TP=4 ITL/TTFT, token/s, quality, or acceptance-rate gain is claimed.

**294 engine tests passed, zero skips, in 100.180 seconds**, including four
new helper/cache/graph tests. Actual-weight probe peak Torch allocation was
1,193,049,600 bytes. Existing services were not stopped or reconfigured.

## Reproduction and evidence

- [quant.json](quant.json): final exactness checks and balanced timings.
- [moe.json](moe.json): actual dispatch, source hashes, all real-weight samples.
- [instructions.json](instructions.json): PTX/cubin/SASS hashes, opcode counts,
  resources and selected SASS excerpts for all 20 compiled artifacts.
- [engine-tests.log](engine-tests.log): complete regression output.
- [provenance.json](provenance.json): image, baseline, tool and source identity.

In the existing CUDA image, from the repository root:

```bash
mkdir -p /work/proof /work/baseline
git show 0990b6e0:engine/kernels/b12x/moe_static_kernel.py > /work/baseline/moe_static_kernel.py
git show 0990b6e0:engine/kernels/b12x/moe_micro_kernel.py > /work/baseline/moe_micro_kernel.py
PYTHONPATH=. python3 probes/engine_fp4_instructions.py --out /work/proof/quant.json
PYTHONPATH=. python3 probes/engine_moe_instructions.py \
  --rank /ranks/rank0of4.safetensors --baseline-dir /work/baseline \
  --out /work/proof/moe.json
python3 probes/engine_fp4_disassembly.py --root /work/proof \
  --nvdisasm /path/to/cuda-13.0/bin/nvdisasm --out /work/proof/instructions.json
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_engine_*.py'
```

Diagnostic wrappers explicitly pass `--keep-ptx --keep-cubin`; environment
variables alone did not preserve artifacts in this image. Diagnostic kernels
launch on the TVM-FFI current stream and verify graph execution. An early
empty-capture development run was discarded and is not used in these results.
