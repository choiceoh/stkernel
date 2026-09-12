# GB10 sparse NVFP4 feasibility and checkpoint comparison

2026-09-12, srv4 GB10, SM121a, CUDA 13.0.88, CUTLASS 4.5.0,
Torch 2.13.0+cu130. Base: `bc61f243`.

**Native sparse FP4 works and reduces standalone projection latency, but
simple pruning of the current checkpoint is not suitable for adoption.**
The synthetic shape sweep is 1.16–1.35x faster after a 64 MiB eviction write.
The real-L3 pilot is 1.34x faster for both projections, when dense and sparse
kernels calculate the **same already-pruned/requantized values**. However,
uncalibrated pruning changes the local projection outputs by about 45% in
relative L2 norm. This is not a measurement of language-model accuracy.

No serving kernel, model file, dispatch policy, or production service was
changed. The implementation is a reproducible diagnostic under `probes/`.

## Additional hardware constraint established by execution

The native sparse operation uses pairwise **4:8 along logical K**: retain two
of four adjacent two-element pairs. Generic scalar 2:4 masks are insufficient.
Only A is sparse, so the probe evaluates `W_sparse @ X.T`, storing the result
in token-major BF16. Accumulation is FP32.

There is a second constraint that was missing from the initial feasibility
discussion: **sparse `m16n8k128` with E4M3 `scale_vec::4X` uses one scale per
32 logical K values**. Four scale values cover K128. The compressed A has 16
retained values in that interval, but the logical interval is 32. The dense
`m16n8k64` path uses scales per K16. Both Red Hat and NVIDIA checkpoints use
the dense group-16 format; pruning alone does not make them directly compatible.

CUTLASS's `SM120_SPARSE_16x8x128_TN_VS` explicitly requires `VS == 32` for
E2M1/E4M3. The probe uses distinct dense/sparse scale layouts, duplicating
each logical K32 scale into two K16 scales for the dense baseline. Every
comparison therefore uses exactly the same mathematical operands.

[NVIDIA PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html),
[CUTLASS sparse MMA definitions](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm120_sparse.hpp),
[CUTLASS sparse NVFP4 example](https://github.com/NVIDIA/cutlass/blob/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu).

The compiled and executed kernels contain:

| Path | Native instruction | Static sites | Registers/thread | Stack/local |
|---|---|---:|---:|---:|
| Dense | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | 128 | 168 | 0 / 0 |
| Sparse | `OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X` | 32 | 168 | 0 / 0 |

Static instruction counts reflect template tiling/unrolling, not dynamic
instruction counts or a speedup ratio. `cuobjdump`'s 1 KiB SHARED field is
static shared memory only; it excludes launch-time dynamic shared memory.
Both kernels use a 128×128×256 CTA tile. The installed sparse template rejected
N32 tiles at compile time; the working N128 tile is not a tuned tiny-decode kernel.

## Correctness and timings

The synthetic sweep uses eight distinct active experts. `N` is tokens **per
expert**; this models balanced/shared expert batches rather than real routing
skew. Shapes are TP=4 GLM projections: w13 `[1024,4096]`, w2 `[4096,512]`.
This is one batched GEMM at a time, not fused FC1/SwiGLU/quant/FC2/scatter.
Input quantization, routing, TP communication and attention are excluded.

All six legal pair masks, both signs, FP4 encodings and varying E4M3 scales
are exercised. Synthetic inputs are already quantized. Both kernels' BF16
outputs are byte-identical to an independently dequantized FP32 Torch BMM
reference in all eight sweep cases and both real-weight pilot cases. TF32 is
disabled. Tests additionally exercise signed-zero pairs, zero weights/scales,
N129 tails, multiple batches, invalid masks, quantization midpoint ties, and
nondefault-stream CUDA Graph replay with poisoned outputs.

**Seven tests pass, zero skips, in 1.451 seconds.** Sweep peak Torch allocation
is 524,812,800 bytes. The pilot's allocation is recorded in `pruning.json`.

Each measurement has twelve rounds, alternating AB/BA. Adjacent rounds form
six balanced cycles. Warm graphs contain sixteen GEMMs; eviction graphs contain
one GEMM after a 64 MiB device write. All graphs must overwrite poisoned outputs.
Compression and both scale-layout preparations happen before timed GEMMs.

| Projection | Tokens/expert | Dense, eviction µs | Sparse, eviction µs | Paired speedup |
|---|---:|---:|---:|---:|
| w13 | 1 | 175.54 | 126.64 | 1.35x |
| w13 | 6 | 164.38 | 127.49 | 1.29x |
| w13 | 32 | 173.69 | 133.99 | 1.30x |
| w13 | 128 | 168.94 | 144.99 | 1.16x |
| w2 | 1 | 87.55 | 73.40 | 1.19x |
| w2 | 6 | 94.90 | 73.18 | 1.31x |
| w2 | 32 | 98.27 | 76.70 | 1.29x |
| w2 | 128 | 116.82 | 95.51 | 1.22x |

Latencies are medians of cycle means; speedup is the median of paired ratios.
The GPU is shared with existing services. Warm results include 2–3x ratios and
sometimes exceed eviction latency; these are not evidence of an architectural
2–3x gain. Raw warm and eviction samples are preserved. An eviction write is a
cache-pressure procedure, not a measured guarantee of every line's residency.
No whole-engine speedup, token/s, TTFT, ITL or dedicated-fleet result is claimed.

Weight traffic can improve as well as compute. For eight w13 experts:
16 MiB dense FP4 + 2 MiB K16 scales becomes 8 MiB compressed FP4 + 2 MiB
metadata + 1 MiB K32 scales: **18 → 11 MiB, 38.9% less**. w2 is 9 → 5.5 MiB.
This includes scale regrouping and is not a lossless conversion of current weights.

## Real-weight pruning pilot

`engine_sparse_nvfp4_prune.py` reads only L3 experts 0–7 from rank0's verified
up|gate layout. It reverses the existing scale swizzle; global scales were
already folded during presharding. Per-tensor sample hashes are recorded.
The original checkpoint is opened read-only and no modified weights are saved.

The pilot keeps the two largest squared-magnitude pairs in each K8 group,
then requantizes with an E4M3 scale per K32 and nearest-even FP4 rounding.
Six Gaussian input rows per expert are quantized once to K32 and shared by
all projection comparisons. There is no calibration, Hessian correction,
finetuning, distillation, or real activation capture.

| Projection | Output relative L2, pruning only | K32 rescaling only | Pruning + rescaling |
|---|---:|---:|---:|
| w13 | 44.98% | 5.86% | 45.26% |
| w2 | 44.88% | 5.64% | 45.12% |

These errors compare local linear projections on synthetic inputs, not task
accuracy or output-token agreement. Activation K16→K32 error is not included:
the same K32-quantized activations are used on both sides. The comparison is
against the existing quantized checkpoint, not an original high-precision model.

Both native kernels correctly calculate the pruned values. On these values the
eviction measurements are w13 **166.37 → 124.35 µs** and w2 **88.28 → 65.87 µs**.
The pruning distortion is much larger than kernel numerical error. Production
adoption would need pair-aware calibrated pruning, scale-aware reconstruction,
and likely selective finetuning/distillation with end-to-end quality gates.

## Red Hat versus NVIDIA checkpoint

The NVIDIA revision inspected/downloaded is
`09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.

Both use ordinary dense NVFP4 expert weights. A read-only Red Hat rank sample
of 100,663,296 FP4 values found 6.84% zeros, 0.0168% eligible K8 groups and
zero fully eligible 16×128 tiles out of 49,152. Sampled scales were finite and
nonzero. The NVIDIA sample is narrower: the first 1 MiB of L3 expert0 gate
weights, with 6.77% zero codes and 35 eligible groups out of 262,144. Its scales
were not inspected in that sample. Neither sample is a whole-model sparsity audit.

NVIDIA additionally quantizes dense MLPs in layers 0–2; shared experts and
attention stay BF16. Nine extra matrices contain 452,984,832 weight elements:

| Extra quantized scope | Red Hat BF16 | NVIDIA NVFP4 + block scales | Saving |
|---|---:|---:|---:|
| All four TP ranks | 864 MiB | 243 MiB | 621 MiB |
| One TP=4 rank | 216 MiB | 60.75 MiB | 155.25 MiB |

Global scale scalars add a negligible amount. 3.56x smaller weights do not imply
3.56x faster whole-model inference. If these MLPs accounted for 5% of elapsed
time and their actual implementation became 3.56x faster, Amdahl's law predicts
only 1.037x overall. This is an example, not a measured share or speedup.

**The NVIDIA download is larger because its MTP experts are BF16.** All 864
NVIDIA MTP expert weight headers were checked in shards 1–3. Red Hat holds
the corresponding 864 weights as FP8 plus 864 BF16 scale tensors:

| Tensor payload | Red Hat | NVIDIA |
|---|---:|---:|
| MTP expert weights + scales | 7,248,642,048 bytes | 14,495,514,624 bytes |
| Whole checkpoint tensors | 197,816,133,372 bytes | 204,419,110,596 bytes |

The MTP expert increase is 7,246,872,576 bytes, offset by 651,165,696 bytes of
dense-MLP savings. Another 7,270,344 bytes of other tensor differences remain
unattributed, yielding a net increase of 6,602,977,224 bytes. ST's current rank
plan uses layers 0–44; the extra MTP payload is not part of that target arena.
The linked NVIDIA recipe says MTP was not exported, but the pinned checkpoint
actually contains 889 layer-45 tensor entries. Header evidence takes precedence.

NVIDIA's model card reports GPQA 92.11% against its own BF16 92.17%; Red Hat
reports 90.57% averaged across three seeds. Their published protocols differ;
there is no controlled head-to-head result here. Official provenance is not
proof of superior quantization accuracy. The current GLM loader is specific
to Red Hat's compressed-tensors layout; ModelOpt names and global-scale
semantics require validation before comparing models in ST.

[NVIDIA checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4),
[NVIDIA quantization recipe](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml),
[Red Hat checkpoint](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4).

## Reproduction

Use the existing `st-engine:9391` image, an isolated task directory, the rank
directory mounted read-only, and an explicit entrypoint. Compilation needed
more than an 8 GiB container limit; the successful CPU build was capped at
16 GiB and two CPUs. Runtime probes were capped at 4–6 GiB container memory
and 2–3 GiB Torch device allocations. Existing services remained running.

Inside the image, with repository `/repo`, output `/work` and rank files `/ranks`:

```bash
cutlass_path=/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass
mkdir -p /work/build
nvcc -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
  --keep --keep-dir /work/build --expt-relaxed-constexpr --expt-extended-lambda \
  -Xcompiler=-fPIC -shared -I "$cutlass_path/include" \
  -I "$cutlass_path/tools/util/include" /repo/probes/engine_sparse_nvfp4.cu \
  -o /work/sparse.so
cd /repo
PYTHONPATH=. ST_SPARSE_PROBE_LIBRARY=/work/sparse.so python3 -m unittest \
  discover -s tests -p test_engine_sparse_nvfp4.py -v
python3 probes/engine_sparse_nvfp4.py --library /work/sparse.so --out /work/results.json
python3 -m probes.engine_sparse_nvfp4_prune --library /work/sparse.so \
  --rank /ranks/rank0of4.safetensors --out /work/pruning.json
```

`cutlass-128.json` and `pruning.json` retain all raw timings, numerical checks,
source/tensor hashes and allocations. `instructions.json` contains SASS excerpts,
hashes and resource reports. `checkpoint-comparison.json` contains byte accounting.
`redhat-audit.py` reproduces the CPU-only rank sparsity sample. The NVIDIA
download runs separately into `/home/choiceoh/models/glm53-nvidia-nvfp4`, with
revision pinned and `hf cache verify --fail-on-missing-files` after completion.
