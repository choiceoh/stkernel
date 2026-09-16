# Prepared cuBLASLt MXFP8 pipeline — 2026-09-16

Implemented an executable cuBLASLt comparison lane for the FP8 products reviewed in
[the CUDA 13.2 feasibility study](../st_cuda132_packed_20260914/CUBLAS_REVIEW.md).
The outcome is **unmeasured**. No GPU was used, no fleet ticket was submitted,
and no service was restarted. There is no claimed decode, prefill or acceptance gain.

The existing FP8Linear/DeepGEMM serving reader remains selected. The comparison
lane has explicit preparation and a reproducible GPU probe; it does not run a
search inside a forward or silently switch an unmeasured cell. This matters after
the observed compile-skew NCCL failures: per-rank algorithm search inside a model
forward would put a new unbounded wait in front of the next collective. Its native
binding is classified in `natives.PROBE_MODULES`, so an unused lane adds no boot
compile cost. Adoption requires a measured cell and the normal boot build barrier.

## Changes

* **Exact scale format without activation conversion kernels.** The existing
  group-128 power-of-two recipe directly publishes four equal MX32 UE8M0 scales.
  Quantized FP8 values are unchanged. Each four-row producer computes both values
  and scale words. At M >= 128, the four 32-row quarters of the scale tile share a
  producer so its scale words are contiguous; short decode batches use adjacent
  rows; a partial final tile launches only ceil(tail/4) producers. The last producer initializes the
  scale padding. Q and BF16 output retain real M.
* **Packet producer.** TP4 FP8-v3 packets can produce the same MX layout in one
  launch. The intermediate BF16 rounding is kept in registers. There is no full
  BF16 unpack buffer, scale conversion launch or change to the wire ABI. Ordinary
  and routed packet strides and partial final rows are covered.
* **Weight scales prepared once.** Positive normal power-of-two block scales are
  checked, encoded and replicated before execution. Prepared projections can
  share this immutable MX buffer. W stays in its existing FP8 row-major layout.
* **No physical transpose.** Column-major TN computes `Yᵀ = W Xᵀ` from existing
  row-major operands and writes directly to the existing BF16 output layout.
  FP32 compute and `FAST_ACCUM=0` are explicit; BF16 partial SplitK reductions are
  excluded. C and D use the same descriptor and address with beta zero.
* **Bounded algorithm search.** Heuristics at 0/1/8/32/64 MiB workspace plus tile,
  stage, swizzle, custom and FP32 SplitK variants are deduplicated and checked,
  capped at 96. Numeric rejects are recorded. Three finalists receive B/A/A/B
  CUDA-graph brackets of the complete producer-to-BF16-output pipeline. Both
  candidate samples must beat their paired baseline by >2%; otherwise the
  explicit prepared choice remains DeepGEMM. Search is GB10-only. SM120 is allowed
  for numerical checks with no timing.
* **Prepared execution.** Native run has no allocation, algorithm lookup or matrix
  transpose. A prepared Python projection never searches. Workspaces are isolated
  by CUDA stream and grow geometrically; old generations remain owned for graph
  lifetime. Scratch allocation may happen at warmup/capture, never graph replay.
  Preparation scratch is temporary; retained workspace is below twice the largest
  rounded allocation per stream (at most 64 MiB each), not 64 MiB per layer or per shape.

## Cost and limits

The additional weight scale buffer is `N*K/32` bytes for padded N: about 3.125% of
FP8 weight payload, while the original DeepGEMM scales remain available. Activation
scales occupy `ceil(M/128)*(K/128)*512` bytes. At M=8,K=4096 this is 16 KiB versus
1 KiB for the original FP32 scales. These costs are explicit in the probe output;
there is no claim that MX storage reduces residency.

Prepared weights are immutable and preparation belongs after storage compaction.
Geometry plans can be shared, but each prepared weight earns its own numerical
verdict. The paired timings are warm component-pipeline evidence, **not** cold
weights, TP4 serving throughput or acceptance. Weight-format equality is not proof
of identical accumulation: real cuBLAS numerics, changed-input graph replay,
actual packed model weights and matched serving measurements remain GPU gates.
The W4A8 decode products are not replaced or inflated to FP8.

## Reproduction

Offline checks use the existing `st-engine:cuda13.2.1-runtime` image with
`--runtime runc --network none`, `NVIDIA_VISIBLE_DEVICES=void`,
`CUDA_VISIBLE_DEVICES=` and no NVIDIA device nodes. Image identity:
`sha256:a9b53fd066bb4fa0c4d12982f7c5dcdd5a2591900a0088c3ffeb41ba0868425c`.
The compile receipt pins source and binary hashes and records the actual library
version (cuBLASLt 130400 in this CUDA 13.2 runtime).

```sh
python3 probes/engine_cublaslt_compile.py --output /out/compile.json
TRITON_INTERPRET=1 python3 -m unittest -v tests.test_engine_cublaslt tests.test_engine_glm53_natives
python3 tools/check.py --list --jobs 2
```

The Triton interpreter executes the actual producers and checks byte equality,
independent scale layout, every normal exponent (1..254), row and K tails, metadata
padding, redzones and packet rank/stride handling. Its cast tests use exactly
representable FP8 values because the interpreter has a known halfway-rounding
limitation; compiled native rounding still needs GPU validation.

**Not executed in this task:** on an owned idle GB10, the following probe records
algorithm metadata, workspace, per-candidate numerical outcomes and full-pipeline
brackets; it also verifies changed-input graph replay on a different capture stream.
It does not reserve a GPU or control services.

```sh
python3 probes/engine_cublaslt_check.py --gpu --shape 8x4096x20480 --shape 8x38784x4096 --output /out/cublas-decode.json
python3 probes/engine_cublaslt_check.py --gpu --shape 512x6144x4096 --producer packets --output /out/cublas-packets.json
python3 probes/engine_cublaslt_check.py --gpu --numerics-only --shape 8x512x4096 --output /out/cublas-numerics.json
```

No speed verdict is permitted from the last command, including on non-GB10 cards.
The default probe uses synthetic weights and cannot establish model acceptance.

## Sources

* [NVIDIA cuBLAS 13.2 documentation](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html): MX block scale layout, matmul descriptors and algorithm attributes.
* [NVIDIA MXFP8 cuBLASLt sample](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuBLASLt/LtMxfp8Matmul): scale format and column-major TN operand contract.

## Validation receipts

* `compile.json`: host binding compilation/loading and **10 SM121 producer variants
  PASS**, no GPU/context initialization. Source hashes match this implementation.
* `interpreter.log`: **16 tests PASS**, including actual producer execution on the
  CPU Triton interpreter and graph/workspace decision contracts.
* `focused.log`: **36 tests PASS, 3 GPU/interpreter skips**, including the package,
  native-cache and boot-builder integration checks after fixing the two findings
  from the first full CPU run.
