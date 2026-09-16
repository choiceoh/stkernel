# cuBLASLt pipeline follow-through — 2026-09-16

This change removes work from the MX producer's GPU execution and expands the
prepared cuBLASLt search. It builds on #1063. **Performance remains unmeasured**;
there was no GPU execution, fleet queue, reservation, service restart or serving
reader change. These are optimizations of the explicit comparison lane.

## GPU work removed

The MX producer receives BF16 values (including the packet consumer's explicit
BF16 rounding). Its clamped group maximum can therefore replace
`exp2(ceil(log2(amax/448)))` with an exponent/mantissa comparison. For
`amax = significand * 2**e`, the scale exponent is `e - 8 + (significand > 1.75)`.
The scale and its reciprocal are exact normal powers of two throughout the
finite, clamped BF16 range. The producer multiplies by that reciprocal instead
of dividing every input value. Infinity and NaNs remain nonfinite; this is not a
clipping or precision change. The existing DeepGEMM producer stays the reference.

`BF16Producer` and `PacketProducer` now bind private scale buffers before capture.
The initial call writes neutral scale padding; repeated calls update only real
rows. At M=8,K=4096 this changes scale stores from 16,384 to 1,024 bytes per
execution (93.75% fewer scale-store bytes); K=20480 changes 81,920 to 5,120 bytes.
This **does not shrink scale-buffer residency or imply a 93.75% kernel speedup**.
An arbitrary producer callback keeps its existing contract and padding behavior.

Bound kernels specialize row count and packet geometry at preparation. The
packet producer's per-row rank/stride divisions become constant arithmetic.
Binding warms the specialized kernel before it can enter a captured graph.
The dynamic convenience APIs remain available and correctly initialize arbitrary
caller-supplied output storage; they skip padding code for exact 128-row tiles.

## Search gaps closed

* Query both 16-byte and 256-byte alignment preferences. Previously every
  algorithm requiring more than 16-byte alignment was discarded, even when
  all actual buffers were aligned. Candidates now carry per-operand alignment
  requirements, are filtered against actual tuning pointers, and are checked
  again by the native binding. A sliced buffer never silently enters an
  incompatible algorithm.
* Enumerate the datatype-compatible algorithm catalog, rather than only IDs
  returned in workspace heuristic shortlists. Grow the catalog buffer until
  complete, bounded at 4096 IDs. Missing implementations contribute seeds even
  if their default tile fails; supported tile/stage configurations can qualify.
* Include 3/6/12-way FP32 Split-K alongside 2/4/8/16-way splits. Workspace stays
  capped at 64 MiB; admitted/timed candidates stay capped at 192. This remains
  a bounded search, not a claim of exhaustive global optimality.
* Capture 16 complete producer-to-output operations per timing graph. Normalize
  by both graph unroll and replay count, reducing Python submission gaps in
  tiny-GEMM timing. B/A/A/B admission still requires both paired samples to win
  by more than 2%, with the same numerical gate and GB10-only timing rule.

The CUDA 13.2 [algorithm catalog contract](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html#cublasltmatmulalgogetids)
and alignment capability attributes are the basis for the native search changes.
Catalog enumeration and AlgoCheck still require a GPU-backed plan; compilation
alone does not prove that any extra algorithm is usable or faster on GB10.

## Verification

The CPU Triton interpreter checks all 32,768 positive BF16 encodings (including
zero, subnormals, every scale boundary, infinity and NaNs) against the previous
scale formula, and checks the inverse separately. Producer tests compare FP8
bytes, independent NVIDIA scale layout, row/K tails, packet rank/stride handling,
redzones and initialized padding reuse. Native FP8 tie rounding and actual
cuBLAS accumulation still require GPU numerical proof.

`probes/engine_cublaslt_compile.py` compiles/loads the native binding and compiles
26 SM121 producer variants with GPUs hidden. It rejects floating-point
log/exp/div/reciprocal instructions in the MX producers and dynamic integer
quotient/remainder instructions in the bound variants. Compilation receipts are
code-generation evidence, not timings.

Reproduction uses the existing image
`st-engine:cuda13.2.1-runtime`, identity
`sha256:a9b53fd066bb4fa0c4d12982f7c5dcdd5a2591900a0088c3ffeb41ba0868425c`,
with `--runtime runc --network none`, no device nodes,
`NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, and bounded CPU/memory.

```sh
python3 probes/engine_cublaslt_compile.py --output /out/compile.json
TRITON_INTERPRET=1 python3 -m unittest -v tests.test_engine_cublaslt tests.test_engine_glm53_natives tests.test_engine_prefill_fp8_consumer tests.test_engine_native_cache
```

The existing `probes/engine_cublaslt_check.py` now uses the bound producers in its
full-pipeline search and changed-input graph replay. It was not run here.

Final receipts:

* `compile.json`: native compile/load plus 26 SM121 variants PASS, source hashes
  checked against this worktree; no GPU initialized.
* `focused.log`: 37 tests discovered, **31 passed and 6 CUDA-only checks skipped**.
  The all-BF16 scale/inverse comparison and actual producer interpreter tests ran.
