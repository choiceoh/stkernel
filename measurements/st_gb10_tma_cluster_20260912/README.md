# GB10 TMA and cluster optimization campaign

Baseline: `815dc9551a9bc251c5c85dc85e0c34554ad35671`, ST GLM TP=4.
The same mHC and MLA baseline files are available byte for byte at public
`main` revision `d881f8dc`; the reproduction commands below use that revision
so they do not depend on the local experiment branch.
The baseline MLA and mHC sources also matched the concurrent native-serving
adoption checkout byte for byte. Initial tests used a private srv4 container and the existing
`st-engine:9391` image (Torch 2.13.0+cu130, TileLang 0.1.12, CUDA 13).

**Adopted into engine source defaults at the operator's explicit request to
proceed without benchmark results.** GPU experiments stopped when the
serving-recovery owner requested the fleet. No fleet service was restarted or
deployed by this change. GPU execution, graph replay and performance validation
of the newly adopted kernels remain pending.
The initial results below are screening measurements, not a fleet acceptance
gate. GPU clocks were not fixed and the machine was not exclusively reserved.

## Adopted defaults

- mHC post: hidden tiles of 512 columns, grid `(tokens, hidden/512)`, 128
  threads per CTA. Two explicit `T.tma_copy` operations share one transaction
  barrier. The elected TMA producer also performs the arrival, so another
  thread cannot arrive before the producer registers its expected bytes.
  The post decorator fixes TMA on and warp specialization off; other mHC
  kernels retain their own pass settings. FP32 channel accumulation order
  and BF16 rounding are unchanged in the source.
- MLA: BF16 `ldmatrix.x4` Q/P loads and transposed 8-bit `ldmatrix` PV loads
  replace separate fragment reads in both ordinary and cluster kernels.
  The existing FP8 ring, split plan, MMA order, DSMEM reduction, and shared
  memory allocation remain in use.

GPU-free checks on the final source:

- `mhc-adopted-compile.json`, `mhc-adopted-generated.cu`: TileLang 0.1.12
  lowered the actual post body, then NVCC 13.0.88 compiled it for SM121a.
  The generated code contains two tensor-map loads for the 4,096-byte
  residual tile and one 1,024-byte bulk load for the layer output. One wait
  covers all 5,120 bytes. The compiler combines the elected producer's last
  expected-byte update and arrival into `arrive_and_expect_tx`.
- `mhc-adopted-nvcc.log`: 48 registers, zero spills; 5,120 bytes of dynamic
  shared storage plus the compiler's 1,024-byte static shared allocation.
- `mla-adopted-nvcc.log`: ordinary/cluster kernels compile with 111/64
  registers and zero spills, retaining 46,976 bytes of dynamic shared memory.
- `mla-adopted-host-check.json`: the CUDA headers' **CPU** implementations
  preserve all 65,536 packed/strided FP8 pair results. Q/P/PV address equations
  extracted from the adopted source match the MMA fragment coordinates.
  This does not prove GPU execution or GPU NaN encoding.
- `*-adopted-sass.json`: actual TMA and matrix-load opcode sites in the final
  compiled cubins. Static instruction counts are not speedup measurements.

Reproduce the local checks with TileLang 0.1.12, Torch (CPU is sufficient),
and a CUDA 13.0.88 toolkit:

```sh
python3 probes/engine_mhc_compile_check.py --output /tmp/st-mhc-compile
python3 probes/engine_mla_host_check.py --cuda-root /usr/local/cuda --output /tmp/st-mla-host
```

## Initial results

- `mhc-auto-initial.json`: four token counts, post and pre-with-norm, four
  changed-input graph replays. The stock, BF16 shared-memory declaration,
  and TMA-enabled pass variants preserve output storage bits. Enabling warp
  specialization fails to compile pre-with-norm (`no available layout found`).
  Post compiles with it, but emits no TMA load.
- Generated CUDA for matching shapes is identical across the automatic-pass
  settings. Nine cached CUDA files have only four unique contents, corresponding
  to the post kernel and prenorm split specializations. The compiler already
  narrows the pre-mix shared copy to BF16. Timing differences between these
  identical kernels must not be counted as improvements.
- The installed TileLang `cuda/op/copy_analysis.cc` selects automatic TMA with
  `allow_load=false, allow_store=true` for an ordinary `T.copy`. Clearing
  `tl.disable_tma_lower` alone therefore does not create the desired loads.
  Explicit `prefer_instruction="tma"` or `T.tma_copy` is needed for the next
  comparison. The former supplies a synchronous barrier/wait; the latter lets
  two independent loads share one completion wait.
- `mla-tma-serial-initial.json`: one producer issues sixteen 512-byte bulk
  copies per tile. Outputs match, but the candidate loses to the established
  load pipeline. It is not selected.
- `mla-tma-warps-initial.json`: eight producer warps each issue two bulk
  copies into a shared transaction barrier. All six shapes and graph replays
  preserve BF16 output bits. Some evicted-cache cases improve, but warm
  cluster latency and unsplit prefill regress. It is not selected.
- `mla-dsmem-vector-initial.json`: replacing coalesced scalar DSMEM reads
  with per-lane float4 reads preserves output bits but slows all tested
  cluster shapes. Warm regression is 7.8–26.0%. It is not selected.

MLA screening also exhausts all 65,536 FP8 byte pairs and 4,096 warp max
fixtures. Full, ragged, empty and duplicate slot lists are checked, along with
changed-input graph replays and an independent FP32 attention reference.
The candidate cluster is compared with the **existing cluster**, as well as
the ordinary kernel; existing cluster gains are not credited to a new change.

## Additional candidates and follow-up comparisons

- mHC post: partition independent hidden tiles between blocks; compare 512
  and 1,024 elements per tile. At six tokens this exposes 48 or 24 blocks,
  versus six in the baseline. This is a work-count fact, not a measured speedup.
- mHC post: explicit TMA loads, including larger full-hidden tiles and a
  shared barrier for the two input loads. Preserve the four-channel FP32
  arithmetic and BF16 output rounding.
- mHC post: a two-stage `T.Pipelined` loop with warp specialization, which
  provides the producer context required for automatic TMA load selection.
  Actual lowering and performance still require the GPU/compiler run.
- MLA: adjust shared row padding and test invertible 16-byte XOR copy-atom
  layouts. Preserve KV selection, split count, MMA arithmetic and reduction
  order. Compare both ordinary and cluster execution.
- MLA: independently validate an integer FP8-to-BF16 encoding candidate;
  all finite FP8 values are exactly representable. This candidate is not
  selected; its GPU byte-pair validation, including NaN encoding, is pending.
- MLA: transposed 8-bit `ldmatrix` gathers two output tiles' packed FP8
  fragments per instruction. Its row-address permutation preserves the MMA
  B register order. BF16 `ldmatrix.x4` also replaces separate Q and P fragment
  reads. The combined candidate is now adopted; GPU execution is still pending.

## GPU-free compiler screening

`mla-compile-screening.json` records source hashes, register counts, spills,
and static SASS opcode sites from a local x86_64 NVCC 13.0.88 compilation with
`-O2 -std=c++17 -arch=sm_121a --cubin`. This matches the server's compiler
version and device target. It is not a timing result. Site counts include NOP
padding and must not be interpreted as utilization or executed instruction
counts.

| Candidate | Ordinary registers | Cluster registers | Ordinary instruction sites | Cluster instruction sites |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 113 | 64 | 1,424 | 1,600 |
| FP8 `ldmatrix` | 113 | 64 | 1,368 | 1,576 |
| BF16 Q/P `ldmatrix` | 107 | 64 | 1,400 | 1,576 |
| FP8 + BF16 Q/P `ldmatrix` | 111 | 64 | 1,352 | 1,544 |
| Integer FP8 conversion | 108 | 64 | 3,104 | 3,288 |

These candidates spill no registers. The `ldmatrix` candidates keep the
46,976-byte shared allocation. The integer candidate more than doubles code
sites because of subnormal/NaN handling, so it is lower priority. The XOR
layout also increases code sites and cluster registers, with no apparent
baseline bank conflict to remove.

Generate the principal `ldmatrix` candidates with:

```sh
git show d881f8dc:engine/kernels/mla/glm53_megakernel.cu > /tmp/st-mla-baseline.cu
python3 probes/engine_mla_ldmatrix_candidates.py --baseline /tmp/st-mla-baseline.cu --output /tmp/st-mla-candidates
```

Run `probes/engine_mla_hardware_check.py` with that baseline and a generated
candidate only after the serving owner releases a GPU test window. It compares
the existing ordinary and cluster paths with both candidate paths. Passing
coordinate checks and compilation does not establish GPU numerical correctness.
The follow-up harness also exhausts packed-versus-strided FP8 conversion bits,
and covers full prefill through 8,192 rows with a 768 MiB Torch allocation cap.
It allocates no unused split partials in the unsplit prefill cases.

`probes/engine_mhc_tma_check.py` writes each transformed source and its hash,
all per-variant failures, correctness results and alternating graph samples.
Follow-up timings use 16 kernel calls per graph to amortize Python graph-launch
cost. Each post case also checks partition boundaries against an independent
CPU FP32 `fmaf` reference. The initial mHC timing method used separate graph
launches; compare variants within a run, not across the two timing methods.
No environment knob or automatic tuning is added to serving.

Local verification also covers nine kernel-package/source-boundary tests,
four MLA driver contract tests, and eight knob tests (three FlashInfer-dependent
tests skip because the package is unavailable). CPU Torch 2.9.1 was installed in a private local venv for these
checks; no Spark GPU or UMA process was started. The earlier 19 candidate
Python syntax checks and source-generation checks also passed. No checkpoint
or presharding file was changed here.

For the later GPU comparison, supply the baseline explicitly so that "stock"
cannot accidentally refer to the newly adopted default:

```sh
git show d881f8dc:engine/kernels/mhc/tilelang_kernels.py > /tmp/st-mhc-baseline.py
python3 probes/engine_mhc_tma_check.py --source /tmp/st-mhc-baseline.py --variants stock,adopted --output /tmp/st-mhc-ab
```

Reference: [TileLang copy operations](https://www.tilelang.com/autoapi/tilelang/language/copy_op/index.html).
Installed compiler source, generated CUDA, and execution are authoritative for
the pinned image; newer public documentation alone is not execution evidence.

The fragment layout is specified by [NVIDIA PTX ISA, ldmatrix](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix).
The 8-bit row/register ordering was also checked against [NVIDIA CUTLASS copy traits](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm100.hpp).
