# W4A8 shared input and packed-pair evidence

Scope: the explicit ST tree decoder's staged W4A8 executor and its queued
numerical oracle. The ordinary production dense lane, W4A4 MoE, KDA FP32
state, weights and deployment packages are not changed. No GPU context,
fleet reservation, boot or serving measurement was used.

The baseline is `de8bfff6092c759cb5bca1d47270e68f38f7e03c` (#956). Both
implementations are compiled by the same compiler in each comparison.
`compile-triton371.json` uses the deployed frontend version, Triton 3.7.1;
`compile-triton380.json` is a second compiler check. Both use CPU Torch
2.14.0, not the deployment's CUDA Torch 2.13.0. These are code-generation
checks, not full runtime qualification. The reports record ptxas versions,
source hashes, cubin hashes, actual MMA operand types and assembled resources.

## Changes and costs

- At H=4096/I=3072, input group quantization executes once instead of being
  repeated by 24 FC1 tiles. This adds one stream-ordered launch (2 -> 3).
- Each packed byte is loaded in its native [128,64] tile and both nibbles
  share the signed-scale/table decision. The up weight is materialized
  after the gate dot. The native byte tables and numerical boundaries remain.
- At 16 rows, workspace grows 181,760 -> 249,344 bytes (+67,584), below
  256 KiB. All views are prepared once, disjoint within a plan and 16-byte
  aligned. A same-geometry output can feed the next invocation. Other views
  overlapping an input-publication plane are rejected before launch.
- Triton 3.7.1 incorrectly promotes FP8 dot operands on SM121 to FP16.
  Its native FP8 predicate admits exactly 89/120, omitting 121:
  [upstream source](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L910-L915).
  W4A8 GEMMs explicitly select consumer Blackwell lowering; the actual PTX
  and cubin target stays sm_121a. No precision option is relaxed. The queued
  oracle selects the same native FP8 arithmetic for a meaningful GPU diff.

At eight rows, the final assembled comparison is:

| Compiler | Stage | Registers before -> after | Shared bytes before -> after | MMA operands before -> after |
|---|---|---|---|---|
| 3.7.1 | FC1 | 255 -> 175 | 32,768 -> 4,096 | FP16 -> FP8 |
| 3.7.1 | FC2 | 173 -> 128 | 32,768 -> 4,096 | FP16 -> FP8 |
| 3.8.0 | FC1 | 206 -> 158 | 34,816 -> 4,096 | FP8 -> FP8 |
| 3.8.0 | FC2 | 179 -> 126 | 16,384 -> 4,096 | FP8 -> FP8 |

All 36 candidate stage/geometry combinations per compiler have zero stack
bytes and zero local-memory instructions. Triton 3.7.1's bundled assembler
is CUDA 13.1.80; Triton 3.8.0's is CUDA 13.3.33. No inference about hardware
latency follows from this table.

## Validation and limits

`final-focused.txt` runs ten CPU tests with Triton 3.7.1, including actual
kernel interpretation of all 65,536 packed-byte/signed-scale combinations,
quantizer group layout/tail masks, output feedback admission, workspace
alignment and the existing full-target W4A8 CPU numerical oracle. FP8 values
in the publisher test deliberately avoid the interpreter's known halfway
conversion error; general native RTNE remains an unexecuted GPU check.

The offline comparison covers 12 geometries (rows 1/4/8/15/16/17/24/31/32
at H4096/I3072 plus small and maximum admitted widths), three candidate
stages each. It rejects spills, queue atomics/polling, accidental FP16/FP4
MMA and a changed hardware target. Quantization/accumulation fidelity on a
real GPU, graph replay, quality, acceptance and serving tok/s remain
unmeasured. Static resource/instruction reductions are not timing claims.

The first full CPU suite invocation lacked Docker `--init`; two unrelated
host-reclaim process-exit tests failed because orphan brokers were not
reaped. That module passed all 18 tests with `--init`, without source changes.
`engine-check.txt` records the final full suite using `--init` and Triton 3.7.1:
205 files, 1,888 tests, zero failed/cannot-run, 374 explicitly skipped.

The opt-in GPU test is extended for rows 1/4/8/16/17/32, repeated graph
replay and output feedback. It was skipped here under the no-queue instruction.

Reproduction (no CUDA device needed):

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 PYTHONPATH=. python -m unittest tests.test_engine_w4a8_pipeline tests.test_engine_w4a8_dataflow
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_w4a8_pipeline_compile.py --baseline de8bfff6092c759cb5bca1d47270e68f38f7e03c --output /tmp/w4a8-compile
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python tools/check.py --list
```

Use `--init` if running the whole CPU suite in Docker. Keep raw cubins/PTX
outside the repository; the JSON hashes identify those compiled artifacts.
