# Decode V reads and owned FP8 head output

The normal DFlash attention now reads V directly from the packed QKV projection, using compile-time row and token strides. Each cell's heads remain packed. Q/K, attention arithmetic, accumulation, cache contents and split geometry are unchanged. Both the one-row and batched callers use the direct read by default.

The captured target's FP8 head writes directly into an owned, padded output buffer shared by every context-capacity graph of one `(n, tokens)` shape. The same logical vocabulary view remains the sampler input. The head's activation quantization, weight pack, BF16 output and observation hook are unchanged. Invalid output shape, dtype, device, stride or alignment is refused before GEMM initialization. Existing head/linear graph labels are retained.

GLM TP4 has 38,720 logical vocabulary columns per rank, padded to 38,784 by the existing FP8 recipe. The 64 padding columns are excluded from the logical view and token IDs; greedy vocabulary selection already reads row strides. Stochastic/rich full-vocabulary gathering still packs the logical view for communication. That packing replaces the old target copy for those callers: do not claim its removal there.

## Source work counts, not measured speed

For C1, K=7 (eight target/draft rows), per rank and ordinary target-plus-proposal step:

| Removed operation | Launches | Copied tensor bytes |
| --- | ---: | ---: |
| Five drafter V materializations | 5 | 20,480 |
| Target logits copy before greedy sampling | 1 | 619,520 |
| Total for that greedy path | 6 | 640,000 |

Each copied byte formerly required a read and a write; these are logical work counts, not measured DRAM traffic. The shared head destination adds only the existing GEMM padding (1,024 bytes at eight rows) and replaces the separately materialized padded GEMM result. Graph-pool high-water memory and serving latency are unmeasured. Neither new persistent weight/state storage nor a precision/K change is introduced.

## Completed checks

- Existing ST image `sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`, runc, no network, 2 CPUs/4 GiB, CUDA hidden: 37 tests, 24 passed and 13 GPU-only skips. Output ownership, observer preservation, bad destination rejection, padded vocabulary boundaries, capacity-buffer identity and C1–C4 V wrapper dispatch are covered. See [cpu-tests.log](cpu-tests.log).
- Actual attention/combine kernels in the CPU Triton interpreter: 15 exact strided-versus-contiguous comparisons across C1–C4, startup, wrap and long context. Poisoned adjacent columns remain untouched. The interpreter uses FP32 data to check address coverage and formula equality, not GPU BF16 rounding. See [interpreter.json](interpreter.json).
- Actual SM121 PTXAS compilation: six BF16 variants for token widths 1/8/32 and contiguous/packed V pitches, all passed. Shared memory remains 28,800 bytes in each compiled case; no CUDA context initialized. See [compile.json](compile.json).
- Added deferred CUDA checks for exact contiguous/strided attention, changed-input/slot graph replay, FP8 output equality/replay at widths 129 and 38,720, and sampling from padded row pitches with poisoned vocabulary tails.

No GPU reservation, model boot, image build, numerical GPU run or performance claim is part of this change. Default activation and proof status are separate. Actual step/s, tok/s, quality and acceptance remain pending. [Manifest and source hashes](manifest.json).

## Reproduce in an existing CPU-only ST container

Mount the checkout at `/work` and an output directory at `/out`; use `--runtime=runc --network=none --cpus=2 --memory=4g --pids-limit=256`, `CUDA_VISIBLE_DEVICES=`, `NVIDIA_VISIBLE_DEVICES=void`, and one thread each for OMP/OpenBLAS/MKL.

```sh
python3 -m unittest tests.test_engine_decode_buffers tests.test_engine_draft_attention tests.test_engine_sampling tests.test_engine_draft_acceptance -v
TRITON_INTERPRET=1 python3 probes/engine_decode_buffers_check.py --mode interpreter --output /out/interpreter.json
python3 probes/engine_decode_buffers_check.py --mode compile --output /out/compile.json
```
