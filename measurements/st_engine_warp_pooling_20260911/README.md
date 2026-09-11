# GB10 one-warp key pooling — 2026-09-11

The GLM return-only `kpool_compress` lane now runs one warp per 128-channel pool,
uses XOR partner exchanges for the Hadamard transform, and allocates only its
returned FP8 keys and FP32 scales. The dummy cache, location array and unused
write mask are gone. It accepts input strides directly instead of materializing
contiguous copies. The separate cache-writing entry point retains its original
rotation and default four-warp launch.

Measured code: `05d0196ea7ef03fbd2c00a254fdaa4d5c57eaffc`.
Baseline: main `d44e3825` (merged PR #546). The probe loads an exact exported copy
of the baseline `engine/kernels/kpool.py` and reproduces the original
`lanes.served().kpool` adapter body. The baseline file hash is in `final.json`.
`source-sha256.json` pins 160 current Python files, verified byte-for-byte on
srv1. Subsequent evidence commits do not change the measured implementation.

## Hardware and numerical behavior

- One warp owns all 128 channels, four values per thread. Fixed XOR permutations
  replace the reshape/transposition butterflies in this return-only path.
- Loaded SM121 kernel resources: **46 registers/thread, zero shared memory,
  zero spills**. Baseline: four warps, 32 registers/thread, 512 shared bytes,
  zero spills. Resource values are reported for every measured pool count.
- Both BF16 rounding boundaries remain: after softmax-weighted pooling and
  after the normalized Hadamard transform. FP8 clipping, power-of-two scales
  and the per-channel slot accumulation order are unchanged.
- Pool/slot/channel input strides are explicit compile-time facts; no input
  copy is needed even for strided views. The original cache-writing wrapper
  uses default unit channel strides after its existing contiguous conversion.
- `WARP_LOCAL_ROTATION` is an internal compile-time choice fixed by the caller,
  not a runtime setting, autotuner or fallback. The serving lane always calls
  `compress_pool_keys`; cache-writing callers keep the original transform.
- An empty input returns correctly shaped empty tensors. No cache, locations
  or write mask are read through the disabled cache-writing branches.

## Paired GB10 measurements

`final.json` contains all samples' median/p95 summaries and per-round eager
results. Five rounds alternate A/B then B/A. Eager cases have 100 samples/round
and 40 warmups. Each graph captures 100 consecutive calls; 50 replays/round after
10 warmups are timed with CUDA events and divided by 100. CUPTI starts only after
all latency measurements. No random generation or correctness comparison is
inside a timed sample.

The **complete lane** includes its allocations/preparation kernels. The
**arithmetic-only** pair preallocates outputs and compares the baseline
four-warp arithmetic kernel with the new one-warp kernel. This separates launch
preparation savings from arithmetic/layout savings.

| Pools | Complete lane, old → new µs | Lane reduction | Arithmetic only, old → new µs | Arithmetic reduction |
|---:|---:|---:|---:|---:|
| 1 | 4.315 → 2.012 | 53.4% | 2.075 → 2.012 | 3.1% |
| 2 | 4.325 → 2.030 | 53.1% | 2.092 → 2.030 | 3.0% |
| 16 | 4.433 → 2.137 | 51.8% | 2.195 → 2.137 | 2.6% |
| 64 | 4.623 → 2.278 | 50.7% | 2.383 → 2.278 | 4.4% |
| 512 | 6.714 → 3.263 | 51.4% | 4.210 → 3.270 | 22.3% |
| 4,096 | 22.423 → 9.670 | 56.9% | 20.319 → 9.650 | 52.5% |

Small-pool improvements mainly come from removing preparation work. The complete
lane changes from **three CUDA kernels to one**, with no copy events in either
path. Example eager wall times: two pools 187.224 → 92.568 µs; 512 pools
186.559 → 93.008 µs.

Peak additional eager allocations **include returned outputs**: two pools
10,752 → 1,024 bytes; 512 pools 80,896 → 67,584 bytes; 4,096 pools
582,656 → 540,672 bytes. Output allocation is required and is not described as
zero allocation. The optimized path has no additional temporary tensor.

The real-weight L3 indexer compares identical current lane tables except for
`kpool_compress`. It includes projection, quantization, pool writes, scoring,
selection and address generation. Five rounds, 50 samples/round, ten warmups:

| Eager L3 phase | Baseline | Optimized | Reduction |
|---|---:|---:|---:|
| Decode, context 2,048, six tokens | 2.059962 ms | 1.967074 ms | 4.5% |
| Prefill, 256 tokens | 2.146504 ms | 2.053596 ms | 4.3% |

The L3 peak allocations remain 252,416 bytes for decode and 6,948,352 bytes for
prefill: larger parts of the indexer still set the overall peak. These are
same-process component measurements with real weights and synthetic activations;
they are **not full-model ITL, throughput, generation quality or fleet scaling**.
Absolute times from the previous PR's measurement session are not compared or
added to this session's results.

## Validation and environment

- `final-engine-tests.log`: **138 GPU tests passed without skips** at the final
  implementation. `local-tests.log`: 96 CPU tests passed, 42 CUDA tests skipped.
- `xor-tests.log`: four focused GPU tests. Random cases cover 0/1/2/31/32/33/64/512
  pools, pool sizes 1/4/8, magnitudes 1e-6/1/100, and BF16/FP32 scores. FP8 bytes
  and scale values are exact against the original-rotation four-warp path.
- Independent torch quantization matches uniform zero/positive/negative,
  alternating-sign and impulse patterns. Tests retain both BF16 boundaries.
- Strided pool, slot, channel and bias views match without modifying inputs.
  CUDA graph replay reads changed keys, scores and position biases correctly.
- Actual baseline-module checks also compare every benchmark shape exactly.
- Real L3: contexts 63/256/2,048 × prefill/verify/rollback = nine exact selected
  slot/count, paged KV byte and state-ring comparisons. Seven real L3 indexer
  tensors total 15,207,424 bytes; no routed expert weights or compatibility
  checks are bypassed.
- srv1 NVIDIA GB10 SM121, driver 580.159.03, torch 2.13.0+cu130, CUDA 13.0,
  Triton 3.7.1; native image `st-engine:9391` ID
  `sha256:0d781f0a8f77d4735d0d09d57b081c9489b3a46131dc72773fd40fcbf267b446`.
  No vLLM is installed or imported in the test container.
- Two CPUs/two OpenMP threads; focused tests 4 GiB, engine tests 6 GiB,
  final benchmark 8 GiB memory caps. Sources are mounted read-only at `/repo`
  with `PYTHONPATH=/repo`; `/cache` and `/evidence` are task-owned.
- The existing `glm53-worker` stayed running and memory-resident. GPU snapshots
  before the run and in `environment.log` show 0% utilization; this is not a
  claim of exclusive ownership of the machine. Only this task's containers
  were started and removed. No other service was changed.

## Reproduce

In the native runtime with the source, cache and evidence mounts above:

```bash
git show d44e3825:engine/kernels/kpool.py > baseline-kpool.py
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
python3 probes/engine_warp_pooling.py \
  --baseline-kpool /repo/baseline-kpool.py \
  --checkpoint /checkpoint \
  --rank-file /indexer-only.safetensors \
  --output /evidence/final.json
```

Config SHA-256:
`29c9f4171196910e99b9c069d6b76c56e3cdcd0f436dc1bacbc9513c9a7529ac`.
Existing srv1 inputs:
`/home/choiceoh/st-engine-f4d7-20260911/config.json` and
`/home/choiceoh/st-engine-f4d7-indexer/indexer-only.safetensors`.
Original tensor hashes are in `../st_engine_indexer_20260911/weights-sha256.json`.
