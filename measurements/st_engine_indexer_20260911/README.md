# Fused GLM indexer lanes — 2026-09-11

Baseline behavior: `c28b047672be0477d9bdedf39a8a5184dc91d172`, the incremental
block-table implementation in PR #540. This additional change connects two
existing served kernels to the engine's explicit lane table.

## Change

The served indexer previously called the PyTorch reference helpers directly
for Hadamard-128 rotation/FP8 quantization and pool expansion. It now calls
`Lanes.indexer_quant` and `Lanes.expand_pools`. The served table binds the image's
`fwht128_quant_fp8` and `expand_pools_and_append_tail`; the reference table binds
the unchanged Python formulas. The two new fields are required, and imports
and execution failures propagate without falling back. Both use the existing
LocalTP main-thread dispatch wrapper, like other served lanes.

The input contract is the GLM profile's contiguous BF16 `[rows,128]` queries,
int32 pool IDs and sequence lengths, and pool size 4. This is a binding change,
not a new quantization format or a rewrite of the served Triton kernels.
Weights, KV layout, indexer scoring/top-k and sampling are unchanged.

## Integrated indexer measurements

The probe loads **seven real L3 indexer tensors, 15,207,424 bytes**, from the
aligned GLM rank1 checkpoint, and uses seeded synthetic hidden/query inputs.
Both paths run the same `net._indexer` and served scoring/pooling kernels; the
baseline replaces only the two new lanes with the previous reference helpers.
There is no model collective inside this replicated indexer.

srv1 GB10, torch `2.13.0+cu130`. Median wall times over five alternating rounds
of 50 calls per implementation, after ten warmup calls:

| Complete L3 indexer call | Previous helpers | Fused helpers | Reduction |
|---|---:|---:|---:|
| Decode/verify, 6 tokens at context 2,048 | 3.490 ms | 2.301 ms | 34.1% |
| Prefill, 256 tokens | 3.610 ms | 2.408 ms | 33.3% |

These times cover the **indexer**, not a full attention block or the 45-layer
model. Inputs are synthetic activations with real weights, rather than a user
prompt traced through the model. They do not qualify full-model quality,
four-server ITL, throughput, or DFlash acceptance.

## Isolated helper measurements

The query quantizer uses 32 indexer heads of width 128 per token. Expansion
uses 512 selected pools, pool size 4 and sequence length 2,047. Five alternating
rounds of 100 calls follow 40 warmup calls per implementation.

| Helper | Tokens | Previous µs | Fused µs | Reduction |
|---|---:|---:|---:|---:|
| Hadamard + FP8 quantization | 1 | 826.87 | 74.14 | 91.0% |
| Hadamard + FP8 quantization | 6 | 828.91 | 73.92 | 91.1% |
| Hadamard + FP8 quantization | 24 | 839.70 | 73.98 | 91.2% |
| Hadamard + FP8 quantization | 512 | 1,367.73 | 77.81 | 94.3% |
| Pool expansion + tail | 1 | 519.48 | 70.38 | 86.5% |
| Pool expansion + tail | 6 | 523.17 | 70.60 | 86.5% |
| Pool expansion + tail | 24 | 520.74 | 71.38 | 86.3% |
| Pool expansion + tail | 512 | 535.38 | 71.40 | 86.7% |

CUDA traces show **35 kernels → 1** for quantization, and **19 kernels plus
two device copies → 1 kernel with no copies** for expansion. Copy events are
counted separately from kernels. The 512-token quantizer's additional peak
PyTorch CUDA allocation falls from 24.125 MiB to 2.0625 MiB. This is the isolated
helper's temporary allocation; complete-indexer peak allocations were unchanged
because other phases determine its peak.

All latency measurements finish before CUDA operator profiling starts, so
profiler initialization cannot affect later timed cases. Each sample waits on
its end event; wall times include event submission and that wait. CUDA stream
spans include host launch gaps, not just kernel execution time. Setup, loading
and JIT warmup are outside measurement. Per-round summaries and allocator
accounting are retained in `lanes.json`. Host clocks and other work were not
controlled, so absolute latency is specific to this measurement environment.

## Correctness

- **96 CUDA engine tests passed, zero skips** (`engine-tests.log`). The two new
  regressions check the lane input contracts, actual dispatch from `_indexer`,
  and propagation of failures without a reference fallback.
- FP8 bytes and FP32 scales match exactly for 21 randomized combinations:
  1/31/32/33/192/768/16,384 rows, with magnitudes 1e-6/1/100. Six additional
  zero, constant, alternating, impulse and ramp rows also match exactly.
- Expanded token IDs match exactly for 1/2/512 pool columns and twelve sequence
  lengths covering empty/short contexts, complete/incomplete pool boundaries,
  and long contexts. Invalid and future pool IDs retain `-1` padding.
- The fused helpers run through LocalTP with four logical ranks and match the
  direct-call outputs exactly.
- With real L3 weights, prefill at contexts 63/256/2,048, six-position draft
  verification with four modified draft inputs, and rollback accepting two
  positions produce **identical selected slots, valid counts, pooled KV bytes,
  scales and tail-ring bytes** across both paths. The same comparisons pass
  again after the integrated timing runs. No tolerance was relaxed.

The prior [block-table report](../st_engine_cache_20260911/README.md) records the
O_DIRECT and allocator measurements at `c28b0476`. The new 96-test suite includes
those cache-publication regressions; this lane-only follow-up does not change
the allocator or I/O implementation.

## Evidence and reproduction

- `source-sha256.json`: 82 engine/test/probe Python files, verified against the
  submitted source. `weights-sha256.json`: the exact seven checkpoint tensors
  read by the probe, including shape, dtype and raw-byte hash.
- `lanes.json`: contracts, real-weight checks, integrated/helper measurements,
  operator traces, and the imported served module's path/hash.
- `lanes.log`, `engine-tests.log`, `environment.log`: raw execution evidence
  with only trailing whitespace removed.

The isolated checkout is `/home/choiceoh/st-engine-f4d7-indexer` on srv1. The
served image is `glm53:v13-b12x-it`, immutable ID
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The probe used an 8 GiB container, two CPUs and `OMP_NUM_THREADS=2`, with its
own JIT cache. It read the existing aligned rank file and model config through
read-only mounts. The CUDA suite used a separate 4 GiB container.

```bash
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
# Inside the served image with this checkout's composed overlays and inputs:
python3 probes/engine_indexer_lanes.py \
  --checkpoint /config --rank-file /ranks/rank1of4.safetensors \
  --output /evidence/lanes.json
```

On the host, `probes/run_mk_probe.sh` composes/mounts the matching overlays.
`PROBE_CACHE=1`, an isolated `CACHE_HOST`, `MAX_JOBS=2`, and
`MK_PROBE_DOCKER_ARGS` supply the bounded container and read-only checkpoint
mounts. No running serving container or production checkpoint was modified.
