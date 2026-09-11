# Fused GLM indexer lanes — 2026-09-11

Integrated with main at `a5b68c84` (PR #541, standalone native kernels).
Both measured paths use this native runtime and the same incremental block
tables; the baseline substitutes the two original PyTorch indexer helpers.
This change connects the native fused kernels to the explicit lane table.

## Change

The served indexer previously called the PyTorch reference helpers directly
for Hadamard-128 rotation/FP8 quantization and pool expansion. It now calls
`Lanes.indexer_quant` and `Lanes.expand_pools`. The served table binds
`engine.kernels.kpool.fwht128_quant_fp8` and
`engine.kernels.kpool.expand_pools_and_append_tail`; the reference table binds
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
The seven unchanged replicated tensors are extracted byte-for-byte into a
separate indexer-only safetensors file with 256-byte-aligned tensor offsets.
It contains no expert tensors. The probe uses the profile's guarded
`rank_loader`; the new up|gate expert layout check remains in force.
Both paths run the same `net._indexer` and served scoring/pooling kernels; the
baseline replaces only the two new lanes with the previous reference helpers.
There is no model collective inside this replicated indexer.

srv1 GB10, torch `2.13.0+cu130`. Median wall times over five alternating rounds
of 50 calls per implementation, after ten warmup calls:

| Complete L3 indexer call | Previous helpers | Fused helpers | Reduction |
|---|---:|---:|---:|
| Decode/verify, 6 tokens at context 2,048 | 0.722 ms | 0.482 ms | 33.3% |
| Prefill, 256 tokens | 1.612 ms | 0.846 ms | 47.5% |

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
| Hadamard + FP8 quantization | 1 | 163.93 | 21.23 | 87.0% |
| Hadamard + FP8 quantization | 6 | 163.83 | 21.25 | 87.0% |
| Hadamard + FP8 quantization | 24 | 167.33 | 21.23 | 87.3% |
| Hadamard + FP8 quantization | 512 | 1,339.50 | 37.39 | 97.2% |
| Pool expansion + tail | 1 | 95.47 | 18.50 | 80.6% |
| Pool expansion + tail | 6 | 96.16 | 18.45 | 80.8% |
| Pool expansion + tail | 24 | 97.07 | 18.64 | 80.8% |
| Pool expansion + tail | 512 | 301.62 | 26.59 | 91.2% |

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

- **105 CUDA engine tests passed, zero skips** (`engine-tests.log`). The two new
  regressions check the lane input contracts, actual dispatch from `_indexer`,
  and propagation of failures without a reference fallback.
- The runtime has no installed `vllm` package and loads no `vllm` module.
  Native import paths and the up|gate weight guard from PR #541 are preserved.
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
O_DIRECT and allocator measurements at `c28b0476`. The final 105-test suite includes
those cache-publication regressions; this lane-only follow-up does not change
the allocator or I/O implementation.

## Evidence and reproduction

- `source-sha256.json`: 142 engine/test/probe Python files, verified against the
  submitted source. `weights-sha256.json`: the exact seven checkpoint tensors
  read by the probe, including shape, dtype and raw-byte hash.
- `lanes.json`: contracts, real-weight checks, integrated/helper measurements,
  operator traces, and the imported served module's path/hash.
- `lanes.log`, `engine-tests.log`, `environment.log`: raw execution evidence
  with only trailing whitespace removed.

The isolated checkout is `/home/choiceoh/st-engine-f4d7-indexer` on srv1. The
standalone image is `st-engine:9391`, immutable ID
`sha256:0d781f0a8f77d4735d0d09d57b081c9489b3a46131dc72773fd40fcbf267b446`.
`PYTHONPATH=/repo` selects the mounted current engine over the image's baked
copy. The imported native kpool module is `/repo/engine/kernels/kpool.py`, SHA256
`49537e2ea75ffe5a63c76f22769d3d9e4b0ee0b249b51c50665a227d310c4a34`.
No overlays are mounted. These native-runtime measurements replace the earlier
seed-image measurements; no cross-image latency comparison is claimed.

The probe used an 8 GiB container, two CPUs and `OMP_NUM_THREADS=2`, with its
own native JIT cache. It read the indexer-only rank file and model config
through read-only mounts. The CUDA suite ran first in a separate 4 GiB
container. No running serving container or production checkpoint was modified.

After placing this checkout, `config.json` and an aligned, compatible rank
file on the host, run the probe with bounded resources and writable evidence:

```bash
repo=/absolute/path/to/stkernel
inputs=/absolute/path/to/probe-inputs  # config.json and indexer-only.safetensors
cache=/absolute/path/to/isolated-native-cache
evidence=/absolute/path/to/evidence
mkdir -p "$cache" "$evidence"
docker run --rm --gpus all --memory=8g --memory-swap=8g --cpus=2 \
  -e OMP_NUM_THREADS=2 -e PYTHONPATH=/repo -e PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$repo,dst=/repo,readonly" \
  --mount "type=bind,src=$inputs,dst=/inputs,readonly" \
  --mount "type=bind,src=$cache,dst=/cache" \
  --mount "type=bind,src=$evidence,dst=/evidence" \
  --entrypoint python3 st-engine:9391 /repo/probes/engine_indexer_lanes.py \
  --checkpoint /inputs --rank-file /inputs/indexer-only.safetensors \
  --output /evidence/lanes.json
```

For the suite, use the same source/runtime/cache mounts, `-w /repo`, a 4 GiB
limit, and replace the Python arguments with
`-m unittest discover -s tests -p 'test_engine_*.py' -v`.
`probes/run_engine_probe.sh` is also available when inputs/output are under its
standard model/cache mounts.
