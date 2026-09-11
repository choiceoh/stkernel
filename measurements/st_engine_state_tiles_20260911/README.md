# Direct-state memory tiles on GB10 — 2026-09-11

**Decision: retain main's 256-element tile.** Enlarging the recurrent read/write
tiles helped the large streaming write workload modestly but consistently hurt
the single-layer, single-sequence six-token write. Neither candidate is promoted.
The two added state tests and reproducible comparison probe remain useful.

Baseline: main `879b8942` with the direct-state architecture from PR #549.
`baseline-256.py` is its exact `engine/kernels/state.py`. The two candidate
snapshots change only the recurrent read/write launch tiles to
`min(512 or 1024, next_power_of_2(width))`. Convolution tiles stay at 256.
The final production state kernel matches the baseline byte for byte.

## Measurements and interpretation

Native `st-engine:9391` on srv1 GB10, torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.
Each candidate ran in three independent processes. Each process performed five
alternating AB/BA rounds of 30 CUDA-graph samples per variant and condition.
The raw JSON retains every sample, including slow and negative comparisons.

The probe uses actual GLM arena layouts, padded physical slots, a 1 MiB FP32
KDA state per layer, one or all 34 KDA layers, and one or four active sequences.
It measures history reads and writes separately. Inputs and snapshots are
prepared outside timing. Both variants receive identical data; history values
and the entire state arena must match exactly before measuring writes.

Positive percentages mean less time than the 256-element baseline. Ranges
include all three independent processes.

| Operation / working set | 512 elements | 1,024 elements |
|---|---:|---:|
| Six-token write, 34 layers, one sequence | +1.68% to +1.98% | +3.37% to +3.84% |
| Six-token write, 34 layers, four sequences | +1.46% to +1.63% | +3.09% to +3.18% |
| Six-token write, one layer, one sequence | **−2.08% to −0.83%** | **−13.10% to −1.53%** |
| History read, 34 layers, one sequence | +0.46% to +1.23% | +0.57% to +1.88% |

The large write working set suggests a small streaming benefit, while the
smaller working set is sensitive to launch geometry and caching. No hardware
DRAM counters were collected, so this does not prove bandwidth saturation or
identify the exact cause of the regression. A production forward pass also
interleaves these transfers with model arithmetic; it is not this transfer-only
loop. The evidence is insufficient to justify changing the shared kernel or
claiming a complete-model speedup. Runtime dispatch flags were not added.

## Final validation

- `gpu-tests.log`: 170 engine tests passed on GPU, no skips.
- `graph-check.log`: 17 real-weight KDA/DSA/MoE conditions passed using isolated
  rank 0 arithmetic. Eager/graph maximum relative difference was zero; state and
  paged bytes matched exactly. This is not a four-node collective check.
- New state tests cover padded slot strides, nonzero storage offsets, small
  and partial tiles, actual KDA width, zero-context masking with NaN storage,
  row-strided inputs, ring wrap and inputs longer than the ring. Existing
  remapping and rejected-future tests remain in the suite.
- `cpu-tests.log`: 170 discovered, 107 passed, 63 skipped for missing local
  PyTorch/CUDA. The GPU run above covers those paths.
- `runtime-final.json`: native ABI and source manifest, with vLLM absent.
- `summarize.py`: checks candidate/baseline source hashes, all six run files,
  all samples, final production source hashes and validation logs.

Runs used bounded private containers (four CPUs, 8 GiB), with the launcher's
shared fleet lock held during validation. These were shared-machine microbenchmarks,
not a reserved fleet throughput run. No sanitizer result or new full-model
quality, latency or throughput result is claimed.

## Reproduction

Use a throwaway checkout so the candidate snapshot cannot affect a serving
checkout. Mount its root at `/repo`, GLM config/tokenizer metadata at `/meta`,
and a writable output directory at `/evidence` in the native image. Run from
`/repo` with `PYTHONPATH=/repo`:

```bash
cp measurements/st_engine_state_tiles_20260911/candidate-512.py engine/kernels/state.py
python3 probes/engine_state_tiles.py \
  --baseline measurements/st_engine_state_tiles_20260911/baseline-256.py \
  --ckpt-meta /meta --output /evidence/tiles512-1.json
```

Repeat in three fresh processes per candidate; use `candidate-1024.py` for the
other variant. Restore `baseline-256.py` before running the final suite and
`probes/engine_decode_graph_check.py --ranks /ranks --ckpt-meta /meta` with the
existing real layer-slice rank files. Validate archived evidence locally with:

```bash
python3 measurements/st_engine_state_tiles_20260911/summarize.py
```

The older scratch-ring experiment in `../st_engine_graph_state_20260911/`
predates PR #549 and is explicitly archived. Its large traffic reduction must
not be counted again as an improvement over this direct-state baseline.
