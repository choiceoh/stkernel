# Drafter post-convolution residual/RMS fusion — 2026-09-17

Default path; K remains 7. Five post-attention and four interlayer MLP
boundaries now each use one Triton launch instead of a tap kernel followed
by residual/RMS. The last MLP still feeds the existing fused head producer.
The convolution, residual sum, scale product and norm weight retain their
former BF16 rounding points. No new persistent weights, tuning knobs,
communication changes or precision changes.

## Evidence and limits

The baseline is `tap_mix` followed by `add_norm` from main `13ce718f`.
The candidate uses the same inputs and dtype. The RTX 5050 is the owner's
explicitly authorized component comparison device, not a GB10 surrogate.
GPU source hashes are in `draft-post-norm.json`; SM121 lowering hashes are
in `draft-post-norm-compile.json`.

- RTX 5050 SM120, 20 SMs, driver 595.79; Torch 2.13.0+cu132,
  CUDA 13.2, Triton 3.7.1.
- Seven GPU cases pass bit-exact residual and normalized output at input
  magnitudes 0, 0.01, 1 and 10, with strided coefficient views.
- The dependent nine-boundary chain also passes bit-exact comparison.
- Two changed-input CUDA graph replays per case pass exact output checks
  and no Torch allocation-counter growth during replay.
- CPU suite: 49 tests, **41 passed / 8 skipped** (`cpu.log`).
- Offline SM121 compile: seven specializations passed, no GPU device access.

Times below are medians across all baseline/candidate positions of matched
B/A/A/B brackets. Single-boundary times use three brackets and 12 graph
repeats; nine-boundary times use two brackets and eight repeats. The existing
`cublaslt._measure` graph unroll amortizes host submissions. All raw samples
are retained, including noisy observations.

| Rows / block / group / taps | Boundary B → A (µs) | Nine boundaries B → A (µs) |
|---|---:|---:|
| 6 / 6 / 256 / 2 | 2.349 → 1.808 | 21.329 → 16.245 |
| 7 / 7 / 256 / 2 | 2.367 → 1.826 | 21.442 → 16.263 |
| 8 / 8 / 256 / 2 | 2.478 → 1.794 | 22.614 → 16.269 |
| 16 / 8 / 256 / 2 | 2.855 → 1.822 | 25.514 → 16.378 |
| 32 / 8 / 256 / 2 | 3.330 → 2.412 | 30.533 → 22.097 |
| 8 / 8 / 16 / 2 | 2.462 → 1.819 | 24.622 → 17.365 |
| 16 / 8 / 64 / 4 | 3.027 → 2.235 | 27.808 → 21.676 |

Every case uses width 4096. The nine-boundary chain models only repeated
post-convolution seams; it omits attention, GEMMs, communication and the
head. It is **not a full drafter measurement**. These savings are measured
in microseconds, not milliseconds, and do not explain the historical
2–3 ms marginal cost from increasing K. GB10/TP4 step/s, tokens/s, acceptance
and the 24 step/s goal remain **unmeasured**. No queue, deployment or restart.

## Reproduction

`run-gpu.sh` records the exact immutable image, owned-container lock and
command used on `ost-97x`. Populate its scratch checkout with this commit's
`engine/`, `probes/`, and `tests/` before running; outputs go into `out/`.
The script stops only its own named container.

Offline compile, in that image with no GPU devices and
`CUDA_VISIBLE_DEVICES=`:

```sh
python3 probes/engine_draft_post_norm_compile.py --output /out/draft-post-norm-compile.json
python3 -m unittest tests.test_engine_draft_post_norm tests.test_engine_drafter tests.test_engine_draft_conv tests.test_engine_cublaslt_producer tests.test_engine_drafter_storage tests.test_engine_draft_tuning_integration tests.test_engine_early_observe -v
```
