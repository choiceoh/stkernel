# Target KDA verification weight layout

Candidate base: `55b36d13`; branch `codex/st-target-cta-layout-0917`.
K=7 and all eight target rows are unchanged. The target's 34 KDA input weights
(6416 x 4096) use resident W4 tiles of 16 output columns rather than 128.
GLM preparation selects this layout by default, before arena relocation.

The change only permutes packed nibbles and scale bytes. Row scales, GPTQ
calibration, FP8 expansion, MMA, split boundaries and accumulation order remain
unchanged. The canonical disk pack remains 128-row; preparation makes one
permutation copy per tensor and replaces that pack. No second resident pack is
retained. The separate FP8 prefill reader is unchanged; the native 1..32-row
readers understand both explicit tensor layouts. Boot fastpath evidence includes
`resident_w4_tiles`. Source hashing invalidates the native build normally.

## Why investigate it

A C1 CTA consumes 16 output columns across K. At K4096 the same 32 x 1KiB W4
records occupy an address span of 254,976 bytes in the original layout and
32,768 bytes in the new one. Scale spans change from 31,872 to 4,096 bytes.
Payload stays 36,864 bytes per CTA: this is locality, not compression or a proven
reduction in memory transactions. The C1 bound cell specializes the new layout
at compile time. Other readers translate logical row addresses at the same
existing launch geometry; their performance also needs a regression check.

## Validation

- CPU: `tests.test_engine_dense_cta_layout`, `test_engine_producer_pack_rows`,
  `test_engine_dense_store_digest`, `test_engine_decode_fastpaths`,
  `test_engine_forward_reduce`, `test_engine_fixed_k_cost`,
  `test_engine_dense_l2_prefetch`: 29 run, 27 passed,
  2 CUDA-only tests skipped. Full real-shape byte round trip, padded tail,
  dequantized FP8 equality and cache contracts are covered.
- Owned RTX5050: same synthetic weight, baseline/candidate BF16 output equality,
  changed inputs and interleaved CUDA graph replay: 17 cells, 51 replay pairs.
  Signed-zero bits are compared through INT16 views as well as numeric equality. See `gpu.json` and `gpu.log`.
  Arena relocation includes W4, scales and a separate FP8 reader, with guards
  and byte equality. No RTX timings are used as GB10 evidence.
- Full served SM121a native compile: PASS, see `sm121-compile.json`. The C1
  specialization has 80 registers and zero stack bytes. An earlier context-copy
  variant used 192 stack bytes and was removed; see `address-specialization.json`.

No fleet queue, production boot, restart or deployment was requested or run.
GB10 component speed, 34-distinct-layer chain latency, full decode step/s,
acceptance and prefill latency are **unmeasured**. This is a default-connected
candidate, not a confirmed performance win. The first performance continuation
gate is >=0.3ms over the 34 target input projections with matched real packs,
then 32K/128K decode and acceptance; that threshold is not an expected saving.

Reproduce the correctness gate:

```sh
python -m unittest tests.test_engine_dense_cta_layout -v
python probes/engine_dense_cta_layout_check.py --gpu --output gpu.json
CUDA_VISIBLE_DEVICES= python probes/engine_decode_native_compile.py \
  --output sm121-compile.json --build-root /tmp/st-cta-compile
```
