# Rank-consistent one-shot sums

The old local-first FP32 fold gives different BF16 outputs across ranks for
the same four operands. `[2^24, -2^24, 1, 1]` gives `[2, 2, 1, 1]` instead of
one replicated result. Fold ranks 0, 1, 2, 3 on every rank; rank 0's arithmetic
is unchanged. The vector lanes and scalar tail use the same CPU/CUDA helper.

The actual helper passes four cancellation fixtures and 65,536 random BF16
vectors, with bitwise agreement across all four rank orderings. The existing
signed MAX/publication oracle passes alongside it (two tests, 1.777 seconds).

`compile-240dd876.json` records full Torch/CUDA extension compile and load from
source `240dd876746cdae931ea3dad8e9b0b11b7fd5cc8`, with no GPU initialized or
opened. The image was `st-engine:main-ff728f43`, Torch 2.13.0+cu130, CUDA 13.0.
Reproduce with `CUDA_VISIBLE_DEVICES=` and a private `ST_ONESHOT_BUILD_ROOT`:

```
python3 -m unittest tests.test_engine_oneshot_sum tests.test_engine_oneshot_integer
python3 probes/engine_oneshot_cpu_check.py --output /out/compile.json
```

GPU qualification remains pending. The production constructor now checks the
four cancellation columns at 1, 7, 24 and 64 rows, both ordinary and PDL entry
points where supported, plus seven-row graph replay at changed input scales.
Transport publication/fences and MAX packet arithmetic are unchanged.

The production sequence-5518 stall has not been reproduced with a traced first
divergence. This independently demonstrated numerical defect is not yet proof
of its cause or repair. The consumer's separate parked-row `KeyError` is fixed
in the base PR #792 and reproduced by a four-row CPU scheduler test. No engine
speedup or completed consumer-quality result is claimed here.
