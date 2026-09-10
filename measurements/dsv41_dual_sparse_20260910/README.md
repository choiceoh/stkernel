# V4.1 dual-pool sparse attention

The official reference's 38 compressed-attention layers concatenate the
sliding-window KV and full compressed prefix before selecting at most 640
positions. This change reads the same ordered positions from the original two
buffers, retaining one 64-slot online-softmax sequence and one sink denominator.
The small index concatenation remains. No KV cache or indexer is replaced.

This is an opt-in official-reference adapter that composes with #521/#522.
It is not a complete vLLM model or a change to serving defaults. GPU numerics,
model equivalence, tok/s, step/s and TTFT remain unmeasured.

## Reference and static copy budget

Official `deepseek-ai/DeepSeek-V4.1-Flash` revision:
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`.

- `inference/model.py` SHA256:
  `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
- `inference/kernel.py` SHA256:
  `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.

`copy-budget.json` derives every layer's old concatenation size from the
released config: D512 BF16 KV, 18 ratio-two layers, 20 ratio-one layers.
At B1/128K, original decode concatenation outputs sum to 3.62964 GiB per rank
per step; at 1M they sum to 29 GiB plus 4.75 MiB. Reading and writing gives
twice those logical byte counts. Prefill uses the whole input as its window,
so its 128K concatenation outputs total 8.375 GiB per rank.

These are static allocation/copy counts, not measured device traffic,
allocator memory savings or latency. Existing KV buffers and other model
temporaries remain. The established vLLM sparse-attention backend already has
separate pools; this change targets the official-reference path specifically.

## Reproduction

Use a CPU Torch environment and the pinned reference source files. No weights
are needed for the arithmetic and integration fixtures.

```sh
python -m unittest discover -s tests -p 'test_dsv41_*.py' -v
python probes/dsv41_dual_sparse_diff.py \
  --reference-dir /path/to/pinned/inference --output /tmp/fresh-dual-result.json
bash launchers/compose-overlays.sh dsv41
python tests/test_logic.py --component core
```

The normal-fleet `cpu_compile_runner.py` uses a clean, frozen revision and an
already-present pinned image to compile H8/H16/H32/H64 on SM121. It binds no
GPU devices or model, has no network, caps CPU/memory and asserts CUDA stays
uninitialized. Source hashes before/after and compiler artifacts are retained.
Offline code generation cannot establish real-device equivalence or speed.

## CPU results

- `cpu-unit.log`: 56 tests passed, zero skips, including 20 new dual-pool tests.
- `cpu-oracle.json`: 40 independent comparison rows passed: 16 arithmetic
  fixtures under each of FP32 and BF16 global defaults, four original window
  fixtures, and four actual source-extracted Attention sequences. The first
  three sequences use all 38 dual-pool layers alongside the actual packed
  indexer adapter; the last restores the original attention while preserving
  cache history and the packed adapter. Finite BF16 values and NaN positions
  match the CPU oracle; NaN payload identity is not claimed.
- The model fixtures use synthetic projections/compressors and documented
  quantizer stand-ins. These are call-path checks, not model-weight validation.
- `cpu-core.log`: 71,731 core checks and 74 megakernel regressions passed.
- Profile composition passed: 22 overlays from seven modules, including ten
  files in `dsv41_model`.

CPU results do not establish TileLang/Triton GPU GEMM equivalence, full-model
quality, or speed. The standalone Torch backend is a bounded correctness
implementation; the optimized Triton path still requires its device gate.

The next GPU gate must compare the original TileLang path with direct reads,
including selected KV values, attention output, changed inputs/cache updates,
full-model quality and matched consumer speed. Current fleet policy requires
canonical consumer campaigns, and V4.1 serving integration remains incomplete.
