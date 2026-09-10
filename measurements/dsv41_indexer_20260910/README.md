# V4.1 CED compact indexer validation

This is a component implementation with an opt-in official-reference adapter.
It is not a running vLLM model or a serving performance result. Model tok/s,
step/s, TTFT, GPU numerics and actual distributed reduction are unmeasured.

Reference release: `deepseek-ai/DeepSeek-V4.1-Flash`, revision
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`, `inference/model.py` SHA256
`4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.

## CPU evidence

- `cpu-oracle.json`: 18 exact BF16 score/selection comparisons plus 12 actual
  adapter call sequences against source-extracted official reference code,
  all passing. Each sequence traverses layer 20 and consumers 24/28/32/36.
  World sizes 1 and 4 use a deterministic CPU sum, not NCCL. Q/K and projection
  outputs are synthetic; the quantizer is not tested. CUDA stays uninitialized.
- `cpu-unit.log`: 17 tests pass, no skips. Includes strided cache storage,
  malformed input/collective rejection, runtime width specialization, opt-in
  installation/restoration, dense fallback and TP state mismatch rejection.
- `cpu-logic-first.log`: the first complete repository regression run reached
  the fleet suite but had one failure in a pre-existing asynchronous launch
  fixture. It is retained as a failure, not counted as a complete gate pass.
- `readiness.json`: read-only source/runtime inventory. At observation, only
  46/48 checkpoint shards were present; Engram shards 47/48 were absent. The
  profile still lacks a complete model, loader and paged attention integration.

Run with a CPU PyTorch environment:

```sh
python -m unittest discover -s tests -p 'test_dsv41_*.py' -v
python probes/dsv41_indexer_diff.py \
  --reference /path/to/pinned/inference/model.py --output /tmp/fresh-result.json
bash launchers/compose-overlays.sh dsv41
python tests/test_logic.py
```

`cpu_compile_runner.py` is the normal-fleet CPU payload for offline SM121
compilation of the exact Triton score kernel. It requires a clean committed
source, the pinned image already on the node and 12 GiB available host memory.
The container has no GPU devices, network or model mounts; CPU/memory are
bounded. Its eventual compile receipt is a code-generation result only.

## What the optimization changes

Only B1/Q1 decode beyond 16,384 positions is admitted. Source layer 20 retains
dense scoring; four consumers score/all-reduce at most 16,384 positions each.
Full-width final top-k remains. The theoretical score/collective element count
is 1/8 at 128K and 1/64 at 1M for those four consumers. These are neither
latency estimates nor measured network bandwidth reductions.

GPU activation remains explicit. A matched canonical consumer campaign must
establish selected IDs/attention results, TP correctness, quality and actual
speed before any default adoption. The current fleet policy does not admit
standalone component GPU probes, and this change does not weaken that policy.
