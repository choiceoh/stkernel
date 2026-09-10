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
- `cpu-core.log`: the separated core gate passes 71,655 checks and 74
  megakernel regressions. `cpu-fleet-retry.log` passes the original 385 fleet
  tests in two isolated shards. No test or fleet source was changed. A
  controlled reproduction of an existing process-exit race is retained in
  `existing-fleet-exit-race.json`; the first failure's truncated output does
  not establish that this was its exact cause.
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
bounded. Its compile receipt is a code-generation result only.

## Offline SM121 compilation: PASS

Normal CPU fleet session `dsv41indexercpu0910v1`, on srv3, compiled the frozen
source `582eb1aecadd3b6454cb9e215ff81c68a3fb3e34`. Its terminal receipt is
`aot-cpu1/fleet-ack.json`; the device-free execution, source hashes before/after
and original compiler log are retained alongside it. The local source hashes
and fetched artifact hashes were checked against these receipts.

- Already-present image:
  `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
- Torch `2.13.0+cu130`, Triton `3.7.1`, target `cuda/SM121`, four warps,
  `enable_fp_fusion=False`; runtime-width `i32` argument in both variants.
- H8/D128 and H32/D128 compile; static shared memory is 12,288/16,384 bytes.
- `aot-cpu1/ptx-audit.json` checks the fetched PTX against the compiler hashes
  and identifies the runtime-width load/bounds tests, BF16-input MMA, and
  separate dot/product/head-sum BF16 conversions. This checks generated code,
  not numerical equivalence on a device.
- No device nodes; CUDA remains uninitialized. The bounded compiler container
  completes in 2.51 seconds with 16.35 GiB host memory available at admission.
  This elapsed time is **not** a kernel timing.
- The full remote evidence, including cubins/PTX and the Triton cache, remains
  at `/home/choiceoh/dsv41-indexer-cpu1-evidence` on srv3. `SHA256SUMS` describes
  that complete directory; the committed evidence is its textual subset.

Later evidence/documentation commits do not change the compiled kernel or
adapter sources. No serving defaults, GPU reservations or admission policy
were changed.

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
