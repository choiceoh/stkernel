# V4.1 packed index-key cache

This follow-up to #521 stores the original FP4 quantizer's packed bytes instead
of BF16 fake-quantized keys. All eight reference indexers use direct packed
scoring; long B1/Q1 decode also retains compact candidate scoring. It is an
explicit reference-model experiment, not a bootable vLLM model or serving
default. GPU numerics, NCCL, model tok/s, step/s and TTFT are unmeasured.

Reference: `deepseek-ai/DeepSeek-V4.1-Flash` at
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`:

- `inference/model.py` SHA256:
  `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
- `inference/kernel.py` SHA256:
  `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.

## Work changed

D128 keys use 64 E2M1 payload bytes and four E8M0 scales, versus 256 BF16 bytes:
73.4375% fewer allocated key bytes. Sources 2, 8 and 14 store half-resolution
positions; source 20 stores full resolution. At 1,048,576 context positions,
the calculated per-rank total is 640 to 170 MiB for B1, or 2.5 GiB to 680 MiB
for B4. These are allocation and operand byte counts, not measured GPU memory
reservation, traffic, bandwidth or speedup.

The Triton dense path reuses a decoded key tile across four queries. Compact
mode loads only candidate IDs. Neither path reconstructs a full BF16 key
prefix. Width, output columns, query count and strides are unspecialized
runtime arguments; context growth must not create a new kernel each token.
The reference's BF16 arithmetic boundaries and full-width final top-k remain.
Dense score tensors and their collective are still full width.

Installation releases the displaced BF16 tensors rather than hiding them in a
rollback handle. It discards cache history and requires a fresh prefill.
`restore(reset=True)` first allocates all replacement buffers, then removes
packed state and requires another fresh prefill. Allocation failure leaves the
active packed adapter intact. Per-step tensor-content checks do not add a host
synchronization. Nonfinite producer behavior remains outside CPU validation.

## CPU evidence: PASS

- `cpu-oracle.json`: 4,096 nibble/scale combinations, 256 distinct packed
  bytes, all 32,640 nonnegative finite BF16 scale inputs and 26 original-wrapper
  boundary rows pass. Signed zero, subnormals, finite-input overflow and the
  decoder's canonical NaN scale policy are included. The original Python
  quantizer wrapper executes with an independent CPU arithmetic replacement
  for TileLang, so its actual GPU byte layout and rounding remain unverified.
- Ten dense/compact score and full-width top-k cases match exactly. Fourteen
  source-extracted adapter sequences cover all eight layers, world sizes 1/4,
  short prefill, odd/even shared-slot publication and changing long-context
  candidates. Distributed sums are simulated on CPU, not NCCL. Projections
  and RoPE inputs are synthetic. The long 32,776-token prefix is seeded into
  both implementations; a full long prefill is **not** executed.
- Weak references confirm all 20 displaced BF16 owners are released across
  those rank fixtures, and all 20 packed owners are released on reset. This
  proves tensor lifetime, not a GPU allocator-reserved-memory reduction.
- `cpu-unit.log`: 36 tests pass, no skips, including the prior #521 suite.
  Coverage includes source/default-argument drift, storage and collective
  mutation, bounded scratch, runtime widths, install rollback, restore OOM,
  fresh-prefill guards and capture rejection.
- `cpu-core.log`: 71,693 repository checks and 74 megakernel regressions pass.
  Fleet code is unchanged and its suite is not rerun for this component-only
  change. `compose.log` records 19 composed overlays from seven modules.

## Reproduction

Use an environment with CPU Torch and the pinned reference source files:

```sh
python -m unittest discover -s tests -p 'test_dsv41_*.py' -v
python probes/dsv41_packed_index_diff.py \
  --reference-dir /path/to/pinned/inference --output /tmp/fresh-packed-result.json
bash launchers/compose-overlays.sh dsv41
python tests/test_logic.py --component core
```

`cpu_compile_runner.py` is a normal-fleet CPU payload for a clean, frozen
source revision. It compiles H8/H32 dense and compact variants for SM121 in an
already-present image. The bounded container has no GPU devices, network or
model mounts and must leave CUDA uninitialized. It records compiler artifacts,
source hashes before/after execution and an execution receipt. Offline
compilation does not establish numerical correctness or performance.

Before adoption, a canonical consumer campaign must validate actual TileLang
packed-byte layout and arithmetic, scores/selected IDs, distributed collectives,
attention results, quality and matched end-to-end speed. The current fleet
policy does not admit standalone component GPU probes; this change does not
alter that policy or the unfinished V4.1 serving integration.
