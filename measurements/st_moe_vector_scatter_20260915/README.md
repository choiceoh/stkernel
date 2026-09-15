# C1 MoE vector scatter

Implementation and native compilation record, 2026-09-15. Base: `937666be`.
The default SF6 staged-output path now combines four adjacent FP32 scatter
reductions into two. GPU latency, replay and serving results are pending.

## Implementation

Each staged-output lane owns eight consecutive BF16 values. The kernel still
multiplies each by its route weight, rounds each contribution with
`cvt.rn.satfinite.bf16x2.f32`, and widens it for the existing FP32 accumulator.
Two `red.relaxed.gpu.global.add.v4.f32` instructions replace four `v2` reductions.
Per-element atomicity, FTZ accumulation, the final BF16 boundary and all
publication barriers are retained. The accumulator and all scratch buffers
keep their original sizes. No launches are added.

The output allocation is 16-byte aligned; the 4096-column row stride,
256-column tile stride and eight-column lane spans keep every vector aligned.
The kernel rejects incompatible row alignment before compilation.

`scatter_vec4` is enabled by default for SF6 reform tiles with staged FP32
output (production C1, M1..8). C2's default direct-register scatter owns
noncontiguous pairs and keeps its `v2` instructions. A C2 comparison explicitly
disabling direct scatter also uses the new staged-output path. Private
route-owned output and other tile/precision paths retain their existing code.
`scatter_vec4=False` selects a distinct control handle; it is a private
comparison setting, not another serving environment variable.

## Native result

Actual CuTe lowering, PTXAS, TVM-FFI and disassembly on srv2, using the pinned
ARM64 image `st-engine:bracket-3acae0170bb2`
(`sha256:1eccea22a434b863c5f146e8616f18971f92e64f516cfbb0b7dc091ea402ef9a`).
Container: runc, no network, two CPUs, 4 GiB, CUDA devices hidden. No GPU
context was initialized. The exact source hashes are in `native-compile.json`.

| Handle | Vector scatter | Registers | CTA shared bytes | Static instructions | Global FP32 RED instructions |
|---|---|---:|---:|---:|---:|
| M8 control | off | 113 | 91,136 | 3,547 | 16 x F32x2 |
| M8 default | on | 113 | 91,136 | 3,539 | 8 x F32x4 |
| M7 default | on | 113 | 91,136 | 3,539 | 8 x F32x4 |
| M16 production direct scatter | off | 96 | 100,352 | 4,013 | 64 x F32x2 |
| M16 staged-output comparison | on | 113 | 91,136 | 3,539 | 8 x F32x4 |

All five handles compiled in 40.9 seconds. Stack/local usage is zero. The
generated code retains FTZ FP32 reductions. Apart from the RED replacement,
the M8 opcode counts differ by two more PRMTs and two fewer NOPs. These are
instruction counts, not a latency or tok/s gain.

## Checks and reproduction

Forty focused CPU tests passed. The new test executes the production staged
epilogue and compares every output address and weighted contribution against
an independent enumeration, including partial tiles, repeated token
destinations, zero/negative weights and the first/last output tiles. It checks
16-byte vector alignment, halved RED calls, default dispatch bounds, repeated
normalization and separate control cache keys.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest -q \
  tests.test_engine_moe_vector_scatter tests.test_engine_moe_sync_cleanup \
  tests.test_engine_moe_batch_reform tests.test_engine_moe_sf6_staging \
  tests.test_engine_moe_scatter_config tests.test_engine_moe_register_scales \
  tests.test_engine_moe_compact_staging tests.test_engine_moe_fc1_reuse \
  tests.test_engine_moe_activation_store
# Existing ST image, no GPU:
CUDA_VISIBLE_DEVICES= PYTHONPATH=/repo python3 probes/engine_moe_sf6_compile.py \
  --scatter-vec4 --sass --output /out/native-compile.json
# GPU comparison must be submitted through bench/fleet.sh run --gpu:
python3 probes/engine_kernel_check.py --lanes moe_pair_vec4 \
  --ranks EXACT_CONSUMER_RANK_FILE --output /cache/moe-vec4.json
```

The existing same-pack comparison changes only `scatter_vec4`, captures the
served FFN consumer at M8/M16, exercises changed activations/routes and real
L3 router fixtures, and runs B/A/A/B component timing after numerical checks.
It does not establish full-model TP4 acceptance or serving speed.
