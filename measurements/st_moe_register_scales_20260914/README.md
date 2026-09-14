# C1 MoE: reconstruct SF6 scale operands directly in registers

This candidate replaces the ordinary SF6 expand/store/barrier/reload sequence in
both FC1 and FC2. Four adjacent lanes share the same scale-copy view; each lane
reconstructs one exact four-byte word and broadcasts it within that quad. The
existing pipeline wait/release owns all packed reads. Weight bytes, MMA order,
activation rounding, scatter, FP32 KDA and K=7 are unchanged.

The route is **ON by default** for the existing compact C1 SF6 geometry: 1–8 input
rows, an even FC1 ring, input reuse and separate packed stages. Other geometries
retain their ordinary path. Internal `sf6_registers=False` is the explicit
same-build control; the kernel cache and disk names distinguish it. Prior
FC1-reuse and compact-staging probes pin this new axis OFF in both arms.
Base: `4dbc0713f159f8b2ab316776e1b34c7f1f6944df` (#938).

## Mechanism and its limits

For one H4096 expert/M16/128-intermediate work item, FC1 has 32 scale stages
(16 K256 tiles × gate/up) and FC2 has 16. Direct reconstruction removes **48
cross-warp scale-publication barriers and 98,304 bytes of expanded shared
stores per item**. These are source-level dynamic work counts, not 48 barriers
per whole engine step. Packed global transfers, pipeline barriers and matrix
multiplications remain. The unused expanded backing is retained, so allocation
and occupancy do not change.

The warp-to-scale mapping is not linear across all 128 threads. The compiler
checks every ordinary copy byte, K64 block and quad against the actual dense
CuTe layouts before accepting the route. The operand probe also uses the real
register fragments and copies, with an independent packed-byte encoder. The
first simple per-thread reconstruction was revised because it duplicated work
and increased register use; only the cooperative version is the candidate.

Native M8 comparison, same source and all other axes fixed (`compile.json`):

| Metric | Ordinary expanded scales | Direct register scales |
|---|---:|---:|
| Registers/thread | 121 | 103 |
| Non-NOP static instructions | 3,474 | 3,724 |
| Static BAR.SYNC instructions | 28 | 22 |
| Static SHFL.IDX instructions | 0 | 96 |
| Staged shared bytes | 99,328 | 99,328 |
| Stack/local bytes | 0 / 0 | 0 / 0 |

Instruction count rises **7.2%** while cross-warp waits and shared stores fall.
Register reduction alone does not increase occupancy at this shared-memory
size. Whether the tradeoff improves the whole kernel is **unknown**. No
component timing, consumer step/s, output tok/s, acceptance or quality result
was collected. This does **not** establish progress from 20 to 24 step/s; that
target still requires reducing 50 ms/step to 41.67 ms/step, about 8.33 ms.

## Completed validation

- 27 CPU tests passed (`cpu-tests.log`): every base byte, all SF6 codes,
  wraparound, every warp/quad, changed packed-ring contents, canaries,
  production FC1 operands/release order, default scope and cache isolation.
- Seven full SM121 native handles compiled: M1/M7/M8 candidate, M8 control,
  stamped M8 candidate, and unchanged-path M16/M32. All have zero stack/local
  bytes. `compile.json` retains actual copy mappings, binary/SASS hashes and
  resources. `native-summary.json` is the M8 comparison.
- Four native operand helpers compiled: original/direct × FC1/FC2.
  The actual mapping's 24,576 operand words matched the independent CPU encoder
  at six boundary bases across all lanes/K blocks (`operands.json`). GPU mode
  is prepared to check every base, untouched shared bytes and 64 graph replays.
- The existing real-weight MoE probe now has `moe_register_scales` for
  M1/6/7/8, duplicate routes spanning multiple M16 items, zero weights, changed
  inputs, repeat spread, and warm/evicted timing. It has not run.

All remote validation used the existing image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`
on srv2 with runc, CUDA hidden, no network, two CPUs/four GiB. Existing CUDA 13
nvdisasm was mounted read-only. No GPU context, queue submission, model boot,
image/engine build or service restart occurred.

```sh
python3 -m unittest -v tests.test_engine_moe_register_scales tests.test_engine_moe_compact_staging tests.test_engine_moe_fc1_reuse tests.test_engine_moe_sf6_staging tests.test_engine_moe_activation_store tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --register-scales --sass --output /out/compile.json
python3 probes/engine_moe_register_scales.py --cpu --output /out/operands.json
# Prepared only; not submitted or run:
python3 probes/engine_moe_register_scales.py --gpu --output /out/gpu-operands.json
python3 probes/engine_kernel_check.py --lanes moe_register_scales --ranks EXACT_CONSUMER_RANK_DIRECTORY
```
