# C1 MoE initialization and synchronization cleanup

Implementation `eae5c36d`, based on `0bde26db` (merged #923). This is a
separate follow-up PR, as requested. It removes unused setup and redundant
synchronization within the existing C1 SF6 kernel; it adds no arithmetic,
DMA, temporary allocation or serving dependency. The default is ON.

## What was redundant

1. SF6 forbids the legacy A ring, but the kernel still allocated and
   initialized its four unused full/empty barriers and built both A states.
   The C1 cleanup omits that storage, initialization and state creation.
2. Each of three pipeline constructors ran its own initialization fence and
   CTA sync, followed by another explicit CTA sync. FC1 and FC2 now use the
   installed API's `defer_sync=True`, followed by one initialization fence
   and the existing CTA sync. The two used rings still initialize every
   full/empty barrier with unchanged groups and transaction counts.
3. C1 has one FC1 half. After its quantization loop, the old first barrier
   was followed by the final fence/publication barrier before any FC2 read
   or next-item sC1 reuse. Normal execution retains only that final pair.
   **Stamped execution retains the earlier barrier too**, so the FC1-end
   timestamp cannot precede another warp's final quantization write.

The constructor fails if the cleanup geometry is no longer one FC1 half
and a single-CTA cluster. SF6 already rejects the legacy A-ring recipe.
The internal `sync_cleanup=False` control preserves the old synchronization
and has a distinct cache key. `sync` appears in the enabled native name.
Eligibility is M1–8 SF6 decode; other recipes/rows retain their behavior.
There is no public serving knob. K=7, FP32 KDA, precision/rounding, prefill
and the compact staging changes from #923 remain intact.

## Work and resource counts

| Default M8 handle | Current-source control | Cleanup |
|---|---:|---:|
| Pipeline initialization CTA sync calls | 4 | 1 |
| Initialized full/empty barriers | 12 | 8 |
| Post-quantization rendezvous per work item, unstamped | 2 | 1 |
| Post-quantization rendezvous per work item, stamped | 2 | 2 |
| Native static instructions excluding NOP | 3474 | 3456 |
| Native static barrier instructions | 28 | 24 |
| Registers | 121 | 121 |
| Staged shared allocation, including alignment | 99328 B | 99328 B |
| Static shared / stack / local allocation | 1024 / 0 / 0 B | 1024 / 0 / 0 B |

The omitted A metadata is 32 bytes before alignment; total shared allocation
stays the same. Arithmetic/MMA and DMA opcode counts do not change. These
are source/native work counts, **not a measured latency improvement**.

## Verification and reproduction

`cpu-tests.log` records 25 passing focused tests. In addition to #923's
exact-byte, stage-lifetime and changed-item gates:

- Execute the actual pipeline initialization block against the inspected
  installed API contract. Both used rings initialize before the common
  fence/sync; zero-length A storage cannot be accessed. Arrival groups and
  transaction budgets are unchanged.
- Execute the actual post-quantization tail with 128 simulated threads,
  uneven writes, 20 randomized schedules per configuration, and stamps
  both ON/OFF. FC2 cannot read until every thread has published its writes.
  A timestamp also requires every peer's final write to have completed.
- Check default/control cache separation, idempotent normalization and
  recipe/row scope.

`native-compile.json` records nine actual CuTe/PTXAS/TVM-FFI handles: M1/M7/M8
cleanup, M8 control, M8 with compact staging OFF, M8 with a one-stage FC1
ring, stamped M8, and unchanged-path M16/M32. The one-stage case only checks
compatibility; the default remains two stages. Source, binary and SASS
hashes and exact resource/opcode records are retained. The PR's latest-head
engine CPU CI and onepass checks supply the complete-suite result.

The installed `cutlass.pipeline.PipelineTmaAsync.create` source was inspected:
`defer_sync=True` skips only its initialization fence/sync, after initializing
both full and empty sync objects. The explicit final fence/sync replaces
those skipped operations. Native compilation uses that actual API.

```sh
python3 -m unittest -v tests.test_engine_moe_sync_cleanup tests.test_engine_moe_compact_staging tests.test_engine_moe_fc1_reuse tests.test_engine_moe_sf6_staging tests.test_engine_moe_activation_store tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --sync-cleanup --sass --output /out/native-compile.json
# Prepared only; not submitted or run:
python3 probes/engine_kernel_check.py --lanes moe_sync_cleanup --ranks EXACT_CONSUMER_RANK_DIRECTORY
```

The prepared real-weight probe compares explicit cleanup OFF/ON at
M1/6/7/8 with duplicate routes/multiple M16 items, changed inputs, zero
weights, graph repeat spread and warm/evicted B/A/A/B timing. Its numerical
threshold remains 0.001. Previous FC1-reuse and compact-staging probes pin
this new axis OFF in both arms to preserve their comparisons.

CPU/native work used the existing srv2 image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`,
runc, CUDA hidden, no network, two CPUs/four GiB and one build worker.
Existing CUDA 13.0 nvdisasm was mounted read-only. No new image or baseline
engine was built, and no GPU queue, GPU context, model boot or service
restart was used. GPU numerical/replay, quality, acceptance and step/s are
still unmeasured.

## Oracle

The upgraded #875 tool at `e2bfbb9afdcc6e8fe1e5fe47ddddd23180b278b3` compares
`0bde26db` with `eae5c36d` at C1 2K/32K/128K using the retained checkpoint
configuration. `--acc 0` is a timing-only assumption, not acceptance.
`oracle-c1.json` leaves total decode delta null because no matched MoE
coefficient exists for this change. No coefficient is inferred from barrier
counts. `paired-profile-c1.json` preserves the unmeasured source-bound template.
