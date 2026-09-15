# C2 batch-tile MoE synchronization cleanup

Implementation compiled and tested as `3be24709` on `d3c7cbea` (merged #925),
then rebased onto `209e9458` (#968). The five hashed MoE kernel sources in
`native-compile.json` are byte-identical after the rebase. #925 removed
unused A-ring setup and redundant synchronization from the SF6 decode
kernel. Its merge with #955 limited the cleanup to C1 rows, because the C2
batch M16 tile had not been exercised with it. This change enables the same
cleanup on that tile. It adds no arithmetic, storage, DMA or serving knob,
and `sync_cleanup=False` still selects a separate control handle.

**Scope.** The served recipe `t,r,sf6,q0` has no `batch` cell. Served
handles do not change: C1 has had the cleanup since #925, and the served C2
decode stays on the M32 tile. The change applies only where `batch` is
declared, which is the K7/C2 M16 tile under qualification (#955, #962).

## Why the C2 tile qualifies

- **Same geometry.** `decode_reform` fixes tile M16, one FC1 half and one
  CTA whether the tile holds C1 rows or the 16 C2 rows. The constructor
  still refuses the cleanup for any other geometry.
- **Every C2-only read follows the retained publication.** Scatter reuse
  loads two route rows at the start of Phase B, and those rows are written
  by other lanes at item start. A2/SFA2 copies and direct-scatter writes
  come later still. All of them follow the FC1 `fence_proxy` +
  `epilog_sync_barrier` pair that the cleanup keeps. The removed barrier
  sat before that pair, with no shared read in between.
- **Initialization is unchanged in kind.** The FC2 prefetch ring has three
  slots. `create(defer_sync=True)` initializes all six of its full/empty
  barriers, and one fence plus the existing CTA sync publishes both used
  rings.
- **Stamped runs keep the earlier barrier**, as on C1.

## CPU verification

`cpu-tests.log`: 56 focused tests pass (10 modules). The changed file,
`tests/test_engine_moe_sync_cleanup.py`, does the following:

- **Initialization.** Executes the kernel's actual initialization block for
  C1 and for the C2 prefetch geometry (FC2 3 stages). Cleanup gives
  `init fc1, init fc2, fence, sync`. The control gives three self-publishing
  inits plus the final sync. Omitted A storage is never touched.
- **Publication tail.** Executes the actual code from the end of
  quantization through the first A2 read, on 128 simulated lanes. It covers
  C1 and C2, cleanup ON and OFF, stamps ON and OFF, with 20 random schedules
  each. On C2, lanes 0–15 store their route rows and each lane's Phase B
  loads two rows owned by other lanes, with a random number of valid rows.
  The test checks:
  - No load sees an unpublished row.
  - Retained bases and weights equal the published values.
  - All-lane rendezvous: 1 with cleanup unstamped, 2 otherwise.
- **Negative control.** Drops the retained item barrier; the C2 loads then
  reach unpublished rows. This shows the simulation can fail, and that the
  cleaned-up handle depends on exactly the barrier it keeps.
- **Scope.** `batch` enables the cleanup at M1–8 and M16 only. The OFF
  control keeps a distinct cache key, and normalization is idempotent over
  rows 0–128.

Four defects injected into a scratch copy were each caught, and none was
committed:
- route loads moved before the publication
- the barrier also omitted when stamped
- C1-only scope restored
- the deferred initialization fence removed

## Native compile (actual CuTe lowering, PTXAS, TVM-FFI; no GPU)

The `native-compile.json` report holds 14 handles, all PASS. They were
compiled on srv2 from the frozen commit, using image
`st-engine-seed:a9b53fd066bb…` (CUDA 13.2.1, CuTe DSL 4.6.2, flashinfer
0.6.18.dev20260819, nvdisasm 13.3). Container settings: `runc`, no network,
2 CPUs / 4 GiB, `CUDA_VISIBLE_DEVICES=` empty. Compilation took 81 s in
total. The per-lane `sf6_register_offsets` lists were removed from the
committed report; its `raw_sha256` names the full report, which is kept
with the SASS in srv2 `~/c2sync-3be24709-out`.

| # | M | overrides | cleanup | registers | C2 scatter/reuse/prefetch | FC2 stages | A barriers | shared B | REG | static instr |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1 | default | ON | ON | -/-/- | 2 | 0 | 91136 | 113 | 3523 |
| 2 | 7 | default | ON | ON | -/-/- | 2 | 0 | 91136 | 113 | 3547 |
| 3 | 8 | default | ON | ON | -/-/- | 2 | 0 | 91136 | 113 | 3547 |
| 4 | 8 | sync_cleanup=False | off | ON | -/-/- | 2 | 4 | 91136 | 113 | 3563 |
| 5 | 8 | sf6_registers=False | ON | off | -/-/- | 2 | 0 | 99328 | 121 | 3486 |
| 6 | 8 | compact_staging=False | ON | off | -/-/- | 2 | 0 | 100352 | 121 | 3501 |
| 7 | 8 | fc1=1 | ON | off | -/-/- | 2 | 0 | 75776 | 126 | 3459 |
| 8 | 8 | stamps=True | ON | ON | -/-/- | 2 | 0 | 91136 | 111 | 3627 |
| 9 | 16 | default (served M32 path) | off | off | -/-/- | 2 | 4 | 101376 | 117 | 6164 |
| 10 | 32 | default | off | off | -/-/- | 2 | 4 | 101376 | 117 | 6180 |
| 11 | 16 | batch | ON | ON | Y/Y/Y | 3 | 0 | 100352 | 96 | 4013 |
| 12 | 16 | batch, sync_cleanup=False | off | ON | Y/Y/Y | 3 | 4 | 100352 | 96 | 4029 |
| 13 | 16 | batch, c2_direct_scatter=False | ON | ON | -/-/- | 2 | 0 | 91136 | 113 | 3547 |
| 14 | 16 | batch, stamps=True | ON | ON | Y/Y/Y | 3 | 0 | 100352 | 114 | 4077 |

Control → cleanup, static instruction deltas (the same −16 on both tiles):

| Opcode | C1 M8 (#4→#3) | C2 M16 (#12→#11) |
|---|---:|---:|
| `BAR.SYNC.DEFER_BLOCKING` | −4 | −4 |
| `SYNCS.EXCH.64` | −4 | −4 |
| `USHF.L.U32` / `UIADD3` / `UMOV` | −4 / −2 / −2 | −4 / −2 / −2 |
| `FENCE.VIEW.ASYNC.S` / `BRA` / `NOP` | −1 / −1 / +2 | 0 / 0 / 0 |

`sync-sequences.json` keeps the ordered synchronization-class instructions
from the retained SASS. On both tiles:
- **Initialization.** The three pipeline-level `BAR.SYNC.DEFER_BLOCKING`
  syncs disappear, leaving the one common sync. The A ring's four barrier
  exchanges disappear too.
- **Work loop.** The `BAR.SYNC.DEFER_BLOCKING` just before `MEMBAR.ALL.CTA`
  disappears, which removes one all-lane rendezvous per work item.
- **Stamped C2 (#14).** This handle keeps that barrier.

Shared allocation, registers, stack and local usage are unchanged between
control and cleanup. The 32 B of A metadata disappears before the 1024 B
alignment.

These are work counts, **not a latency result**. GPU numerics, replay,
component time, step/s, acceptance and quality are unmeasured. No fleet
ticket, GPU context, boot or service restart was used.

## Reproduce

```sh
python3 -m unittest -v tests.test_engine_moe_sync_cleanup tests.test_engine_moe_batch_reform \
  tests.test_engine_moe_sf6_staging tests.test_engine_moe_scatter_config tests.test_engine_moe_register_scales \
  tests.test_engine_moe_compact_staging tests.test_engine_moe_fc1_reuse tests.test_engine_moe_activation_store \
  tests.test_moe_reform_sf_pack tests.test_moe_static_sf6_direct
# CPU-only ST image, repository at /repo, owned output at /out:
CUDA_VISIBLE_DEVICES= PYTHONPATH=/repo \
  python3 probes/engine_moe_sf6_compile.py --sync-cleanup --sass --output /out/native-compile.json
# Prepared GPU pair comparison, not submitted or run:
python3 probes/engine_kernel_check.py --lanes moe_pair_sync --ranks EXACT_CONSUMER_RANK_FILE
```

`moe_pair_sync` compares cleanup OFF against ON on the `batch` recipe, at M8
and at the M16 C2 tile. It uses the rank file's own weights and scales, changed inputs and
routes, duplicate-route occupancy, zero weights and the L3 router, under
the existing 0.001 relative gate, and records B/A/A/B timing after all
numerics.
