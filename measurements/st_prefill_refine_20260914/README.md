# Refine the enabled prefill lanes

Dense-prefix attention already knows that every visible pool is selected, but
the ordinary indexer still projected, quantized and scored those queries. This
change shares the coverage boundary between attention and indexer dispatch.
Covered queries now construct their exact pool IDs directly. Query projection,
head gates, quantization, scoring and top-k run only for the remaining queries.
Key/gate projections and all pool/tail cache writes still process every row.

The existing `prefill_dense_prefix=1` and `prefill_absorb_tiles=1` defaults remain
on. The independent query-sharding default stays off. There is no new switch.
Decode, capture, probes, multiple segments, steps outside 128–32,768 rows and
covered prefixes below 128 retain their previous dispatch. Runtime readiness
now requires covered-query bypass on every DSA layer when dense prefix is on.

## Buffer and addressing changes

- Share the bounded query-input padding helper with query sharding. A short
  residual range is padded to 64 only for projection, preserving the existing
  FP8 prefill lane; padding never reaches cache writes or selection.
- Fill covered IDs and the selected suffix into disjoint parts of one output.
  `_select_pools(out=...)` avoids a second full suffix result for multi-pass
  selection. Single-pass top-k still has its bounded result temporary.
- Query shards allocate the final collective packet once. Only covered rows
  and ghost padding are initialized; scored rows are copied directly, removing
  the overwritten ID fill/mask and the ragged packet concatenation.
- MLA query/output relative offsets use int32 within their proven bounds;
  paged KV offsets retain int64. The largest absorb element offset is
  `32768*16*512-1 = 268435455`, safely below the int32 limit. CPU kernel-body
  checks now use actual int32 aranges. This change did not lower compiler
  register/shared-memory use and is not counted as a measured speedup.

## Validation

The existing ST image was used with no GPU devices in read-only, network-disabled
runc containers: CPU=2, memory/swap=4 GiB, pids=256,
`NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=''`.
Image: `sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`.

At `1d452e11`, **47 existing tests passed without skips**. Four of the five new
tests also passed; the fifth test fixture omitted head-gate input for the
multi-pass mock (`cpu-tests-r1.log`, 52 tests, 29.720 s). The fixture was corrected
without changing production code. At `066b74f6`, **all 5 new tests passed without
skips**, 8.218 s (`cpu-new-tests.log`).

```sh
python3 -m unittest -v \
  tests.test_engine_prefill_covered_queries tests.test_engine_prefill_absorb_tiles \
  tests.test_prefill_dense_prefix tests.test_prefill_indexer_shards \
  tests.test_engine_execution_plans tests.test_engine_native_execution tests.test_engine_knobs
```

The tests verify exact selected slots, valid counts, pool/tail/cache bytes and
full-row cache projections for 128/129/130/131/132/133/195 rows, covered-prefix
boundaries and seven-token fallback. TP4 semantic model checks compare hidden
and auxiliary outputs, KDA/KV state and the following seven-token decode exactly,
with independent and combined sharding/absorb/dense-prefix lanes and tiled
259-row execution. Additional cases check output sentinels, single/multiple
selection passes, short FP8 input padding, 16/32-bit ID packets and initialized
ragged wire rows. New tests participate in the normal engine CPU CI selection.

`compile_kernel.py` compiled both selected absorb geometries and all eleven
dense-prefix layer offsets for SM121 using the retained actual checkpoint config.
All **13 variants passed**, CUDA remained uninitialized, elapsed **9.104 s**.
`compile.json` records source/config hashes and resource reports; artifacts remain
under `/home/choiceoh/glm53-logs/st-prefill-refine-20260914/compile-r1`.
Absorb query/output use 128/151 registers and 8,192 dynamic + 1,024 static shared
bytes. Dense prefix uses 178–180 registers and 65,536 dynamic + 1,024 static shared
bytes. All have zero stack/local spill, matching the previous resource counts.

Merge `bd85f3ce` incorporates main `91280525` and retains its enabled MoE decode
output. Every production source hash in the compiler record still matches after
the merge; none of this change's kernel, indexer, net or boot files changed.

## Work counts and Oracle

`work_budget.py` uses the actual checkpoint and chunk=32,256. Counts below are
per rank per DSA layer, for the default replicated indexer (sharding off).

| Input tokens | Queries bypassed | Scoring rows removed | Selection passes, before -> after |
|---|---:|---:|---:|
| 2,000 | 2,000 | 100% | 2 -> 0 |
| 2,672, historical short prompt | 2,051 | 76.76% | 3 -> 1 |
| 32,000 | 2,051 | 6.41% | 32 -> 30 |
| 128,000 | 2,051 | 1.60% | 127 -> 125 |

These are removed query operations, not total prefill speedups. Short suffix
projection padding is included in `work-budget.json`.

Oracle #875 at `e2bfbb9a` compared **main `91280525` -> candidate `bd85f3ce`** with
the actual config and no setting overrides. Both enabled prefill defaults,
chunk=32,256 and resident cache/state layout are preserved. The timing delta
remains **null** because matching GPU timings are unavailable. The paired
profile template is source/settings/model-bound and intentionally unmeasured.

No GPU queue job, engine build/boot, live restart or GPU launch was issued. CPU
and compiler proof do not qualify GPU numerics, quality/acceptance or consumer
TTFT/tok/s. The 2K 3,300 and 128K 4,000 targets remain unmeasured for this change.
