# Operator-enabled token-major MLA contractions

The user explicitly requested immediate enablement of the new kernel. Commit
`45ada81d` sets `prefill_absorb_tiles=1` in both production and experimental boot
defaults. Experimental rollback remains `STK_prefill_absorb_tiles=0`; production
rejects either environment override. Bare net/ExecutionPlan construction remains
neutral; serving fills the actual plan from the boot declaration.

At that commit, **21 focused CPU tests passed, no skips, 4.477 s** in the same
bounded ST CPU image used by the implementation gate (`cpu-tests.log`):

```sh
python3 -m unittest -v \
  tests.test_engine_prefill_absorb_tiles.AbsorbRoutingTests \
  tests.test_engine_native_execution tests.test_engine_knobs
```

The tests cover both serving defaults, rollback, fixed production behavior,
per-layer execution evidence and decode/capture/probe routing. The actual kernel
and native wrapper AST are unchanged from compiler-qualified `fc2167b8`, excluding
docstrings. Prior arithmetic/model/compiler results retain their original pins.

Oracle #875 (`e2bfbb9a`) compared `6e08213f` with `45ada81d` **without `--set`**,
using the retained actual checkpoint config. The evaluated candidate plan enables
both `prefill_absorb_tiles` and `prefill_dense_prefix`; indexer sharding stays off.
Chunk size is 32,256 and resident cache/state-layout deltas are zero. Prefill
timing remains unpriced/null at 2K, 32K and 128K. The new paired profile template
is bound to this default-on source and settings.

This is selection by operator request, not GPU numerical/performance proof.
It takes effect on the next startup from the merged source. No running engine
was restarted and no GPU queue job, engine build/boot or GPU launch was issued.
