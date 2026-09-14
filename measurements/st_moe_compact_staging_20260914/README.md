# Compact C1 MoE staging

Implementation `2034a63b`, based on `119f881d` (merged #920, including #921).
The audit of #913/#915/#916/#918/#920 found that gate/up register reuse had
left up's A/SFA shared allocation unused. Reclaim that allocation and use
part of the space for disjoint FC2 packed scales. The C1 SF6 default is ON;
`compact_staging=False` retains a separately cached current-source control.
No GPU speedup is claimed.

## Ownership and exact work

With an even FC1 ring, gate always occupies an even stage index and up an
odd one. Inputs map to `stage // 2`; an even gate slot is reused only after
all four MMA warps release that same original B/SFB slot. Gate has already
loaded all four K64 input fragments by then. Up reads those registers, so
the producer may refill shared input while up executes. The B/SFB pipeline,
its waits/releases/transaction counts, and its prefetch depth are unchanged.
Odd FC1 rings keep their original allocation and indexing. Native layout
guards require an identical per-stage A/SFA consumer layout and exactly
half the staged extent; the retained-register guard from #920 also runs.

Each FC2 stage now owns both a packed input and an expanded scale buffer
until the existing consumer release. DMA still transfers 1552 scale bytes
under the same transaction barrier. Expansion reads disjoint storage, so
it omits the read-before-write barrier and retains the publication barrier.
There is no alias between packed and expanded buffers, no early release,
and no new pipeline. For H4096/N256, this removes **16 four-warp barrier
rendezvous per work item** (one expert M tile and one intermediate slice).
Floating-point/MMA order, scales, quantization and rounding remain unchanged.

| C1 resource | Current-source control | Compact staging |
|---|---:|---:|
| FC1 A/SFA staged bytes | 8192 | 4096 |
| FC2 separate packed bytes | 0 | 3104 |
| Total staged shared bytes including alignment | 100352 | 99328 |
| M8 registers | 121 | 121 |
| M8 static instructions excluding NOP | 3485 | 3474 |
| M8 static named-barrier instructions | 32 | 28 |

Header padding absorbs part of the packed allocation; the net shared-memory
saving is **1024 bytes/CTA**. The separately reported static shared allocation
remains 1024 bytes. These are layout and code-generation counts, not a timing
estimate or a claim that occupancy changed.

`compact_staging` is effective only for M1–8 SF6 decode with FC1 register
reuse, separate FC1 scales and an even FC1 ring. It is normalized before
cache lookup and included in the native name as `compact`. Larger rows and
legacy recipes retain their paths. There is no public serving knob. K=7,
FP32 KDA state, prefill paths and the other enabled decode changes remain.

## Verification

- `cpu-tests.log`: 22 tests pass in the existing CPU-only ST image. The
  actual FC1 consumer loop matches the uncompacted register-reuse control
  with poisoned released buffers and changed payloads across work items.
  The actual producer checks its compact destinations and exact expected
  transaction bytes, including all 32 DMA lanes and skip-A controls.
- An independent four-warp pipeline model permits producer run-ahead and
  adversarial warp skew over three work items, without resetting the ring.
  It covers 1/2/3/4/6-stage rings, 20 seeds each and both configurations.
- The actual FC2 producer/consumer address-selection blocks run on prefilled
  1/2/3-stage rings across changed experts and intermediate slices. The real
  expansion helper runs under randomized 128-thread memory interleavings.
  All bytes match the independent encoder; neighboring queued inputs and
  canaries remain intact. Each compact stage executes one barrier; the
  in-place control executes two. Existing all-base/all-code tests also pass.
- `native-compile.json`: seven actual CuTe/PTXAS/TVM-FFI handles pass: M1/M7/M8
  compact, M8 uncompacted, M8 without FC1 reuse, M16 and M32. Native fragment
  guards pass, no stack/local spill appears, and the M8 floating-point/MMA
  opcode counts do not change. Source, binary and SASS hashes are retained.
- The PR's full engine CPU CI and onepass contract check supply the complete
  suite result. See its latest-head check before merging.

The prepared `moe_compact_staging` real-weight probe compares explicit
`compact_staging=False/True` on the same L3 TP4 weight pack at M1/6/7/8,
changed and duplicate routes, partial M16 tiles, zero weights, graph repeat
spread, and warm/evicted B/A/A/B timing. Its numerical gate remains 0.001.
**It was not run.** The previous `moe_fc1_reuse` probe now pins compact
staging OFF in both arms so it continues to isolate #920's change. Native
compiler cases use named overrides instead of positional Boolean tuples.

```sh
python3 -m unittest -v tests.test_engine_moe_compact_staging tests.test_engine_moe_fc1_reuse tests.test_engine_moe_sf6_staging tests.test_engine_moe_activation_store tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --compact-staging --sass --output /out/native-compile.json
# Prepared only; not submitted or run:
python3 probes/engine_kernel_check.py --lanes moe_compact_staging --ranks EXACT_CONSUMER_RANK_DIRECTORY
```

CPU/native work used existing image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`
on srv2 with runc, CUDA hidden, no network, two CPUs/four GiB and one build
worker. Existing CUDA 13.0 nvdisasm was mounted read-only. There was no image
or baseline-engine build, GPU context, model boot, service restart or queue
submission. Actual GPU numerics/replay, quality, acceptance and step/s remain
unmeasured.

## Oracle

The upgraded #875 Oracle at `e2bfbb9afdcc6e8fe1e5fe47ddddd23180b278b3`
compares `119f881d` with implementation `2034a63b` at C1 2K/32K/128K using
the retained checkpoint configuration. `--acc 0` is a timing-only assumption,
not a measured acceptance rate. MoE has no matched coefficient for this
change, so total decode delta is null. No speed coefficient is inferred
from shared-memory or barrier counts. `oracle-c1.json` and the unmeasured
`paired-profile-c1.json` preserve source identities and the missing proof.
