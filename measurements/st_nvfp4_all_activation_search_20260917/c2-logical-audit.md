# FC search x C=2: input publication and buffer-lifetime audit

Follow-up on 2026-09-18. The concrete finding is a missing **generic-global
store -> TMA async-proxy load ordering step** after FC1 input packing. The
patch adds `fence.proxy.async.global` after the second resident-grid barrier.
This closes the identified proxy-ordering gap. GPU reproduction of stale reads
and consumer quality/performance after the patch have not been measured.

## The invariant to preserve

For token `t`, expert `e`, and 16-value input block `b`, FC1 must consume the
pair returned by the quantizer for that same block:

```
(P[t,e,b], S[t,e,b]) = search(X[t,b], global_scale[e])
FC1 operand         = decode_fp4(P[t,e,b]) * decode_fp8(S[t,e,b])
```

Concurrency does not enter this function. A C=2 call can change which thread
writes the pair, its expert-local row, and when it becomes visible; it cannot
mix the pair's token, expert, block, or invocation. The search chooses packed
bits and scale together. It has no shared/global side effects. The pinned
FlashInfer packing helpers also use separate local output registers instead
of modifying the input values passed to the search.

The same static FC2 radius (1) is used in ss1 and as1 at both widths. The new
static decode arithmetic is FC1 search. The production folded-scale path
supplies unit input global scales, so its hot path is the same-scale branch.

## Where C=1 and C=2 differ

Both widths use the same M16 reform MMA tile: FC1 `(16,128,256)` and FC2
`(16,256,128)`. C=2 makes rows 8..15 live; it does not introduce a different
FC1 reduction shape. Its frontend and epilogue do differ:

| Phase | C=1, M8, reuse=3 | C=2, M16, reuse=4 |
|---|---|---|
| Before grid barrier 1 | Quantize/search each token block into a global cache; prepare routes | Prepare routes |
| After grid barrier 1 | Copy cached packed bits and scale into expert rows | Quantize/search into registers, then fan out to eight expert rows |
| Publication | Grid barrier 2 | Grid barrier 2 |
| FC1 read | TMA reads packed A and SFA separately | TMA reads packed A and SFA separately |
| FC2 output | Shared BF16 tile, FP32 vector-4 reductions | Register pairs, FP32 vector-2 reductions |

Thus in the unit-scale serving path, enabling FC1 search adds its work before
the first barrier at C=1, but between the two barriers at C=2. C=2 interleaves
search and expert-row writes and has a different thread-to-write assignment.
The exposure mechanism is this different publication schedule, not a
cross-request maximum inside the search.

## The broken ordering edge

The unfixed source executes:

```
st.global.u64 packed_A[...] = P
st.global.u8  SFA[...]      = S
resident grid barrier: CTA sync, membar.gl, counter/epoch, CTA sync
cp.async.bulk.tensor ... packed_A
cp.async.bulk.tensor ... SFA
```

`membar.gl` and the epoch protocol coordinate ordinary global accesses. TMA
uses the async proxy. Existing `fence.proxy.async.shared::cta` instructions
cover shared memory, and `fence.mbarrier_init` covers barrier initialization;
neither establishes this global-memory proxy edge. Waiting for TMA completion
publishes what TMA read, rather than ordering the preceding generic source
writes. This follows the [PTX async-proxy and fence contracts](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

Consequently the code lacks the guarantee that a TMA read sees both members
of the current pair. For example, if it sees `P_new` and `S_old`, its effective
operand is `decode(P_new) * S_old`. That is neither the search winner nor the
baseline operand. Lower SSE for `(P_new,S_new)` says nothing about this mixed
operand. No out-of-bounds address or NaN is necessary for that failure mode.
This is a permitted failure mode under the missing ordering contract, not a
claim that the consumer run was observed to take this exact interleaving.

The gap exists in both widths and both recipes. Its existence therefore does
not, by itself, select C=2 as the failing case. The C=2-specific interaction is
the changed placement of search, fanout writes, and the separately loaded
packed/scale pair. A shared latent ordering gap can be exposed differently by
those schedules; C=1 improvement does not validate C=2's publication path.

## Patch location

The change in `engine/kernels/b12x/moe_static_kernel_v4.py` is immediately
after grid barrier 2, before any CTA begins TMA input reads:

```
all producers store P and S
    -> resident-grid publication/acquisition
    -> each consuming CTA: fence.proxy.async.global
    -> TMA reads P and S
```

Placing the fence only in CTA 0, only in the quantizer helper, or before the
inter-CTA publication would not express the required consumer-side ordering.
The [NVIDIA mixed-proxy paper, section 5.4](https://d1qx31qr3h6wln.cloudfront.net/publications/ISCA_2022_MixedProxy.pdf)
requires the proxy fence on the causal path in the CTA performing the
non-generic access. No recipe, search objective, rounding, routing, or tile
layout is changed by this patch.

## Other interaction boundaries checked

- **Packed/scale address association:** executed both production address
  expressions and compared scale destinations with the installed CuTe
  `tile_atom_to_shape_SF` consumer layout. All 36,864 checked coordinates
  agree, including every live row/block in M8 and M16, experts 0/1/287, and
  both minimum and production row capacities. Packed addresses match across
  the two modes and retain 8-byte alignment. No scale-address collisions.
- **Thread ownership and scratch:** all token blocks have exactly one writer
  across 160 threads/CTA for grids 32/36/40/44/48. The reserved cache/route tail
  stays outside every reachable compact expert plane even for all-distinct
  top-8 assignments. This checks ownership and layout, not memory visibility.
- **Scale representation:** searched activation SFA remains full E4M3 bytes.
  SF6 compression applies to the weight SFB planes; searched activation codes
  are not truncated into the six-bit weight format.
- **FC1 -> FC2 publication:** all MMA warps rendezvous after writing A2/SFA2;
  C=2 route metadata is read after that barrier. The existing adversarial CPU
  test detects unpublished metadata when its barrier is removed.
- **FC2 early stage release:** the final B/SFB operands have already been
  copied into registers before the stage is released. The four-warp consumer
  group governs reuse of the three C=2 stages. The final item-retirement
  barrier prevents another item from overwriting A2/route metadata early.
- **Output rounding:** both epilogues round the down projection to BF16,
  multiply by the FP32 route weight, round that contribution to saturated
  BF16, then widen for FP32 atomic addition. C=2 does not omit the route
  weight or an activation scale. Accumulation order can still differ; this
  remains a separate boundary after input publication is repaired.

## Reproducible evidence

The read-only source snapshot and local pre-patch files match by SHA-256 for
the quantizer, dispatcher, static v4/v5 kernels and common helpers. Native
audits used serving image
`sha256:e9e80b94d41277b171cef5785483989acd1c91d450d0d1068269e99f60bc75bd`,
CUDA hidden, no GPU device mount, network disabled, and disposable containers.
Native PTX targets `sm_121a`, PTX 9.3.

- `audit_c2_addresses.py` / `c2-addresses.json`: source-expression and actual
  CuTe-layout checks above. Run as a real Python file because CuTe traces its
  source; the layout inspection runs inside a CPU-only JIT trace.
- `audit_c2_native_ptx.py` / `c2-native-ptx.json`: original M8/M16 x ss1/as1,
  zero global async-proxy fences in all four generated kernels.
- `c2-native-ptx-fenced.json`: four patched kernels compile; each has one
  global proxy fence after the second grid membar and before TMA instructions.
  Source inspection puts that fence in the common unconditional path.
- `c2-native-sass.json`: the C2/as1 machine code gains a real
  `FENCE.VIEW.ASYNC.G` instruction; the cubin changes. The fence inventory loses
  no instructions. Resource counts remain REG 96, STACK 0, LOCAL 0, with
  unchanged shared/constant counts. This is instruction/resource evidence,
  not a measured latency result.
- `python3 -m unittest tests.test_engine_moe_sync_cleanup
  tests.test_engine_activation_scale_search tests.test_engine_moe_batch_reform -q`:
  all 13 tests pass after the patch.

The earlier [receipt/configuration audit](c2-path-audit.md) remains the record
of completed consumer grades, prefill chunking, and original test coverage.
No GPU fleet was started for this audit. Post-patch GPU quality and latency
remain unmeasured.
