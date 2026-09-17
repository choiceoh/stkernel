# FC activation publication repair

The follow-up repairs the prefill proxy-ordering gap and the final-arrival
acquire gap identified in [the remaining-publication audit](remaining-publication-audit.md).
It also adds identical-input M8/M16 checks with real rank weights. These checks
validate kernel publication and width equivalence; consumer score recovery,
acceptance and the C=1 throughput guard have not been remeasured.

## Publication changes

There are two distinct ordering requirements:

1. A resident-grid barrier must acquire every arriving CTA's stores before
   its final leader releases the new epoch. An entry `membar.gl` followed by
   a relaxed counter RMW publishes that CTA's writes, but the last counter
   arrival need not be the last entry fence. Add a GPU-scope fence after the
   final relaxed RMW, before counter reset and epoch release.
2. Once generic-global stores have been acquired, each TMA consumer must
   bridge them to the async proxy. Add `fence.proxy.async.global` after the
   final route/pack publication and before the first TMA operand load.

The first change applies to all six resident-grid barrier owners: static v4,
stock static, micro, direct micro, generic dynamic and gated dynamic. The
direct-micro M1 chunk publisher also gains a CTA sync and entry fence before
its counter RMW, plus the final-arrival acquire fence. Its consumers already
acquire the completed epoch. The separate W4A16 barrier already has the final
fence and requires no change.

The second change applies to stock static/micro, generic/gated dynamic,
common SF6, the prefill reuse kernel and the generated private M64 kernel.
Static v4 retains the proxy fence added by PR #1133. The common SF6 owner
covers served short Q0-word and long SF6 prefill, as well as the packet path.
Parent-source pins and imported-source provenance are refreshed together.

Regenerating the private M64 body also closes its stale copy of the activation
search branches: it now inherits the current gated/Q0 parents. The generator
pins those parents and reproduces the checked-in body. M64 remains private;
its search-disabled arithmetic is unchanged.

The ordering follows the PTX [release/acquire patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-and-acquire-patterns)
and [memory/proxy fence contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar).
The patch expresses the missing edges explicitly; the original missing edges
were found by protocol/native-code inspection, not by a reproduced stale-read
GPU failure. It does not change expert count, routing, served search radius,
FC1/FC2 reduction shapes or the C=1/C=2 dispatch choices.

## CPU and native validation

- **74 focused CPU tests pass** in the serving image, including the new four
  publication tests. The tests execute the actual barrier control flow in an
  adversarial schedule where CTA 0 fences first and increments last. Removing
  the final acquire is a negative control that exposes missing writes.
- The imported-source provenance check passes separately.
- **30 existing activation-search kernel compile cases pass**, plus private
  M64/as1 and direct-micro M1/M8 native compiles. CUDA is hidden and remains
  uninitialized throughout these checks.
- Static M8/M16, ss1/as1 native audits contain both final-arrival fences and
  one global proxy fence. All four use 96 registers, zero stack and zero local
  memory. The earlier C=2/as1 receipt also used 96 registers; this is a resource
  check, not a latency measurement.
- Short M=2304 and long M=32256 prefill both change from zero global proxy
  fences to one in PTX and one `FENCE.VIEW.ASYNC.G` in SASS.

Receipts: [CPU tests](publication-repair-cpu.log),
[30 compile cases](publication-repair-compile.jsonl),
[three additional compiles](publication-repair-extra-compile.jsonl),
[static native audit](publication-repair-static-native.json), and
[prefill native audit](publication-repair-prefill-native.json).
The native audit scripts are `audit_c2_native_ptx.py` and
`audit_prefill_publication.py` in this directory; the compilation lane is
`engine_kernel_check.py --lanes fp4_scale_search_compile` with
`ST_PROBE_NO_GPU=1 CUTE_DSL_ARCH=sm_121a`.

## Live GPU validation

`probes/engine_moe_publication.py` is an optional `publication` section of the
existing `moe_c2_cells` lane. It uses layer 3 of the real folded-scale
`rank3of4.safetensors`, the served chunk size 256 and `as1` on GB10. The same
input/route tensors are evaluated as two M8 calls and one M16 call. Serving
kernels contain no diagnostic instrumentation or arithmetic changes.

The initial successful receipt, [publication-repair-gpu.jsonl](publication-repair-gpu.jsonl),
records:

| Check | Cases | Observed result |
|---|---:|---|
| Two seeds, shared/disjoint/router routes, unit/unequal global scales | 12 | All 1,536 packed A/SFA route pairs identical |
| Full native output | 12 | BF16 differences 0; maximum absolute FP32 difference 4.76837158203125e-7 |
| One nonzero route, one FC2 K128 slice | 8 | M8/M16 output bit exact |
| CUDA graph replay with three changed payloads | 3 | BF16 differences 0; maximum absolute FP32 difference 5.960464477539063e-8 |
| Zero routes after output poisoning | 1 | Finite, exactly zero outputs |

The existing `fp32_max_ulps` field is **RMS-floored ULP normalization**, not an
integer bit-distance between each pair of FP32 values. Its denominator is
the FP32 spacing at `max(abs(got), abs(want), RMS(want))`; this avoids inflating
the addition-order metric when contributions cancel near zero. The initial
full-width maximum is 2 in that metric and the graph maximum is 1. Absolute
error and BF16 differences above are the direct numerical observations.

The strengthened final probe additionally checks nonzero fixtures, output
hash changes, reconstruction of each one-route result from its four isolated
K slices, and repeated-row prefill at 337/2304/32256 rows with changed and zero
payloads. Ticket `fc-pub18c` passed the nonzero/hash guards and both exact
reconstructions, then rejected the probe's decode-only finalizer before
prefill launched. The [failed-run receipt](publication-repair-prefill-harness-failure.jsonl)
is retained. The corrected probe uses the ordinary BF16 prefill output buffer.

The [final GPU receipt](publication-repair-final-gpu.jsonl), ticket `fc-pub18d`,
**passes** on integrated source `744d08ec2275398a55f042b940432f8fb7849ac2`:

| Final check | Cases | Observed result |
|---|---:|---|
| Identical-input M8/M16 | 12 | 1,536 packed A/SFA pairs identical; BF16 differences 0; FP32 max absolute difference 4.76837158203125e-7 |
| Isolated FC2 K128 contributions | 8 | Exact, nonzero outputs at both widths |
| Sum four K slices back into one route | 2 | Exact reconstruction |
| Changed-payload and zero-route CUDA graphs | 4 | BF16 differences 0; changed hashes; exact zero-route output |
| M=337/2304/32256, two changed payloads and zero routes each | 9 | All repeated input rows produce identical BF16 output; changed hashes and exact zeros |

The final width and graph maxima are both 1 in the RMS-floored ULP metric.
The graph maximum absolute difference remains 5.960464477539063e-8. These are
layer-local observations, including casting the static FP32 sums to BF16;
they are not a whole-model or consumer-response equivalence claim.

## Concurrent-merge follow-up

PR #1141 merged at `a2fd1ccd`. PR #1142 independently added the same fence to
stock static, micro and generic dynamic, producing two fences in each owner.
The next deduplication (#1143, `cd8cc080`) removed both copies. The existing
publication test reproduces three missing-fence failures on that exact main
snapshot; [the regression receipt](publication-repair-merge-regression.json)
records the affected owners.

PR #1145 restores exactly one fence per owner and its provenance hash. Those
three files are byte-for-byte identical to the earlier native-validated repair.
All eight publication, source-contract and targeted provenance tests pass on
the integrated source, `744d08ec`, and the PR's complete CI check passes. Both
independent source-contract suites are retained. PR #1145 subsequently merged;
the final GPU receipt completes the queued validation.

The integration also includes #1139's separate FP32 long-prefill accumulation
change. The [final prefill native audit](publication-repair-prefill-final-native.json)
recompiles the current short and long paths, confirming one global proxy fence
in both PTX and SASS with the updated source hashes. The earlier prefill native
receipt applies to the pre-#1139 implementation. The final GPU ticket is pinned
to the integrated source; it does not reuse a pre-#1139 prefill result.

## Runtime and lifecycle

Serving changes are commit `629bcc118af5b6298a9ae59c0c0a8fae65887374`; the
strengthened probe is `631904c9f016fc58f98f8f35b8de0507eb468e06`. The latter
changes no serving code. The native audit and GPU receipts retain source hashes.

- CPU compile image: `sha256:e9e80b94d41277b171cef5785483989acd1c91d450d0d1068269e99f60bc75bd`.
- GPU image on srv4: `sha256:926d2267c0683bade767f640108b4afe6ab3060ea803555d4e606800fc7d3fb5`.
- Both inspected images have PyTorch 2.13.0+cu132, CUDA 13.2, CUTLASS 4.6.2,
  FlashInfer 0.6.18.dev20260819 and `fp4_common.py` SHA-256
  `a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1`.
  The whole images have different identities; this is not a claim of image
  equivalence.

The [runtime receipt](publication-repair-runtime.json) records both inspections;
neither inspection mounted a GPU or initialized CUDA.

The first ticket `fc-pub18` failed before kernel execution because its pinned
CPU image was absent on srv4. Ticket `fc-pub18b` used the explicitly pinned
srv4 image and passed in 29.3 seconds, with peak allocation 1,932,266,496 bytes.
The runner removed its container and returned the single-GPU reservation at
completion. No four-node serving fleet was started for this repair.

Ticket `fc-pub18c` ended and released its reservation after the finalizer
contract rejection. The corrected ticket `fc-pub18d` uses the same
queue/runner and pinned GPU image. It was paused before admission to integrate
the concurrent changes, updated through `fleet.sh edit`, and resumed with its
original queue ticket. The admitted source is pinned at `744d08ec`.

The run completed successfully in **45.9 seconds**, with peak allocation
4,466,283,520 bytes. The runner removed its container and returned the
reservation on completion. The [release receipt](publication-repair-release.json)
records the finished ticket, zero remaining containers mounted from this
probe's source tree, and all 50 deployed source hashes matching the pinned
workspace. Other fleet jobs were not stopped.

```bash
ST_IMAGE=sha256:926d2267c0683bade767f640108b4afe6ab3060ea803555d4e606800fc7d3fb5 \
ST_PROBE_TREE=st-probe-fc-publication18 \
bash bench/fleet.sh run --gpu --detach fc-pub18d 5 \
  "FC publication repair: current FP32 prefill, one fence per owner, BF16 probe output" \
  -- bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
  --lanes moe_c2_cells:publication:layers=3:chunks=256 \
  --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors \
  --output /cache/fc-publication18-verified-gpu.jsonl
```

Consumer quality, natural-EOS acceptance, throughput and a causal explanation
of the earlier C=2 score loss still require matched consumer measurements.
Passing these publication checks does not supply those measurements or prove
that all possible kernel defects are absent.
