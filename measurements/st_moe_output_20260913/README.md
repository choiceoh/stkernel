# Decode MoE output finalization

Explicit candidate; serving selection remains unchanged. GPU timing and TP4
consumer qualification are pending. This work does not modify KDA state storage,
draft depth, projection precision or correction bias. It is independent of #895.

The routed expert currently accumulates in FP32, copies into a BF16 tensor,
adds the BF16 shared expert into another BF16 tensor, then exchanges that tensor.
The candidate consumes the completed FP32 accumulator before workspace reuse:

- At ordinary MoE-to-MHC boundaries, the existing OneShot packet grid performs
  `BF16(BF16(accumulator) + shared)` directly into its TX slot. Its local rank
  descriptor points to TX. ACK guarding, writer fences, 48 publication tickets,
  rank ordering and the immediate consumer remain in the existing protocol.
  The two separate cast/add launches and both BF16 intermediates disappear.
- At auxiliary-feature layers and the final layer, a Triton finalizer writes
  the ordinary BF16 tensor in one launch; the existing all-reduce and feature
  handling continue. It removes one intermediate and one launch.
- C=1 keeps its shared-expert side stream. It joins exactly once before the
  callback reads the shared result. Failure and missing/double-consumer checks
  preserve the stream and borrowed-workspace lifetime. Wider batches keep their
  existing sequential shared-expert computation.

The packet candidate is selected only by `decode_direct(..., moe_output=True)`;
no serving caller supplies that option. The lower-level finalizer refuses any
geometry outside the explicitly prepared tiled TP4 NVFP4 decode contract.
The admitted probe uses the existing D11 campaign expiry of 2026-09-16.

For the current five auxiliary layers, 36 MoE boundaries use packets and six
retain tensors. The source-level change removes 78 launches per target forward
and 9.75 MiB of local intermediate store/read traffic at C=1 K=7 (39 MiB at
C=4). These are operation/byte counts, not elapsed-time or arena savings.
Expert weight streaming, router selection and network payload size are unchanged.

## Qualification

- The focused Linux CPU suite passes 22 tests; two GPU-only tests are skipped.
  `cpu-tests.log` is the raw result.
- `compile.json` and `compile.log` record five SM121 Triton variants plus full
  native OneShot and producer-oracle extension builds with CUDA inaccessible.
  No CUDA context was initialized. `integration.json` verifies that all compiled
  inputs remain unchanged after main integration; the two Python composition
  files received main's separate prefill changes. These results do not establish
  GPU execution.
- `cpu-partial-tree.log` retains an initial wider-suite attempt: 13 modules
  lacked launcher/overlay/template fixtures in the CPU copy, and one exposed
  the already-fixed main #896 `fc_bias` reference default. The complete fixture
  copy and #896 are included for final validation; none is a speed measurement.
- The wider pre-integration pass covers 1,541 tests, with 269 GPU skips. Its one
  remaining failure was missing recipe-evidence directories in the CPU copy;
  all 33 shape tests pass after copying those actual directories. Both raw logs
  are retained. The additional 27 dispatch/transport checks pass.
- After integrating main `f7b2cbe2`, all 27 related CPU tests pass with one GPU
  skip (`integration-check.log`). The native/Triton source is unchanged.

`tests/test_engine_moe_output.py` checks rounding ties and cancellation,
unchanged inputs, output guards, shared-stream join/failure order, packet
ownership, and CPU four-rank C=1/C=4 state and auxiliary-feature ordering.
`test_engine_moe_output_transport.py` runs the real fused exchange through all
four local-rank positions, replayed ring wraps and mixed BF16/int64 exchanges
using an owned CPU proxy. It is not a real NIC or TP4 performance result.

`probes/engine_moe_output_check.py` loads the exact L3 rank pack and its ModelOpt
scales (when present), hashes the source tensors before in-place packing, then
tests M=8/16/24/32 and 8/32/all-distinct expert routes. It compares each candidate
against the **same FP32 accumulator** byte for byte, so cross-launch atomic
variation cannot hide a changed BF16 rounding boundary. Full routed/shared
component timings use captured warm and 128-MiB-evicted B/A/A/B measurements.
Those timings cover the tensor finalizer; packet exchange and NIC timing remain
part of the later TP4 consumer. No synthetic memcpy is substituted for exchange.

The GPU bundle isolates KDA commit, K7 projection and MoE output into three
300-second child processes. A failed child does not discard the others' results.
It runs one component reservation with no full-model boot:

```bash
bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
  --lanes k7_output_bundle \
  --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391
```

Accepted reservation: `st-decode-output-bundle0913v2`, ticket
`17893109571846042`, frozen source `12c9ce9a7` in the dedicated controller
checkout. The official replacement preserves the original `enqueued_at`
`1789303400` (21:43 KST). Estimate/cap is three 5-minute components; the foreign
fleet owner is untouched. `admission.json` records both the queued replacement
and the cancelled predecessor. The initial preparation was refused before any
GPU hold because newer relevant main changes were missing; integration resolved
that refusal.

Oracle #875 is used with K=7, FP32 state, 32K/128K and C=1/C=4. New source lacks
paired pricing, so the total time delta is unknown. Its `--acc 0` scenario is
explicitly timing-only; it is not measured acceptance. Default cache allocation
is unchanged, and the Oracle does not activate this explicit candidate.

The later candidate consumer retains the operator's one-boot workload: C=1 twice,
C=4 once, decode step/s, actual output tok/s and acceptance, with answer grades
recorded only as observations. No claim of reaching 22 step/s is made here.
