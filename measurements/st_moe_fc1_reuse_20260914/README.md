# C1 FC1 gate/up input reuse

Implementation `7898f7e7`, base `bec3b6dd` (merged PR #918). C1 M1–8 SF6
decode now retains gate's A/SFA register fragments for up and omits up's
duplicate A/SFA DMA. The existing B/SFB pipeline ring remains. No separate
A pipeline is enabled and no additional shared buffer is allocated.

## Dataflow and transaction contract

Gate and up consume identical packed input/activation-scale tiles. Their
weights and weight scales remain independent. The full K256 A/SFA fragments
already provide storage for four K64 blocks. A native layout guard proves
that blocks do not overlap: each block owns 32 A elements and 4 SFA elements.
Broadcast within a block is allowed. Gate fills every block before up
reuses it; MMA only reads these fragments. The next K tile refills them.

The ordinary FC1 transaction barrier originally expects B/SFB plus A/SFA
for both stages. Reuse sets its base expectation to B/SFB (17936 bytes).
For gate only, the DMA leader adds 4096 expected bytes with the installed
CuTe `mbarrier_expect_tx` operation **before any TMA issue**. The base
expectation is already positive, so the phase cannot finish before this
addition. Gate transfers 22032 bytes; up transfers 17936. No transaction
is expected without a matching transfer. Both stage waits/releases and
SF6 publication barriers remain, and the skip-A diagnostic adds no bytes.

For one H4096 work item (one expert M tile and one 128-column intermediate
slice), 16 K256 tiles previously issued 32 A plus 32 SFA transfers. They now
issue 16 of each: **32 fewer transfer issues and 65536 fewer bytes** per
work item. FC1 input/scale traffic is halved; its total DMA traffic changes
from 705024 to 639488 bytes. Shared-to-register A/SFA loads also halve.
These are exact work counts, not a throughput percentage.

The internal `fc1_reuse_a` default is ON only for M1–8 SF6 decode reform.
`fc1_reuse_a=False` is a separately cached control; enabled handles include
`a1reuse` in their lane log. Larger rows, legacy recipes, K=7, FP32 KDA state
and existing decode fastpaths retain their behavior. No public serving knob
is added.

## Native code evidence

`native-compile.json` records six actual CuTe/PTXAS/TVM-FFI builds:
M1/M7/M8 enabled, M8 control, and unchanged-path M16/M32. The M8 comparison
uses the same current source and differs only in `fc1_reuse_a`.

| M8 handle | Static instructions | Excluding NOP | Registers | Stack/local bytes |
|---|---:|---:|---:|---:|
| Repeated-input control | 3595 | 3540 | 126 | 0/0 |
| Gate/up reuse | 3533 | 3485 | 121 | 0/0 |

Static `LDSM.16.M88.4` count drops 106→102, scalar `LDS` 42→38, and total
`UTMALDG.3D`/`UTMALDG.4D` count 14→10. Transaction-expect instructions are
added for gate. Floating-point/MMA opcode counts and named-barrier count
remain unchanged. C1 staged allocation remains 100352 bytes, with the
binary separately reporting 1024 static shared bytes. Wider rows retain
117 registers and no local/stack bytes. Source, binary and SASS hashes,
opcode histograms, actual layout byte counts and fragment ownership counts
are in the report. There is no GPU latency/throughput verdict.

## Validation and reproduction

- `cpu-tests.log`: 18 focused tests pass. The actual production consumer
  gate/up loop is executed across 32 K tiles, changing item/thread payloads,
  warp-boundary lanes and 1/2/3-stage wraparound. Released buffers are
  poisoned, and up's omitted input stage is unreadable. Every MMA operand
  and wait/release event matches the control while A/SFA loads halve.
- The actual producer loop and transaction-byte calculation are executed
  for all 32 producer lanes. Exactly the leader adds gate's 4096 bytes
  before any transfer, and both stages issue their exact expected bytes.
  The skip-A diagnostic retains zero input transfers/expectation.
- The fragment guard rejects partial K tiles and cross-block register
  aliasing. The native compiler runs the guard on the real A/SFA layouts.
  Existing scale, activation and cache-normalization checks also pass.
- Six full native handles compile. The PR's full engine/onepass CPU CI
  provides the complete-suite result.
- `moe_fc1_reuse` is prepared in the existing real-weight GPU probe. It
  explicitly captures control/candidate configurations at M1/6/7/8, changed
  routes (including 1/2/4 experts and multi-M16 work), zero weights, repeat
  spread and warm/evicted B/A/A/B timing. The existing 0.001 numerical gate
  is unchanged. This probe was **not run**.

```sh
python3 -m unittest -v tests.test_engine_moe_fc1_reuse tests.test_engine_moe_sf6_staging tests.test_engine_moe_activation_store tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --fc1-reuse --sass --output /out/native-compile.json
# Only under an existing admitted GPU hold; not executed for this change:
python3 probes/engine_kernel_check.py --lanes moe_fc1_reuse --ranks EXACT_CONSUMER_RANK_DIRECTORY
```

CPU/native checks used existing image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`
on srv2, runc, CUDA hidden, no network, two CPUs/four GiB and one build
worker. Host CUDA 13.0 nvdisasm was mounted read-only. No image/baseline
engine build, GPU context, model boot, service restart or queue submission
was issued. Full GPU numerical/replay, quality, acceptance and consumer
step/s remain unmeasured.

## Prior alternatives and Oracle

The old direct-register scatter is not enabled: its recorded C1 evicted
cases regressed 0.22–1.59% (`../st_decode_scatter_20260913/README.md`). The
old separate A ring also showed no useful gain (archived 38차 §8/§10).
This change instead retains registers and removes the duplicate DMA while
keeping the existing FC1 stage ring.

PR #875's upgraded Oracle at `e2bfbb9afdcc6e8fe1e5fe47ddddd23180b278b3`
compares base `bec3b6dd` to implementation `7898f7e7` at C1 2K/32K/128K
with the retained checkpoint configuration. `--acc 0` is a timing-only
assumption, not acceptance. Without a paired MoE coefficient, total decode
delta is null. No latency coefficient is inferred from bytes/instructions.
`oracle-c1.json` and the empty paired template retain the identities.
