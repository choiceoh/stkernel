# GB10 execution plans

Three independent experiments for GLM53 on four GB10s. Default execution and
FP32 KDA storage remain unchanged. These are not measured speedups.

| Arm | Decode TP overlap | Early DFlash2 observe | Prefill window |
| --- | --- | --- | --- |
| baseline | off | off | one native tile |
| tp-overlap | C=4 as 2+2 | off | one native tile |
| early-observe | off | on | one native tile |
| layer-prefill | off | off | two native tiles |
| combined | C=4 as 2+2 | on | two native tiles |

## Ordered compute and communication

`execution.decode_overlap` captures one target graph. Compute stays on one
stream because the existing native kernels share scratch. A second stream
reduces the first group's output while the second group computes. Every rank
enqueues collectives in `(layer, attention/ffn, group)` order. Each reduction
owns its input until the NIC and consumer finish. Events protect consumers and
join both streams before graph capture returns.

The full batch still samples and commits once, using the existing RNG and row
lifecycle. C=1/2/3 keep their original target batch. C=4 changes GEMM row shape
and doubles per-layer reduction count; communication overlap may fail to pay
for those costs. Numerical CPU agreement alone is not adoption evidence.

## Early DFlash2 context preparation

Target auxiliary layers are zero-based `[4, 13, 23, 32, 41]`. Once layer 41
finishes, the ordinary context projection can run while target layers 42--44
and the vocabulary head finish. Proposal generation still waits for its anchor.

The native W4 kernel originally used process-global partial sums and arrival
counters. The early FC projection owns a separate initialized workspace;
the same packs, K-slice order and arithmetic use those pointers. Shared-MLP
pair state and low-rank correction are not admitted on the private entry point.
The existing fused `write_context` commits only the final retained count,
including EOS/limit trimming. Calibration observes these committed rows once;
tentative projection never updates calibration or the drafter ring.

The target graph rejoins the observation stream before returning. Prepared
context lives in the target graph pool and is consumed before its next replay.
Synchronous rich sampling also consumes the prepared context.

## Layer-major prefill

A scheduler window contains two or four existing 9,216-token tiles. At each
layer all tiles run attention, then all tiles run FFN, in token order. This
retains the existing per-tile kernel shapes, SP transport and numeric guards
while improving the opportunity to reuse that layer's weights in the 24 MiB
GB10 L2. No claim is made that a layer or its experts fit in L2.

Only the idle-prefill budget grows. With live decoders the 2,304-token prefill
budget and alternating decode step remain unchanged. Prefill and decode never
share a model step. The existing 12 GiB runtime workspace ceiling still applies;
additional carries and auxiliary features must fit it during qualification.

Per-layer KDA rings advance in token order. Interior marks use the existing
kernel outputs. A mark at a tile end is copied before that layer's next tile
overwrites its ring. The runner publishes a prefix only when the whole window
finishes, so publication may be delayed relative to one-tile execution.

## Selection and evidence

Non-production knobs expire on 2026-09-30:

```
STK_execution_overlap=1
STK_early_observe=1
STK_prefill_tiles=2
```

Production rejects knob overrides and fixes all three experiments off. The
selected plan is included in `st:lane_info`; private workspace bytes are
reported at capture. All experiment combinations require FP32 KDA state.

CPU tests compare real reference KDA/DSA paths: changed C=4 grouping, physical
request order, resumed prefill, output/auxiliary values, state/KV bytes and
prefix snapshots. The focused CUDA gate exercises concurrently running native
W4 GEMMs with distinct inputs and repeated changed-input graph replay.

The canonical fleet gate is a committed-arm `st-chain` with a same-code
baseline and each experiment separately, followed by the combined arm only
when individual arms qualify. Each boot runs full onepass twice at C=1/C=4,
2K/32K/128K with prepared kernels, fresh prefixes and separate diagnostics.
Judge actual per-request TTFT/ITL, output tok/s, acceptance, quality, verbosity
and workspace peaks. Do not infer an engine win from compilation or the
focused scratch test.
