# Exact compact vocabulary merge

Item 2 from the Atlas review: merge the 64 TP4 vocabulary candidates without
restoring a 154,880-column array and running dense CUDA top-k. The candidate set,
selector, proposal distribution, verification, and RNG draws are unchanged.
This reduces proposal work; it is not an acceptance-rate improvement.

## Serving change

`Drafter.capture_decode` selects the compact buffer on the qualified GB10 runtime:
Torch git `cf30153c4c131c8164ee7798e5022d810682e2cb`, CUDA 13.2, capability 12.1,
TP4 packet width 64, vocabulary 154880, top-16, and at most 28 proposal rows.
Other geometries and builds retain the existing dense merge. The runtime check
runs before graph capture. It must be requalified before widening the gate.

Packed-key ordering alone would alter tied candidates. The Triton kernel instead
recreates CUDA radix-gather order (above-cutoff candidates by token ID, followed
by cutoff ties by token ID), then calls the same small unstable CUDA sort. At a
negative-infinity cutoff it includes the implicit dense background. There are
no host reads, new random draws, or changes to logit arithmetic.
The ordering reconstruction follows the pinned PyTorch
[radix gather](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/TensorTopK.cu)
and [small sort](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/Sort.cu),
with dense CUDA top-k retained as an independent executable oracle.

The old reusable buffer already cleared only touched columns. This change saves
its dense workspace and full-vocabulary selection reads, not a full-array write
per token. At seven rows the eliminated dense buffer is 4,336,640 bytes (4.14 MiB)
per rank; the gathered packet is 3,584 bytes.

## Validation and limits

`gb10-component.json` is the first same-input CUDA graph B/A/A/B comparison,
run by fleet session `vocab-merge-0917-7e62` beside production, without an engine
boot. All 15 tests ran and passed on the recorded runtime. The initial C=1 merge
component median was 198.980 us dense versus 24.022 us compact; shared production
load makes this a component observation, not an engine tok/s claim.

The tests compare exact candidate IDs and scores for random, tied, signed-zero,
nonfinite and decodable-mask cases, changing-packet graph replay, actual serving
greedy and T=1 walks, full proposal probabilities, and block-verification results
with fixed uniforms. T=1 tests establish equivalence, not T=1 traffic acceptance.

Reproduce on an admitted fleet source:

```sh
ST_PROBE_GIB=1 bash bench/fleet.sh run --gpu --detach SESSION 5 \
  'Compact merge current-main serving equivalence' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
  --lanes vocab_merge --output /cache/SESSION.json
```

The current-main probe additionally exercises the fused sampled walk and checks
that the qualified runtime enables the serving default. CPU-only coverage uses
`tests.test_engine_vocab_merge.CompactRuntimeTests`; the independent radix-order
reference also runs with `TRITON_INTERPRET=1`.

## Current-main qualification

`gb10-main-component.json` and `gb10-main.log` record session
`vocab-merge-main-0917-7e62`, candidate `ae1f33fb` based on `f5ebf857`.
All **16/16 tests passed with zero skips**, including the runtime gate and the
latest fused sampled walk. `default_compact=true` on the actual serving runtime.
Peak allocator reservation was 343,932,928 bytes within the 512 MiB cap.
The CPU-only runtime gate also passed separately, and PR #1113's engine CI passed
on `7f03bf40` (same engine and tests; documentation commit).

| Proposal rows | Dense median us | Compact median us |
|---:|---:|---:|
| 1 | 72.469 | 18.405 |
| 7 (C=1) | 102.532 | 19.045 |
| 14 | 143.075 | 19.915 |
| 28 | 233.391 | 20.994 |

These are paired graph measurements on identical packets beside production.
The different absolute times in the first run demonstrate why component numbers
must not be converted into a service speedup claim.

## Full adoption measurement

Fleet session `vocab-merge-adopt-0917-7e62` was admitted with
`ST_BRACKET_VALIDATION=full`, candidate `ae1f33fb` and baseline
`f5ebf857e66f07c8ebeddf07d984e4c82dee7770`. It runs the canonical full
onepass A/B, including 32K/128K and quality/acceptance. The capture-only TP4
replay ticket was replaced before execution to avoid redundant fleet work.
At this record's creation, the full A/B is queued behind production's quiet gate;
there is no live tok/s or final adoption verdict yet.
