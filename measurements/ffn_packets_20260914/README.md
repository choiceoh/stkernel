# Direct packet consumption in ordinary prefill, PR #895

PR #895 now retains **packet FFNs only**. Mixed decode/prefill and compact KDA
execution code have been removed. Ordinary decode, KDA state, cache and graph
behavior retain integrated main `77d80b6c`, including its deferred FP32 KDA
default. The [scope decision](../../bench/ST_GB10_PACKET_ONLY_20260914.md) records
what was retired and why. Historical S/M measurements remain at their original
source revisions; they do not qualify this implementation.

The existing `Glm53Net.forward()` FFN branch can pass its four rank-ordered
FP8-v3 packets directly to router, routed expert and shared gate/up. It retains
one all-gather and the existing reduce-scatter ordering. No mixed scheduler,
route planner, ticket or whole BF16 FFN input is introduced.

The experiment remains **OFF by default** (`STK_prefill_ffn_packets=1` opts in).
It requires native eager TP4, chunk-ordered prefill, H4096/E288/I512/top-8,
SF6 M128 and `8192 < real rows <= 32768`. A control-group vote agrees all
supported layers before transport. Short prefill, decode, unsupported packs and
calibration observers retain ordinary execution. Production continues to
reject experiment overrides.

## Numerical contract and router repair

Readers preserve FP8 → FP32 scale multiplication → BF16 rounding. Expert
group-16 quantization remains per expert, including unequal input scales.
Shared group-128 quantization crops real rows before its GEMM. The activation,
down projection and reduction order remain unchanged.

The first GPU checks failed the exact router-logit gate, before real-weight
FFN timing. At 8193 rows, the byte-load router changed 1,558,421 FP32 logits,
with maximum absolute error `3.0517578125e-05`; pipeline depths 1, 2 and 3
produced the same differences. The compiler IR identified dot operand
`kWidth=4` for the packet input versus `kWidth=2` for ordinary BF16 input.
Triton's [operand-packing heuristic](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)
traces original load widths when choosing that layout.

`0641070e` reads aligned 16-bit packet words and extracts the original bytes.
Both BF16 dot operands now use the ordinary packing, with no extra BF16
buffer. The compiler probe checks both operand widths, and GPU tests require
exact router logits, route IDs and weights. The tolerance was not relaxed.
The subsequent expert input implementation reads four FP8 values per
transaction and shares their rank address and transport scale, retaining
FP32 multiplication and BF16 rounding for each element.
`57e63935` additionally loads one transport scale per router row and uses a
single pipeline stage. Router shared memory falls from 48 KiB to 16 KiB;
the original BF16 dot packing and exact GPU gates remain unchanged.

## Evidence and limits

All reports below include source hashes. [source_revisions.json](packet_only/source_revisions.json)
checks those hashes against each immutable commit, rather than treating older
CPU or compiler results as measurements of a newer source.

- Packet-only cleanup `a61644a4`: **136 CPU tests passed, 22 CUDA skips** in 15
  isolated Linux modules, and five actual SM121 compiler variants passed.
  [CPU](packet_only/cpu.json), [compiler](packet_only/compile.json).
- Main integration and diagnostics `ff877806`: **146 CPU tests passed, 23
  CUDA skips** in 17 isolated modules, including main's new draft QK and
  common-kernel contracts. [CPU](packet_only/cpu-postmerge.json).
- The rejected byte-load router is retained in
  [GPU v3](packet_only/gpu-v3-failure.json),
  [GPU v4](packet_only/gpu-v4-failure.json) and its
  [diagnostic log](packet_only/gpu-v4-failure.log).
- Aligned router `0641070e`: all **16 GPU unit/transport/router tests** and
  five actual-weight FFN cells passed the numerical gates.
  [GPU v5](packet_only/gpu-v5.json), [compiler](packet_only/compile-aligned.json).
  This version was **2.7–12.2% slower** than its same-run ordinary FFN and is
  not a speed win.
- Vector expert input `fa2050ea`: all numerical gates passed, but complete FFNs
  were still **2.6–24.3% slower** within their own brackets. At 32K, separate
  warm diagnostics measured ordinary/packet router at **2.009/10.977 ms**,
  expert at **44.509/44.726 ms**, shared quantization at **2.002/1.285 ms**,
  and ordinary unpack at **1.914 ms**. These identify the router as the next
  target; they are not an additive latency model.
  [GPU v6](packet_only/gpu-v6.json), [compiler](packet_only/compile-vector.json).

The GPU probe starts at received packets and ends at the L3 FFN sum, using
synthetic activations, actual TP4 rank weights, and a shared FP8 pack prepared
from those weights. Each cell checks exact router/routes/shared Q/S/output
and expert FP4/SFA by `(expert, token)`, including unequal expert scales.
BF16 atomic scatter uses fixed **2% relative-max / 0.4% relative-RMS** component
limits and records ordinary cross-launch variance. Every correctness gate
must pass before a cell's B/A/A/B timing starts. Compilation and warmup are
excluded; device and synchronized wall times are both retained.

This is **one GB10 beside production**, within an 8 GiB probe allocation
limit. Pack/NIC/all-gather/reduce-scatter, serving TTFT, output tok/s,
calibration-store quality and model acceptance are outside the measurement.
Cross-cell and cross-run absolute times must not be compared as speedups.
The 32K warm consumer diagnostics are separate from the complete-FFN bracket.

At 32,256 rows the removed BF16 intermediate is 252 MiB/rank; the received
packet is 126.24609375 MiB and shared Q/S is 129.9375 MiB. These are declared
buffer sizes, not a measured engine peak or memory-traffic saving. Matched
32K/128K C=1/C=4 onepass, decode and quality/acceptance gates remain necessary
before adoption. The earlier **52 ms D8/D32 + 32K complete mixed-FFN target**
was not met and is not redefined as a standalone prefill-kernel target.

Two CI boot-contract tests originally matched the final keyword text and broke
when main added `reasoning_opener`. They now inspect the production `fleet()`
server call by AST, asserting the same lease and retention arguments
independently of keyword order. All **86 tests passed** in the two affected
Linux modules ([record](packet_only/cpu-ci-fix.json)); the initial unscoped AST
check failure is also [retained](packet_only/cpu-ci-unscoped.json).
The [runtime continuity record](packet_only/runtime_continuity.json) applies
to those test fixes before the final main merge. `b29b4083` subsequently
integrates main `77d80b6c`, retaining main's matching AST checks, sampled-draft
agreement, C1 MoE scale preparation and reasoning-opener changes. Its **238
CPU tests passed, 23 CUDA skips** in 20 isolated Linux modules, including the
lease/retention and draft-agreement contracts. Five compiler variants passed.
[Final CPU](packet_only/cpu-final.json), [final compiler](packet_only/compile-final.json).

## Current same-run result

Source `b29b4083`, reservation `st-ffn-packets0914v8`, ticket
`17893689403134005`: **16 GPU tests and all five real-weight numerical cells
passed** after integrating main `77d80b6c`. Both router operands retain
`kWidth=2`; all five compiler variants passed.
[Raw GPU record](packet_only/gpu-v8.json),
[compiler record](packet_only/compile-final.json).

Each median uses eight samples per arm in a B/A/A/B bracket. Negative changes
mean shorter latency:

| Real rows | Ordinary device ms | Packet device ms | Device change | Ordinary wall ms | Packet wall ms | Wall change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,193 | 38.248 | 37.988 | -0.68% | 40.573 | 40.334 | -0.59% |
| 8,194 | 41.005 | 38.034 | -7.24% | 43.337 | 40.370 | -6.85% |
| 8,195 | 38.223 | 37.851 | -0.97% | 40.569 | 40.243 | -0.80% |
| 9,216 | 42.357 | 42.179 | -0.42% | 44.702 | 44.497 | -0.46% |
| 32,768 | 137.279 | 132.456 | -3.51% | 139.368 | 134.822 | -3.26% |

At 32K, the recorded complete FFN decreased by **3.51% device / 3.26% wall**.
All five medians favored packets in this run, but several differences are
below 1%; this shared-GPU bracket does not establish a robust universal win.
The earlier `57e63935` bracket recorded **3.38% device / 3.31% wall** at 32K
([GPU v7](packet_only/gpu-v7.json)); these runs are not pooled or compared
across different ordinary baselines.

In the final run, warm router was 1.948/4.964 ms (ordinary/packet), expert
100.196/100.034 ms, shared quantization 1.967/1.245 ms, and unpack 1.874 ms.
Do not compare those absolute times against earlier runs under different
production load or sum the warm diagnostics into a predicted FFN latency.

Final review added a reference-only fallback at `22c993bc`: selecting the
reference `moe` lane also disables its packet reader before rank agreement.
Its [CPU regression test](packet_only/cpu-reference-fallback.json) passed.
[Structural continuity](packet_only/runtime_continuity_final.json) verifies
that the packet kernels, ordinary binding and all previously measured test
bodies are unchanged; only that fallback and its CPU test were added.
The whole [main-integrated CI run](https://github.com/choiceoh/stkernel/actions/runs/34815435486)
also passed, including engine, onepass and oracle contracts.

Default adoption is still pending full-model quality and serving measurements.

## Reproduction

CPU/compiler image: `sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`
(Torch 2.13.0+cu130, Triton 3.7.1). The source is read-only, CUDA is hidden, and
compiler outputs are outside the checkout:

```sh
CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a PYTHONPATH=. \
  python3 probes/engine_ffn_packets_compile.py --output /out/compile
```

From the immutable candidate checkout, use the canonical **single-GPU** lane:

```sh
ST_IMAGE=sha256:8190d08e822e1f9d18dda5a127a5d9a9e8c53ef4b5136d1154f9ea4b7727f1ed \
ST_PROBE_TREE=st-packet-final-b29b4083 \
  bash bench/fleet.sh run --gpu --detach st-ffn-packets0914v8 20 \
    'Packet consumer full FFN qualification' -- \
    bash probes/run_engine_probe.sh probes/engine_ffn_packets_check.py \
      --ranks /path/to/exact/st-ranks --ckpt-meta /path/to/exact/st-ranks \
      --samples 8 --output /cache/ffn-packets0914v8.json
```

Earlier [CPU](cpu.json) and [compiler](compile.json) records describe their
own frozen revisions. The old `st-ffn-packets0914v2` reservation, ticket
`17893163563368864`, ended after a **720-minute queue timeout before GPU
execution**. Its [admission receipt](admission.json) is not a numerical result.
