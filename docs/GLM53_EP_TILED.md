# EP tile-major decode and prefill

This candidate retains EP4 expert ownership and shares one tile-major weight
allocation between a dedicated static decode kernel and the EP-local prefill
kernel. It is experimental and defaults off. Neither recovered decode speed
nor retained prefill gains have been established for this implementation.

The first full-model SF6 run completed on 2026-09-09. Its actual-weight
canaries and SF6 release passed on all four ranks, but every arm failed the
existing Korean gate. The candidate also had lower observed fixed-decode
throughput than the second baseline. It is not ready for default adoption.

The later A-ring follow-up also missed the user-selected absolute decode
target of 67 tok/s: TP B measured 71.3369 tok/s and EP A measured 61.3553
tok/s over three complete fixed-1024 requests. B passed Korean 0/8; A failed
1/8. Both passed facts 18/18. A's engine throughput was 18.2028 step/s,
versus 18.2190 in the earlier EP run, so sharing FC1 activation transfers
did not demonstrate a decode gain. The target does not waive quality gates
or establish relative non-regression against TP.

Enable `ENABLE_EP=1 VLLM_GLM53_EP_TILED=1 VLLM_GLM53_TP_SF6_Q0=0` on the GLM
profile, retaining its `t,r,sf6` scale-compression setting. The TP Q0 owner/canary does not apply to EP weights. Keep the old
`VLLM_GLM53_EP_PREFILL_LOCAL`, `VLLM_B12X_EP_ZERO_WEIGHT_MICRO`, and
`VLLM_B12X_EP_WARM_COMPACT` experiments off. The new flag also preserves the
EP-compatible attention/MHC prefill SP configuration. Attention remains TP4.

## Implementation

- Exact SM121, BF16 activations, E72/H4096/I2048/top8, unpadded local experts,
  SwiGLU `(1,0,10)`, and maximum batch capacity 8192..16384.
- Weight loading permutes NVFP4 bytes in place into the TP v5 tile-major
  layout. Both kernels alias those same bytes; no second model weight copy
  or request-time EP-to-TP redistribution is introduced.
- Native M1..32 uses the EP static kernel, retaining the v5 TMA descriptors
  and v4 FC1/activation/FC2 pipeline. Remote IDs and zero route weights are
  discarded before expert indexing. BF16-rounded contributions accumulate
  in FP32 and are converted once to the caller's BF16 output.
- M33..capacity uses EP-local M128 prefill with the existing route/Q0/task
  publication and FP32 scatter. Its tile-major weights and direct SF6 scale loaders share the existing compute body.
- All M1..32 static compiler keys and the runtime-shaped prefill kernel are
  prepared before inference. One-launch remapping handles both ranges.
- Both decode geometries and prefill read the same lossless SF6 scale owner.
  Each 2048-byte scale stage occupies 1552 bytes (24.22% less scale storage,
  not total model memory). Existing on-device packing verifies every byte by
  roundtrip; an unrepresentable plane refuses this experimental owner.
- SF6 M1..8 restores four scale bytes per integer word. Low-nibble and
  high-two-bit fields are spread into separate byte lanes; the base addition
  preserves modulo-256 output without carries between lanes. Volatile reads,
  all 128 threads' byte ownership and both expansion barriers remain the same.
  Raw scales and M9..32 retain the inherited restoration path. The selected
  word-unpack specialization has its own 18-field cache key, required by the
  startup canary and compiler receipt.
- Original scale Parameters and loader aliases survive the startup canary.
  The final model hook releases them only after the full checkpoint walk and
  reseals the owner generation. Inference cannot start with an unfinished
  release, and no full-size scale decompression buffer is retained.
- Prepared owners reject changed weights/scales, unsupported geometry,
  overlapping output storage, and conflicting forced backends. They cannot
  fall back to a row-major kernel after relayout.

Rank imbalance, sparse-row padding, and collective costs remain. The proposed
gain comes from the changed weight access and decode implementation; there
is no justified numerical speedup estimate before measurement.

## Validation and decision

The startup canary collects independent stock compact outputs on the first
actual layer before relayout, then checks the new owner on the same inputs.
It retains the existing numerical limits and stock-repeat control checks.
It includes short decode, small/large prefill, concentrated and remote routes,
changed input contents at fixed addresses, and current/side-stream graph
replay. It additionally binds both actual packed plane hashes/addresses and
requires the SF6 kernel cache keys. The final release receipt records actual
raw and packed byte counts separately from numerical acceptance.
Failure prevents readiness and cannot be cleared by calling weight
finalization again. The canary covers one actual layer per rank, not all model
layers or sanitizer acceptance.

CPU lowering uses `probes/run_glm53_ep_tiled_cpu.py` through normal fleet
`--cpu` admission, an immutable serving image and CUDA bindings capsule, a
no-device/no-network container, 4 GiB memory and 2 CPU limits, and the existing
12 GiB available-memory guard. Its result is compilation evidence only.

The decision run must compare same-source TP and EP-tiled arms with matched
capacity/runtime settings, warm cache classification, the original quality
checks, and direct 2K/32K/128K prefill TTFT/tok/s plus fixed-1024 decode tok/s.
An old TP baseline or a component timing cannot establish non-regression.
Default adoption remains conditional on both retained prefill improvement
and resolved decode regression.

## First SF6 onepass result (2026-09-09)

Normal fleet session `eptiledsf60909v1` ran B0, B1, then A from frozen source
`f6b0934eb3d14b46cc58c29f6c9983f776eed250`, using the same immutable image,
capacity, request hashes and direct consumer workload. B0 was compile-cold;
B1 and A were warm. Baselines kept TP4 and TP SF6 Q0 enabled. A used the
three candidate flags above. These are observations from failed quality
arms, not an accepted improvement comparison:

| Metric | B0 | B1 | EP tiled + SF6 A |
|---|---:|---:|---:|
| 2K best-warm prefill tok/s | 2432.31 | 2424.96 | 2805.85 |
| 2K best-warm TTFT, s | 0.875 | 0.878 | 0.758 |
| 32K prefill tok/s | 3071.11 | 2998.01 | 3277.65 |
| 32K TTFT, s | 10.597 | 10.856 | 9.929 |
| 128K prefill tok/s | 3136.95 | 3124.71 | 3267.98 |
| 128K TTFT, s | 40.982 | 41.143 | 39.339 |
| Fixed-1024 pooled decode tok/s | 58.1072 | 69.7002 | 64.4654 |
| Fixed-window pooled engine step/s | 14.4213 | 20.4549 | 18.2190 |
| Fact checks | 18/18 | 18/18 | 18/18 |
| Korean-dirty responses | 1/8 | 1/8 | 1/8 |

Actual prompt lengths are 2121/2128/2128 for the three 2K questions and
32545/128559 for the single combined 32K/128K requests. Pooled decode is
`sum(completion_tokens - 1) / sum(decode_s)` over three complete 1024-token
requests. A's repetitions were 60.8794/77.9237/57.8784 tok/s, with stable
18.28/18.16/18.20 engine step/s. B1's repetitions were
70.9865/70.3075/67.8839 tok/s. B0 slowed within its fixed-decode window and
must not be used to manufacture a candidate win. No fixed-window speculative
acceptance counter was collected, so output-rate variation cannot be
attributed to acceptance alone.

All three Korean failures were two CJK characters in `Halvorsen博士` in the
reasoning channel of one fixed-decode response. No content-channel output
was produced for those responses. This shared symptom does not establish
kernel corruption or clean final-answer quality. The original judge stopped
on A's Korean failure (exit 4); no extra baseline or acceptance was produced.

All four ranks passed nine actual-weight canary cases and 54 candidate
comparisons each, including changed-input graph/side-stream replay. This
covers the first actual layer per rank, not every model layer or sanitizer
acceptance. Every rank finalized 42 SF6 layers: 4,756,340,736 raw scale bytes
were replaced by 3,604,414,464 packed bytes, saving 1,151,926,272 bytes
(1.073 GiB/rank, 4.291 GiB total). Both TP baselines already used SF6; this is
EP packing versus its own uncompressed scales, not extra savings over TP.

[Original onepass records and four-rank evidence](../measurements/glm53_ep_tiled_20260909/onepass1/README.md)
preserve the failed gates and distinguish payload completion from service
recovery. The remaining decode cost requires another bounded implementation
and same-runtime consumer validation before this path can become a default.

## A-ring follow-up (2026-09-09)

Session `eptiledring0909v2` first failed before A readiness because its
startup canary still required a 16-field cache key while SF6 M1..8 selected
the new 17-field A-ring artifact. Each rank passed its first eager numerical
comparison before the exact-key check stopped startup. The correction keeps
the numerical gates and uses the real compiler key construction in the CPU
regression fixture across M1..32. The failed attempt and its B record remain
in [ring_onepass2](../measurements/glm53_ep_tiled_20260909/ring_onepass2/README.md).

The corrected frozen source `57914a3f8bb01a76a20099b9c2605be3ea15b7f4`
passed 101 CPU tests and six actual no-device lowerings, then ran the normal
B-to-A onepass `eptiledring0909v3`. The new cache key, all nine actual-weight
cases and 54 candidate comparisons per rank, graph replay and SF6 finalization
passed on all four ranks. These checks cover the first actual layer per rank;
every layer separately completes SF6 roundtrip and finalization.

| Metric | TP + SF6 B | EP + SF6 A-ring A |
|---|---:|---:|
| 2K best-warm prefill tok/s / TTFT | 2426.78 / 0.877 s | 2983.94 / 0.713 s |
| 32K prefill tok/s / TTFT | 3059.88 / 10.636 s | 3241.81 / 10.039 s |
| 128K prefill tok/s / TTFT | 3137.83 / 40.971 s | 3259.17 / 39.445 s |
| Fixed-1024 pooled decode tok/s | 71.3369 | 61.3553 |
| Fixed-window pooled engine step/s | 20.4551 | 18.2028 |
| Fact checks | 18/18 | 18/18 |
| Korean-dirty responses | 0/8 | 1/8 |

Decode repetitions were B 70.1128/71.6497/72.2834 and A
60.0517/63.4765/60.6452 tok/s. Pooled decode uses all 3069 timed output tokens
divided by all three decode durations. B carries `cold_compile=true`; A does
not. Prefill values are descriptive and do not establish a matched warm
full-model speedup. The chain retained the Korean failure and exited 4.
The 67 tok/s target was not achieved; defaults remain TP.
