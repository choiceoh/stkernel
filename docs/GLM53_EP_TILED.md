# EP tile-major decode and prefill

The GLM53 default retains EP4 expert ownership and shares one tile-major weight
allocation between a dedicated static decode kernel and the EP-local prefill
kernel. SF6 scale compression and verified input preparation remain enabled,
with DFlash K5. The accepted absolute decode target is 67 tok/s; this does
not mean EP has matched TP decode speed.

## Default acceptance (2026-09-10)

Normal fleet session `eptiledprep0910v8`, frozen source
`388aabdd79e874808262e15c0b974dc8ccca6f07`, measured **72.62743 tok/s**
for EP across three complete fixed-1024 requests (72.33144 / 81.65081 /
65.64184 tok/s). The acceptance contract uses the pooled rate, not the
fastest repetition or a minimum per-request rate. TP measured 80.49311 tok/s;
EP remains 9.8% below it. Both arms passed facts 18/18, Korean 0/8 and all
required serving proof (TP 1/1, EP 2/2). The accepted user-selected absolute
target does not establish relative non-regression or statistical superiority.

| Prefill observation | TP B | EP A |
|---|---:|---:|
| 2K best-warm TTFT / tok/s | 0.838 s / 2539.12 | 0.708 s / 3004.23 |
| 32K single-request TTFT / tok/s | 10.673 s / 3049.31 | 10.060 s / 3235.21 |
| 128K single-request TTFT / tok/s | 41.168 s / 3122.79 | 39.443 s / 3259.40 |

B retains `cold_compile=true`; A does not. The original prefill compatibility
rule rejects this as a matched full-warm prefill comparison. The long-context
values are individual observations, not an accepted steady-state speedup.
The original decode judge also remains incomplete with one baseline sample.
Neither limitation is erased by default adoption under the absolute target.

The repaired workload evidence includes 26 fresh TP and 27 fresh EP
preparation checkpoints, all zero drift. The final logged totals were
1664 steps / 27 checks for TP and 1728 / 28 for EP. Startup checks were
excluded from these fresh checkpoint counts. All four EP ranks passed the
12-case actual-weight canary, changed-input graph replay and packed-only
SF6 finalization of 42 layers. CPU admission passed 158 tests and 19 actual
no-device lowerings; the default profile and explicit TP rollback passed
14 additional focused tests.

The profile changes only `ENABLE_EP=1`, `VLLM_GLM53_EP_TILED=1` and
`VLLM_GLM53_TP_SF6_Q0=0`. Restore TP explicitly with
`ENABLE_EP=0 VLLM_GLM53_EP_TILED=0 VLLM_GLM53_TP_SF6_Q0=1 SPEC_K=5`.
The original [v8 records and verdicts](../measurements/glm53_ep_tiled_20260909/prep_onepass8/README.md)
and [CPU admission](../measurements/glm53_ep_tiled_20260909/prep_cpu10/README.md)
remain separate from the subsequent production recovery verification.

## Follow-up: EP decode target 76 tok/s

`VLLM_GLM53_EP_DECODE_OPT=1` is an experimental, separately keyed extension
to the accepted EP path. Its default remains 0 after the first candidate
missed the 76 tok/s target.
Both comparison arms retain EP4, SF6, K5 and verified preparation; the
candidate adds only this switch. The new absolute target is pooled
fixed-1024 x3 decode >=76 tok/s, with the existing quality and direct
2K/32K/128K prefill measurements. A historical 72.63 result does not replace
the same-source EP baseline. Original cold-compile classifications and
single-request long-prefill limitations remain visible.

For SF6 M1..8, FC2 reads its packed scales from two separate shared-memory
slots. Restored scale bytes therefore cannot overwrite another warp's
unread packed bytes, removing one expansion barrier per FC2 stage. The
post-expansion barrier and all producer/consumer releases remain. The
scatter metadata cache uses its actual 16 rows instead of 128. This both
removes unused stores and keeps dynamic shared storage at 100352 bytes;
the compiler receipt must additionally verify static shared allocation
<=1024 bytes and total <=101376 bytes. Baseline keys and M9+/prefill paths
retain their preceding behavior. Arithmetic and lossless SF6 bytes are
unchanged.

Preparation keeps its first-use and every-64-step stock comparison. On
the candidate's C=1/K5 verification steps it compares the cloned fused
preimage directly with live stock views, eliminating only the second
snapshot clone. Successful verification emits fresh workload checkpoints;
turning the switch on alone does not prove this optimization ran.

Both arms explicitly require EP and PREP proof even when their knob delta
is empty. Candidate proof additionally requires the new native cache keys,
unchanged large-shape selection and fresh verified clone-elimination
checkpoints. The CPU gate includes the baseline's 19 lowerings plus four
optimized local/global lowerings.

### First EP76 result (2026-09-10)

Frozen source `b41d0da24059379d41c079626cc67c3e83cd14e3`, normal fleet
session `epdecode76onepass0910v1`, completed the same-source EP B→A pair.
The candidate measured **71.78687 tok/s**, versus baseline **72.76702 tok/s**
(-1.34697%), and did not meet the absolute 76 target. The original judge
remains `incomplete`/`unresolved` with one baseline; this observation does
not establish a statistically significant regression or improvement.
The candidate is not adopted. The accepted EP default and v8 evidence above
remain unchanged.

| Observation | EP baseline B | EP candidate A |
|---|---:|---:|
| Fixed-1024 pooled decode tok/s | 72.76702 | 71.78687 |
| Fixed-window pooled engine step/s | 19.89587 | 18.72490 |
| 2K best-warm TTFT / input tok/s | 0.709 s / 3001.52 | 0.749 s / 2841.01 |
| 32K single-request TTFT / input tok/s | 10.070 s / 3231.86 | 9.947 s / 3271.83 |
| 128K single-request TTFT / input tok/s | 40.178 s / 3199.74 | 39.244 s / 3275.87 |
| Facts / Korean-dirty responses | 18/18 / 0/8 | 18/18 / 0/8 |
| Required serving proof | 2/2 EP/PREP | 3/3 EP/PREP/OPT |

B's three fixed requests measured 76.38589/76.49326/66.38786 tok/s;
A's measured 72.44182/72.30024/70.64654. Each pooled rate uses all 3069
timed output tokens. The engine rate uses the separately recorded fixed
metric windows. Whole-onepass acceptance counters are not those same windows.

All ordered request hashes match. The 32K/128K and all three fixed output
hashes differ across arms; long-request completion lengths were B692/A1200
and B1200/A474 respectively. Fixed requests all completed exactly 1024
tokens. B retains `cold_compile=true`; A has no such field. The long-context
values each come from one request and do not establish matched warm-prefill
superiority. Neither these differences nor the lower engine rate identify
the cause of the observed throughput difference by themselves.

Both arms passed all four ranks' 12-case actual-weight canary and SF6
finalization. Native M4/6/8 used baseline global key23 or candidate key24;
M12/16/24/32 retained key20. Fresh PREP checkpoints numbered B26/A25 with
zero drift. A also recorded 25 verified clone-elimination checkpoints,
ending at fused step1600; these prove execution, not performance improvement.

The first CPU prerequisite failed at source `680a899d` with four existing
test-fixture namespace errors for `_report_decode_opt`: 181 tests, zero
failures and zero skips. Its 23 compiled artifacts remain failed-gate
evidence, and no final runtime recheck is claimed. CPU2 at the measured
source passed all 181 tests and 23 lowerings, with CUDA uninitialized and
the final runtime identity rechecked. [CPU1 failure](../measurements/glm53_ep_tiled_20260909/ep76_cpu1_failed/README.md),
[CPU2 pass](../measurements/glm53_ep_tiled_20260909/ep76_cpu2/README.md),
and [original onepass records and verdicts](../measurements/glm53_ep_tiled_20260909/ep76_onepass1/README.md)
preserve the separate outcomes. An FC1 follow-up is being prepared and has
not yet been measured.

## Implementation and preceding experiments

The combined candidate keeps DFlash K5 and SF6. Native M1..32 maps
global expert IDs inside the existing route-publication kernel, removing the
separate remap launch and its temporary-plane traffic. Dynamic prefill keeps
the original remapper. The local-ID reference entry is retained; global
entries use a distinct cache namespace with the actual map/ID metadata.
No MMA, barrier, TMA layout or scatter arithmetic is changed by route fusion.

The same candidate repairs PREP_FUSED's reviewed KV-zero overlay pin. The
image's original preimage remains pinned separately. Captured runner source
confirms that block zeroing runs in `update_requests`, before input/attention
preparation. Every armed plan is compared with the original preparation
chain on first use and then every 64 fused steps. DISARM remains sticky even
after capture or wake.

The consolidated onepass reservation first ran EP with full shadow
comparison, then same-source TP and EP with preparation armed. Every arm
requires actual same-boot preparation execution proof; defaults keep
`knobs={}` and carry an independent `required_proofs` declaration. Missing
proof or drift rejects a row and excludes it from baseline reuse. Sparse log
checkpoints are not fixed-request counters. The absolute goal remains pooled
fixed-1024 x3 decode >=67 tok/s with existing quality and prefill gates.

Session `eptiledprep0910v7` completed on frozen source
`96a8cab45a1eb049298315a77ed881ee74f04f9b`. Armed EP measured
**77.09304 tok/s**, with three complete repetitions at
77.83371 / 75.66344 / 77.82287 tok/s, facts 18/18 and Korean 0/8.
Its observed absolute throughput exceeds 67, but the canonical verdict is
**UNPROVED**, so defaults remain TP. The preselected TP baseline measured
76.49247 tok/s and failed Korean 1/8; it must not be treated as an accepted
baseline. Neither armed row passed the new preparation evidence gate.

| Metric | TP + SF6 + preparation B | EP + SF6 + preparation A |
|---|---:|---:|
| 2K best-warm prefill tok/s / TTFT | 2519.84 / 0.844 s | 3004.64 / 0.708 s |
| 32K prefill tok/s / TTFT | 3046.68 / 10.682 s | 3259.87 / 9.984 s |
| 128K prefill tok/s / TTFT | 3135.55 / 41.000 s | 3264.50 / 39.381 s |
| Fixed-1024 pooled decode tok/s | 76.49247 | 77.09304 |
| Fixed-window pooled engine step/s | 22.33787 | 19.72401 |
| Facts / Korean-dirty responses | 18/18 / 1/8 | 18/18 / 0/8 |
| Required serving proof | 0/1 | 1/2 |

The preparation evidence failure is a logging integration defect. Armed
preparation checks its first step and every 64 steps, but previously logged
only its first successful check and each 64th successful check (step 4032).
The sole successful checkpoint in each armed boot predates onepass. The
proof correctly refuses to substitute startup evidence for workload evidence.
Shadow S passed with 1808 successful comparisons and zero drift at its last
logged checkpoint, facts 18/18, Korean 0/8 and proof 2/2. Its 58.30594 tok/s
includes full shadow comparison and is excluded from performance comparison.
All four ranks passed startup source, numerical and SF6 checks in each arm;
completed-workload preparation counters were not captured on all four ranks.

The follow-up changes only successful armed-check logging and rejection
diagnostics, retaining actual check cadence, kernel arithmetic, K5 and SF6.
It requires another normal TP/EP onepass with fresh preparation checkpoints.
The original v7 rejection and failed TP quality result remain preserved in
[the complete v7 evidence](../measurements/glm53_ep_tiled_20260909/prep_onepass7/README.md).

The first full-model SF6 run completed on 2026-09-09. Its actual-weight
canaries and SF6 release passed on all four ranks, but every arm failed the
existing Korean gate. The candidate also had lower observed fixed-decode
throughput than the second baseline. It is not ready for default adoption.

The preceding BF16-scatter follow-up missed the user-selected absolute decode
target of 67 tok/s: warm TP B1 measured 72.9244 tok/s and EP A measured
59.8872 tok/s over three complete fixed-1024 requests. Both passed facts
18/18 and Korean 0/8. The matched warm prefill observations favored A, but
decode was 17.88% below B1. A's engine throughput was 18.4447 step/s versus
18.2126 in the preceding EP run; that 1.27% cross-run difference does not
establish a stable optimization gain. The target does not waive quality
gates or establish relative non-regression against TP.

The bounded K3 configuration comparison completed from frozen source
`6977199cf699f82696f925f77bc930d500135532` in fleet session
`eptiledk30910v6`: TP + SF6 with DFlash K5 (B0 preparation, B1 comparison),
then EP + SF6 with DFlash K3 (A). A measured **60.85548 tok/s**, below the
absolute 67 target, with 21.03647 engine step/s, facts 18/18, Korean 0/8 and
proof 2/2. Its three requests measured 60.36829/60.31594/61.90889 tok/s.
The kernel bytes were unchanged from the BF16-scatter run; K3 changed the
complete drafting/verification configuration. Faster engine steps did not
restore the required output throughput.

B1 measured 73.73555 tok/s but failed Korean 1/8. Both B0 and B1 failed that
gate, so the existing chain automatically added ABASE, which measured
74.31471 tok/s with facts 18/18 and Korean 0/8. The original verdict remains
incomplete/unresolved with only one eligible baseline and a -18.1% candidate
delta. The preselected B1 failure is retained. A's 2K/32K/128K warm prefill
observations were 2854.08/3271.27/3187.48 tok/s, but its first 2K request
was slower than B1. These observations do not establish accepted improvement.

Normal CPU admission passed 119 tests and nine real lowerings. All four
candidate ranks passed 12 actual-weight cases, 72 candidate plus 72 control
comparisons, graph replay and SF6 release. The onepass proof binds actual
DFlash K3 argv to the same boot's whole-workload positional counter deltas;
these are not fixed-window acceptance counts. Defaults remain K5/TP.
[All four original arms and validation scope](../measurements/glm53_ep_tiled_20260909/k3_onepass6/README.md).

Read-only follow-up found an independent default integration conflict:
`VLLM_GLM53_PREP_FUSED=1` is configured, but all v5/v6 arms log DISARM because
its whole-file guard expects the image's worker-utils SHA `3dcd6ad34ee1…`.
The deliberately mounted KV-zero safety overlay has SHA `fd27b906f336…`.
Removing exactly its KV-zero additions reconstructs the expected original
SHA; other utilities are unchanged. The new integration repair described
above confirms the runner order and updates only the reviewed installed-source
pin. This feature was not rearmed during v5/v6. Its historical benefit is not
a current performance result.

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
  discarded before expert indexing. The current M1..8 SF6 candidate scatters
  BF16-rounded contributions directly into BF16 output with the stock vector
  atomic. Raw scales and M9..32 retain FP32 accumulation and the final BF16
  copy. All modes retain the existing publication barriers.
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
  word-unpack specialization used an 18-field cache key in the measured run.
  The new direct-BF16 specialization additionally changes the output ABI field
  and appends its own tag, forming a 19-field key required by the startup
  canary and compiler receipt. Other native modes retain their 16-field keys.
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
For the acceptance decision, preserve the user's absolute decode target and
the explicit prefill comparison limits documented above.

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

## Word-parallel SF6 follow-up (2026-09-09)

Frozen source `549fd55319c3379438c3cb99694df9f28c01aefb` passed 107 CPU
contracts and six actual no-device lowerings. The M6 PTX scale-restoration
interval changed from 130 to 80 integer/address instructions, with four
stores, 123 registers and zero stack/local storage unchanged. Those static
counts did not translate into a demonstrated engine throughput gain.

Normal fleet session `eptiledword0909v4` completed a same-source B-to-A
direct onepass. All four A ranks passed nine actual-weight canary cases and
54 candidate comparisons each, including changed-input and graph replay.
The selected M6 key had the required 18 fields; M12/24/32 retained 16.
All four ranks completed SF6 finalization before readiness.

| Metric | TP + SF6 B | EP + SF6 word-unpack A |
|---|---:|---:|
| 2K best-warm prefill tok/s / TTFT | 2557.19 / 0.832 s | 2804.52 / 0.759 s |
| 32K prefill tok/s / TTFT | 2983.97 / 10.907 s | 3290.47 / 9.891 s |
| 128K prefill tok/s / TTFT | 3134.64 / 41.012 s | 3261.84 / 39.413 s |
| Fixed-1024 pooled decode tok/s | 69.8109 | 66.3037 |
| Fixed-window pooled engine step/s | 20.4737 | 18.2126 |
| Fact checks | 18/18 | 18/18 |
| Korean-dirty responses | 0/8 | 1/8 |

Decode repetitions were B 67.5897/70.5231/71.4368 and A
62.8145/65.6278/70.9773 tok/s. The pooled result includes all three complete
requests (3069 timed output tokens); the fastest repetition does not meet
the pooled acceptance contract. B carries `cold_compile=true`; A does not.
Prefill timings are descriptive, not a matched warm speedup verdict.
The original quality gate failed on A's first fixed-decode repetition and
the completed chain retained exit 4. Holder release completed at
23:49:46 KST. Defaults remain TP.

[Word-unpack CPU evidence](../measurements/glm53_ep_tiled_20260909/word_cpu4/README.md)
and [direct onepass evidence](../measurements/glm53_ep_tiled_20260909/word_onepass4/README.md)
preserve the source identity, full measurements and failed gates.

## Native BF16 scatter candidate (2026-09-10)

The BF16-scatter candidate limits direct BF16 output to SF6 M1..8.
It uses the stock `scatter_add_v4_bf16x2` helper and keeps each contribution's
existing `satfinite` BF16 rounding. It removes the subsequent eight BF16-to-FP32
conversions, the extra FP32 vector reduction and the final output copy.
Collective input was already BF16; this does not reduce communication bytes.
Output must be 16-byte aligned and disjoint from all owned workspace buffers
before remapping or kernel admission. Larger native batches and prefill
retain their existing FP32 accumulation path.

The numerical contract is unchanged. Stock compact references produce BF16
pair outputs and then sum tokens; direct scatter changes summation order
and repeatedly rounds atomic additions. It therefore cannot claim bitwise
equivalence from the code change alone.

Frozen source `e88f5fd368ef8c895000496101fc7fcf8e7fb344` passed 109 CPU
contracts and six actual lowerings, including the installed stock helper's
complete source SHA and M6's actual BF16 output ABI. Normal fleet session
`eptiledbf160910v5` then ran B0, B1 and A. B0 was the preselected compile
preparation arm; B1 and A both had no `cold_compile` field. All arms retained
the same image, capacity and fixed-1024 x3 workload. B1/A is the comparison.

| Metric | TP + SF6 B1 | EP + SF6 BF16-scatter A |
|---|---:|---:|
| 2K best-warm prefill tok/s / TTFT | 2429.39 / 0.876 s | 3013.56 / 0.706 s |
| 32K prefill tok/s / TTFT | 3075.14 / 10.583 s | 3133.60 / 10.386 s |
| 128K prefill tok/s / TTFT | 3142.21 / 40.914 s | 3246.81 / 39.595 s |
| Fixed-1024 pooled decode tok/s | 72.9244 | 59.8872 |
| Fixed-window pooled engine step/s | 20.5276 | 18.4447 |
| Fact checks | 18/18 | 18/18 |
| Korean-dirty responses | 0/8 | 0/8 |

B1 repetitions were 70.5244/78.9832/69.9394 tok/s; A repetitions were
59.9839/59.9423/59.7361. All six responses completed 1024 tokens, with the
pooled rate using all 3069 timed output tokens. B0 measured 79.0671 tok/s
with the same quality results; it remains preparation rather than a selected
comparison. There is no fixed-window per-position speculative-acceptance
counter, so its whole-onepass counters cannot explain the fixed-window rates.

All four A ranks passed nine actual-weight cases and 54 candidate plus 54
stock-control comparisons, with zero bad rows, the exact BF16 key19 for M6,
FP32 key16 for M12/24/32, packed-scale identity, graph replay and SF6 finalization
of 42 layers. This covers the first numerical layer per rank, not every model
layer. The onepass and supervisor exited 0 and released the holder at
00:36:25.811 KST. Successful execution and quality do not imply the absolute
67 tok/s objective passed: **it failed, and defaults remain TP**.

The measured prefill input-rate differences were +24.05% / +1.90% / +3.33%
at 2K / 32K / 128K; the corresponding TTFT reductions were 19.38% / 1.87% /
3.22%. These are this matched warm pair's observations, not a stable 40%
campaign improvement. [CPU evidence](../measurements/glm53_ep_tiled_20260909/bf16_cpu5/README.md)
and [original onepass evidence](../measurements/glm53_ep_tiled_20260909/bf16_onepass5/README.md)
preserve the checks and failed performance target.
