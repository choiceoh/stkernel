# Adopt SF6 direct reads and raw-scale release by default

On 2026-09-09 the operator requested "기본값으로 도입해" and
"그리고 원본 날려서 메모리까지 줄이고" after reviewing the candidate and clean
baseline measurements. The GLM53 profile now defaults to
`VLLM_GLM53_B12X_STATIC_V2=t,r,sf6`.

Eligible MoE layers pack both FP8 scale planes losslessly and use packed scales
for decode, static prefill and dynamic prefill. Before the first model forward
and profiling/KV sizing, the existing owner finalizer clears registered raw
scale Parameters, quantization descriptor scales, MMA views and known raw-cache
aliases. There is no global raw reconstruction buffer. A packed-only owner
cannot return to a raw-scale kernel or regenerate the original scales on later
calls. Layers that cannot represent both planes losslessly, or have an
unsupported backend/geometry, keep their original planes.

This adopts the already measured kernel and owner implementation without
changing its bytes. The operator's explicit decision supersedes the earlier
hold-off on default promotion. Historical INVALID or incomplete measurements
are preserved; this is not a newly validated SF6-only speedup claim.

## Memory accounting

The candidate's four ranks each observed 42 eligible layers, one packed-owner
finalization, zero SF6 fallbacks, and the following tensor sizes:

| Per rank | Bytes | GiB |
|---|---:|---:|
| Original scale allocation released | 4,756,340,736 | 4.429688 |
| Packed scale allocation retained | 3,604,414,464 | 3.356873 |
| Net scale storage reduction from raw-only | 1,151,926,272 | 1.072815 |

Across four ranks, net scale storage arithmetic is **4.291260 GiB less**.
Releasing 4.429688 GiB after packing is not a 4.429688 GiB net saving versus
raw-only: the compressed allocation must be counted. Host MemAvailable and
allocator-reserved memory can differ from live tensor storage; no matched
net-memory benchmark is claimed. This removes in-memory scale references,
not checkpoint files or FP4 model weights.

## Measurement decision

Old SF6 candidate A: 20.281702 step/s, 49.305528 ms/step; isolated baseline B:
20.206613 step/s, 49.488748 ms/step. The raw step difference is 0.371602%,
0.183221 ms/step. Window medians are 19.880271 vs 19.873337 step/s; output
throughput is 61.954794 vs 62.843273 tok/s. Both individual records passed
quality18/18 and Korean0/8 with exclusive own requests. Candidate 2K warm TTFT
was 0.838278 vs baseline0.888751 seconds; 32K and128K were slightly slower.

Old A had fused AR-consumer MHC disabled after a T16 selftest mismatch; new B
passed and captured its T6 consumer. The reason for that gate difference is
unresolved despite identical kernel bytes. Endpoint and serving identities
also differ, so the original A/B remains INVALID and this cross-run comparison
cannot establish SF6's isolated performance effect. No further GPU measurement
is added for this requested default promotion.

See [completed baseline](../glm53_sf6_baseline_retry_20260909/README.md),
[original candidate evidence](../glm53_sf6_direct_onepass_20260909/STATUS.md),
and [compiler/owner evidence](../glm53_sf6_direct_prefill_20260909/README.md).

## Change and validation

Only the GLM53 profile default, its existing assertion/audit hashes and current
documentation change for adoption. Before the merge-time main update, all 63 sources matched the measured
commits. Main PR#504 then added boot-phase device-memory stamps; 62/63 remain
byte-identical, with only `deneb_boot_stamps.py` inherited from main changed.
SF6 kernels and owners remain identical. See `source-equivalence.json`.
GLM53 and DSV4 compose successfully; generated files match their module sources. Compact AR and inline RDMA
remain default0. The full source stack includes their dormant implementation,
existing MHC FP32 fallback and SF6 dependencies from PR#498; PR#501 is retargeted
to main for one atomic merge of the already measured source tree.

Owner audit found no missing raw references or required new kernel edits.
Focused owner/dispatcher/pack tests passed29/29 without skips. The initial
macOS system-Python core gate was INCOMPLETE (6863checks and50regressions,
torch unavailable) and is retained in `local-incomplete/`; it is not used as
complete admission evidence. A torch-enabled isolated Python runs the full
core, sensitivity and affected contracts, with final results recorded below.
The prior production-image compiler/installed-wrapper evidence applies to
unchanged source bytes; local CPU checks do not substitute for GPU numerics.

To retain raw scales, restart with `VLLM_GLM53_B12X_STATIC_V2=t,r`. Packed-only
models require a fresh load for that change because original Parameters have
already been released. Profile adoption applies to subsequent normal boots;
the historical experiment checkouts and their recorded runtime are unchanged.

Final CPU gate: **71,159 core assertions, 50 megakernel regressions, 7
sensitivity cases and 54 targeted tests PASS**, no skips. Raw per-suite
logs and counted reports are retained in `cpu-complete/`. Python3.12 with
torch2.14.0 was used for these CPU-only checks. No serving/GPU process was
created. The logic audit also incorporates main PR#504's two boot-memory assertions;
audited helper access graphs and dependency closure are unchanged.

Merge-time main synchronization: PR#504's boot-memory instrumentation and its
two assertions are retained. Final core **71,161 assertions / 50 regressions**
and sensitivity7 pass without skips (`cpu-main-sync/`). The 54 earlier targeted
tests cover unchanged transport/MHC/SF6 kernel and owner sources. This merge
adds no new SF6 timing or net-memory claim.
