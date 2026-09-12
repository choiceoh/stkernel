# TP SF6 Q0 prefill

> 그대로 두는 기록 — **캠페인 산출물.** 당시의 기록이라 고치지 않는다 — 고치면 기록이 거짓이 된다.

The default choice uses TP Q0 producer fusion while retaining TP decode.
The [campaign record](GLM53_EP_PREFILL_LOCAL.md) reports the completed
onepass25/27 measurements and their unresolved canonical performance verdicts.
This choice is not a claim of statistical noninferiority, a confirmed prefill
win or 40% improvement.

## Runtime contract

`MoEGatedDynamicKernelSF6Q0` overrides only
`initialize_route_q0_and_publish` in the existing SF6 dynamic backend. The
exact gate is SM121, E288/H4096/I512, top8, M128, NVFP4 tile-major packed SF6,
unshared SwiGLU-OAI (1,0,10), and 4096..8192 executed token rows. Other calls
keep their existing path. Static decode and packed weight layout are unchanged.
This source property does not guarantee unchanged end-to-end latency.

Expert-scale, selected-route equality and scale-offset caches are local to
the invocation. All eight routes, including zero weights, preserve row
reservations and token mapping. Stock BF16 zeroing/quantization, task splitting,
publication, synchronization and communication remain. Existing shared backing
holds the four-row stage and routing caches; no extra global workspace or
second model weight copy is introduced. The candidate alone adds the dynamic
cache suffix `glm53_tp_sf6_q0_v1`.

## Profile and rollback

| Setting | Selected default | Rollback |
|---|---|---|
| `VLLM_GLM53_TP_SF6_Q0` | `1` | `0`: stock SF6 Q0, same TP decode |
| `VLLM_GLM53_STARTUP_TRIM` | `1` | `0`: omit the one-time post-warmup trim |
| `VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE` | `1` | `0`: retain the unused graph-memory estimate |

EP/local/compact-warmup/zero-weight and stock-topk micro remain 0. The other
prefill experiments remain 0, and static TP stays `t,r,sf6`. Common prefill
SP1/FP8-v3/min4096, image4/video0, graph/capacity/prefix/spec settings remain.
The image4/video0 policy is already main PR #508, not a Q0 benefit.

The skip flag affects only the V2 GLM5-next dry estimate when it would not be
applied; real model profiling and graph capture remain. Trim synchronizes,
collects garbage, releases inactive allocator cache and calls `malloc_trim`
once after final warmup. Live owners remain; the first request may allocate
again. Observed reclamation was MiB-scale, not a multi-GiB headroom claim.

For the next normal authorized launch, `VLLM_GLM53_TP_SF6_Q0=0` is the focused
rollback. To restore all three pre-adoption settings use
`VLLM_GLM53_TP_SF6_Q0=0 VLLM_GLM53_STARTUP_TRIM=0 VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=0`.
Disabling the common startup controls also changes startup conditions. These
settings do not themselves mutate an existing service or submit a fleet job.

## Evidence limits

Onepass27 B→A completed with fixed-output decode 67.4398→69.2342 tok/s
(+2.66%), quality 18/18 and Korean 0/8 on both arms. Its original judge remains
`incomplete/unresolved` with one baseline boot and no noise floor. Candidate
repetitions span 61.87..79.66 tok/s. Prior onepass25 was −6.69% against pooled
quality-valid B0/B1/B3 and its canonical result was inconclusive. A positive
latest pair does not erase the earlier negative result or establish no loss.

Descriptive onepass27 input rates changed 2K warm −3.64%, 32K+1.44%, 128K+0.85%.
The baseline's `cold_compile` flag makes it incompatible for canonical prefill
acceptance, even though decode is compatible. Two warm 2K samples and one
32K/128K request per arm do not establish a stable prefill gain.

Startup canary coverage is one first eligible actual-weight layer per rank,
with four fixtures and eager/two graph-stream comparisons, not every layer
or a sanitizer suite. Actual-model launch markers can precede HTTP requests.
The strict snapshots and canonical request records establish separate runtime
and serving evidence; native channel diagnostics do not waive any quality gate.

CPU24's 165 tests/30 compiled kernels remain the original receipt. The default
profile is the sole changed file among 37 contract sources; its two new loader
tests are separately scoped. All 19 mounted sources remain identical. GPU27
ran frozen `ea413ac4c39ba3e6e4009c73587b0d536053b4bf`; this is not a claim
that the later default commit or public serving runtime was GPU-tested.
