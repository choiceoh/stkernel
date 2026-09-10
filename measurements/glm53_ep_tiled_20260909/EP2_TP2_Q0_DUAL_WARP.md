# EP2/TP2 dual-warp Q0 candidate

v9 completed on frozen source `407271d9cf73f5db6192a8410c68f152e67301cb`:
**A72.50972 decode tok/s missed the absolute76 target**, versus B76.54038.
Both passed facts18/18 and Korean0/8. The32K prefill observation improved,
but128K did not; this is not a consistent prefill gain. The canonical judge
remains `incomplete` / `unresolved` with one baseline sample and no noise floor.
The candidate is not adopted; both HYBRID and Q0 public profile defaults remain0.
[CPU12/13 originals](ep76_cpu12_cpu13/README.md) and
[v9 measurements](ep76_onepass9/README.md) preserve the distinct evidence scopes.

The preceding same-source onepass reached 78.93348 decode tok/s with EP2/TP2,
but lost 5.21% / 3.26% prefill throughput versus EP4 at 32K / 128K. This
experiment keeps the EP2/TP2 weight layout and changes only its dynamic Q0
producer. These previous numbers motivate the work; they do not substitute
for the same-source EP2/TP2 baseline below.

`VLLM_GLM53_EP_HYBRID_Q0_DUAL_WARP=1` is opt-in and requires the exact
E144/I1024 tiled SF6 owner with `VLLM_GLM53_EP_HYBRID_TP2=1`. Both flags remain
off in the public profile. The static decode kernels, weight allocation,
FP4 quantization functions, SF6 layout, task publisher and FP32 output scatter
are retained. The previously rejected decode optimization stays off.

The M128/N128 Q0 staging area fits four H4096 BF16 tokens. The baseline uses
one math warp per token and leaves four of its eight math warps idle during
this phase. The candidate uses two warps per token, each processing four of
the token's eight 32-block SF iterations. Only the even warp's lane zero
allocates routes. A CTA barrier outside all token/warp guards publishes its
metadata to both consumers, including partial batches; the asynchronous
input-copy wait and subsequent batch/final fences remain separate.

This adds no shared allocation, global buffer, input copy or kernel launch.
It does add one CTA synchronization per four-token batch. The quantized-row
fanout and its bytes do not decrease: two warps share the work. A speedup is
not implied by the doubled active warp count. CPU13's actual control/candidate
lowerings both report REG168 / STACK112 / LOCAL0 / static SHARED1024.
The CPU12 scoped PTX audit found metadata reads after the common publication
barrier inside the batch loop, and CPU13's audited PTX/cubin bytes, keys,
specializations and resources are identical. This is not a final SASS,
GPU race-freedom or numerical proof; stack size alone is not a spill count.

The owner seals the selected mode before relayout, gives it a separate
workspace/prewarm identity and supplies an explicit bool to the dynamic
dispatcher. The candidate appends `glm53_ep2tp2_q0_dual_warp_v1` to the existing
21-field hybrid dynamic cache key; native keys remain unchanged. Startup
schema2 retains all original numerical/graph/input/scale checks and binds the
candidate's 22-field dynamic keys. A dedicated serving marker is emitted only
after the actual prefill call outside canary/capture.

The no-device CPU gate retains the previous eight lowerings and adds one
E144/I1024 candidate lowering, with explicit false/true selectors for the
hybrid dynamic pair. CPU13 completed **239 tests, 0 failures/errors/skips and
9 fresh lowerings**, with23 mounted/84 contract source hashes, isolated
contracts and CUDA uninitialized. Its actual receipt admits source407271d9;
CPU12's earlier b36dc7f1 receipt remains separate. All27 emitted artifact files
are byte-identical between the runs and stored once with explicit per-run
references. Earlier local core checks remained incomplete because of missing
Torch/skips, even when their child process returned0; they are not relabelled
as the complete image gate. Compiling and host contracts do not establish
GPU numerical correctness or performance.

## GPU attempts

[v8](ep76_onepass8/README.md) completed B only: pooled fixed1024×3 decode
73.77020 tok/s, facts18/18, Korean1/8. Fixed rep1 reasoning included
`Halvorsen博士`; this remains a quality rejection. A was refused before boot
because pinned GMU0.6329 exceeded the fresh preflight limit0.61. There is no A
measurement or matched pair, and copied older A logs are excluded.

The [supplement](ep76_onepass8/supplement/README.md) binds seven real head JIT
warnings to exact canonical prefix/after bytes. No compile warning occurs in
the attributed32K segment, but this does not prove absence of all JIT or
explain its TTFT. The original GMU review's uncertainty is retained. A separate
follow-up on the pinned image source confirms the1056-block override is
applied before maximum-length admission; it does not prove physical headroom
or a successful new boot.

v9 was a fresh matched B/A retry with common GMU0.60, unchanged1056 KV blocks,
max length1048576, max sequences4 and max batch8192. The four-rank source/image,
command and environment audit found only Q0_DUAL_WARP0→1 different. Both arms
retained SF6/K5/PREP1/OPT0/HYBRID1. It ended17:28:30 KST with payload/terminal rc0
and released ownership. Completion and quality/proof success do not satisfy
the separate absolute76 target.

| Metric | EP2/TP2 B, Q0 off | EP2/TP2 A, Q0 on | Observed change |
| --- | ---: | ---: | ---: |
| fixed1024×3 pooled output tok/s | 76.54038 | 72.50972 | −5.2661% |
| fixed-window pooled engine step/s | 21.53830 | 21.41215 | −0.5857% |
| 2K best-warm TTFT / input tok/s | 0.788482s / 2698.86 | 0.784304s / 2713.24 | input +0.5328% |
| 32K single TTFT / input tok/s | 12.370252s / 2630.91 | 11.116860s / 2927.54 | input +11.2747% |
| 128K single TTFT / input tok/s | 40.801576s / 3150.83 | 41.130386s / 3125.65 | input −0.7994% |
| Facts / Korean contamination | 18/18 / 0/8 | 18/18 / 0/8 | Both pass |
| Canonical execution proofs | 3/3 | 4/4 | Both pass |

All three fixed samples are retained: B80.87967/72.55165/76.64205 and
A75.54754/76.10326/66.67989. Pooled output uses3069 post-first tokens divided by
the sum of three decode durations; engine speed is a separate pooled fixed-window
metric. All eight request hashes match and all eight output hashes differ.
Canonical facts/quality pass does not establish exact numerical identity.
The larger output-rate change is not attributed solely to Q0 or native kernel
cost from these measurements.

Both rows omit `cold_compile`; that field is preserved, not reinterpreted as
proof of no JIT. The exact canonical prefix/after head logs report **seven
request-time JIT warnings in each arm**: six in the first2K request and one
`_gumbel_sample_kernel` warning in fixed rep0. The attributed32K segments
(B delta lines48–90, A49–92) contain zero compile-pattern messages. Monitor
suppression and hook coverage were not proved complete; this is not a global
no-JIT guarantee, and no warning duration or TTFT adjustment is inferred.
Exact accepted/drafted counters per fixed request are unavailable; whole-onepass
and10-second aggregates crossing request boundaries cannot supply them. The
[source-bound matched/log audit](ep76_onepass9/supplement/README.md) preserves
these findings and their limits.

Long32K/128K each contain one combined request; their cold/warm
compatibility fields repeat one TTFT. No variance, statistical noninferiority,
repeatable32K gain or global no-regression claim follows. The original judge
records `-5.3% with no floor yet (n=1): one more baseline sample`; its duplicate
original entries remain in the archive. This is retained evidence, not an
instruction to issue another run.

Both four-rank ready sets retain the first eligible actual-weight layer's
12case/72candidate+72stock comparisons, selected Q0 mode and SF6 FINALIZED42.
Ready launch markers may include profiling calls; completed canonical records
separately retain the actual execution proof. Rank-local canaries are not
independent proof of the final distributed sum.

[v9 originals and audit](ep76_onepass9/README.md) remain separate from v8B and
the historical78.93 result. Public defaults and all earlier rejected/incomplete
verdicts remain unchanged.
