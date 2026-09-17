# NVFP4 activation scale search extension — 2026-09-17

The user requested the same three-candidate search at all three identified
NVFP4 paths: routed MoE FC1, dynamic-prefill FC2 and the first three dense MLPs.
The `as1` recipe selects that extension. `ss1` retains its original meaning:
search only the routed static FC2 input. They cannot be combined. `as2` retains
the optional five-candidate reference. No experts, weights, global scales,
activation/clamp semantics or accumulation arithmetic change.

## Coverage

- Static v4/v5 FC1: ordinary pack, vector loads, C=1 token cache and C=2
  register fanout, including the per-expert unequal-scale branch.
- Dynamic prefill: FC1 and FC2 in the generic and gated bodies; Q0, raw-scale
  Q0, short/long SF6 and FP8 packet-input route producers. Packet transport is
  unchanged; search scores the existing BF16-rounded inputs to FP4 packing.
- Dense MLP: FC1 and FC2 in the micro and stock-static kernels. The admitted
  dense geometry is E=1, H=4096, I=3072/rank. Long-prefill W4A16 still uses BF16
  activations, so it has no FP4 activation scales to search.
- The bound routed/dense NVFP4 geometry gates the extension. Unrelated shapes
  and formats retain their quantizers. Both in-memory and persistent compile
  keys include the option. Inherited-source pins and provenance were updated.

The live production environment points to `st-glm53-9391-up-gate-full`, whose
layout is `st-glm53-b12x-up-gate-v1`. Its dense MLPs use the separate dense path,
not the E=1 NVFP4 adapter. Thus this consumer A/B measures the routed FC1 and
prefill extension. The dense extension is for the separate ModelOpt NVFP4
checkpoint; changing the current dense precision is outside this change.

## Validation

GPU-free checks used the existing serving image
`sha256:e9e80b94d41277b171cef5785483989acd1c91d450d0d1068269e99f60bc75bd`,
PyTorch 2.13.0+cu132 and SM121a native compilation.

- 74 focused CPU tests passed (policy, quantizer oracle, profile defaults,
  source/dependency contracts, route producers, packets, SF6 staging and cache).
- 30 kernels compiled. Ten existing-ss1/extended-as1 pairs cover routed
  8/16/32 rows, dense micro/static, generic dynamic, raw/SF6 Q0, long prefill
  and packet prefill. Every enabled binary differs from its ss1 control;
  register, stack, shared and local resource counts are unchanged in all ten.
- C=1/C=2 routed kernels use 96 registers and zero stack/local memory.
  Dense/static and dynamic kernels retain their existing nonzero stack usage;
  this is not a claim that every kernel is spill-free.
- The disabled quantizer is byte-identical to the original quantizer.

Receipts: `cpu.log`, `compile.jsonl`, `compile-summary.json`. The compile source
snapshot precedes a helper docstring update and the probe's expanded fingerprint;
no compiled arithmetic changed after that check.

The default-flipped candidate passed another 26 focused CPU tests. GPU ticket
`fp4-all-native17v2` also passed: 151,322 finite blocks had no worsened SSE beyond
the quantizer's tolerance; exceptional-scale/disabled parity and poisoned-output
graph replay passed. Actual layer-3 routed output remained finite and zero-route
outputs remained exactly zero for both C=1/C=2 shapes. Relative output changes
against ss1 are differences, not errors against a ground-truth model.

| Single routed MoE kernel cost, as1 vs ss1 | Warm | Evicted |
|---|---:|---:|
| C=1 | +0.474% | -0.003% |
| C=2 | +0.435% | -0.169% |

Positive means slower. These are four B/A/A/B brackets on one rank, not consumer
throughput. Receipts: `gpu.jsonl`, `candidate-cpu.log`.

## Additional error ablation

`probes/nvfp4_all_activation_projection.py` reads nine routed experts from the
actual production rank (layers 3/20/40, experts 0/73/287) and the three dense MLPs
from the separate ModelOpt rank. Each cell uses 16 synthetic normal BF16 input
rows. It honors the rank's **up|gate** layout and keeps the same dequantized
checkpoint weights, clamped SiLU, BF16 intermediate and K128 BF16 down-projection
contributions in every arm. The reference omits FP4 activation quantization.
This isolates activation-induced SSE, not weight error against the original
model, captured activation behavior, native CUDA exactness or answer accuracy.

| Routed experts, pooled SSE reduction | Reduction | Improved cells |
|---|---:|---:|
| FC1 activation: max-based to search | 17.715% | 9/9 |
| FC1 projection: max-based to search | 17.649% | 9/9 |
| FC2 activation, identical FC1 output | 9.652% | 9/9 |
| FC2 projection, identical FC1 output | 9.565% | 9/9 |
| Full MLP: existing FC2 ss1 to as1 | **12.547%** | **9/9** |
| Full MLP: no search to as1 | 14.922% | 9/9 |

The separate ModelOpt dense fixture shows only 0.069% less FC1 activation SSE
and 0.005% less FC2 activation SSE; pooled full-MLP SSE increases by 0.043%.
That fixture does not establish a dense improvement. Input distributions and
checkpoint global scales matter, and smaller local SSE need not improve the
whole MLP. `projection.json` retains all per-cell values, seeds and hashes.

## Consumer gate

The implementation was merged in #1127 while these measurements were pending.
After reviewing the C=1 improvement, unchanged reported primary decisions and
C=2 certificate issues, the user explicitly elected to retain **as1 as default**.
This follow-up records the evidence without changing the engine or its default.
Tickets `fp4-all-base17` and `fp4-all-candidate17` compare existing `ss1` with
`as1` using the same implementation and harness, production shape, C=1/C=2
fixed 1024-token throughput, natural-EOS acceptance and clarified ko-reasoning-v3
quality, including 32K and 128K C=1. Quality must be preserved and C=1 output
throughput must remain at least 95% of the matched baseline.

- Baseline: `b2faf8169284ee62b74e456042b4d97dda28a2a3`.
- Candidate: `1fd46c9ee5c88fcdfb6577361e86d61ad93a0261`.
- Both bench trees: `63ae431979ba63a37c8126936b77597c9a743447`.
- One boot per arm: C=1 natural EOS twice; C=2 and fixed length once.
- Baseline records: `20260917T130020-d15c307f7bb5` and
  `20260917T131947-0bfbe0518c6e`.
- Candidate records: `20260917T133200-b3eb2a4aaa98` and
  `20260917T135029-d3b4bfc0952f`.
- All four records are complete. Harness, quality protocol, workload, budgets,
  served shape and measurement policy match. All four timing preparation checks
  pass, with profiler off, fresh prefixes and no observed compile/graph capture.

| Consumer metric | Existing ss1 | All-path as1 |
|---|---:|---:|
| Fixed 1024-token C=1 output tok/s | 87.132 | **95.358 (+9.44%)** |
| Fixed 1024-token C=2 aggregate tok/s | 109.169 | **111.158 (+1.82%)** |
| C=2 / C=1 | 1.253x | 1.166x |
| Natural C=1 acceptance, runs 1 / 2 | 55.125% / 56.468% | 56.312% / 55.059% |
| Natural C=1 tokens/step, runs 1 / 2 | 4.859 / 4.953 | 4.942 / 4.854 |
| C=1 strict score, runs 1 / 2 | 52/57 / 53/57 | 54/57 / **57/57** |
| C=1 strict score, combined | 105/114 | **111/114** |
| C=2 strict score | 72/76 | 62/76 |
| All strict points / complete certificates | 177/190; 23/30 | 173/190; 22/30 |
| Parsed primary results | 30/30 | 29/30 |
| Reported primary decisions, including malformed response | 30/30 | 30/30 |

The C=1 speed guard passes. C=2 absolute throughput rises slightly, while the
ratio falls because C=1 improves more; the 1.7x target is not reached. Fixed
throughput has one sample per arm, so these are observations, not a replicated
speedup estimate. Natural acceptance is similar; two C=1 repeats do not establish
reduced variance. They use the pinned runs' counters, before the later EOS/limit
accepted-draft accounting correction in #1129, with the same convention in both
arms. These are not measurements of that later main build.

| Fresh-prefix prefill, run 2 | ss1 tok/s (TTFT) | as1 tok/s (TTFT) |
|---|---:|---:|
| 2K | 2473.55 (1.120 s) | 2483.10 (1.116 s) |
| 32K | 3541.62 (9.587 s) | 3503.91 (9.691 s) |
| 128K | 3484.92 (37.278 s) | 3446.10 (37.698 s) |

The candidate's missing closing JSON brace accounts for five failed C=2 checks;
its reported primary answer is right. C=2 also adds five certificate deductions
for constraints, intermediate fields and a minimal core. Both arms lose four
C=2 points to bit-string transcription. The [quality audit](quality-audit.md)
retains these distinctions and every original grade. No case is excluded.

The canonical judge reports **NO BASE**, because the baseline has no fully
passing warm certificate sample; the candidate's second C=1 run passes 9/9.
This is not a canonical WIN or a claim that all quality gates passed. Adoption
is the user's explicit decision after reviewing the mixed evidence.

`consumer-baseline.json` and `consumer-candidate.json` preserve raw-record hashes,
per-request timing/output hashes and original quality details.
`consumer-comparison.json` is generated by `compare_consumer.py` from the four
completed raw records. Raw artifacts remain under the recorded paths on srv2.
`audit_quality.py --raw-root DIR --output FILE` reproduces the reviewed audit
from folders `base-1`, `base-2`, `candidate-1`, `candidate-2`, each containing
`record.json`, `quality.jsonl`, `requests.jsonl` and `workloads.json`.

At measurement completion the bracket stopped all four candidate containers
and dropped its temporary tiers by 23:00:48 KST. All four recorded container IDs
were subsequently verified absent; the lease passed to the next queued job.
`fleet-stop.json` records that release. No further GPU experiment was started
for this campaign.

The follow-up [C=2 receipt audit](c2-path-audit.md) records the concurrent
prefill chunks and decode selectors. The deeper [logical and native-code
audit](c2-logical-audit.md) identifies a missing generic-global -> TMA proxy
fence at FC1 input publication and includes its correction. CPU checks and
native compilation pass; post-patch GPU quality and speed remain unmeasured.

The subsequent [remaining-publication audit](remaining-publication-audit.md)
confirms the same missing global proxy fence in both served SF6 prefill paths
and reopens the generic grid barrier's final-arrival acquire contract.
`prefill-publication-native.json` contains the two GPU-free native receipts;
this follow-up changes evidence only, not serving code.

The [publication repair](publication-repair.md) then fixes both ordering
boundaries across the kernel families and adds real-weight, identical-input
M8/M16 validation. It keeps kernel correctness evidence separate from the
still-unmeasured consumer quality and performance after these repairs.
The final GPU check passes, including all nine short/long prefill fixtures;
its container and single-GPU reservation were released at completion.
