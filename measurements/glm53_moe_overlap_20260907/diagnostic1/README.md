# Component attribution: splitting cost and FP8 error

Diagnostic session `moeoverlapdiag10908` acquired the normal fleet hold at
00:33:29 KST on 2026-09-08. Collection ran 00:34:21–00:36:06; exact incoming
container/config/image/mount/source and healthy-endpoint recovery completed
at 00:39:05, exit 0. **This exit means diagnostics and recovery completed;
it is not a GPU correctness pass or a serving acceptance result.**

Frozen source is `6b8c2b3ec561f962282142e6ede123e2ed48d702` at
`/home/choiceoh/stkernel-moe-overlap-diagnostic1-0908` on all four nodes.
The remote job is `/tmp/glm53-moe-overlap-diagnostic1-0908`; original rank
logs are `/tmp/glm53-moe-overlap.RyxS5U/{bf16,fp8-v3}-rank-{0..3}.log`.
All 72 records contain four rank reports; source/API checks, both transport
completion markers, source hashes and exact recovery were verified.
The formal numerical comparator and runtime code were unchanged.

## Numerical attribution

All 16 size/routing/transport cases returned identical repeated all-gather
inputs, and all 64 rank/case checks returned bit-identical results when
replaying a fixed partial through reduce-scatter. The observed source of
variation precedes transport: repeated MoE on the same gathered input has
up to roughly 0.54% maximum row-relative L2 difference. Transporting varying
partials can then exceed the unchanged per-row FP8 comparison bounds.
This identifies stages; it does not by itself prove a specific atomic
instruction or exclude every possible stream defect.

At FP8 4143 skew (where overlap is disabled), both the nominal candidate and
independent stock control exceed the same gate in this collection. Other
stock-control false comparisons appear at 4096. Thus the original fallback
failure is not candidate-specific and a single repeat pair incompletely
characterizes this stock path's variation. The threshold has not been
relaxed, nor has the failed full gate been reclassified as passing.

There is also a separate candidate problem. At FP8 6912 skew, **887 valid
rows** fail across four ranks, and at 8192 skew **708 rows** fail, while the
whole-stock repeat is exact in both cases. Maximum peak-relative error
reaches about 9.68% / 7.73%, exceeding the original 4% floor. These failures
cannot be dismissed as the fallback control variation. FP8 overlap must
not be enabled or used for a serving comparison in this form. The changed-
input serial-versus-overlap comparator also has a one-row failure at 8192
skew; no complete FP8 correctness or stream-safety claim is made.

BF16 whole-path candidate and independent controls have zero failing rows
in all eight diagnostic cases. The earlier formal BF16 gate also passed,
but these diagnostic repetitions do not validate a new runtime revision.

## Splitting versus overlap cost

The following six-order, slowest-rank median timings preserve the original
-0.75 changed-activation phase. Stock processes the full chunk once; serial
uses the same two stripes without stream overlap; overlap is the candidate.
All route counts use actual gathered inputs and the dispatcher's selected
M128 tile. Full counts exactly match the sum of both stripe counts.

| Transport | Tokens-routing | Stock / split physical tiles | Stock ms | Serial stripes ms | Overlap ms |
| --- | --- | ---: | ---: | ---: | ---: |
| bf16 | 6912-balanced | 592 / 604 | 18.788 | 25.809 | 25.934 |
| bf16 | 6912-skew | 432 / 432 | 14.976 | 13.609 | 13.947 |
| bf16 | 8192-balanced | 650 / 733 | 20.121 | 28.517 | 28.705 |
| bf16 | 8192-skew | 512 / 512 | 17.717 | 16.349 | 16.774 |
| fp8-v3 | 6912-balanced | 518 / 636 | 15.091 | 21.659 | 20.365 |
| fp8-v3 | 6912-skew | 432 / 432 | 13.963 | 12.747 | 12.500 |
| fp8-v3 | 8192-balanced | 614 / 672 | 16.705 | 23.218 | 22.502 |
| fp8-v3 | 8192-skew | 512 / 512 | 16.401 | 15.702 | 14.934 |

BF16 overlap does not beat serial stripes in any measured case. Balanced
splitting raises total latency even without overlap: 18.79→25.81 ms at 6912
and 20.12→28.52 ms at 8192. Added physical tiles are only about 2.0% / 12.8%,
so padding alone does not explain the entire delay. FP8 hides some of the
split cost but remains slower than full-chunk stock for balanced routing,
and its concentrated-routing numerical failures block serving.

The concentrated-routing speed gains already occur with serial stripes;
they are not proof of benefit from communication overlap. Keep this path
default-off and do not rerun the unchanged serving gate. A useful next
optimization should preserve one full-chunk MoE call and reduce work inside
that call, rather than assuming token splitting provides a general win.
Actual per-expert tails and stock M64/M128 dispatch are a bounded next
investigation; the archived route counts permit quantifying the padding
budget before designing or testing a new candidate.

Raw compressed rank/lifecycle logs and `records.jsonl` are retained with
`summary.json` and checksums. The serial component spans are separate,
instrumented measurements and must not be summed to predict overlapping
runtime. No full-model MoE TTFT or cumulative 40% improvement is claimed.
