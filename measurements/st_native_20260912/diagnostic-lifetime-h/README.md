# Native long-context repeat and first-divergence evidence

H's native 128,559-token runner is not repeatable: the first token changes among
785, 220 and 154842 after clearing prefix entries and caches between requests.
A GPU event after each target forward also failed on repetition (111721), and
its 1200-token answer retrieved only one of three facts. No event or global
synchronization workaround was adopted.

Instrumenting the first two 6912-token chunks located the first different
linear input at L4.kda.in_proj, immediately after L3 (the first MoE). Earlier
dense and shared-expert projections matched. Replaying that actual L3 input,
route ids and route weights isolated the nondeterminism to the local MoE call.

Across seven repeats after the first output, changed BF16 element counts were:

| Rank | Minimum | Maximum | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| 0 | 973966 | 1150454 | 0.046875 |
| 1 | 633113 | 871919 | 0.046875 |
| 2 | 994921 | 1360871 | 0.046875 |
| 3 | 983247 | 1231662 | 0.0625 |

Each output is [6912,4096]. The dynamic TP prefill epilogue atomically sums
weighted contributions in BF16, rounding at each arrival; static decode already
uses a FP32 accumulator. This establishes local numerical nondeterminism, but
full-answer correctness must be checked separately after the repair.

`moe-repeat.py` is the command executed inside the loaded diagnostic session.
Each node retains its actual input in
`/home/choiceoh/st-native-long-lifetime-h/moe-L3-input-rank<R>.pt` (not committed).
The JSON files retain the repeat differences and first-token/answer controls.
The failed live patch attempt used the wrong workspace helper and exited before
producing repaired results. The wrapper restored the pinned serving release,
active head systemd service, and removed the ingress drain rule.
