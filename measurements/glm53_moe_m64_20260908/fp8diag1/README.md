# FP8 diagnostic: candidate excess remains after the stock control

Normal fleet session `moem64diag10908` received GO at 2026-09-08 03:34:09 KST.
The four-rank diagnostic ran 03:35:19–03:36:29 and completed all 72 trials.
Exact incoming four-container identity/config/source/image recovery and outer
exit 0 completed at 03:39:07. A 03:40 live check found fleet FREE, queue 0,
health 200. The incoming MK_INPUT_CTA=2/PACK_SHA256=1 experiment profile was
restored; this was not a deployment of standard defaults.

Source: `c50a6e78b174f3e89700e3d5595e4654c89b0e11`, frozen on all four nodes at
`/home/choiceoh/stkernel-moe-m64-fp8diag1-0908`. Kernel and all 56 build/glm53
files are unchanged from check5. Job directory is
`/tmp/glm53-moe-m64-fp8diag1-0908`; raw rank logs came from
`/tmp/glm53-moe-m64.a628xO`. The diagnostic emitted only its dedicated completion
marker, with numerical acceptance and serving admission both false. Exit 0
means collection and recovery completed, not that the numerical gate passed.

## Fixed comparison results

Each row below covers eight alternating trials across all four ranks. Counts
are failing row-trial observations, not unique errors or independent samples.
All failures were in the peak comparison; no L2 or nonfinite failure occurred.
Every local MoE comparison passed before TP transport.

| Case | Seed | Candidate bad rows | Control bad rows | Candidate/control failed trials |
| --- | ---: | ---: | ---: | ---: |
| 4096 concentrated, M64 disabled | 13307 | 5 | 5 | 4 / 4 |
| 4096 concentrated, M64 disabled | 118036 | 8 | 7 | 5 / 4 |
| 4096 concentrated, M64 disabled | 223066 | 1 | 2 | 1 / 2 |
| 8192 balanced | 17403 | 24 | 2 | 8 / 2 |
| 8192 balanced | 122132 | 15 | 4 | 8 / 2 |
| 8192 balanced | 227162 | 9 | 0 | 4 / 0 |
| 6144 balanced | 15355 | 0 | 0 | 0 / 0 |
| 6144 balanced | 120084 | 1 | 0 | 1 / 0 |
| 6144 balanced | 225114 | 0 | 0 | 0 / 0 |

The fallback's 14/14 total confirms that stock repetition can fail this gate.
However, 8192 has candidate excess in all three seeds (48 versus 6 total;
20/24 versus 4/24 failed trials). It cannot be excused by the stock control.
The 6144 active control also has one candidate failure in the expanded seeds.
These are one-boot diagnostic observations, not independent production runs.

Thirteen of the 48 candidate failures at 8192, and the sole 6144 failure, are
exactly one float32 representable step above their peak limit. For example,
8192/seed17403/row7905 has error 0.09836065769195557 and limit
0.09836065024137497 in all eight trials. This suggests a normalized arithmetic
boundary that needs raw-numerator evidence; it is not proof that a row should
pass. All these failures remain counted. Even excluding them descriptively
leaves 35 candidate observations versus six controls at 8192. No threshold,
comparison or serving gate was changed. The maximum candidate peak error was
0.09836065769195557, with maximum L2 among failed rows 0.017205622047185898.

## Evidence and validation

`fp8-v3-rank-{0,1,2,3}.log.gz` preserve all four process logs; rank 0 retains
every failing row's values, thresholds, repeat noise and candidate/control
intersection across all four ranks. `summary.json` is reproducible with
`python3 summarize.py`. Its per-seed trial counts are descriptive. The original
diagnostic completion was independently recomputed from all 72 records and
matched exactly. All four restored identities were compared with `before.json`.

Focused CPU checks passed 5 diagnostic tests in the pinned image (including
actual torch tensor math), 3 probe/API tests, 10 lifecycle tests and 10 serving
admission tests. Local host diagnostic tensor math was skipped for missing
torch; the pinned image covers it. Python compilation, shell syntax and diff
checks passed. Actual CPU compilation without GPU/network passed M128/M64
both before submission and inside the queue before stopping serving. Compiler
times are not throughput evidence. Logs and request/worker/lifecycle records
are archived here, with SHA256SUMS covering all final files.

## Next bounded diagnostic

Keep the kernel, original gate and limits fixed. Capture the exact local
partial immediately before reduce-scatter from the same invocation being
compared; the current separate local-MoE invocation only narrows the scope.
Use the same three cases, three seeds and eight alternating trials. For every
arm, replay the frozen partial through FP8 transport and retain bitwise replay
agreement. Compare the same partials through native BF16 transport, and retain
the per-rank pre-quantization values, packed FP8 values/scales and FP32 sums for
the union of failed rows. Also retain raw peak/L2 numerators and reference
denominators to distinguish a normalization boundary from a material excess.
Preserve the original normalized failure counts alongside these diagnostics.

This separates local accumulation differences, FP8 scale/rounding effects,
reduction amplification and transport repeat instability without changing the
acceptance rule. Do not assume FP8 transport itself is faulty merely because
the separately run local comparison passed. Add focused CPU validation of the
trace/replay evidence, freeze a new source and use a new normal fleet turn.
This follow-up is planned, not yet implemented or submitted. The completed
diagnostic must not be polled or rerun unchanged. Sanitizers/direct TTFT remain
blocked, M64 remains default-off, and cumulative 40% remains unproven.
