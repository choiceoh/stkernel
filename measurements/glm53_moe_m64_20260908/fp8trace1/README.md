# Actual partial and FP8 packet trace

The trace confirms deterministic FP8 transport of frozen partials and exposes
rounding amplification of small MoE output differences. It does not approve
M64 numerics or serving. All original failure counts remain unchanged.

Submitted 2026-09-08 04:05:41 KST in normal fleet session `moem64trace10908`.
GO was 04:12:44, probe execution 04:13:56–04:15:20, and exact incoming four-node
recovery plus outer exit 0 completed at 04:17:57. The next owner received the
fleet at 04:18:04. No further GPU job was submitted for this experiment.

The immutable four-node source is `383894cf46df9185f3be6e9be5098c095bbcc337`
at `/home/choiceoh/stkernel-moe-m64-fp8trace1-0908`. The persistent job is
`/tmp/glm53-moe-m64-fp8trace1-0908`, worker 1368231. Kernel/runtime files are
unchanged from check5, and the original numerical gate remains unchanged.
The separate `--fp8-trace` mode cannot admit serving. Raw process logs came
from `/tmp/glm53-moe-m64.sFgn4I` and are preserved as four compressed files.

The fixed three cases, three seeds and eight alternating trials capture each
actual baseline/repeat/control/candidate call's pre-transport partial. Each
frozen partial is replayed through production FP8 packing, exchange and unpack,
compared with the unmodified helper, and reduced with native BF16. The trace
retains packet bytes/scales, FP32 sums and selected original output bits from
all four ranks for every failed row, plus raw comparison numerators and original
normalized failure counts. Its fixed 128-row budget fails explicitly if exceeded.

Six actual pinned-image CPU trace tests passed, including packet boundary/empty
selection, per-arm tensor row selection, raw rounding boundaries, compression
corruption, all-rank coverage and failure-preserving completion. CPU compilation
of M128 and M64 also passed. The local validation record distinguishes observed
test results from raw stdout and records host numpy/torch skips. Lifecycle,
collector and API checks passed. `cpu-analyzer.log` records two actual CPU
known-data/corruption tests for the offline packet/sum reconstruction reader.

Run the analyzer with numpy/torch and the frozen `probes` directory on PYTHONPATH:

```
python3 analyze.py /path/to/raw-rank-logs summary.json
```

It verifies every payload's shape/ownership, reconstructs FP8 packing from the
stored BF16 partials and reconstructs the source-order FP32 sums, requiring exact
agreement with the GPU trace. Failure summaries retain scale changes and the
per-rank values at the largest changed output component. Raw-numerator checks
are diagnostic only; they do not change any original normalized failure.

## GPU and CPU reconstruction results

All 72 trials and all 288 rank payloads completed. Every frozen replay matched
the original output and the unmodified helper bitwise. The stored FP32 sum
converted to BF16 matched the production unpack/store. Captured partials and
gathered inputs were unchanged. Both the actual partial and native BF16
comparison phases passed on every rank/trial; all values were finite.

The unchanged FP8 peak comparison still fails:

| Case | Seed | Candidate/control failing row-trials | Candidate/control failed trials |
| --- | ---: | ---: | ---: |
| 4096 concentrated, M64 disabled | 13307 | 5 / 5 | 4 / 4 |
| 4096 concentrated, M64 disabled | 118036 | 2 / 2 | 2 / 2 |
| 4096 concentrated, M64 disabled | 223066 | 3 / 3 | 2 / 3 |
| 8192 balanced | 17403 | 21 / 1 | 8 / 1 |
| 8192 balanced | 122132 | 10 / 0 | 5 / 0 |
| 8192 balanced | 227162 | 52 / 1 | 8 / 1 |
| 6144 balanced | 15355 | 0 / 0 | 0 / 0 |
| 6144 balanced | 120084 | 6 / 1 | 3 / 1 |
| 6144 balanced | 225114 | 1 / 0 | 1 / 0 |

These are descriptive row-trial counts in one boot, not independent samples.
The trace instrumentation changes timing, so counts are not a matched trend
against the previous diagnostic. Candidate excess persists in all three 8192
seeds, totaling 83 versus 2; 6144 totals 7 versus 1 and fallback 4096 totals
10 versus 10. No sanitizer or full-model TTFT was run.

The CPU analyzer verified all 1,152 rank/arm packet sets and 1,152 destination/
arm sum sets (including empty selections) against the saved trace. The actual
selected population is 106 failed-row-trial unions across cases. It reconstructed
FP8 bytes/scales from the saved BF16 partials and reconstructed FP32 sums in
source-rank order with exact agreement. Full raw payloads allow reproduction.

For example, 8192/seed122132/trial7/row5074/component1098 changes the sum of
unquantized partials by -0.21875, but the decoded FP8 and stored BF16 sums change
by -4.25. Block scales are unchanged. One rank's partial moves 38.0 to 37.75,
crossing an E4M3 rounding boundary (byte 122 to 121 at scale .125). This is a
quantization effect, not a mismatched packet or a frozen-transport race in this
diagnostic. Scale changes occur in only 5/83 candidate failures at 8192 and none
at 6144; scale jumps alone cannot explain the observed excess.

Raw numerators also confirm a normalized comparison boundary: row7905 can have
raw peak error 6 and repeat error 2, yet float32 division produces an error one
representable step above the three-times-repeat limit. Of 83 candidate failures
at 8192, 23 do not exceed the algebraic raw-peak boundary; the other 60 do.
At 6144 these counts are 2 and 5. All remain failures under the original rule.

## CPU-only INT8 alternative, on selected rows

`int8_replay.py` evaluates symmetric INT8 with per-2048-element power-of-two
scales, round-to-nearest-even, FP32 source-order summation and BF16 storage.
It would use the same one-byte value payload plus existing FP32 scales, but
no GPU runtime implementation or speed measurement exists yet.

| Previously selected row-trials | FP8 candidate/control failures | INT8 candidate/control failures |
| --- | ---: | ---: |
| 4096: 15 | 10 / 10 | 0 / 0 |
| 8192: 84 | 83 / 2 | 0 / 0 |
| 6144: 7 | 7 / 1 | 0 / 0 |

Against each arm's own unquantized FP32 partial sum, the median relative L2
quantization error at 8192 is 2.669% for FP8 and 1.283% for INT8; median peak
error is 3.341% versus 1.129%. This is a selected, failure-biased subset of 336
arm-row-trial observations at 8192, not full-model accuracy or a speedup.
The INT8 recipe passes CPU tests for symmetric ties-to-even, scale boundaries,
zero and empty rows. Its exact implementation and results are archived.

Next implement an explicit default-off, reduce-scatter-only INT8 experiment;
keep all-gather FP8, short-chunk BF16 and current defaults unchanged. Validate
GPU packing against the CPU recipe and compare all rows in the original fixed
cases/seeds, preserving original FP8 results and tolerance rules. Only complete
full-row/quality evidence can justify progressing to a full gate and direct
TTFT. Do not infer acceptance from these 106 selected rows, relax thresholds,
or rerun completed trace jobs unchanged. M64 remains off and 40% is unproven.
