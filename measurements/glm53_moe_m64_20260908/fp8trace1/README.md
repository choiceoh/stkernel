# Actual partial and FP8 packet trace

Submitted 2026-09-08 04:05:41 KST in normal fleet session `moem64trace10908`.
At 04:12 it was first in the queue behind another owner's boot. No GPU trace
result is available in this preparation snapshot.

The immutable four-node source is `383894cf46df9185f3be6e9be5098c095bbcc337`
at `/home/choiceoh/stkernel-moe-m64-fp8trace1-0908`. The persistent job is
`/tmp/glm53-moe-m64-fp8trace1-0908`, worker 1368231. Kernel/runtime files are
unchanged from check5, and the original numerical gate remains unchanged.
The separate `--fp8-trace` mode cannot admit serving.

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

GPU execution, reconstruction and exact incoming recovery are pending.
