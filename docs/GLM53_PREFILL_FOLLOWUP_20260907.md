# GLM prefill follow-up verification

PR #439 adds FP8 unpack/MHC post fusion, independent AG/RS thresholds, and
direct grouped PyNCCL packet exchange. Both new execution paths remain off
by default; the separate thresholds inherit the existing shared boundary.

## First GPU gate: rejected

Fleet `spfused0907` ran 2026-09-07 11:46:31–11:47:10 KST on source
`66747d596f8ce1be786f13383b00128f3ac994b9`, using immutable image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The helper SHA was `bed829bc85bc94d93b500538116990e62a30df8e53de181c8e938ae720e9d056`.

Explicit SM121 compilation passed for all five listed kernels, including
`_unpack_sum_mhc_post`. All 28 simulated-rank transport cases passed.
The fused MHC zero-input case passed at 32 rows/rank, but the random case
failed bitwise comparison with the mounted production TileLang MHC post.
The batch stopped there; TP4 direct exchange, threshold sweeps and serving
throughput were **not tested** by this invocation.

Read-only inspection of the production TileLang cache's embedded PTX found
the cause: the compiler rounds `comb[0,j] * residual[0]` first and then uses
`fma(post[j], x, product)`. The initial fusion rounded `post[j] * x` first.
Those mathematically equivalent expressions can differ after BF16 rounding.
The correction follows the compiled order, preserving the preceding BF16
decode rounding as well. A deterministic cancellation fixture uses x=1,
r0=1.5, comb0=1+2^-23, and post=-(1.5+2^-22): stock gives +0, while the
incorrect order gives -2^-24. The GPU probe now checks this boundary.

Raw log: `srv2:/tmp/glm53-prefill-followup.aRyhm1/fleet.log`.
Traffic audit: `srv2:/tmp/glm53-prefill-followup.aRyhm1/traffic.jsonl`.
Reference PTX is embedded in the production cache entry
`tilelang/0.1.12/linux-aarch64/kernels/ca8d6b8e3e7290c732146d1a8a6e7b4f79530eb813f6a3d86cfe3f78a47af006/executable.so`.

## Corrected GPU gate: numerics passed; memory incident

The retry on source `52b23e229231e470236d36ea74bfebc352d512ea` (main
`e1a88d7` included) started under the same fleet session at 12:01:38 KST.
All 28 fused MHC cases passed, including the deterministic rounding boundary,
bit-exact post/pre continuation at 33/532/1728 local rows with and without
RMSNorm, repeated consumers, and nondefault streams. Actual TP4 passed 180
raw-codec cases plus 80 independent-threshold cases (260 total). Both actual
direct exchanges and deferred packet/MHC consumers were exercised.

However, the probe's admission was insufficient: an idle serving process
still held most of the GB10 UMA memory. At 12:01:25, available memory was
9,579 MiB; at 12:01:58 it fell to 6,048 MiB (4.93%). `earlyoom` sent SIGTERM
to serving worker PID 777975. The API exited at 12:02:06. The resulting
microtimings are exploratory offline results, not clean serving evidence.
The arithmetic comparisons remain valid. No serving throughput claim is
derived from this invocation.

Recovery ran through fleet session `spfusedrestore` at 12:06:37–12:11:12,
using the pinned image and production defaults. The launcher observed health
200 before the fleet was handed to the next queued offline experiment.
Probe launchers now check MemAvailable on every participating node before
starting a GPU container, reserving 8 GiB for the probe plus 10% of host RAM
(at least 8 GiB) for existing work. Missing counters and inadequate headroom
fail closed; there is no override. Each subsequent single-GPU phase rechecks.
The observed incident values and threshold edge are covered by CPU tests.

Exploratory observations used to choose the serving candidate:

| Operation / rows | Existing path | Candidate | Decision |
| --- | ---: | ---: | --- |
| Decode + MHC post / 1728 local | 842.11 us | 735.23 us fused | Test fusion in serving |
| All-gather / 2128 global | 1013.62 us BF16 | 795.68 us v3 | Test AG threshold 2048 |
| Reduce-scatter / 2128 global | 1049.87 us BF16 | 1111.54 us v3 ProcessGroup | Retain RS threshold 4096 |
| Reduce-scatter / 4095 global | 1494.93 us ProcessGroup | 39723.73 us direct | Keep direct path off |
| Reduce-scatter / 4096 global | 1281.31 us ProcessGroup | 11458.94 us direct | Size cliff; no promotion |

These are per-arm medians; mirrored per-round ratios and all samples are in
the [raw evidence](GLM53_PREFILL_FOLLOWUP_20260907.json). The direct arm also
has high variance, so ratios of independent medians must not be described as
its paired estimator. No per-kernel percentage is an end-to-end gain.

A matched serving pair is queued as `spfusedserv` on the tested source,
with `FUSE_MHC=1`, AG minimum 2048, RS minimum 4096, and direct exchange off.
It uses 2K/4K/8K/32K/128K onepass requests with exclusivity required. Quality,
Korean corruption and traffic contamination are gates. The actual serving
result and original 40% target remain unresolved.
