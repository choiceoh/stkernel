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

Corrected GPU verification and a matched, exclusive serving prefill A/B are
pending. No additional speedup or original 40% target claim is established.
