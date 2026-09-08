# CPU readiness-vote preparation

`pinned-cpu.json` records the image, bounded CPU-only Docker invocation, source
SHA-256 values, timestamps and machine-readable test result. `pinned-cpu.log.gz`
is the complete combined stdout/stderr. All 30 startup-artifact tests passed with
zero skips and no CUDA initialization. The four ranks ran as separate processes
inside one network-isolated container; this is not four-node network evidence.

`installed-source-audit.json` contains source hashes and selected excerpts from
the restored public head image, plus the relevant WORLD call sites and selected
nonsecret environment values. It is a read-only source audit, not runtime
allocation ownership or performance evidence. Captures are timestamped; they
must not be represented as live state after a later deployment.

These files contain no model request, TTFT, quality, routing or profiler result.
The candidate remains default off. See
[`GLM53_RANK_CACHE_CPU_VOTE.md`](../../../docs/GLM53_RANK_CACHE_CPU_VOTE.md)
for the remaining live gate and cache-construction limitation.
