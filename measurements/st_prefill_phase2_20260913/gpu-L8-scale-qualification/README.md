# L8 component decision

The admitted `st-prefill-scale-L8r2` job measured candidate
`975f1fb963f127945fe3a6eb82071c659f2666d8` on 2026-09-13. It passed the
actual rank-0 layer-3 numerical gate, then deliberately declined the improved
engine boot because temporary scale expansion regressed short prefill. The
supervisor finished with exit 1 and released its fleet lease. No consumer
onepass ran and these numbers are not TTFT or served tok/s.

| Input rows / route fixture | Packed median ms | Expanded median ms | Throughput ratio |
| --- | ---: | ---: | ---: |
| 2,672 / balanced | 12.0570 | 13.0116 | 0.9266 |
| 2,675 / zero weights | 11.9827 | 12.9601 | 0.9246 |
| 8,192 / duplicate experts | 11.4837 | 11.3189 | 1.0146 |
| 32,256 / concentrated experts | 41.6817 | 38.3063 | 1.0881 |

Each median has six event-timed launches from alternating packed/expanded
controls. They include both temporary expansions, allocations, MoE execution,
and the final cast. The fixtures use real weights with synthetic activations
and routes, including changed inputs on the same storage and a second CUDA
stream. Both complete original scale planes matched exactly (113,246,208
bytes), both dedicated CUDA byte/capture tests passed, and the actual weight
identity was preserved. This establishes component correctness in the tested
scope, not generation quality or end-to-end speed.

The follow-up selector retains the packed reader for all automatic launches
through 8,192 rows; the lossless expansion remains a private long-prefill
candidate. Explicit numerical controls still support the measured short path
so the rejected result remains reproducible. No GPU result from this directory
applies to the follow-up revision, and it has not been requeued or deployed.

`scale-expansion.json` and `scale-expansion-gate.log` are unmodified GPU outputs;
`fleet-run.log` records the decision and release. `qualification-summary.json`
retains their SHA-256 values, the admitted candidate identity, and the derived
median arithmetic. CPU evidence for the measured revision remains in the
remote `L8-scale-cpu-r5` directory.
