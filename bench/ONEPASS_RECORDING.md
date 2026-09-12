# Onepass recording (harness 41)

> 살아 있는 참조 — **원패스가 무엇을 기록하는지. 하니스가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

`python3 bench/onepass.py --name NAME` runs the Korean context ladder at C=1
and C=4. C=4 sends four independent streaming requests for each canonical
question with a start barrier. Request rates and aggregate rates have different
denominators; C=4 never enters the C=1 decode-window pool.

Each arm first replays its complete workload to exercise the actual shapes and
generation path. Every request uses a different `cache_salt`, leaving the model
prompt unchanged. The measured arm must show zero reused prompt tokens, no
observed new Triton/CuTe/C++ specialization or ST graph capture, complete
traffic counters, and the requested decode width on every rank. These are
observed preparation conditions, not a claim that every external compiler or
OS page cache is known. The compiler observers report their coverage. Missing
evidence invalidates comparison. Harness 40 records are incompatible.

The printed first/median TTFT values are prepared, fresh-prefix requests.
`cold_s` and `warm_s` remain compatibility aliases for first/median; they do not
describe compiler warmth. The compiler's first-use time is in the preparation
artifacts. There is no implicit OS-cache flush or production-server restart.

Performance requests have the profiler **off**. After both concurrency arms,
one representative question per context and concurrency is replayed for GPU
diagnostics. A diagnostic collects one prefill chunk and up to four decode
steps at the requested width. This bounds profiler overhead and trace size;
the recorded step positions say exactly which work was sampled. Diagnostic
latencies describe an instrumented replay and are excluded from performance
comparisons. The full host step sequence remains available for each arm.

The default ledger is `~/glm53-logs/bracket-onepass.jsonl`. Every invocation
also creates an immutable `onepass-runs/<UTC-time>-<id>/` next to the ledger:

| Artifact | Contents |
| --- | --- |
| `record.json` | Checkpointed run identity, shape, policy, C=1/C=4 results and validity |
| `requests.jsonl` | Completed requests, raw answer/channel deltas, hashes, first reasoning/content times, SSE gaps and token usage; flushed and fsynced per completion |
| `<phase>/server.json` | All-rank preparation counters, request-to-row mappings, host steps, sampled device stages and trace checksums |
| `<phase>/latency.jsonl` | Individual host/device/kernel latency rows with phase, rank and step |
| `<phase>/latency-summary.json` | Per-operation/kernel count, sum, mean, median and range; per-rank, per-step GPU interval unions |
| `<phase>/rank-N/*.trace.json.gz` | Original Chrome/CUPTI trace plus captured node labels; verified by SHA-256 during transfer |

Each rank also retains its own manifest, latency rows and raw traces under the
boot's dump directory, `onepass-latency/<token>/rank-N/`. A failure retains
completed requests and a partial manifest; the ledger marks an incomplete run.
A hard-killed client leaves the last checkpoint and server running manifest.

CUDA graph labels come from CPU inspection of the capture DAG at semantic
boundaries, joined to CUPTI graph node IDs on replay. This inserts no GPU
timing nodes and does not change kernel arguments or graph work. Eager kernels
use CPU annotation/external-ID joins. Unknown nodes remain `unmapped`; their
duration and raw event are retained. Inspect `graph_attribution_errors` and
each trace's `unmapped` count before making layer-level claims.

Host launch spans, device spans and cross-rank times are separate. Kernel sums
can exceed elapsed time because streams overlap. The union is computed within
one rank and one profiled step; cross-rank clocks are never subtracted. Existing
asynchronous device-stage samples retain their original step context. Pending
samples are declared, not synchronized in the normal request path. SSE event
gaps are not token ITL, since a speculative event can contain several tokens.
Small samples have counts and ranges, without misleading tail percentiles.

The ST endpoint is `GET/POST /v1/engine/latency`. Controls require loopback
access, idle serving and a unique token. All ranks participate;
each rank must be a separate serving process; threaded LocalTP is refused
because its ranks share one CUDA profiler. While recording,
inference requests must carry `X-ST-Latency-Token`. `end` requires idle serving;
`abort` with the owning token releases an abandoned recording and marks it
incomplete. An abandoned idle recording expires after 120 seconds; active
inference is never interrupted by this timeout. Trace downloads are restricted to the completed session's declared
files and bounded chunks. Servers without this endpoint still produce client
records, with detailed/steady-state evidence marked unavailable.

Validation commands:

```sh
python3 -m unittest tests.test_engine_latency tests.test_onepass_recording
python3 -m unittest discover -s tests -p 'test_onepass*.py'
python3 tools/check.py --list
bash bench/fleet.sh run --gpu SESSION 5 'onepass attribution contract' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes latency
```

The GPU contract verifies graph attribution, profiler-off normal recording and
unchanged arithmetic at widths 1 and 4. It is an instrumentation proof, not a
GLM throughput measurement. Actual consumer results come from onepass on the
candidate serving build.
