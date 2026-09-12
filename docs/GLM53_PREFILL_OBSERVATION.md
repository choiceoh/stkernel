# Current-default prefill attribution

> 그대로 두는 기록 — **캠페인 산출물.** 당시의 기록이라 고치지 않는다 — 고치면 기록이 거짓이 된다.

Status: worker instrumentation, isolated boot/restoration, canonical quality/TTFT
requests and all-rank trace transfer/analysis are implemented. CPU contracts pass;
the first capture is queued as `glm53observe0908v1` with frozen source
`75686447d5cca72904b7050e333f8c319884a28d` on all four nodes. Preflight passed;
at the 09:52 KST submission it was first behind `deploycache0908v6`. **No live
result or new speedup is claimed.** M64 and INT8 remain off and deprioritized. The
observation branch starts from current main separately from preserved PR #455.

## Implemented pieces

`probes/glm53_prefill_observer.py` is a diagnostic mount, not a production overlay.
The pinned vLLM worker-extension interface can load its `WorkerExtension`; its
middleware exposes only the local POST `/glm53/prefill-observe` with status,
begin and end operations. No B12x hook exists before begin or after a successful
end. An idle worker reports its source hash and rank.

Begin requires TP4/PP1/DP1 without EP, one identifiable target `Glm5NextModel`,
and an unambiguous mapping from expert weight pointers to target layer names.
The temporary B12x wrapper hook forwards the original input objects and output
identity. It records actual wrapper row counts, layer, call ordinal, rank and
request ID. Missing mapping, capture, unknown geometry or exceeding 4096 calls
invalidates collection. Failure never selects a different model kernel.

- `profile` adds a `GLM53_PREFILL_OBSERVER` annotation and CPU metadata only.
  It does not read expert IDs or copy tensors to CPU.
- `routes` copies selected expert IDs to CPU in a separate request and records
  288 expert counts, all top-8 slots, zero-weight slot count and M64/M128 padding
  budgets. Counts are observational; padding ratios are not predicted speedups.

No tensors, DLPack capsules, outputs or expert weight tensors are kept in records.
End restores the exact original method and refuses to overwrite an unrelated
replacement. Complete ordered layer cycles are required. Reported groups identify
**executed MoE rows**, not scheduler tokens inferred from a maximum batch size.
All four reports must match source/request/mode/layer/group coverage. Actual
routing histograms remain rank-specific; equality is not silently assumed.

`bench/prefill_observation.py` wraps one already-attested request. Its HTTP client
requires the normal fleet boot holder and loopback port 18000. It validates idle
observers and all four begin acknowledgments before sending model traffic. If
begin or start_profile partially succeeds or loses its response, finally still
attempts stop_profile and observer end, then checks every rank is idle. Missing
completion or failed cleanup remains incomplete evidence. The outer runner must
recover the private boot if those cleanups cannot be proven. There is no
standalone CLI that bypasses the outer serving lifecycle.

`probes/glm53_prefill_trace.py` requires exactly one new regular trace per rank
and refuses changed old files, symlinks and ambiguous captures. Inventory must
be stable before the runner invokes this check. It compares every request/rank/
layer/call annotation with the worker report, rejects mixed GPU devices, and
retains category/kernel duration sums, interval unions, gaps and unknown kernels.
Categories are name-based heuristics. Overlapping sums or category unions cannot
be added into TTFT shares. Capture GPU span is not request wall time.

## Validation and remaining live checks

`tests/test_glm53_prefill_observation.py` has eleven CPU tests, including real BF16
tensor input/output identity, detached routing counts, restored hook identity,
partial/duplicate/rank/source coverage, missing cleanup, NaN padding evidence,
trace replacement and overlap, plus request/RPC/profile failure cleanup.
All eleven pass in the pinned image with `--runtime runc --network none`, 4 GiB
RAM and two CPU cores, without GPU access. Raw logs and source hashes are under
`measurements/glm53_prefill_observation_20260908/preparation/`.

The connected runner adds fourteen tests covering cloned configuration, foreign-name
refusal, settled partial failures, failed boot/request recovery, retried original
restoration, short-request coverage, quality and prompt identity. Together with
the eleven observer, six trace and eight fleet handoff tests, all 39 pass in the same pinned
CPU-only runtime without skips. Exact source hashes and raw logs are under
`measurements/glm53_prefill_observation_20260908/runner-preparation/`. The separate
Docker API create-only fixture passed requested Config/HostConfig preservation
without starting either container. Its earlier failed assertion is retained and
unattributed; this preparation does not prove live model boot behavior.

CPU tests do not prove that the live compiled serving path visits this hook, that
the target weight mapping resolves, that all ranks emit the expected trace schema,
or that probe overhead fits memory. Live coverage must fail closed if any of those
assumptions fail. Do not promote synthetic CPU checks into runtime acceptance.

`bench/prefill_observation_run.py run --revision SHA --out DIR` requires the normal
fleet boot holder and `FLEET_OBSERVATION_CLONES=1` at fleet submission. It first uses the fleet's approved-default restoration helper,
then snapshots the four persistent originals. `probes/glm53_observation_host.py`
creates uniquely labelled, stopped clones through the Docker API. It preserves
image, environment, model/overlay/cache mounts, capacity and GPU settings; only
the loopback endpoint, diagnostic mount, Python extension, profiler/log output
directories differ. It validates the requested configuration before stopping
originals. Workers start first, then head. Exact IDs/config/source and endpoint
health are required at every readiness check and original restoration.

Only owned clones may be removed. Partial prepare, failed boot, request failure
and termination pass through cleanup and original recovery. Logs survive failed
collection; a successful original recovery is recorded even when the request
failed. The existing fleet supervisor remains responsible for public restoration
after the payload, including interrupted jobs and handoffs. The opt-in fleet
supervisor hook runs `fleet_observation_cleanup.py` independently of the payload
and removes only the exact session-labelled clones before restore or handoff.
It retries failed teardown while retaining the ticket. This also covers payload
SIGKILL after the supervisor's 20-second cancellation grace; an interrupted run
does not claim exact-original restoration merely because public health recovers.

One canonical priming pass is excluded, followed by five measured passes. Each
pass retains the existing onepass workload: separate questions at 2K and combined
questions at 32K/128K. Thus there are 15 short TTFT samples and five per long
context, with 45 measured quality checks. Every request has a distinct cache salt,
zero prefix hits, raw salted JSON and gzip SSE, token usage and exclusive traffic
checks. All serving requests use the existing 12 GiB memory guard; all nodes must
have 128 GiB disk headroom before the boot. No capacity is reduced to fit a run.

After the profiler-off passes, each context gets one profiled request and a
separate routes request, both limited to one generated token. Prompt and sampling
identity must match the canonical request (first question at 2K). Rank-specific
traces must be stable, newly created, copied and rehashed before leaving the
hold. No routes pass may create profiler traces.

`bench/prefill_observation_run.py analyze DIR` runs after GPU release. It verifies
raw request/response hashes and fresh identities, joins all-rank annotations to
worker coverage, and reuses `tools/trace_prefill_attribution.py` for explicitly
annotated pure-prefill GPU ranges. It preserves sums, unions, overlap, idle/unknown
work and per-layer routing/padding records. Baseline TTFT is reported separately.
These are actual model routes for canonical Korean synthetic requests, **not a
production traffic distribution**. Instrumented category shares and padding
budgets are not a predicted full-model gain. No experimental kernel is enabled.
