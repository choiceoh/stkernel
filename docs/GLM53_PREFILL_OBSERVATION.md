# Current-default prefill attribution

Status: worker instrumentation, request cleanup and CPU trace validation are
implemented. The frozen isolated boot, profiler-off quality/TTFT arm and all-rank
trace transfer/inventory runner are not yet connected. **No GPU run is submitted
and no new speedup is claimed.** M64 and INT8 remain off and deprioritized.

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
standalone CLI that bypasses the unimplemented outer serving lifecycle.

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

CPU tests do not prove that the live compiled serving path visits this hook, that
the target weight mapping resolves, that all ranks emit the expected trace schema,
or that probe overhead fits memory. Live coverage must fail closed if any of those
assumptions fail. Do not promote synthetic CPU checks into runtime acceptance.

The next implementation step is an isolated runner using current default serving
with exact incoming container/configuration/source restoration, all-rank identity,
memory/disk guards and normal fleet ownership. It must preserve profiler-off
fresh 2K/32K/128K TTFT, raw responses and quality independently, then collect
separate profiled and routing requests with unique cache salts, request/prefix/
traffic counter proof, stable fresh trace inventories and source-hashed archives.
Start from the existing serving/lifecycle/client helpers. Do not reuse the failed
M64 gate pin or enable the existing dev-lab replay/reload endpoints.
