# Private rank-cache vote memory comparison

This diagnostic combines the default-off CPU vote from PR #469 with the private
clone/recovery fixes in PR #468. It runs no model request and makes no TTFT or
quality claim. The source branch explicitly includes both parents; it does not
assume either draft PR is in main.

`bench/glm53_cpu_vote_memory_pair.py --revision SHA --out DIR` uses the existing
normal fleet boot holder, all-rank frozen-source check, approved-default
normalization, original snapshot, private clone and exact original stop/start
lifecycle. Submit only with `FLEET_OBSERVATION_CLONES=1` through `fleet.sh run
--gpu`, so the frozen supervisor can also remove an owned clone after payload
cancellation. Existing public containers are not deleted or reconfigured.

The fixed sequence is PRIME=0, BASE0=0, CPU=1, BASE1=0. PRIME constructs the
new-source artifacts and is excluded from the warm comparison. Each remaining
arm requires an unambiguous rank-cache hit on all four ranks, target/drafter FP8
hits without errors or misses, and NCCL initialization diagnostic log lines.
The repeated device-vote baseline brackets time-dependent memory variation; one
CPU arm is still insufficient for a performance verdict.

The dedicated host helper accepts only literal vote policies 0 and 1. Relative
to the original, it rebinds exactly `glm53_rank_cache.py` and
`glm53_startup_cache.py` to read-only files in the frozen checkout, changes the
private observer class/middleware, and fixes `NCCL_DEBUG=INFO` and
`NCCL_DEBUG_SUBSYS=INIT,NET` in every arm. The only difference between warm arms
is the vote flag. All other image, environment, model, hardware, capacity and
overlay identities must match. No arbitrary environment/mount override is
accepted. The manifest still identifies the unchanged public composition; the
two explicit replacement file hashes are separately attested in every arm.

Later clones may be prepared only against the exact same paused original
container ID/config/image. Writable, duplicate or missing cache-module binds,
disabled rank cache, split groups, preexisting CPU vote, foreign clone names,
external NCCL log files and source/composed-file mismatch are refused. Initial
preparation requires 192 GiB disk free (128 GiB reserve plus 64 GiB construction
headroom); later arms still require 128 GiB free. Existing cache eviction policy
is unchanged. No service or model capacity is reduced to create room.

The read-only memory RPC is a local POST with no options. It requires idle
observer hooks and plain TP4, and records all four workers' selected policy,
loaded rank-cache source hash, probe source hash, process PSS, VmPin, pinned
allocator/device counters, host MemAvailable and aggregated `/dev/zero
(deleted)` mapping sizes/counts/PSS. It also records API process memory without
initializing CUDA. No allocator trim, cache purge, model invocation or GPU
collective is added by this receipt; the existing engine control RPC transports
the reports.

After HTTP readiness and idle attestation, each arm settles for 15 seconds and
records three snapshots spaced five seconds apart. Serving logs, configuration,
source identities, cache/NCCL receipts and per-arm completion are saved before
clone removal. Mutable runtime identity, partial ranks, wrong policy/source,
missing counters or cold fallback invalidate the arm. NCCL log counts and the
9408 KiB mapping fingerprint do not establish mapping ownership.

Cleanup must complete before originals restart. The shared paused lifecycle now
accepts a cleanup callback; persistent clone-removal failure stops that restart
and leaves recovery to the fleet supervisor. The outer retry also requires
cleanup before attempting restoration. Memory-only completion has its own
experiment name and marker; the prefill analyzer explicitly rejects it.

Validation: the pinned image's isolated `runc`, network-none, 4 GiB/two-CPU
container passed 39 tests (12 new pair tests, 16 clone/lifecycle/request tests,
11 observer tests), with zero failures/errors/skips and CUDA uninitialized.
Eight additional stdlib lifecycle tests passed locally. The initial local
pair test run had a mock serialization error and missing-dependency skips;
that fixture was corrected before the pinned run. Evidence and exact source
hashes are in
[`pair-preparation/`](../measurements/glm53_rank_cache_cpu_vote_20260908/pair-preparation/).
This validates CPU contracts, not a live boot or live memory saving.

After the normal fleet hold is released and exact restoration is verified,
compare each worker and API PSS/VmPin/mapping totals against both bracketing
baselines. Report the baseline drift and host MemAvailable separately. If the
candidate leaves every node above the unchanged 12 GiB request requirement,
the next step is a separately guarded direct TTFT/quality/profile/routing
capture. This diagnostic itself never enters that request path, and cannot be
counted toward the campaign's unproven 1.40x throughput target.
