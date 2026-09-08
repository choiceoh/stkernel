# GLM53 rank-cache CPU readiness vote

Status: experimental, default off. This removes a possible startup allocation
trigger to unblock private prefill observation. It is not a measured prefill
optimization, and the campaign's 1.40x direct-throughput target remains unproven.

The preceding host-cache reclamation experiment stopped before its first model
request because the head and srv3 remained below the unchanged 12 GiB request
guard. Worker pinned-cache reclamation saved less than 1 MiB per rank; API PSS
dropped about 377 MiB. That experiment and its restoration receipts are preserved
in [PR #468](https://github.com/choiceoh/stkernel/pull/468). Repeating that same
reclamation is not the next experiment.

## Why this candidate

The installed-source audit is recorded in
[`installed-source-audit.json`](../measurements/glm53_rank_cache_cpu_vote_20260908/preparation/installed-source-audit.json).
It read source files and selected nonsecret configuration from the restored
public head, without importing torch or running a collective. The image is
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`,
vLLM `0.1.dev20051+g487ecf187`, NCCL `2.29.7`.

- WORLD already has a CPU Gloo process group. Its device group is created
  without `device_id` in the current non-split-group path, and its coordinator
  sets `use_device_communicator=False`.
- The rank-cache readiness vote nevertheless creates a CUDA int32 and performs
  a MIN all-reduce on WORLD's device group, including on all-rank cache hits.
- The inspected warmup paths use WORLD object broadcasts and barriers; those
  methods use its CPU group. The InstantTensor source loader uses WORLD's device
  group and must retain it for cold or partial-miss fallback.
- Removing the EP group is inappropriate: the installed MoE weight-quantization
  helper performs its TP-wide MAX on EP's device group. Group lifetime and
  collective ordering also preclude simply aliasing TP and EP.

This makes a CPU readiness vote a concrete way to avoid initializing a GPU
communicator solely for cache control. Source inspection does not establish how
much retained memory it saves, or exclude every indirect consumer. The earlier
9408 KiB mapping-size fingerprint does not prove mapping ownership, communicator
count, or reclaimability; no fraction of its 3.45 GiB total is booked as savings.

## Behavior and limits

`VLLM_GLM53_RANK_CACHE_CPU_VOTE=0` preserves the device-group vote. Explicit `=1`
uses the existing WORLD CPU group with the same int32 MIN reduction. All ranks
must select the same policy before loading. A missing or non-Gloo CPU group
fails before collective submission instead of choosing a different local path.
Single-rank loading needs neither group. No group is created, removed or shared
by the candidate; no object collective is added.

The control flag is excluded from artifact environment identity, so a same-source
0/1 comparison can reuse the same exact artifacts. The implementation's file
hash remains in the rank-cache key: the first boot of this new source still
invalidates old rank artifacts. Neither an old-source warm boot nor this initial
cold boot is a matched baseline for the candidate's warm-hit memory effect.
Cache restoration, loaded bytes, alias handling, post-load hooks, and source
fallback behavior are unchanged.

`VLLM_DISTRIBUTED_USE_SPLIT_GROUP` is unset on the inspected public head and
defaults to false in the installed image. The experimental CPU vote requires
the plain Gloo group; mixed-backend split groups are not supported. It does not
promise to avoid WORLD NCCL allocations on a cold/partial-miss source load, nor
to remove TP/EP communication buffers. CPU consensus tests do not prove GPU
stream ordering, inter-node Gloo latency, live quality or startup memory savings.

## Validation

The pinned image ran `tests/test_glm53_startup_artifacts.py` in an isolated
`runc` container with no network/GPU access, 4 GiB RAM and two CPU cores:
**30 tests passed, zero failures/errors/skips**, torch `2.13.0+cu130`, CUDA
uninitialized. Raw output, command and exact input hashes are in
[`preparation/`](../measurements/glm53_rank_cache_cpu_vote_20260908/preparation/).

The real four-process Gloo test uses distinct legacy and CPU process groups.
Both yield `[true, false, false]` for all hits, a rank-2 miss, and all misses on
every rank. During CPU votes the device-group reference is deliberately invalid
and CUDA device selection raises. Additional checks cover unsupported groups,
single-rank behavior, exact artifact reuse across 0/1/0, implementation-source
invalidation, existing partial-miss fallback, aliases and post-load hooks.

The GLM overlay was recomposed. Local core checks passed after recomposition,
including profile forwarding/readers and source/composed-file consistency.
Those broad local checks skipped tensor-dependent paths because host torch is
absent; the focused pinned-image result above has no skips.

## Next live gate

Before a new request experiment, prepare a bounded same-source warm-hit 0/1
boot comparison through normal fleet ownership. Freeze source, image and all
four nodes' configuration, and vary only this control flag. Account explicitly
for first-source cache construction and disk reserve; retain original-serving
restoration and supervisor cleanup. Do not silently extend the private observer's
current exact-environment clone contract to allow arbitrary overrides.

For both warm arms require all four rank-cache hits, readiness, exact selected
policy and unchanged capacity. Capture worker PSS, VmPin, `/dev/zero` mapping
sizes/counts, available host memory and NCCL initialization evidence at matched
phases. Cold or partial-miss fallback remains a separate compatibility check.
API reclamation, if used, must be identical in both arms and reported separately.

Only proceed to the existing direct 2K/32K/128K TTFT, quality, profile and routing
capture when every node passes the unchanged 12 GiB request guard. Keep the
128 GiB disk reserve and all other services/capacity intact. An allocation
reduction alone is not a throughput gain; no new boot or GPU job was submitted
for this CPU preparation.

## September 8 memory attempt

The normal fleet run glm53cpuvotemem0908v1 reached its private PRIME server but
failed before the first memory sample because the client had not registered
the new memory endpoint. No paired memory, TTFT or quality result exists.
Exact original recovery passed on all four nodes; the subsequent supervisor
public refresh and release completed at 13:34:54 KST.

The dedicated memory client now registers only its empty-body JSON endpoint.
A test exercises its actual HTTP request/parsing implementation. The local
13-test run has two environment skips; pinned-image validation and any retry
remain pending. [Raw failure and restoration evidence](../measurements/glm53_rank_cache_cpu_vote_20260908/attempt1/README.md)
close this attempt without claiming a performance gain.
