# Host reclaim returned too little memory to admit PRIME

Session `glm53observemem0908v1`, frozen code
`37db5eb1fb17b1ed03d5116f02e317fc64ffee4c`. Normal fleet GO was
2026-09-08 11:36:19 KST; exit 1 and release were at 11:49:50 KST.

The approved-default normalization boot completed at 11:43:21. Its launcher
automatically selected GMU0.6329 while retaining KV2000000/maxlen1048576/
1056 blocks. The subsequent private clone preserved the new original's exact
model, overlay, configuration and capacity. This boot is not a matched pair
with the previous observation attempt.

All four reclaim RPCs and the API-process reclaim completed and passed the
source/rank/counter validator. The worker measurements are distinct from the
later API measurement; their host-availability samples are not simultaneous.

| Process | PSS before GiB | PSS after GiB | PSS reduction MiB |
|---|---:|---:|---:|
| Rank 0 | 9.629865 | 9.629373 | 0.504 |
| Rank 1 | 9.722777 | 9.722476 | 0.309 |
| Rank 2 | 9.748816 | 9.748503 | 0.320 |
| Rank 3 | 9.739424 | 9.739096 | 0.336 |
| API | 4.107803 | 3.739773 | 376.863 |

Each worker's PyTorch pinned allocator owned about 96.134 MiB before reclaim,
of which about 96.004 MiB was active. Only about 133.5 KiB was returned per rank.
Device allocated/reserved counters were identical before and after. This rules
out a multi-GiB unused PyTorch pinned pool in this boot. It does not establish
GPU tensor numerical correctness or attribute other native allocations.

The next request-guard sample still refused head (9,323,740 KiB available) and
srv3 (10,402,480 KiB), both below 12,582,912 KiB. PRIME's client was never spawned;
no model request, TTFT, quality, routing report or profiler trace was collected.
The API's PSS decrease is not a one-to-one increase in host MemAvailable and
provides no prefill speedup evidence. Do not repeat this experiment unchanged.

All four exact post-normalization original IDs/config/source identities were
restored, public health passed, the private endpoint was down, and every owned
clone was removed. The supervisor confirmed cleanup and healthy approved
defaults. `post-original-check.json` retains these checks and the release record;
the holder and queue were empty at that read. `remote-source-hashes.json` verifies
all 27 archived remote files byte-for-byte. Reclaim validation and archive syntax/
hash checks passed; no new model code or numerical test was run in this follow-up.

## Read-only allocation follow-up

`post-head-smaps.json` and `post-head-zero-mappings.json` describe the restored
head processes, not the stopped private workers. The head worker has 797
`/dev/zero (deleted)` mappings totalling 3,898,820 KiB. Of these, 384 mappings
are exactly 9,408 KiB each: 3.4453125 GiB in total, plus 384 separate 8 KiB
mappings. The named NCCL shared-memory mappings are a separate, smaller category.

The image reports `nvidia-nccl-cu13` version 2.29.7. Its matching upstream
[initialization code](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/init.cc#L690)
and [protocol constants](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/include/device.h#L87)
give default LL512 KiB + LL1284800 KiB + Simple4096 KiB = 9408 KiB. The
[network transport](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/transport/net.cc#L811)
allocates per-protocol dedicated ring/tree buffers. This is a strong size
fingerprint for investigating NCCL allocations, not a proven attribution of
every mapping, a count of live communicators, or proof that buffers can be freed.

Local `parallel_state.py` creates an EP group for MoE even when expert parallel
execution is disabled, and `cuda_communicator.py` creates PyNccl communicators
for multi-rank groups. The next source audit should identify which groups and
protocol buffers are actually required. No communicator, NCCL setting or other
service was changed. Any later change needs its own correctness and direct
serving evidence; buffer capacity is not a speedup forecast.
