# C2 M16 with direct register scatter — 2026-09-14

This records the first direct-output candidate. The current candidate also
[retains scatter state across the FC2 sweep](../glm53_c2_scatter_reuse_20260914/README.md).

**Retain this C2 candidate in PR955; production remains unchanged.** In the
final actual-weight FFN comparison, two independent groups of eight related
inputs are 5.45% faster with warm caches and 4.44% faster after eviction than
ordinary M32. The additional scatter change alone improves the prior M16
candidate by 3.57% / 3.48% in that fixture. This is component evidence, not
full-engine throughput or speculative-acceptance evidence.

The explicit `t,r,sf6,batch` recipe selects M16 plus direct register scatter
at exactly 16 target rows (C2/K7). With the existing engine prefill option it
is `t,r,sf6,batch,q0`; `q0` does not change this static decode comparison.
Production stays `t,r,sf6,q0`. C1 and other row counts keep their original
effective configuration and cache keys within the same source build. A short
prefill of exactly 16 rows also qualifies if `batch` is explicitly selected.

The output path reuses the existing validated register-pair scatter: it skips
the FC2 output shared-memory store, publication barrier and reload. Both BF16
rounding boundaries, saturated route contributions, FP32 atomic accumulation,
shared-expert execution and final output conversion remain in place. The
barrier that protects input and route-metadata reuse is retained. Duplicate
routes still use the complete expert M-tile loop.

The tested direct work-map alternative did not pay for its extra register and
branch. Its execution code was removed; its [rejection evidence](../glm53_c2_work_map_20260914/README.md)
and frozen source remain available.

## Native compiler and CPU proof

Engine source: `3ecbfd14d1569fa128a477a616d1945e302e5d85`.
Final paired-request probe: `82f5a121f1e54137d9eba2ccacb44357cb24e65e`.
Only the input fixtures change in the latter; every recorded engine hash
matches the final source tree.

- CPU image: `sha256:4f5f6e884e9a9e4ef4af532546a8b2c8a8e7dd710ce5d761bfe16d1377e4fb25`.
- GPU image: `sha256:a0709718a4b26894d5cac5f417a0f74fde4198531e97c9b3171b0bcf513a6a9f`.
- CUDA 13.2.1, PTXAS 13.2.78, Torch 2.13.0+cu132, GB10 sm_121a.
- Actual ModelOpt rank-3 L3 expert/router/shared weights; their hashes and all
  four scale tensors are recorded in each GPU identity record.
- [25 focused CPU tests pass](cpu-tests.txt). [All four native handles](cpu.json)
  compile, including actual output-coordinate validation and SASS inspection;
  [compiler log](cpu.log). CPU compilation creates no GPU context.

| Resource | Ordinary C2 M32 | Previous C2 M16 | C2 M16 + direct output |
|---|---:|---:|---:|
| Registers/thread | 117 | 115 | 96 |
| Stack/local bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| Dynamic shared memory bytes | 101,376 | 91,136 | 91,136 |
| FC1 input staging bytes | 24,576 | 4,096 | 4,096 |
| Static SASS instructions | 6,148 | 3,590 | 4,518 |
| Static BAR.SYNC instructions | 48 | 22 | 18 |

Direct output reduces registers 16.5% relative to the prior M16 candidate;
its unrolled stores increase static instruction count. This resource change
alone is not a latency verdict. The C1 handle remains at 115 registers,
91,136 shared bytes and 3,590 static instructions.

## Same-runtime FFN results

The two canonical single-GPU tickets succeeded and released their holds:
`st-c2-direct0914` (77.6 seconds) and `st-c2-paired0914` (39.7 seconds).
The first has 42 exact numerical cells; the final has 44, for **86 cells with
maximum relative difference zero**. Peak allocated GPU storage stays below
2.1 GB. Poisoned outputs, repeated/reversed replay order, changed inputs,
changed routes, duplicate-route overflow and zero routed weights are included.

Each bracket is B/A/A/B. Each arm entry is the median of 32 graph replays,
and the values below average the two arm entries. Timing v3 places events
inside the FFN graph; 128 MiB eviction and host enqueue gaps are outside.
Actual shared experts, output cast/add and finalization are timed. Router
selection is outside timing; collectives are identities.

Final bundle comparison against ordinary M32, in microseconds:

| Actual L3 router on synthetic inputs | Cache | Ordinary | Candidate | Change |
|---|---|---:|---:|---:|
| Two groups of 8 related inputs, U20 | warm | 393.120 | 371.680 | -5.45% |
| Same | evicted | 447.032 | 427.184 | -4.44% |
| One group of 16 related inputs, U11 | warm | 265.472 | 230.600 | -13.14% |
| Same | evicted | 327.160 | 290.128 | -11.32% |
| 16 independent inputs, U98 | warm | 1580.864 | 1553.152 | -1.75% |
| Same | evicted | 1633.320 | 1606.128 | -1.66% |

The explicit U8 fixture improves 24.78% warm / 17.14% evicted. This concentrated
case is not a generic C2 speedup. Across all final C2 fixtures the combined
candidate's descriptive mean is lower than ordinary M32; small differences
are not treated as robust wins. C1 is an unchanged-code control and still
shows timing variation up to 2.28% in these runs.

Direct scatter by itself does not help every occupancy: in the final separate
M16 comparison, independent U98 warm is +0.48% and U128 evicted is +0.77%.
The initial combined U128 warm bracket was +1.80%; repeating this unresolved
case with the added two-request fixture gave -1.54%. Both records are retained,
and no noisy sample is silently discarded. The supported additional benefit
is on concentrated/related routes, with the strongest C2-specific evidence
coming from the two-request grouping above.

Artifacts: [final bundle](gpu-paired.json), [final scatter-only](direct-paired.json),
[final log](paired.log), [initial bundle](gpu-first.json),
[initial scatter-only](direct-first.json), [initial log](first.log),
[all derived comparisons](summary.json).

No ST serving container was present, but Nemotron and Chronos retained GPU
contexts; this is not exclusive isolation. [Post-run device facts](gpu-after.json)
show 0% GPU utilization and no ST serving container. The TP4 fleet remained
leased to another session, so no consumer boot was taken. Full-model output
quality, acceptance, TTFT, NIC behavior and output tok/s require a matched TP4
consumer run before promotion. No production default or concurrency limit is
changed by this PR.

## Reproduce

CPU, in the pinned image with devices hidden:

```sh
CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a PYTHONPATH=/repo \
  python3 probes/engine_moe_sf6_compile.py --batch-reform --sass --output /out/cpu.json
```

From the frozen final probe checkout on srv2, through normal fleet admission:

```sh
ST_IMAGE=sha256:a0709718a4b26894d5cac5f417a0f74fde4198531e97c9b3171b0bcf513a6a9f \
ST_PROBE_TREE=st-c2-paired-82f5a121 \
bash bench/fleet.sh run --gpu --detach st-c2-paired0914 10 'C2 paired-request FFN' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
    --lanes moe_pair,moe_pair_direct \
    --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors \
    --output /cache/st-c2-paired-82f5a121.json
```

The scatter-only arm uses `c2_direct_scatter=False` for its M16 reference; that
switch preserves the old M16 cache key. The combined arm uses ordinary M32.
Both run the actual `Glm53Net._moe` FFN consumer on one shared weight pack.
