# C2 direct work mapping — rejected, 2026-09-14

The work-map change was removed from the execution path: its additional
register and runtime branch did not establish a useful latency improvement.
These artifacts preserve the tested implementation and its numerical proof.

At C2/K7, unique top-8 routing bounds each expert to 16 rows. The candidate
computed expert/slice ownership directly instead of scanning `row_counts`.
The existing device counter recorded experts exceeding 16 rows on each replay;
the existing grid barrier published it, and duplicate routes selected the
original complete tile loop. No host dispatch or extra allocation was added.

Source `984e4eccf9077492a0c0ea27aa77638ac682185a` contains the experiment.
`72a6a881` failed native CuTe lowering because branch results were not
initialized before staged control flow; [the failure report](cpu-initial-failure.json)
is retained. The corrected [CPU report](cpu.json) and [log](cpu.log) pass all
four handles. The candidate uses 116 registers, compared with 115 for M16 and
117 for ordinary M32, with zero stack/local storage. Shared storage remains
91,136 bytes. [All 27 focused CPU tests passed](cpu-tests.txt).

Fleet ticket `st-c2-work0914` ran for 76.1 seconds on one srv4 GB10 and succeeded.
The [work-only comparison](work-only.json) is M16 versus M16 plus work mapping;
the [bundle comparison](gpu.json) is ordinary M32 versus that combined candidate.
[The complete GPU log](gpu.log) retains both. All 42 numerical cells have
maximum relative difference zero; all 18 replay overflow-counter checks match.

Mean of the two medians per arm in each B/A/A/B bracket, in microseconds:

| Work map only, C2 fixture | Cache | M16 | M16 + map | Change |
|---|---|---:|---:|---:|
| Actual L3 router, independent synthetic inputs, U98 | warm | 1581.648 | 1594.352 | +0.803% |
| Same | evicted | 1637.216 | 1616.744 | -1.250% |
| Actual L3 router, correlated synthetic inputs, U11 | warm | 267.776 | 269.064 | +0.481% |
| Same | evicted | 331.288 | 331.736 | +0.135% |

Other U8–128 work-only evicted changes range from -0.479% to +0.100%.
The U98 bracket itself drifts; its isolated negative average is not a winner.
The combined tile/map candidate still regresses the correlated fixture.

Images are the same pinned CUDA 13.2.1 images as the preceding
[M16 experiment](../glm53_c2_moe_20260914/README.md): CPU
`sha256:4f5f6e884e9a9e4ef4af532546a8b2c8a8e7dd710ce5d761bfe16d1377e4fb25`
and GPU `sha256:a0709718a4b26894d5cac5f417a0f74fde4198531e97c9b3171b0bcf513a6a9f`.
The GPU had no serving ST container, but existing Nemotron/Chronos processes
retained GPU allocations; this was not exclusive GPU isolation. Pre-run device
utilization was 0%. The TP4 fleet remained held by another session.

Reproduce from the frozen source on srv2 through fleet admission:

```sh
ST_IMAGE=sha256:a0709718a4b26894d5cac5f417a0f74fde4198531e97c9b3171b0bcf513a6a9f \
ST_PROBE_TREE=st-c2-work-984e4ecc \
bash bench/fleet.sh run --gpu --detach st-c2-work0914 10 'C2 work ownership' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
    --lanes moe_pair,moe_pair_work \
    --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors \
    --output /cache/st-c2-work-984e4ecc.json
```

These are captured FFN component comparisons, including shared experts and
output finalization. No full-model tok/s, TTFT, acceptance or quality verdict
is implied. The production recipe was not changed.
