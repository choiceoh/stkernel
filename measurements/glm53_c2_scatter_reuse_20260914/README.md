# C2 scatter state reuse — 2026-09-14

The explicit C2 candidate now retains output route metadata across the FC2
sweep and synchronizes MMA warps once at sweep retirement. It keeps **96
registers with no spills**, reduces static SASS from 4,518 to 3,861 instructions
(-14.5%), and passes all 44 final numerical cells exactly. In the two-request
fixture, the additional mean latency reduction versus the preceding direct
output candidate is only **1.28 / 1.30 us (0.35% / 0.31%)**, warm / evicted.
This small observation is not presented as a robust serving-speed gain.

The production recipe remains `t,r,sf6,q0`; this remains an explicit
`t,r,sf6,batch,q0` candidate at exactly 16 target rows. No concurrency limit,
speculative width, weight format or arithmetic boundary changes.

## What changes and why it is bounded

The real CuTe output-coordinate validator enumerates every register pair in
all 128 MMA lanes. Each M16 lane owns two output rows. It derives and verifies
the same pair-to-row map in every lane; no unverified lane-coordinate formula
is substituted. The kernel reads each row's token destination and route
weight once, using volatile shared loads on every work item, then retains the
four values through all 16 FC2 output-column tiles. It does not cache them
across experts, M-tiles or graph replays.

Direct output has no shared output buffer to publish per column. FC2's
two-stage producer/consumer pipeline continues to protect B and packed SFB.
A2 is read into registers before the sweep and remains unchanged throughout
it. The final MMA retirement barrier prevents any warp from overwriting A2
or route metadata for the next item. Thus the output-retirement barrier moves
from once per output-column tile (16 times) to once per work item. FC1,
quantization, pipeline and grid barriers remain in place.

`c2_scatter_reuse=False` compiles the preceding direct-output reference. The
normalizer requires direct scatter and the complete M16 register-SF6 operand
path, and is idempotent. C1, ordinary C2 M32, tile-only M16 and the preceding
direct-output reference retain identical SASS hashes to the earlier proof.

## Frozen proof

- Source: `87eabd5d2d6c1bffeb9cd643e133afec4cab6dc5`.
- CPU image: `sha256:4f5f6e884e9a9e4ef4af532546a8b2c8a8e7dd710ce5d761bfe16d1377e4fb25`.
- GPU image: `sha256:a0709718a4b26894d5cac5f417a0f74fde4198531e97c9b3171b0bcf513a6a9f`.
- CUDA 13.2.1, PTXAS 13.2.78, Torch 2.13.0+cu132, actual GB10 sm_121a.
- [25 focused CPU tests](cpu-tests.txt), [five native handles](cpu.json),
  [compiler output](cpu.log). No GPU context is used for the native CPU gate.
- GitHub engine check on this source passed in 6m32s:
  [run 34850056823](https://github.com/choiceoh/stkernel/actions/runs/34850056823).
- GPU ticket `st-c2-reuse0914b` succeeded in 82.3 seconds and released its hold.
- The actual rank-3 L3 ModelOpt weights, router and shared experts have the
  same hashes as the previous experiment. All recorded source hashes match
  the frozen tree.

| Resource | Previous direct output | Sweep reuse |
|---|---:|---:|
| Registers/thread | 96 | 96 |
| Stack/local bytes | 0 / 0 | 0 / 0 |
| Dynamic shared memory bytes | 91,136 | 91,136 |
| Static SASS instructions | 4,518 | 3,861 |
| Static plain LDS instructions | 134 | 10 |
| Static BAR.SYNC instructions | 18 | 15 |

The LDS row counts that specific static opcode, not all memory operations.
Weight traffic is unchanged. Static instruction savings do not translate
directly to the same percentage of FFN latency.

## Same-runtime FFN comparison

Both comparisons include shared experts and output cast/add, using actual
`Glm53Net._moe` with identity collectives. All numerical checks precede timing:
changed inputs and routes, repeated/reversed graph replay, poisoned output,
duplicate-route multi-tile stress and zero routed weights. **44 cells have
maximum relative difference zero**, with peak allocation below 2.1 GB.

The real-router fixture has two independent groups of eight related synthetic
inputs, selecting 20 experts. Mean of the two medians per arm in a B/A/A/B
bracket (32 replays per entry), in microseconds:

| Comparison | Cache | Reference | Candidate | Change |
|---|---|---:|---:|---:|
| Previous direct output -> sweep reuse | warm | 366.360 | 365.080 | -0.35% |
| Same | evicted | 417.712 | 416.416 | -0.31% |
| Ordinary M32 -> complete C2 candidate | warm | 379.560 | 368.760 | -2.85% |
| Same | evicted | 432.624 | 417.472 | -3.50% |

The complete candidate also measures -13.74% / -11.93% for one group of related
inputs (U11), and -1.95% / -2.41% for independent inputs (U98). Those are distinct
fixtures, not interchangeable C2 speed claims. Cross-run absolute times must
not be compared to the preceding experiment. The reuse-only U32 evicted and
U98 evicted means are +0.18% and +0.08%; the unchanged C1 control varies from
-0.77% to +0.93% in that comparison. The small additional latency benefit is
therefore less decisive than the native instruction reduction.

[Complete candidate](gpu.json), [reuse-only comparison](reuse-only.json),
[all derived results](summary.json), [complete GPU log](gpu.log).
[Post-run GPU facts](gpu-after.json) retain resident processes and container
state. Other GPU contexts remain; there is no exclusive isolation. The TP4
fleet remains assigned to another session, so full-model tok/s, TTFT, output
quality and acceptance are still unmeasured.

## Compiler failures retained

The [initial CPU report](cpu-initial.json) and [second report](cpu-second.json)
retain a CuTe staged-variable collision in the reference path: output
coordinates and row variables were also introduced in an optional pre-loop
branch. Binding immutable coordinates before the loops and using a distinct
cache-row variable fixes both paths. The new candidate compiled throughout;
all five reference/candidate handles pass on the final source.

The premature first GPU launch `st-c2-reuse0914` reached the same reference
compile failure and was cancelled; it finished with code 1 after 35.9 seconds.
[Its log](initial-gpu-failure.log) and [partial artifact](initial-gpu-failure.json)
are retained. It supplied no C2 timing evidence. The successful replacement
was launched after confirming the complete five-handle native gate.

## Reproduce

In the pinned CPU image, with devices hidden:

```sh
CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a PYTHONPATH=/repo \
  python3 probes/engine_moe_sf6_compile.py --batch-reform --sass --output /out/cpu.json
```

From the frozen checkout on srv2 through normal fleet admission:

```sh
ST_IMAGE=sha256:a0709718a4b26894d5cac5f417a0f74fde4198531e97c9b3171b0bcf513a6a9f \
ST_PROBE_TREE=st-c2-reuse-87eabd5d \
bash bench/fleet.sh run --gpu --detach st-c2-reuse0914b 10 'C2 scatter state reuse' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
    --lanes moe_pair,moe_pair_reuse \
    --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors \
    --output /cache/st-c2-reuse-87eabd5d.json
```
