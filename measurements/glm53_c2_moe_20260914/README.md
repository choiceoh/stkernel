# GLM53 C=2 MoE M16 specialization — 2026-09-14

This records the initial tile-only candidate. The current PR955 candidate adds
[C2 direct register scatter](../glm53_c2_direct_20260914/README.md), with stronger
component results and the same requirement for full consumer validation.

**Decision: retain an explicit candidate; do not change the production default.**
Native resources improve and the real-weight numerical checks pass, but the
component observations do not establish a consistent latency advantage. Full
TP4 consumer throughput and acceptance have not been measured for this change.

The candidate extends the existing C1 packed SF6 operand pipeline to exactly
16 target rows (C=2, K=7), selected by `t,r,sf6,batch`. Other row counts retain
their original compiled handles. The production recipe remains `t,r,sf6,q0`.
Selection is by row count: an explicitly enabled 16-row short-prefill call
would also use the candidate. It is not a decode-only phase switch.

With unique top-8 routes, each expert receives at most 16 rows at C=2. An M32
expert tile therefore spends half its rows on padding even when every request
selects that expert. The existing M16 tile also reuses FC1 gate/up inputs,
stores one input stage, and restores packed SF6 directly into MMA registers.
The expert tile loop still handles repeated routes and counts greater than 16.

## Frozen comparison

- Source: `0b37d5ec54340da4dde53a18b49809705f0a27fe`, based on `b55be73f`.
- Final probe: `825f4f122caebb8cee6b054848b2a305346cad47`; engine code is unchanged.
- CPU image (srv2): `sha256:4f5f6e884e9a9e4ef4af532546a8b2c8a8e7dd710ce5d761bfe16d1377e4fb25`.
- GPU image (srv4): `sha256:a0709718a4b26894d5cac5f417a0f74fde4198531e97c9b3171b0bcf513a6a9f`.
- Shared seed: `sha256:a9b53fd066bb4fa0c4d12982f7c5dcdd5a2591900a0088c3ffeb41ba0868425c`.
- Torch 2.13.0+cu132, CUDA toolkit 13.2.1, PTXAS 13.2.78, GB10 sm_121a.
- Fleet tickets `st-c2-moe0914`, `st-c2-shared0914`, `st-c2-events0914`;
  single GPU, 8 GiB budget; no production restart.

## Native compiler gate

[Raw report](cpu.json), [compiler output](cpu.log), [CPU tests](cpu-tests.txt).
All three native handles compile; all 27 selected CPU tests pass.
The [capture regression test](capture-tests.txt) also passes (28 CPU tests total).

| C=2 resource | M32 baseline | M16 candidate |
|---|---:|---:|
| FC1 input shared storage, bytes | 24,576 | 4,096 |
| Dynamic shared memory, bytes | 101,376 | 91,136 |
| Registers per thread | 117 | 115 |
| Stack / local memory, bytes | 0 / 0 | 0 / 0 |
| Static SASS instructions | 6,148 | 3,590 |

This is a resource comparison, not a serving-speed result. C1 has the same
M16 resource shape and the same compiled cache key with or without `batch`.

## GPU method

The `moe_pair` lane of `probes/engine_kernel_check.py` loads actual rank-3 L3
expert and shared weights, including all four ModelOpt scale tensors. Both
arms use one weight pack, input addresses, the actual `Glm53Net._moe` consumer,
and the production output cast/add. C1 retains shared overlap; C2 retains its
ordinary shared chain. Collectives are identities.

All numerical checks precede timing. They include changed activations and
routing weights, reversed replay order, poisoned outputs, duplicate-route
stress, unique expert counts from 8 through 128, zero routed weights, and the
actual L3 router on independent/correlated synthetic activations. The existing
relative-maximum threshold is 0.001. Timing uses B/A/A/B with warm caches and
128 MiB eviction outside the CUDA event interval.

This scope does not measure full-model quality, speculative acceptance, NIC
traffic, TTFT or output tok/s. Those require a matched consumer bracket.

## Results and rejected conclusions

The corrected v3 probe completes all three variants: 63 numerical cells,
maximum observed relative difference **0**, and peak allocated GPU storage
below 2.1 GB. Each variant compares the ordinary M32/serial-shared FFN against
M16 plus its named shared-expert policy. Shared fusion/overlap are probe-only.

[Tile result](tile-v3.json), [serial fused shared](shared-serial-v3.json),
[overlapped shared](shared-overlap-v3.json), [complete log](v3.log),
[derived comparison](v3-summary.json).

These are descriptive component observations. Another ST engine was active on
srv4 (`st-engine:bracket-c58a8eb534ee`; `/metrics` on srv2:8001 reported one
running request). The single-GPU lane admits probes beside that engine based
on available memory; this is not GPU isolation. Cross-run absolute latencies
therefore must not be compared or promoted to a serving-speed verdict.

For the actual L3 router on the synthetic input fixtures, mean latency change
across the two samples per arm of each B/A/A/B bracket is:

| Candidate | Independent, U=98 warm / evicted | Correlated, U=11 warm / evicted |
|---|---:|---:|
| M16, original shared chain | −0.71% / −1.57% | +1.30% / +0.45% |
| M16, fused serial shared | −2.95% / −1.36% | +0.52% / +0.46% |
| M16, fused overlapped shared | −2.17% / −2.47% | +2.34% / −0.39% |

Positive means slower. The first row changes only the expert tile/pipeline;
the latter two change that tile and shared execution. None proves a universal
C2 win; no candidate is made the default and no 2× throughput claim is made.

The original [v2 tile report](gpu.json) and [summary](tile-summary.json) are
retained. The subsequent [serial v2 report](shared-serial-v2.json) passes
numerics, but [overlap v2](shared-overlap-v2.json) fails with a stream-parent
collision after its numerical cells. [Its log](shared-v2.log) is retained.
Those timings are not adoption evidence. Repeated `_capture` calls allocated
fresh pooled streams, eventually aliasing an existing `SharedOverlap` stream.
The helper now reuses one warm/capture stream per device. Timing v3 also places
both warm and evicted events inside their graphs, excluding Python enqueue
gaps from the warm column; v2 timed those gaps with an outer event interval.

## Reproduce

In the pinned CPU image, without a GPU:

```sh
CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a PYTHONPATH=/repo \
  python3 probes/engine_moe_sf6_compile.py --batch-reform --sass --output /out/cpu.json
```

From the immutable source checkout on srv2, through fleet admission:

```sh
ST_IMAGE=sha256:a0709718a4b26894d5cac5f417a0f74fde4198531e97c9b3171b0bcf513a6a9f \
ST_PROBE_TREE=st-c2-moe-825f4f12 \
bash bench/fleet.sh run --gpu --detach st-c2-events0914 10 'C2 same-runtime FFN comparison' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
    --lanes moe_pair,moe_pair_serial,moe_pair_overlap \
    --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors \
    --output /cache/st-c2-moe-825f4f12.json
```
