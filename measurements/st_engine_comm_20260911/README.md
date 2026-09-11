# TP4 communication investigation — 2026-09-11

**The next bounded architecture candidate is DFlash2's full-vocabulary gather.**
The target forward also has 91 serial sum collectives, but replacing their API
with an asynchronous call would not remove the next block's input dependency.
This investigation changes no production communication policy or model output.

Source baseline: `b4cb126b1908eb796ea4091196bcada21d57dbaf`, incorporating main
through PR #554. The new `probes/engine_comm_profile.py` records its own hash
and the relevant engine/launcher hashes in all four rank reports.

## Observed communication

One four-node pilot, native `st-engine:9391`, GB10, NCCL 2.29.7, production RoCE
environment and 16 channels. Each condition captures a chain of 32 collectives,
warms it, then measures 20 replays with local CUDA events. Values below are the
range of the four ranks' medians, per collective after dividing by 32.

**These are exploratory graph-chain times.** They include NCCL/PyTorch stream
dependencies, graph scheduling and amortized initial rank-arrival skew. They
are not pure network latency or complete-model timing. No independent repeat
or 4-channel/automatic comparison was completed.

| Collective / logical input per rank | Rank median range |
|---|---:|
| int64 MAX, 8 B | 62.73–65.63 µs |
| int64 MAX, 48 B | 63.85–65.35 µs |
| BF16 SUM, 8 KiB / one target token | 72.24–74.59 µs |
| BF16 SUM, 48 KiB / six target tokens | 75.81–77.27 µs |
| BF16 SUM, 192 KiB / 24 target tokens | 200.66–202.17 µs |
| BF16 SUM, 2 MiB / 256 prefill tokens | 836.64–837.65 µs |
| BF16 SUM, 8 MiB / 1,024 prefill tokens | 2,437.31–2,439.50 µs |

All eight shapes on all four ranks passed rank-dependent nonzero SUM/MAX
oracles before timing. Timed sums use zero to prevent repeated in-place
reduction overflow. Every timed condition recorded zero CPU-throttled µs.
This does not exclude OS scheduling, GPU scheduling or other system activity.

## Separating peer wait from transfer work

A second, instrumented graph aligns each iteration with a collective and adds
200,000 GPU sleep cycles on rank 3 before the next 8 KiB SUM. The measured
local delay was 132.29 µs. Ranks 0–2's attributed collective spans grew from
81.02–81.55 µs to 191.17–191.55 µs; rank 3's collective span was 61.28 µs after
its delay. Event durations are computed locally, never by subtracting clocks
from different devices.

This controlled perturbation confirms that a long collective event span can
contain waiting for another rank. It does not quantify the real model's rank
imbalance. Exact model attribution still needs a stable fleet window and
per-stage instrumentation around current KDA/DSA/MoE execution. The injected
graph contains additional timing events and an alignment collective, so its
total time is not directly comparable with the uninstrumented chain.

## GB10 platform constraint

All four logs select NCCL's internal `NET/IB` transport, both RoCE HCAs and
16 communication channels; they report GDR disabled. Independent CUDA driver
queries on all four hosts return success with value 0 for both
`GPU_DIRECT_RDMA_SUPPORTED` and `DMA_BUF_SUPPORTED`.

NVIDIA's Spark-specific support article states that GPUDirect RDMA is not
supported for this unified-memory platform and suggests host-pinned RDMA
buffers for applications needing the supported I/O path. This agrees with
the measured capability flags; forcing GDR or loading `nvidia-peermem` is not
an optimization justified by these results.
[NVIDIA DGX Spark GPUDirect RDMA support](https://nvidia.custhelp.com/app/answers/detail/a_id/5780)

The logs also report unavailable optional mlx5 DMA-BUF symbols. That warning
does not establish the cause of latency, and updating the library would not
establish GPU RDMA capability on a device reporting that capability absent.
The missing external `libnccl-net.so` likewise did not prevent the built-in
IB transport from initializing and completing collectives.

NCCL channel count trades communication parallelism against CUDA resources;
the 16/4/automatic sweep is prepared but has no winner yet.
[NVIDIA NCCL channel documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2243/user-guide/docs/env.html#nccl-min-nchannels)

## Architecture audit and next experiment

`net.forward` calls the vocab-parallel embedding SUM once, then one attention
SUM and one FFN SUM in each of 45 layers: **91 SUM calls per target forward**.
Every output enters the following mHC/input calculation. There is no independent
next-block input available merely by setting `async_op=True`; useful overlap
requires changing the dataflow or scheduling independent requests. Blindly
combining adjacent reductions would change the computation.

The target's greedy sampling already exchanges an int64 MAX candidate per
token. Stochastic sampling still gathers full logits for its distribution and
must retain its sampling semantics.

The drafter has a separate opportunity at `drafter.py:248`: `target.head(h)`
gathers full BF16 logits, converts them to FP32, masks the undecodable tail and
then keeps only 16 candidates per draft position. The real config retained
here declares `selector_top_k=16`; the runtime uses five draft positions.

A proposed local-top-16 → candidate-gather → global-top-16 path would have:

| Representation, five draft positions | Local contribution per rank | Gathered result per rank |
|---|---:|---:|
| Existing BF16 full vocabulary | 387,200 B | 1,548,800 B |
| Candidate FP32 scores + int64 IDs | 960 B | 3,840 B |

The **logical payload reduction is 99.752%**. This is a design calculation,
not measured wire traffic, an implementation, or a speedup result. A simple
two-array implementation adds a second collective; packing into one message
and the local selection cost need measurement.

The global top 16 can be found in the union of local top-16 sets when the same
total ordering is used throughout. Existing `torch.topk` does not promise a
stable ordering of ties. BF16 scores can tie, and candidate IDs feed the next
selector stage. A smaller gathered array can therefore change draft IDs even
when the score values match. A prototype must check masking, nonfinite values,
cross-rank ties and exact candidate/output agreement on real logits, then
measure acceptance and complete-step latency if a tie policy changes.
[PyTorch topk tie contract](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)

This is the preferred next implementation experiment. Channel tuning and
model rank-arrival profiling remain measurements to finish when the fleet is
available; a custom host-buffer communicator is a larger, lower-priority change.

## Run lifecycle, cleanup and remaining limits

The pilot dispatcher acquired the shared `/home/choiceoh/st-fleet.lock` and
used only four private containers, each capped at four CPUs and 2 GiB RAM with
no additional model weights. Initial GPU utilization was 0%; existing GLM
services were present in the initial snapshot. During the pilot period those
services were killed and removed. Available daemon events show the rank-0
service kill beginning at 13:22:02 UTC, before our first diagnostic container
started at 13:22:09 UTC. Our code only removes its explicitly named diagnostic
containers. The actor responsible for the service lifecycle change was not
identified; this dataset must not assume stable service residency.

All four diagnostic processes exited **0** at about 13:22:19 UTC and their
containers were removed. The collection wrapper then failed when attempting
to overwrite srv1's already-written, root-owned result file as the SSH user.
All four completed JSON reports and logs were recovered without rerunning the
experiment; `pilot/dispatch.log` retains the failure. The revised dispatcher
skips that redundant local overwrite, assigns a separate batch name and takes
an after snapshot on failure as well. This revised dispatcher has passed syntax
checking but has not completed a second fleet run.

Our lock was released. Another ST launcher acquired it at 13:22:29 UTC and
started `st-glm53` on the fleet. We did not remove that owner's lock or start
the channel sweep against it. `platform-and-lifecycle.json` records the lock,
active containers, CUDA capability results and relevant daemon events.

`cpu-tests.log`: 190 engine tests discovered, 127 passed, 63 skipped because
local PyTorch/CUDA is unavailable. No new full GPU engine-suite, full-model
quality, acceptance, service latency or throughput result is claimed here.

## Reproduction and evidence checks

`pilot/` retains the original dispatcher, all four raw JSON/log pairs and
the preflight snapshot. `platform_check.py` performs read-only driver and
daemon checks. `summarize.py` validates hashes, all samples, correctness flags,
driver capabilities and successful diagnostic cleanup:

```bash
python3 measurements/st_engine_comm_20260911/summarize.py
```

The revised `fleet_probe.py` runs on srv1 and expects a source archive at
`/tmp/st-tp4-lat-source-f4d7.tar.gz` containing the probe, `engine/base/comm.py`,
the package initializers, `engine/profiles/glm53/net.py`, `launchers/start-st-glm53.sh`
and `launchers/lib/common-tp4.sh`. It fails if another owner holds the fleet
lock. In an available window, its default runs 16/4/automatic in three rounds,
reversing their order in the second round. Reserve a stable fleet window before
using that comparison to select a production configuration.
