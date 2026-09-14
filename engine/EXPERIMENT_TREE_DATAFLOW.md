# GLM tree verification and persistent MLP experiment

This is an explicit **eager, greedy, one-sequence experiment**. It connects
one DFlash candidate pass, expert-cost tree selection, FP32 KDA branch
verification, the existing target projections/MoE, and accepted-cache commit.
It is callable through `engine.profiles.glm53.tree_decode.decode_once`.
There is no serving default or environment flag selecting it.

## Preserve the target's actual precision

The ordinary prepared `DenseLinear` decode lane is **W4A8**. Its `W4Pack`
contains E2M1 nibbles, signed compact group-16 scale codes and FP32 per-row
undo scales. The native lane expands W4 to E4M3 on chip and quantizes BF16
inputs to E4M3 in groups of 128. `PersistentW4A8` reads those same pack
objects, including GPTQ packs after the raw BF16 arena has been retired.
It preserves GATE|UP ordering, scale/reciprocal arithmetic, BF16 projection
and activation boundaries, and the existing TP output reduction.
It does not re-quantize weights or apply smoothing a second time.

ModelOpt fixed-one-expert dense layers and routed experts use a separate
**W4A4 NVFP4** convention: UP|GATE, interleaved E4M3 scales and global
activation/weight scales. `PersistentNVFP4` binds only that fixed-one-expert
dense layout, including tile-major weights and the actual SF6 owner after
raw scales have been retired. Routed MoE retains the target's current lane.
The W4A8 and W4A4 bindings explicitly refuse each other's layouts.
`PersistentDense` is only the unprepared BF16 scheduling oracle/prototype.

## Three implemented pieces

1. `speculative_tree.py` expands DFlash's existing selector lattice using
   parent-conditioned codebook scores and a best-first path-mass frontier.
   A peaked seven-step proposal can reach depth seven within eight nodes;
   breadth-first expansion previously exhausted the budget on shallow
   siblings. An exact integer tie key allows top-k instead of full sorting,
   and only retained edges cross to the CPU. Selection is ancestor-closed and
   ranks proposal mass against additional predicted layer/expert bytes.
   Unknown routes receive a conservative charge. Proposals never prune the
   target router or alter greedy verification. `RouteTable` is a bounded
   empirical baseline keyed by predecessor/token/depth and runtime ID;
   it is **not a trained hidden-state router predictor**. Coverage, actual
   expert usage and pre-update prediction recall/precision are recorded.
2. `tree_kda.py` and `kernels/kda/tree.py` store FP32 key, channelwise decay
   and update factors. Each native CTA owns its head/value slice through
   the whole tree. DFS traversal carries the current FP32 state along chains
   and reconstructs only after backtracking to another branch (8 rather
   than 36 full-state updates for an eight-node chain). One prepared topology
   serves all layers and a fused convolution gathers only the last four
   ancestors/prefix taps. Only ancestors influence a node. The final accepted
   state and any crossed prefix-block checkpoint are materialized before
   canonical writes. Convolution history and DSA pools/tails are private
   to each branch. DSA compresses each newly completed pool once, scores all
   queries against a shared prefix/branch key bank in one indexer call, and
   runs one sparse-MLA query batch over bounded branch-private latent rows.
   Top-k groups retain each path's exact physical column count for tie
   compatibility. Commit reuses the verified pool bytes. KDA stays FP32.
3. `w4a8_pipeline.py` replaces the W4A8 queue with two static stages. FC1
   publishes BF16-boundary SwiGLU output directly as group-128 FP8; FC2
   keeps ordered FP32 partials in registers. Stream ordering replaces device
   polling and the per-layer host completion vote. One prepared workspace
   is reused across layers/steps of matching geometry. The low-level MLP
   executor supports graph capture and returns a borrowed output view;
   concurrent calls sharing that workspace are forbidden. The old queued
   W4A8 executor remains an explicit comparison oracle. W4A4 and BF16
   prototypes still use the bounded acquire/release worker queue in
   `tile_dataflow.py`. Actual MoE route metadata uses one host transfer for
   all layers at commit instead of one synchronization per routed layer.

## Explicit entry point

For a prepared ordinary dense target, create the binding once and pass it
to successive steps; `field` is the existing DFlash feature ring:

```python
from engine.modules.speculative_tree import RouteTable
from engine.modules.w4a8_dataflow import PersistentW4A8
from engine.profiles.glm53.tree_decode import decode_once

binding = PersistentW4A8()
routes = RouteTable(runtime_id)  # must identify model, packs, selector and build
result = decode_once(
    drafter, caches, field, seq=seq, slot=slot, anchor=anchor, context=context,
    budget=remaining, runtime_id=runtime_id, predictor=routes,
    bytes_per_expert=rank_expert_bytes, fixed_node_bytes=rank_fixed_node_bytes,
    nodes=8, width=2, persistent_mlp=binding,
)
context, anchor = result["context"], result["tokens"][-1]
```

All ranks must call the step with the same descriptor. The caller exclusively
owns the sequence and reserves `context + spec_k + 1` tokens first. The root
is an already emitted, uncomputed anchor. Returned tokens are newly emitted
outputs; the last output becomes the next anchor, not a committed input.
EOS and output budget stop branch traversal before extra cache writes.
The accepted target features alone enter the existing drafter ring.

Limits are explicit: up to 32 tree rows (to preserve W4A8 decode dispatch), depth at most
`spec_k`, single folded W4 pack per projection, no active calibration
observers, SM121, and caller-bounded scratch. Sampled rejection, C=4 batching,
production graph capture, prefix snapshot publication and HTTP scheduling
are not implemented by this API. The caller retains those responsibilities.
Proposal/commit metadata still synchronizes with the host; the W4A8 MLP
has no host completion read. Capturing that component does not capture the
full tree step or integrate it with serving.

## Reproducible checks without a GPU queue

Run from the repository root with torch CPU, numpy, safetensors and Triton
(the recorded offline compiler uses torch 2.14.0+cpu and Triton 3.8.0):

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_dataflow_mock.py --output /tmp/tree-mock
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_dataflow_compile.py --output /tmp/tree-compile
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 PYTHONPATH=. python probes/engine_tree_dataflow_interpreter.py --output /tmp/tree-addresses.json
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_fastpath_bench.py --iterations 20 --output /tmp/tree-cpu-ab.json
```

The mock records per-test CPU latency, source hashes and storage arithmetic.
The offline compiler emits SM121 cubins/PTX and checks FP8 MMA for W4A8,
block-scaled FP4 MMA for W4A4, and acquire/release publication. Staged W4A8
also rejects device code containing atomics, polling or local-memory spills. No CUDA context
is initialized. The interpreter checks actual native packed readers against
independent byte references. Its W4A8 activation check executes native
scaling arithmetic and uses torch's FP8 RTNE conversion: Triton 3.8's CPU
interpreter rounds halfway FP8 values upward, unlike the generated `cvt.rn`.
It is not a device-conversion numerical result.
The convolution interpreter checks the actual ancestry gather and FP32 sum;
libdevice activation and native rounding remain device checks. The matched
CPU probe uses #947 and the current implementation with common target
operators, plus ordinary linear target reference calls. It records output
hashes, head-token agreement and per-bracket samples. These small CPU models
are not production default performance, real-weight quality or acceptance.

`tests.test_engine_tree_dataflow_gpu` provides opt-in native numerical and
repeated-invocation checks, exact carry/reconstruction comparisons and W4A8
graph replays with changed inputs for a separately authorized GB10 window. They are
skipped on CPU; neither these commands nor the implementation reserve fleet
resources or launch a server.

For 16 nodes, 16 heads and 128x128 KDA state, per-node state snapshots would
use 16 MiB per layer. Factors use 384 KiB plus one owned 1 MiB initial state,
with accepted-state materialization and other scratch additional. A 16-row,
4096x3072 W4A8 MLP plan now declares 181,760 bytes of scratch versus
6,473,456 bytes for its queued predecessor, with no additional
resident weight bytes. These are allocation formulas, not observed process
memory savings or a speedup forecast.

## Adoption evidence still needed

CPU equivalence and offline code generation do not establish native worker
liveness, real-weight numerical agreement, predictor usefulness or serving
speed. Compare the same build's ordinary path and the candidate on matched
quality, acceptance/tokens per step, C=1 output tok/s, TTFT and peak memory;
then cover C=4 and 32K/128K contexts. Keep compile, graph preparation and
prefix-cache reuse outside the measured bracket. A low-coverage predictor
or the current eager overhead may outweigh expert/state savings.

Motivation: [EcoSpec](https://arxiv.org/abs/2607.12696),
[Bole](https://arxiv.org/abs/2608.01651),
[TreeWY](https://arxiv.org/abs/2608.20961), and
[Mirage persistent kernels](https://arxiv.org/abs/2512.22219).
Their results are not GLM-5.3/ST/GB10 performance evidence for this change.
