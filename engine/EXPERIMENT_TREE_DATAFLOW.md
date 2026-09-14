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
   parent-conditioned codebook scores. Selection is ancestor-closed and
   ranks proposal mass against additional predicted layer/expert bytes.
   Unknown routes receive a conservative charge. Proposals never prune the
   target router or alter greedy verification. `RouteTable` is a bounded
   empirical baseline keyed by predecessor/token/depth and runtime ID;
   it is **not a trained hidden-state router predictor**. Coverage, actual
   expert usage and pre-update prediction recall/precision are recorded.
2. `tree_kda.py` and `kernels/kda/tree.py` store FP32 key, channelwise decay
   and update factors. Each native CTA owns its head/value slice through
   the whole tree. Only ancestors influence a node. The final accepted
   state and any crossed prefix-block checkpoint are materialized before
   canonical writes. Convolution history and DSA pools/tails are private
   to each branch; DSA gathers a bounded top-k latent bank, not a full
   prefix copy for every node. KDA storage remains FP32.
3. `tile_dataflow.py` executes gate/up producer tiles, dependent down
   partials and output reductions in one persistent kernel. This is an
   actual task queue, not repeated CUDA graph replay. All producer tickets
   are assigned before any producer wait, and all down tickets before any
   reduction wait. Dependencies therefore belong to workers already in
   flight; no extra scheduler SM or unscheduled producer CTA is required.
   GPU acquire/release atomics publish data, each output has one writer,
   and bounded polling reports failure before the output collective.

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
Worker completion and proposal metadata currently synchronize with the host.

## Reproducible checks without a GPU queue

Run from the repository root with torch CPU, numpy, safetensors and Triton
(the recorded offline compiler uses torch 2.14.0+cpu and Triton 3.8.0):

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_dataflow_mock.py --output /tmp/tree-mock
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_dataflow_compile.py --output /tmp/tree-compile
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 PYTHONPATH=. python probes/engine_tree_dataflow_interpreter.py --output /tmp/tree-addresses.json
```

The mock records per-test CPU latency, source hashes and storage arithmetic.
The offline compiler emits SM121 cubins/PTX and checks FP8 MMA for W4A8,
block-scaled FP4 MMA for W4A4, and acquire/release publication. No CUDA context
is initialized. The interpreter checks actual native packed readers against
independent byte references. Its W4A8 activation check executes native
scaling arithmetic and uses torch's FP8 RTNE conversion: Triton 3.8's CPU
interpreter rounds halfway FP8 values upward, unlike the generated `cvt.rn`.
It is not a device-conversion numerical result.

`tests.test_engine_tree_dataflow_gpu` provides opt-in native numerical and
repeated-invocation checks for a separately authorized GB10 window. They are
skipped on CPU; neither these commands nor the implementation reserve fleet
resources or launch a server.

For 16 nodes, 16 heads and 128x128 KDA state, per-node state snapshots would
use 16 MiB per layer. Factors use 384 KiB plus one owned 1 MiB initial state,
with accepted-state materialization and other scratch additional. A 16-row,
4096x3072 W4A8 MLP plan declares about 6.2 MiB of scratch and no additional
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
