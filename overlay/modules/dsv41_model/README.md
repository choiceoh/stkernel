# V4.1 CED candidate indexer

The layer planner describes the checkpoint. The indexer implements a separate,
explicitly installed optimization of the official reference model's long-context
decode path. This module does **not** yet implement or register a complete vLLM
model, load a V4.1 checkpoint, or make `profiles/dsv41.env` bootable.

## Work removed

The reference's layer 20 selects up to 2,048 candidate blocks, each containing
eight compressed positions. Layers 24, 28, 32 and 36 nevertheless score **all**
positions and all-reduce those scores before masking out positions outside the
candidate blocks.

The compact path keeps layer 20's selection and gives the four consumers a
sorted position list with a capacity of 16,384. Each consumer scores and
all-reduces only that list. The optional Triton implementation reads keys by
position directly, without materializing a `[batch, queries, candidates, dim]`
key gather. Its arithmetic preserves the reference's separate BF16 rounding
after QK, weighted ReLU and the head sum.

After the collective, scores are scattered into an original-width `-inf`
tensor before the **unchanged** top-k and position-order sort. Calling top-k
on a shorter vector would change how equal scores are selected. Full-width
top-k, the source indexer, the encoder indexers and sparse attention remain.

For one decode query, with the shipped candidate limit:

| Compressed positions | Original values per consumer | Compact values | Reduction |
|---:|---:|---:|---:|
| 16,384 | 16,384 | dense fallback | 0% |
| 32,768 | 32,768 | 16,384 | 50% |
| 131,072 | 131,072 | 16,384 | 87.5% |
| 1,048,576 | 1,048,576 | 16,384 | 98.4375% |

These are score and collective **element counts**, not measured latency,
network traffic, model tok/s or TTFT. In particular, no full-model speedup can
be inferred from them.

## Activation and scope

The reference adapter is opt-in and instance-scoped; importing these files or
composing the profile does not activate it. It verifies the official reference
source before installing any hooks. It can be restored without replacing
global class methods.

Only long-context, single-token decode at batch size one is routed. Prefill,
larger batches, multi-token calls and contexts that fit the candidate capacity
retain the reference forward. The admitted call also preserves the original
`[1,1,S]` top-k shape, since GPU top-k can choose an algorithm by row count.
The original projection, RoPE, FP4 fake quantization, key cache publication
and collective order are retained. Candidate state is invalidated across
incompatible steps. A rank-local state mismatch in TP fails before a consumer
collective: falling back on only one rank would mix full and compact payload
sizes. Runtime replacement is rejected. Inference tensors have no version
counter, so external in-place mutations of the same storage cannot all be
detected. The official reference still uses a global `shared_attn`
object: this adapter does not turn it into a concurrent serving runtime or
claim CUDA graph replay support.

The Torch implementation is a portable arithmetic oracle. The Triton backend
requires an explicit choice and separate GPU numerical validation. Any test
that substitutes already-quantized inputs isolates the indexer calculation;
it does not validate the upstream FP4 quantizer.

Given an existing official reference `Transformer` and its imported `model`
module, installation is explicit and reversible:

```python
from vllm.models.dsv41.dsv41_reference_adapter import install_reference_indexer

handle = install_reference_indexer(transformer, reference_model,
                                   enabled=True, backend="torch")
try:
    # Run the reference model through its normal ordered forward.
    ...
finally:
    handle.restore()
```

`backend="triton"` selects the experimental CUDA implementation. `WIDTH` is a
runtime, non-specialized scalar: growing the context by a token must not
compile another kernel. Geometry and strides remain compile specializations.

## Validation

`probes/dsv41_indexer_diff.py` reads the pinned reference's `Indexer.forward`
and `select_candidate_blocks` AST rather than maintaining another handwritten
reference. It tests BF16 scores, selected positions, causal and partial-block
boundaries, equal scores, negative head weights, offsets and simulated
four-rank reduction. The simulator is not a real NCCL result.

`tests/test_dsv41_reference_adapter.py` exercises installation, fallback,
state ownership, TP failure and restoration. The core boundary tests also
reject invalid tensor/collective contracts and lock the runtime width rule.
The module can be composed through
`bash launchers/compose-overlays.sh dsv41`; the profile has no activation knob.

Before enabling a serving path, the remaining evidence must include GPU score
and selected-ID equivalence, actual TP reduction, changed-input/replay behavior,
attention outputs, and a matched consumer benchmark including decode tok/s,
step/s, TTFT and quality. Current fleet policy admits canonical consumer
campaigns, not arbitrary component GPU scripts. CPU tests or offline compilation
must not be recorded as GPU validation.
