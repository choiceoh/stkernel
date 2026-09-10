# V4.1 CED candidate indexer

The layer planner describes the checkpoint. The indexer implements a separate,
explicitly installed optimization of the official reference model's long-context
decode path. This module does **not** yet implement or register a complete vLLM
model, load a V4.1 checkpoint, or make `profiles/dsv41.env` bootable.

The follow-up [packed index-key cache](PACKED_INDEX.md) removes the BF16 key
buffers and provides direct packed dense/compact scoring across all eight
indexers. It has a separate opt-in handle and cannot be installed alongside
this BF16 adapter.

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

## Reading a pre-sharded rank file

`dsv41_preshard_load.py` is the loader half of `tools/dsv41_preshard.py`. The
builder writes one safetensors per rank -- 84.7 GiB for rank 0 of 4, measured,
against 475.2 GiB for the whole checkpoint -- and leaves expert names at their
GLOBAL ids on purpose: a file that renumbered them is indistinguishable from a
correct one once written. The renumbering happens here, where the rank is
known.

Every rank file has the same dtypes, the same shapes and nearly the same size,
so nothing about the DATA says which rank it is. The builder therefore records
`__metadata__` (rank, world size, `--dense`, `--mtp`, expert counts) and
`require()` refuses a file that disagrees. A file built before that existed
falls back to its filename and is refused unless `allow_unstated=True`.

`probes/dsv41_preshard_load.py` holds both halves to the same partition, and
found a real disagreement on its first run: `wanted_by` sharded the DSpark
block's 128 experts across ranks while the builder replicated them -- 1,728
tensors per rank. It had gone unnoticed because
`probes/dsv41_loader_route.py` excluded `mtp.` from its comparison, with a
comment explaining why that was reasonable. Both sides now take an `mtp` mode,
the default (`replicate`) matches this fleet's `draft_tensor_parallel_size=1`,
and the probe compares the full name set in both modes and requires the two
modes to differ.

| | rank 0 tensors | of which DSpark |
|---|---|---|
| `--mtp replicate` | 26,961 | 2,401 |
| `--mtp ep` | 25,233 | 673 |

`--dense tp` aborts in `build`. It is planned and accounted for, but the model
side does not read a split dense tensor yet, and a file written now would load
and compute garbage.

## The sliding-window KV ring

Every attention layer attends over a window of raw KV; the layers with
`compress_ratio > 0` concatenate compressed positions after it.
`dsv41_window.py` is the window half -- where a token's KV is written, and
which slots a query may read -- and both are pure index arithmetic, so they are
held to the reference exactly rather than within a tolerance.

The whole invariant is `slot = position % window_size`. Decode satisfies it
directly. Prefill does not: it seeds the ring from the last `window` tokens of
the chunk with a rotated two-part copy, and it is the rotation that makes the
invariant hold for the decode steps that follow. Seed it without the rotation
and the cache still holds every surviving token exactly once -- nothing about
it looks wrong -- while every slot is off by `seqlen % window`, so the first
decode step reads real, plausible tokens from the wrong positions.
`probes/dsv41_window_diff.py` checks the ring against the invariant rather
than against a second implementation, and includes that seed as a control:
it misplaces all 64 of 64 slots while losing nothing.

The second thing worth writing down is that the returned ids live in TWO index
spaces, chosen by `start_pos`: prefill ids index the CHUNK (one causal row per
query), decode ids index the RING (one row, oldest first). `window_kv_len`
exists so `dsv41_sparse_contract.check_topk_idxs` can be given the right bound
and the distinction becomes a checked contract instead of a comment.
