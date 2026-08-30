# glm53_tail_slot_persistent

The kpool tail slot mapping is rebuilt every step and read from inside a
captured cudagraph. It was returned as a fresh `clone()`, so the graph recorded
the address of whichever copy existed at capture and every replay since has read
that one.

## Why the original was written that way

`compute_kpool_tail_slot_mapping` says so:

> Pure torch (no Triton, no device sync): the indexer op consumes the tail slot
> mapping **on its eager break**, so the returned tensor need not be the
> persistent `BlockTables` buffer.

The eager break does not happen here. `eager_break_during_capture` returns the
function unchanged when `VLLM_USE_BREAKABLE_CUDAGRAPH` is off — it is, `=0` —
and even when on it steps aside under `CUDAGraphMode.FULL`:

```python
if not is_breakable_cudagraph_enabled():
    return fn
...
if mode == CUDAGraphMode.FULL:
    return fn(*args, **kwargs)
```

`FULL_DECODE_ONLY` is what this model gets, because the sparse indexer backend
reports `AttentionCGSupport.UNIFORM_BATCH` and vLLM downgrades to it. So the
indexer is captured, and `KpoolTailMetadataBuilder` keeps allocating tensors
nothing reads.

## Why this is the shape of the Korean corruption

The tail holds the most recent tokens — the ones not yet folded into a pool. A
frozen slot mapping sends their writes to capture-time slots, so a decode step
can miss the byte it just emitted:

| observation | this mechanism |
|---|---|
| decode corrupts, prefill never does | prefill runs eager; only decode replays a graph |
| exactly one single-byte glue token lost | the tail is precisely the not-yet-pooled recent tokens |
| rare, ~1 per 1000 tokens | the tail matters only when `seq_len % kpool != 0`, and stale slots sometimes coincide |
| non-deterministic at temperature 0 | depends on which descriptor replays and what capture saw |
| worse with speculation | eight tokens per step instead of one, so eight times the tail traffic |
| the replay address assertion never fired | it inspects positional args; attention metadata arrives via the forward context |
| unmoved by `MOE_CUTOVER`, fp8 KV, MoE backend | different subsystem entirely |

## The fix

Write through a caller-owned buffer the builder allocates once, so the address
the graph baked in stays valid while the contents move. This is what the
decorator asks of anything it wraps:

> **In-place output buffer required.** Decorated ops must write into a
> caller-provided output tensor; a fresh tensor returned by `fn` would change
> address each replay and break downstream graph segments.

Allocated lazily on first build, when dtype and length are known — converting a
pre-made buffer with `.to()` would hand back a new tensor and reintroduce the
bug. It grows if a later step needs more room; that reallocation moves the
address, so it must happen before capture, which it does since capture runs the
largest descriptors first.

`out=None` keeps the old clone behaviour for any caller outside a graph.

Base contract from `glm53:v13-b12x`. **Not yet confirmed by measurement** — the
mechanism explains every observation, but the corruption rate after this change
has not been measured.
