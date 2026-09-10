# Packed FP4 index-key cache

The official V4.1 reference calls the indexer's FP4 quantizer in fake-quant mode
and stores the resulting BF16 keys. The packed adapter instead stores the
original quantizer's E2M1 payload and E8M0 scales. Dense and candidate scoring
unpack only the key tile being consumed. There is no full-prefix BF16 key
temporary and no second quantization of fake-quantized values.

This is an explicit reference-model experiment, not a complete vLLM model or
a change to serving defaults. CPU validation and offline compilation do not
establish GPU correctness, actual memory usage or model speed.

## Storage and reads

For an index key with 128 features and 32-feature scale groups:

| Representation | Values | Scales | Bytes per position |
|---|---:|---:|---:|
| Reference BF16 fake-FP4 | 256 B | already folded into values | 256 B |
| Packed E2M1 + E8M0 | 64 B | 4 B | 68 B |

The allocation formula is `B * (3 * floor(S / 2) + S)` positions across
sources 2, 8, 14 and 20. At S=1,048,576, B=1, this is 640 MiB BF16 versus
170 MiB packed per rank, a 73.4375% reduction in index-key storage. With B=4,
the same calculation is 2.5 GiB versus 680 MiB. These are allocation/operand
byte counts, not allocator-reserved memory or measured bandwidth/latency.

All eight indexers must use packed-aware scoring to release the original
buffers. Merely changing the four long-context candidate consumers would
leave the owners' BF16 buffers and short-context path resident. This adapter
therefore implements dense packed scoring for sources, prefill and shorter
contexts, and combines it with candidate scoring for long B1/Q1 decode.
Dense score/all-reduce tensors and the final full-width top-k remain.
The Triton dense path shares each decoded key tile across four queries; the
compact path handles one query's selected positions. Context width, output
width and query count remain runtime arguments rather than per-token compile
specializations.

## Numerical contract

The producer calls the official `fp4_act_quant(k, 32, False, E8M0)` after the
original projection, normalization and RoPE. The query's original in-place
quantization stays intact. Cache writes retain those returned bytes directly.
Packing an already fake-quantized BF16 key again would not establish equivalent
scales or overflow behavior, so that path is not used.

Even features occupy the low nibble. E8M0 raw zero represents `2^-127`, not
zero. Reconstruction must preserve signed zero, BF16 subnormals and overflow;
a finite input to the original quantizer can still dequantize to infinity.
The score path reconstructs BF16 keys before the dot product, then retains the
reference's BF16 dot output, BF16 weighted product and BF16 head-sum boundaries.
Different device GEMM/collective reduction orders still require GPU comparison.

Source publication is conditional in the original: if a ratio-two source has
no completed latent group, it neither writes nor republishes its own key cache.
The packed implementation retains the currently published shared descriptor
in that case. Selecting the owner unconditionally would change the reference's
cross-layer behavior, including an odd decode step that still sees layer 20's
descriptor from the preceding step.

## Lifetime and activation

Installation is opt-in and starts at a fresh-prefill boundary. It discards
existing index-key history and requires the next applicable indexer call to
have `start_pos=0`. The adapter owns all eight indexer instance methods and is
mutually exclusive with the earlier BF16 compact adapter. It does not patch
the global Indexer class or register a vLLM architecture.

Only shape/device/dtype metadata is retained for the old BF16 buffers. Keeping
the tensors in a rollback handle would defeat the memory saving. Packed
buffers are registered as nonpersistent buffers; moving or replacing them
behind an active handle is rejected. The official reference's single-model,
ordered-forward restriction remains. CUDA graph capture is unsupported.

Restoration is a cache reset, not continuation of an active request. It first
allocates all replacement BF16 buffers; an allocation failure leaves the
packed installation active. After a successful reset, a fresh prefill is
required before decode resumes. There is no silent per-rank fallback to an
incompatible score/collective layout.

The Torch implementation is an arithmetic/CPU-validation backend. Selecting
the Triton implementation is explicit. Before GPU adoption, validate actual
TileLang packed-byte layout, key reconstruction, score/selected-ID equivalence,
TP collectives, attention outputs and end-to-end quality and speed under a
matched consumer campaign.

For a newly constructed official reference model, before its first prefill:

```python
from vllm.models.dsv41.dsv41_packed_reference_adapter import (
    install_reference_packed_indexer,
)

handle = install_reference_packed_indexer(
    transformer, reference_model,
    enabled=True, cache_boundary="fresh_prefill",
    compact_decode=True, backend="torch",
)
# The next applicable indexer call must start a fresh prefill at position zero.
# Run the model through the original ordered forward.
# To remove the adapter and discard the request's index-key history:
handle.restore(reset=True)
# The next request must again begin with a fresh prefill.
```

Use `backend="triton"` only for an explicitly scheduled device-validation
campaign. This does not enable the unfinished V4.1 serving profile.
