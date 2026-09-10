# Packed compressed KV with direct sparse attention

The official V4.1 reference already quantizes compressed KV to E2M1 with one
E4M3 scale per 16 values, but immediately dequantizes it to a BF16 cache. This
opt-in path stores the original out-of-place quantizer's bytes and reconstructs
only the positions selected by sparse attention. No already-quantized BF16
cache is requantized.

## Storage and scope

A D512 compressed row occupies 256 payload bytes plus 32 scale bytes instead
of 1,024 BF16 bytes: a 71.875% reduction in allocated compressed-cache bytes.
Owners 2, 8 and 14 store one row per two input tokens; owner 20 stores one row
per token. At batch one and a configured capacity of 1,048,576 tokens, their
combined tensors occupy 720 MiB instead of 2.5 GiB, a reduction of 1,840 MiB
per rank. Capacity is not the current prompt length. Window caches, weights,
index caches and other model temporaries are separate.

For a decode query selecting 512 valid compressed positions in all 38 layers,
the logical selected compressed operands total 5.34375 MiB instead of 19 MiB.
This counts each selected row once. Head-CTA rereads, caches, invalid slots and
padding affect actual device traffic. Neither these byte counts nor offline
compilation establish kernel or full-model speed.

The adapter includes #524's direct window/compressed addressing and removes
the full-prefix KV concatenation. It is an alternative to the BF16 dual-pool
adapter; installing both on the same Attention instances is rejected. It
composes with the independent packed-index adapter from #522.

## Numerical boundaries

The producer still calls the pinned official `fp4_act_quant` after RoPE with
block size 16 and `torch.float8_e4m3fn` scales, selecting its out-of-place result.
The original boundary is an FP32 product of E2M1 values and decoded E4M3
scales, converted to BF16 before QK or PV. The implementation constructs those
exact finite BF16 bits using integer significands and exponent adjustment.
CPU tests cannot establish the vendor TileLang producer's packing,
overflow or rounding behavior on a GPU.

The direct sparse computation retains merged index meaning, duplicate slots,
the 64-slot order, FP32 online accumulators, BF16 probabilities before PV, and
one sink contribution after the last tile. It does not replace the staged
recurrence with a single full softmax. NaN payload identity is not promised.

## Integration and lifetime

Installation is explicit, after device placement, at a fresh-prefill boundary.
The owner publishes its compressed cache even while a ratio-two group is
incomplete. The original indexer consumes the pre-RoPE latent before the
Attention producer rotates and quantizes it. The packed writer stores only
the original quantizer output.

The adapter releases the displaced BF16 owners. It does not retain them in a
rollback handle. `restore(reset=True)` allocates all replacement BF16 buffers
before changing the installed state, then requires fresh prefill. It does not
preserve packed cache history across restoration. Ordered single-model
execution remains the reference contract; graph capture is unsupported.

```python
from vllm.models.dsv41.dsv41_packed_kv_reference_adapter import (
    install_reference_packed_kv_attention,
)

handle = install_reference_packed_kv_attention(
    transformer, reference_model, enabled=True,
    cache_boundary="fresh_prefill", backend="torch",
)
# Begin with start_pos=0 through the ordinary ordered model forward.
handle.restore(reset=True)
# Begin a new prefill before decoding again.
```

Torch is the portable arithmetic implementation. Triton is explicit opt-in
and needs GPU differential validation. This module does not complete the
V4.1 vLLM model, register its architecture, or activate a serving default.
Actual quantizer bytes, attention outputs, full-model quality and matched
tok/s, step/s and TTFT remain device-validation gates.
