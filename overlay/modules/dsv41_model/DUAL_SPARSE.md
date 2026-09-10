# Direct sparse attention over two KV pools

The official V4.1 reference concatenates the sliding-window KV and the entire
compressed prefix before sparse attention. Its sparse kernel then gathers only
the requested positions, at most 128 window slots and 512 compressed slots per
query. The dual-pool path reads those same positions directly from the two
inputs and removes the full-prefix KV concatenation.

This is an explicit reference-model adapter. It does not implement or register
a complete V4.1 vLLM model. Existing serving paths are unchanged; no GPU
numerical or model-performance result follows from CPU tests or compilation.

## Removed work

There are 18 ratio-two compressed layers and 20 ratio-one layers. With BF16
D512 KV, batch one and context length L, the reference's concatenation outputs
occupy this many bytes across a decode step:

```
1024 * (18 * floor(L / 2) + 20 * L + 38 * 128)
```

At 128K this is 3.62964 GiB per rank per step; at 1M it is 29 GiB plus 4.75 MiB.
Reading the source and writing the destination doubles the logical copy byte
count. These are code-derived byte counts, not measured HBM traffic or speed.
The existing KV buffers remain allocated. The small index concatenation also
remains; candidate selection and the set/order of selected positions are not
changed.

Prefill's window input contains the full input sequence, not just 128 rows.
The same direct-read addressing handles it using the actual window length.
At 128K prefill, the original 38 concatenation outputs total 8.375 GiB per
rank. This does not remove the original model's other large prefill tensors.

## Attention contract

Merged indices retain their original domain: positions below the window length
address that input; other valid positions address the compressed input after
subtracting the window length. The original sentinel is -1. Duplicate positions
remain duplicated, and the 64-slot tile sequence is unchanged. Inputs are read
with their real strides, including B>1 compressed-prefix slices whose batch
stride still reflects the full cache capacity.

Both pools participate in one online softmax. The computation retains BF16
Q/KV inputs, FP32 score accumulation and scaling, running maximum/sum, BF16
probabilities before the value dot, FP32 output accumulation, sink contribution
to the denominator only, and final BF16 output. The initial maximum remains
-1e30. Degenerate denominators are not silently changed to zero outputs.
CPU arithmetic agreement does not prove equivalence of TileLang and Triton
MMA, exponential, or reduction lowering on a device.

Head tiles are independent because normalization is per head. The device path
uses 16-head tiles for the released H64/D512 attention: TP4 has H16, TP2 H32,
TP1 H64 and TP8 H8. Runtime lengths and strides must not specialize anew as a
decode context grows. No full-prefix BF16 materialization is used. The Torch
backend gathers only bounded query/64-position tiles and is an arithmetic
validation implementation; selecting Triton is explicit.

## Integration and lifetime

The adapter changes only the 38 compressed Attention instance forwards. It
preserves projections, RoPE, normalization, quantization, cache writes,
candidate/indexer calls, and the small index concatenation. In particular,
the compressed owner publishes its cache even when latent is None; the index
cache has a different publication rule. Both original rules stay intact.

The adapter is separate from the #521/#522 indexer adapters and can coexist
with either. It owns no cache history and restores only its instance forwards.
The original single-model shared runtime and ordered-forward assumptions
remain; CUDA graph capture is not admitted. Installation verifies the pinned
reference implementation and occurs after model device placement.

Given an existing official reference model:

```python
from vllm.models.dsv41.dsv41_dual_sparse_reference_adapter import (
    install_reference_dual_sparse_attention,
)

handle = install_reference_dual_sparse_attention(
    transformer, reference_model, enabled=True, backend="torch",
)
# Run the model through its normal ordered forward.
handle.restore()
```

GPU adoption still requires actual TileLang-versus-Triton output comparison,
attention/model quality checks and a matched consumer campaign with tok/s,
step/s and TTFT. The current fleet policy and incomplete V4.1 serving
integration are not bypassed by this adapter.
