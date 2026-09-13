# SPDX-License-Identifier: Apache-2.0
"""Fuse BF16 expert-output addition into the unchanged FP8 v3 packet math.

Only transport padding sees synthesized zeros. No padded token enters attention,
KDA, routing, or a cache. The sum rounds to BF16 before FP8 scale selection.
"""
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["N", "LOCAL_N", "PAYLOAD_BYTES"])
def _pack_sum_rs_payload(X, Y, Packed, Scales, N, LOCAL_N, PAYLOAD_BYTES,
                     BLOCK: tl.constexpr):
    block = tl.program_id(0)
    local_blocks = LOCAL_N // BLOCK
    rank = block // local_blocks
    local_block = block % local_blocks
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, mask=offsets < N, other=0.0).to(tl.float32)
    y = tl.load(Y + offsets, mask=offsets < N, other=0.0).to(tl.float32)
    # Preserve the existing BF16 add result before measuring/quantizing it.
    x = (x + y).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 1.0e-30) / 448.0)))
    # Each equal-sized destination packet is [FP8 values | FP32 scales].
    # LOCAL_N is whole BF16 rows, so both views and every packet are aligned.
    local_offsets = local_block * BLOCK + tl.arange(0, BLOCK)
    tl.store(Packed + rank * PAYLOAD_BYTES + local_offsets,
             (x / scale).to(tl.float8e4nv))
    tl.store(Scales + rank * (PAYLOAD_BYTES // 4) + LOCAL_N // 4 + local_block,
             scale)
    if local_block == local_blocks - 1:
        # Initialize the alignment gap in this packet. One CTA owns it and
        # its at-most-31 FP32 words never overlap actual scales or values.
        tail = tl.arange(0, 32)
        scale_end = LOCAL_N // 4 + local_blocks
        tl.store(Scales + rank * (PAYLOAD_BYTES // 4) + scale_end + tail, 0.0,
                 mask=scale_end + tail < PAYLOAD_BYTES // 4)
