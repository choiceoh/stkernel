"""The causal prefix where kpool selection necessarily includes every token."""


def covered_prefix(rows: int, context: int, topk: int, pool: int) -> int:
    if any(type(v) is not int for v in (rows, context, topk, pool)):
        raise ValueError('prefix geometry must use integers')
    if rows < 0 or context < 0 or topk <= 0 or pool <= 0:
        raise ValueError('invalid causal prefix geometry')
    # At a query of length L, floor(L/pool) complete pools and L%pool tail
    # tokens are visible. All of them fit until the next complete pool.
    return max(0, min(rows, (topk // pool + 1) * pool - 1 - context))


def mla_dense_prefix_ref(q, latent, block_table, block_size, block_stride,
                         layer_offset, context, scale, ckv_scale):
    """CPU semantic oracle: the exact descending slots of a covered prefix.

    Chunk the reference gather to bound memory. This is not a GPU kernel or
    an oracle for the new kernel's online-softmax rounding order.
    """
    import torch
    from engine.modules.sparse_attention import mla_sparse_mqa
    out = torch.empty_like(q)
    width = context + len(q)
    for begin in range(0, len(q), 32):
        end = min(begin + 32, len(q))
        lengths = context + torch.arange(begin + 1, end + 1, device=q.device)
        positions = lengths[:, None] - 1 - torch.arange(width, device=q.device)[None, :]
        safe = positions.clamp_min(0)
        slots = safe if block_table is None else (
            block_table[(safe // block_size).long()].long() * block_stride
            + layer_offset + safe % block_size)
        slots = slots.masked_fill(positions < 0, -1).to(torch.int32)
        out[begin:end] = mla_sparse_mqa(q[begin:end], latent, slots, lengths, scale, ckv_scale)
    return out
