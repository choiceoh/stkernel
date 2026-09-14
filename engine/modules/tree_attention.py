"""Private-tree attention addresses; integer-only and no canonical KV writes."""
import torch


def check_slots(ids, lengths, pool, table, block, stride, offset, paths, context, out, counts):
    if (ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64)
            or not 1 <= len(ids) <= 32 or ids.shape[1] < 1
            or type(pool) is not int or pool <= 0 or pool & (pool-1)
            or lengths.shape != (len(ids),) or lengths.dtype != torch.int32
            or table.ndim != 1 or table.dtype != torch.int32
            or any(type(v) is not int for v in (block, stride, offset, context))
            or block <= 0 or block % pool or stride < offset+block or offset < 0 or context < 0
            or len(table)*block < context or context >= 2**31-8
            or paths.ndim != 2 or paths.shape[0] != len(ids) or not 1 <= paths.shape[1] <= 8
            or paths.dtype != torch.int64 or not paths.is_contiguous()
            or out.shape != (len(ids), ids.shape[1]*pool+pool-1) or out.dtype != torch.int32
            or counts.shape != (len(ids),) or counts.dtype != torch.int32
            or any(t.device != ids.device for t in (lengths, table, paths, out, counts))):
        raise ValueError('tree slots require bounded paths, lengths and an int32 paged-cache map')


def pool_slots(ids, lengths, pool, table, block, stride, offset, paths, context, out, counts):
    if ids.is_cuda:
        from engine.kernels.indexer import tree_pool_slots
        return tree_pool_slots(ids, lengths, pool, table, block, stride, offset, paths, context, out, counts)
    check_slots(ids, lengths, pool, table, block, stride, offset, paths, context, out, counts)
    from engine.modules.sparse_indexer import pool_slots as reference
    reference(ids, lengths, pool, None, block, stride, offset, out, counts)
    positions = out.long()
    canonical = torch.zeros_like(positions)
    if context:
        safe = positions.clamp(0, context-1)
        canonical = table[safe//block].long()*stride + offset + safe % block
    private = -paths.gather(1, (positions-context).clamp(0, paths.shape[1]-1))-1
    out.copy_(torch.where(positions >= context, private, canonical).masked_fill(positions < 0, 0))


def absorb(x, weight, *, transpose=False):
    """Use the existing small-row BF16/FP32 contraction, without a weight copy."""
    if not x.is_cuda:
        from engine.modules.mla_absorb import mla_prefill_absorb_ref
        return mla_prefill_absorb_ref(x, weight, transpose=transpose)
    from engine.kernels.mla.decode_absorb import tree_absorb
    return tree_absorb(x, weight, transpose=transpose)
