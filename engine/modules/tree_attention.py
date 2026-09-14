"""Private-tree attention addresses; integer-only and no canonical KV writes."""
import torch


def check_key_bank(keys, scales, private, private_scales, table, per, stride, offset, prefix):
    if (keys.ndim != 2 or keys.shape[1] < 1 or keys.shape[1] & (keys.shape[1]-1)
            or keys.dtype != torch.float8_e4m3fn or keys.stride(1) != 1
            or scales.shape != (len(keys),) or scales.dtype != torch.float32
            or private.ndim != 2 or private.shape[1] != keys.shape[1] or private.stride(1) != 1
            or private.dtype != keys.dtype or len(private) > 32
            or private_scales.shape != (len(private),) or private_scales.dtype != scales.dtype
            or table.ndim != 1 or table.dtype != torch.int32
            or any(type(v) is not int for v in (per, stride, offset, prefix))
            or per <= 0 or offset < 0 or stride < offset+per or not 0 <= prefix <= len(table)*per
            or any(t.device != keys.device for t in (scales, private, private_scales, table))):
        raise ValueError('tree key bank requires FP8/FP32 records and a bounded paged prefix')


def key_bank(keys, scales, private, private_scales, table, per, stride, offset, prefix):
    """Gather the paged prefix and append private pools directly into one bank."""
    if keys.is_cuda:
        from engine.kernels.indexer import tree_key_bank
        return tree_key_bank(keys, scales, private, private_scales, table, per, stride, offset, prefix)
    check_key_bank(keys, scales, private, private_scales, table, per, stride, offset, prefix)
    out = torch.empty((prefix+len(private), keys.shape[1]), dtype=keys.dtype, device=keys.device)
    scale = torch.empty(prefix+len(private), dtype=scales.dtype, device=keys.device)
    p = torch.arange(prefix, device=keys.device)
    ids = table[p//per].long()*stride+offset+p % per
    torch.index_select(keys.view(torch.uint8), 0, ids, out=out[:prefix].view(torch.uint8))
    torch.index_select(scales, 0, ids, out=scale[:prefix])
    out[prefix:].copy_(private)
    scale[prefix:].copy_(private_scales)
    return out, scale


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
