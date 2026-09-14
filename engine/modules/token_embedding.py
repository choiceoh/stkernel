"""One rank's token embeddings; nonlocal IDs contribute positive zero to the TP sum."""
import torch
import torch.nn.functional as F


def lookup(ids, weight, start: int):
    if (ids.ndim != 1 or ids.dtype != torch.int64 or weight.ndim != 2
            or min(weight.shape) <= 0 or weight.device != ids.device
            or type(start) is not int or not 0 <= start < (1 << 63)):
        raise ValueError('token embedding requires int64 token rows, a local weight matrix and an int64 shard start')
    if ids.is_cuda:
        if (weight.dtype != torch.bfloat16 or weight.stride(1) != 1
                or weight.stride(0) < weight.shape[1]):
            raise ValueError('CUDA token embedding requires BF16 weights with packed hidden columns')
        from engine.kernels.token_embedding import lookup as native_lookup
        return native_lookup(ids, weight, start)
    local = ids - start
    invalid = (local < 0) | (local >= weight.shape[0])
    return F.embedding(local.masked_fill(invalid, 0), weight).masked_fill(invalid[:, None], 0)
