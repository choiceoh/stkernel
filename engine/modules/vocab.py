"""Exact vocabulary-parallel greedy selection with one int64 MAX per token.

Order FP32 logits in the high word; the low word breaks ties in favor of
the smallest global token id, just like argmax on the gathered vocabulary.
BF16/FP16 logits convert exactly to FP32. NaNs win argmax, signed zeros tie,
and ranks with no decodable tokens contribute the minimum int64 sentinel.
No random draws and no full-vocabulary communication are required.
"""
import torch


def argmax(local_logits, comm, start: int, decodable: int | None = None):
    if local_logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("vocabulary argmax requires BF16, FP16 or FP32 logits")
    width = local_logits.shape[-1]
    if start < 0 or start + width >= 2**31 or (decodable is not None and decodable <= 0):
        raise ValueError("vocabulary ids must fit nonnegative int32 and contain a valid token")
    valid = width if decodable is None else max(0, min(width, decodable - start))
    if valid:
        value, index = local_logits[..., :valid].float().max(dim=-1)
        value = torch.where(value == 0, torch.zeros_like(value), value)
        value = torch.where(torch.isnan(value), torch.full_like(value, float("nan")), value)
        bits = value.contiguous().view(torch.int32).to(torch.int64)
        ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
        key = (ordered << 32) | (0xffffffff - (index + start))
    else:
        key = torch.full(local_logits.shape[:-1], -(2**63), dtype=torch.int64, device=local_logits.device)
    comm.all_reduce_max(key)
    return 0xffffffff - (key & 0xffffffff)
