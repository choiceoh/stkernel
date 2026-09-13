"""Selector projection with an explicit output-rounding contract."""
import torch
import torch.nn.functional as F


def project(x, weight, *, fp32=False):
    if not fp32:
        return F.linear(x, weight).float()
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or x.device != weight.device or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16):
        raise ValueError('FP32 selector projection requires matching BF16 matrices')
    # Same BF16 tensor-core operands as the router; widening a BF16 output
    # afterwards would already have discarded close selector score differences.
    if x.is_cuda:
        return torch.mm(x, weight.T, out_dtype=torch.float32)
    return torch.mm(x.float(), weight.float().T)
