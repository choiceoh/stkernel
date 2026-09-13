"""Semantic reference for token-major MLA absorption; no GPU timing claim."""
import torch


def mla_prefill_absorb_ref(x, weight, *, transpose=False):
    equation = 'thc,hvc->thv' if transpose else 'thd,hdc->thc'
    return torch.einsum(equation, x, weight).contiguous()
