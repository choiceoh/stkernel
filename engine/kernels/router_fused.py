"""The decode router in one launch: IEEE FP32 logits, noaux_tc top-8 and weights (a 2026-09-17 cell).

The served chain is seven launches per MoE layer (`glm_pointwise.router_logits` + `route_weights`). This
kernel takes the same IEEE FP32 products in another projection add order, so its logits move
by a few ulps and a near-tied top-8 boundary can flip. Selection order and weight reduction match the
pinned PyTorch/Triton path; `_align=False` keeps the prior tail for component comparisons only.
The GLM53 consumer can bind it at eight/sixteen
decode rows; adoption requires the full consumer bracket as well as `probes/engine_router_cells.py`.
"""
from pathlib import Path

import torch

EXPERTS, HIDDEN, TOPK, MAX_ROWS = 288, 4096, 8, 16

_EXT = None
_TICKETS = {}


def build():
    """Compile and load; no device touch, so a fleet boot can run this beside the others."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        from engine.kernels.common.native_cache import prepare_cuda_sources
        from engine.kernels.native_root import build_root
        src = Path(__file__).with_suffix('.cu')
        flags = ['-O3', '-gencode', 'arch=compute_121a,code=sm_121a']
        key, directory, sources = prepare_cuda_sources(build_root('router-fused'), [src],
                                                       (flags, torch.__version__, torch.version.cuda))
        _EXT = load(name='st_router_fused_' + key, sources=list(sources),
                    extra_cuda_cflags=flags, build_directory=str(directory), verbose=False)
    return _EXT


def _ticket(device):
    """One arrival counter per execution/capture stream, reset by the kernel.

    Captures made on the same stream share this workspace and must replay in
    order, as their graph pool does. Independent captures use distinct streams.
    """
    index = torch.cuda.current_device() if device.index is None else device.index
    key = (index, torch.cuda.current_stream(device).cuda_stream)
    if key not in _TICKETS:
        _TICKETS[key] = torch.zeros(1, dtype=torch.int32, device=f'cuda:{index}')
    return _TICKETS[key]


def route(x, gate, bias, topk, scale, *, logits=None, ids=None, weights=None, _align=True):
    """(int32 ids [rows, 8], FP32 weights [rows, 8]) for BF16/FP32 x [rows <= 16, 4096] against the resident
    FP32 gate [288, 4096] and bias [288]; `logits` [rows, 288] FP32 is written when given (else allocated)."""
    if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != HIDDEN or not 0 < x.shape[0] <= MAX_ROWS
            or x.dtype not in (torch.bfloat16, torch.float32) or not x.is_cuda or not x.is_contiguous()):
        raise ValueError('fused router requires contiguous CUDA BF16/FP32 x of [1..16, 4096]')
    if (gate.shape != (EXPERTS, HIDDEN) or gate.dtype != torch.float32 or gate.device != x.device
            or not gate.is_contiguous() or bias.shape != (EXPERTS,) or bias.dtype != torch.float32
            or bias.device != x.device or not bias.is_contiguous()):
        raise ValueError('fused router requires the resident FP32 gate [288, 4096] and FP32 bias [288]')
    if topk != TOPK:
        raise ValueError('fused router selects exactly eight experts')
    if _EXT is None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('fused router must be built and warmed before graph capture')
    rows = x.shape[0]
    if logits is None:
        logits = torch.empty((rows, EXPERTS), dtype=torch.float32, device=x.device)
    if ids is None:
        ids = torch.empty((rows, TOPK), dtype=torch.int32, device=x.device)
    if weights is None:
        weights = torch.empty((rows, TOPK), dtype=torch.float32, device=x.device)
    build().run(x, gate, bias, _ticket(x.device), logits, ids, weights, float(scale), _align)
    return ids, weights
