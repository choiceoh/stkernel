"""Bounded CUDA radix selection for the GLM long-prefill sparse indexer.

Select only the valid prefix directly from the scorer's strided FP32 output.
This removes the dense boolean mask, masked write, Torch top-k workspace and
int64-to-int32 padding chain. Ties choose lower pool IDs; downstream pool_slots
owns position order. Decode/capture and other k values retain Torch selection.
"""
from pathlib import Path

import torch

_EXT = None


def _build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        from engine.kernels.common.native_cache import prepare_cuda_sources
        from engine.kernels.native_root import build_root
        src = Path(__file__).with_suffix('.cu')
        flags = ['-O3', '-gencode', 'arch=compute_121a,code=sm_121a']
        key, build, sources = prepare_cuda_sources(build_root('prefill-topk'), [src],
                                                   (flags, torch.__version__, torch.version.cuda))
        _EXT = load(name='st_prefill_topk_' + key, sources=list(sources),
                    extra_cuda_cflags=flags, build_directory=str(build), verbose=False)
    return _EXT


def admits(rows: int, columns: int, k: int) -> bool:
    """The launch shapes `select` takes: the half of its rule that is a row count and a width, not a tensor. A caller
    that scores one step's rows in several launches asks this first -- a row this selection takes in the whole step
    must not fall to torch.topk in a smaller launch, whose equal scores land where its candidates happen to."""
    return k == 512 and 64 < rows <= 32768 and 0 < columns <= 262144


def admits_calls(rows: int, per_call: int, columns: int, k: int) -> bool:
    """`admits` for every launch `rows` rows make when they are selected `per_call` at a time -- a scorer's logits
    workspace cuts a long step into such calls, whole ones and a remainder."""
    if rows <= 0 or per_call <= 0:
        return False
    sizes = {min(rows, per_call)} | ({rows % per_call} if rows > per_call and rows % per_call else set())
    return all(admits(size, columns, k) for size in sizes)


def select(logits, valid, k):
    if (logits.ndim != 2 or not admits(logits.shape[0], logits.shape[1], k) or logits.stride(1) != 1
            or logits.dtype != torch.float32 or not logits.is_cuda
            or valid.shape != (logits.shape[0],) or valid.dtype != torch.int32
            or valid.device != logits.device or not valid.is_contiguous()):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    out = torch.empty((logits.shape[0], 512), device=logits.device, dtype=torch.int32)
    _build().run(logits, valid, out)
    return out
