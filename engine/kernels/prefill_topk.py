"""Bounded CUDA radix selection for the GLM long-prefill sparse indexer.

Select only the valid prefix directly from the scorer's strided FP32 output.
This removes the dense boolean mask, masked write, Torch top-k workspace and
int64-to-int32 padding chain. Ties choose lower pool IDs; downstream pool_slots
owns position order. Decode/capture and other k values retain Torch selection.
"""
import hashlib
from pathlib import Path

import torch

_EXT = None


def _build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        src = Path(__file__).with_suffix('.cu')
        flags = ['-O3', '-gencode', 'arch=compute_121a,code=sm_121a']
        key = hashlib.sha256(src.read_bytes() + repr(
            (flags, torch.__version__, torch.version.cuda)).encode()).hexdigest()[:16]
        build = Path.home() / '.cache/st/prefill-topk' / key
        build.mkdir(parents=True, exist_ok=True)
        _EXT = load(name='st_prefill_topk_' + key, sources=[str(src)],
                    extra_cuda_cflags=flags, build_directory=str(build), verbose=False)
    return _EXT


def select(logits, valid, k):
    if (k != 512 or logits.ndim != 2 or not 64 < logits.shape[0] <= 32768
            or not 0 < logits.shape[1] <= 262144 or logits.stride(1) != 1
            or logits.dtype != torch.float32 or not logits.is_cuda
            or valid.shape != (logits.shape[0],) or valid.dtype != torch.int32
            or valid.device != logits.device or not valid.is_contiguous()):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    out = torch.empty((logits.shape[0], 512), device=logits.device, dtype=torch.int32)
    _build().run(logits, valid, out)
    return out
