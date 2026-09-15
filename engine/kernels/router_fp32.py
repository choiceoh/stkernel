"""IEEE FP32 router projection, with resident FP32 weights at every row count."""
from pathlib import Path

import torch

_EXT = None


def build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        from engine.kernels.common.native_cache import prepare_sources
        from engine.kernels.native_root import build_root
        src = Path(__file__).with_suffix('.cpp')
        flags = ['-O3']
        key, directory, sources = prepare_sources(
            build_root('router-fp32'), [src], (flags, torch.__version__, torch.version.cuda))
        _EXT = load(name='st_router_fp32_' + key, sources=list(sources),
                    extra_cflags=flags, build_directory=str(directory), verbose=False)
    return _EXT


def router_logits(x, weight):
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or not x.is_cuda or x.device != weight.device
            or x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != torch.float32):
        raise ValueError('FP32 router requires matching CUDA BF16/FP32 inputs and FP32 weights')
    if _EXT is None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('FP32 router must be built and warmed before graph capture')
    return build().run(x, weight)
