"""Explicit GB10 pinned host mapping; never alias ordinary cudaMalloc memory."""
from functools import cache
from pathlib import Path


@cache
def build():
    import torch
    from torch.utils.cpp_extension import load
    from engine.kernels.common.native_cache import prepare_sources
    source = Path(__file__).with_name("buffer.cu")
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a"]
    key, directory, staged = prepare_sources(Path.home()/".cache/st/mapped-staging", [source],
                                              (flags, torch.__version__, torch.version.cuda))
    return load(name="st_mapped_staging_"+key, sources=list(staged), extra_cuda_cflags=flags,
                build_directory=str(directory), verbose=False)


def allocate(nbytes):
    if type(nbytes) is not int or nbytes <= 0 or nbytes % 4096:
        raise ValueError("mapped staging requires positive page-aligned bytes")
    return build().mapped_pair(nbytes)
