"""Where the per-module native extensions keep their builds: a cache path, the TRITON_CACHE_DIR class.

It lives outside `common/` because the common kernels read no environment (D11); like the MLA, dense and
one-shot build roots, this is the one environment read of its package, and it names a directory, not a knob.
"""
import os
from pathlib import Path


def build_root(name):
    """Where the native module `name` keeps its builds: ``$ST_NATIVE_BUILD_ROOT/<name>``.

    A cache path, the same class as TRITON_CACHE_DIR. The served image and the
    launcher point it into /cache, which outlives the container. The fallback
    under $HOME is the container's own writable layer, and every launcher stop
    is a `docker rm`: a build kept there is a build the next boot repeats.
    Four modules kept theirs there, so every boot recompiled all four on every
    rank before the door opened -- mapped staging 46.1 s, prefill top-k 45.5 s,
    the bounded graph 56.1 s and the decode queue 47.4 s on rank 3 of the
    2026-09-15 tempab5 boot. The key under this root already names the sources,
    flags, Torch and toolkit, so a kept build is reused exactly when nothing it
    was compiled from has changed.
    """
    return Path(os.environ.get("ST_NATIVE_BUILD_ROOT", str(Path.home() / ".cache/st"))) / name
