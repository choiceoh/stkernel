"""Stable, content-addressed source inputs for ST's native CUDA builds.

Torch/Ninja still own compilation, dependency tracking, locking and loading.
The source snapshot prevents an identical checkout or rsync timestamp from
changing Ninja's inputs. This module neither loads a binary nor touches CUDA.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


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


def cuda_toolchain_identity(cuda_home, nvcc=None):
    """Identify the compiler independently of the CUDA version built into Torch.

    A CUDA 13.0 Torch wheel can compile ST extensions with CUDA 13.2. Compiler
    feature guards then change the device code even when every source, flag
    and Torch version stays unchanged. Do not reuse the old native artifact.
    These version queries do not initialize a CUDA context.
    """
    if not cuda_home:
        raise RuntimeError("native CUDA build requires a CUDA toolkit")
    root = Path(cuda_home)
    paths = (Path(nvcc) if nvcc else root / "bin/nvcc", root / "bin/ptxas")
    return [(str(path.resolve()), subprocess.check_output(
        [str(path), "--version"], text=True, timeout=10).strip()) for path in paths]


def prepare_cuda_sources(root, sources, identity):
    """Key the canonical SDK; the runtime rejects redirected PYTORCH_NVCC."""
    from torch.utils.cpp_extension import CUDA_HOME
    compiler = cuda_toolchain_identity(CUDA_HOME)
    return prepare_sources(root, sources, (*identity, compiler))


def prepare_sources(root, sources, identity):
    """Return (key, build_directory, staged_paths) for flat source siblings.

    Read every input once: the key and compiler inputs refer to the same bytes.
    Include local headers in ``sources`` and all explicit build flags/runtime
    versions in ``identity``. Headers are staged beside their translation unit;
    only translation units should be handed to Torch's extension loader.
    """
    snapshots = [(Path(p).name, Path(p).read_bytes()) for p in sources]
    names = [name for name, _ in snapshots]
    if not names or len(names) != len(set(names)):
        raise ValueError("native build sources must have nonempty, unique basenames")
    record = {"schema": 1, "identity": identity,
              "sources": [(name, hashlib.sha256(data).hexdigest()) for name, data in snapshots]}
    key = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    directory = Path(root).expanduser().resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    # Different processes can prepare this same key before Torch takes its
    # build lock. Only the first writer should change the staged file's mtime.
    with (directory / ".sources.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        staged = directory / "src"
        staged.mkdir(exist_ok=True)
        for name, data in snapshots:
            path = staged / name
            if path.is_file() and path.read_bytes() == data:
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=staged, prefix=".source-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return key, directory, tuple(str(directory / "src" / name) for name in names)
