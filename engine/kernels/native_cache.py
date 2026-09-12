"""Stable, content-addressed source inputs for ST's native CUDA builds.

Torch/Ninja still own compilation, dependency tracking, locking and loading.
The source snapshot prevents an identical checkout or rsync timestamp from
changing Ninja's inputs. This module neither loads a binary nor touches CUDA.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile


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
