"""Where tensors go across ranks (framework): safetensors header walking and
the byte arithmetic shared by every profile's placement rules.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path


GIB = 1 << 30


_ITEMSIZE = {"I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4, "F16": 2}


def _headers(repo: Path):
    index = json.loads((repo / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        path = repo / shard
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        header.pop("__metadata__", None)
        yield header
