"""Giant embedding tables that do not fit in the box, read row-wise from NVMe
(module: DSv4.1 engram, Qwen3.8 PLE).

The reads are dsv41_engram_io.ShardReader (O_DIRECT, one aligned sector per
row, fixed thread pool) -- nothing in it is model-specific, so it is imported.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


OVERLAY = Path(__file__).resolve().parent.parent / "overlay/modules/dsv41_engram"


def _io_module():
    if "dsv41_engram_io" in sys.modules:
        return sys.modules["dsv41_engram_io"]
    spec = importlib.util.spec_from_file_location(
        "dsv41_engram_io", OVERLAY / "dsv41_engram_io.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv41_engram_io"] = mod
    spec.loader.exec_module(mod)
    return mod


class SSDEngramLookup:
    """One table's rows, fetched per step instead of held."""

    def __init__(self, weight_path: "str | Path", scale, block_size: int,
                 queue_depth: int = 32):
        io = _io_module()
        self.reader = io.ShardReader(str(weight_path), queue_depth=queue_depth)
        self.row_bytes = io.EMB_ROW_BYTES
        self.scale = scale                      # resident, [rows, dim/block]
        self.block_size = block_size
        self.rows_read = 0
        self.calls = 0

    def rows(self, local_indices):
        """[N, dim] bf16 for a flat int64 tensor of local row ids."""
        import torch

        flat = local_indices.reshape(-1)
        uniq, inverse = torch.unique(flat, return_inverse=True)
        wanted = uniq.tolist()
        blob = b"".join(self.reader.gather(wanted))
        self.rows_read += len(wanted)
        self.calls += 1
        table = (torch.frombuffer(bytearray(blob), dtype=torch.uint8)
                 .view(len(wanted), self.row_bytes)
                 .to(local_indices.device, non_blocking=False)
                 .view(torch.float8_e4m3fn))
        scales = self.scale.index_select(0, uniq)
        vals = (table.float().unflatten(-1, (-1, self.block_size))
                * scales.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        return vals.index_select(0, inverse).view(*local_indices.shape, -1)
