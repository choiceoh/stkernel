"""DSv4.1-Flash's engram tables on the reference module tree (profile).
"""
from __future__ import annotations

from pathlib import Path

from engine.modules.lookup_table import SSDEngramLookup, _io_module


ENGRAM_DIR = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash-engram")


def prepare(ref):
    """Make `ParallelEngramEmbedding.__init__` cheap, BEFORE the model is built.

    Swapping after construction is too late: `Transformer.__init__` allocates
    both tables on the device as it walks the layers, and 47.21 GiB on top of
    73.25 does not survive to the line that would have freed it. So the class
    is patched to allocate a zero-row weight and keep only the scale, and
    `attach` fills in the readers afterwards.
    """
    import torch
    from torch import nn

    cls = ref.ParallelEngramEmbedding
    if getattr(cls, "_ssd_patched", False):
        return cls
    original = cls.__init__

    def __init__(self, num_embeddings, dim):
        original(self, num_embeddings, dim)
        self.weight = nn.Parameter(
            torch.empty(0, dim, dtype=torch.float8_e4m3fn,
                        device=self.weight.device), requires_grad=False)

    cls.__init__ = __init__
    cls._ssd_patched = True
    return cls


def _load_scale(path: "str | Path", rows: int, width: int, device):
    """The resident half of a table, read once."""
    import torch

    raw = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.uint8)
    if raw.numel() != rows * width:
        raise ValueError(f"{Path(path).name}: {raw.numel():,} bytes, expected "
                         f"{rows * width:,} ({rows:,} x {width})")
    return raw.view(rows, width).to(device).view(torch.float8_e8m0fnu)


def attach(model, rank: int, world: int, engram_dir: "str | Path" = ENGRAM_DIR):
    """Swap every ParallelEngramEmbedding for an SSD-backed lookup.

    The layer id is recovered from the module path (`layers.1.engram...`)
    because the shard files are named for it, and a wrong pairing here reads
    real rows out of the wrong table -- which produces plausible numbers and no
    error at all.
    """
    import torch
    import torch.distributed as dist
    from torch import nn

    engram_dir = Path(engram_dir)
    swapped = []
    for name, module in list(model.named_modules()):
        if type(module).__name__ != "ParallelEngramEmbedding":
            continue
        layer = None
        for part in name.split("."):
            if part.isdigit():
                layer = int(part)
                break
        if layer is None:
            raise ValueError(f"cannot tell which engram table {name} is")
        path = engram_dir / f"engram-l{layer}-r{rank}of{world}.weight"
        if not path.exists():
            raise FileNotFoundError(path)
        scale_path = path.with_suffix(".scale")
        io = _io_module()
        n_rows = path.stat().st_size // io.EMB_ROW_BYTES
        scale = _load_scale(scale_path, n_rows, io.SCALE_ROW_BYTES,
                            module.scale.device)
        module.scale = nn.Parameter(scale, requires_grad=False)
        lookup = SSDEngramLookup(path, scale, module.block_size)
        if lookup.reader.n_rows != module.part_num_embeddings:
            raise ValueError(
                f"{path.name} holds {lookup.reader.n_rows:,} rows, the module "
                f"wants {module.part_num_embeddings:,}")

        module.ssd = lookup

        def forward(self, indices, _lookup=lookup):
            mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
            local = (indices - self.vocab_start_idx).masked_fill(mask, 0)
            values = _lookup.rows(local)
            values = values.masked_fill(mask.unsqueeze(-1), 0)
            if world > 1 and dist.is_initialized():
                dist.all_reduce(values)
            return values

        module.forward = forward.__get__(module, type(module))
        swapped.append((name, layer, lookup.reader.n_rows))
    return swapped
