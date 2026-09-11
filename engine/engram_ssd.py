"""Keep the engram tables on the NVMe, because they do not fit in the box.

The arithmetic is not close. A rank holds 73.25 GiB of weights; its two engram
tables are another 47.21 GiB; the box is 121.63 GiB. Resident engram leaves
0 GiB for the CUDA context, the activations and the KV -- so `budget.py` has
assumed `engram on SSD` from its first line, and this is the file that has to
make that assumption true.

What is on disk and what is not:

    weight   [96,001,542, 256] fp8, 24.58 GB per table per rank   -> SSD
    scale    [96,001,542, 8]   e8m0, 0.77 GB per table per rank   -> resident

The split is the shard builder's, not ours (dsv41_engram_io.py:37). A scale row
is 8 bytes and a weight row is 256; reading both would double the IOPS to save
1.5 GiB out of 121, which is the wrong trade in the direction that matters.

The reads themselves are `dsv41_engram_io.ShardReader` unchanged -- O_DIRECT,
one aligned 512-byte sector per row, a fixed thread pool striding the window
list. That module measured 157K IOPS against 52.6K for the naive dispatch, and
none of that reasoning is model-specific, so it is imported rather than
reproduced here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parent.parent / "overlay/modules/dsv41_engram"
ENGRAM_DIR = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash-engram")


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
