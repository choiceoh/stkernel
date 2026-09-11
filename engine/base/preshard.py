"""Write one rank's tensors, already in the engine's layout, as a safetensors
file (base). The offline half of D1: the loader does no repacking because
this did it once.

The file format is plain safetensors (8-byte header length, JSON header, raw
bytes) written STREAMING: the header is fixed up front from the specs, so a
46 GiB rank file never has to sit in host memory, and the tensors can arrive
in any order (each is written at its own offset). Groups of specs share
their checkpoint sources, so an expert layer's 3.8 GiB is read once and split
four ways rather than read four times.

`RankLoader` (base/loader.py) reads what this writes; the two agree on the
dtype names below.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import torch

from engine.base.arena import ALIGN

_PAD_PREFIX = "__st_padding__."

_NAMES = {torch.uint8: "U8", torch.int8: "I8", torch.float8_e4m3fn: "F8_E4M3", torch.bfloat16: "BF16",
          torch.float16: "F16", torch.float32: "F32", torch.int32: "I32", torch.int64: "I64", torch.bool: "BOOL"}


class RankWriter:
    def __init__(self, path: "str | Path", specs, metadata: "dict | None" = None):
        specs = tuple(specs)
        names = [s.name for s in specs]
        if len(set(names)) != len(names) or any(n.startswith(_PAD_PREFIX) or n == "__metadata__" for n in names):
            raise ValueError("rank tensor names must be unique and outside the reserved metadata/padding namespace")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header, off = {}, 0
        for i, s in enumerate(specs):
            padding = (-off) % ALIGN
            if padding:
                # Safetensors forbids holes. Explicit byte tensors make the
                # padding valid for its standard reader while keeping each
                # real weight aligned for TMA after a coalesced arena upload.
                header[f"{_PAD_PREFIX}{i}"] = {
                    "dtype": "U8", "shape": [padding], "data_offsets": [off, off + padding]}
                off += padding
            n = s.nbytes()
            header[s.name] = {"dtype": _NAMES[s.dtype], "shape": list(s.shape), "data_offsets": [off, off + n]}
            off += n
        if metadata:
            header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * ((8 - len(raw) % 8) % 8)                       # safetensors pads the header to 8
        self.base = 8 + len(raw)
        self.offsets = {s.name: (header[s.name]["data_offsets"][0], s) for s in specs}
        self.done = set()
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        self._write_all(struct.pack("<Q", len(raw)) + raw, 0)
        os.ftruncate(self.fd, self.base + off)
        self.total = off

    def _write_all(self, buf, offset):
        pos, view = 0, memoryview(buf)
        while pos < len(view):
            wrote = os.pwrite(self.fd, view[pos:pos + (1 << 30)], offset + pos)
            if wrote <= 0:
                raise OSError(f"preshard: short write at {offset + pos}")
            pos += wrote

    def put(self, name: str, t: torch.Tensor) -> None:
        off, s = self.offsets[name]
        if tuple(t.shape) != tuple(s.shape) or t.dtype != s.dtype:
            raise ValueError(f"preshard: {name} built {tuple(t.shape)} {t.dtype}, declared {tuple(s.shape)} {s.dtype}")
        if name in self.done:
            raise ValueError(f"preshard: {name} written twice")
        buf = t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        assert len(buf) == s.nbytes()
        self._write_all(buf, self.base + off)
        self.done.add(name)

    def close(self) -> None:
        os.close(self.fd)
        missing = set(self.offsets) - self.done
        if missing:
            raise RuntimeError(f"preshard: {self.path.name} is missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")


def write_ranks(groups, out_paths: "list[Path]", load_sources, world: int, metadata=None, log=print) -> "list[int]":
    """groups: iterable of (label, source_keys, specs_of_rank) where
    specs_of_rank(rank) -> list[Spec] (same names/order for every rank);
    load_sources(keys) -> {name: tensor}. Writes every rank file at once."""
    import time
    groups = list(groups)
    all_specs = [[] for _ in range(world)]
    for _label, _keys, specs_of in groups:
        for r in range(world):
            all_specs[r].extend(specs_of(r))
    writers = [RankWriter(p, all_specs[r], metadata) for r, p in enumerate(out_paths)]
    t0 = time.perf_counter(); read = 0
    for label, keys, specs_of in groups:
        t1 = time.perf_counter()
        src = load_sources(keys)
        read += sum(t.numel() * t.element_size() for t in src.values())
        for r in range(world):
            for s in specs_of(r):
                writers[r].put(s.name, s.build(src, r, world))
        log(f"  {label:<14} {len(keys):>5} sources -> {len(specs_of(0)):>3} tensors/rank  {time.perf_counter() - t1:6.1f} s")
        del src
    for w in writers:
        w.close()
    log(f"  read {read / 2**30:.2f} GiB, wrote {world} x {writers[0].total / 2**30:.2f} GiB in {time.perf_counter() - t0:.0f} s")
    return [w.total for w in writers]


def _selfcheck() -> None:
    import tempfile
    from engine.base.params import Spec
    from engine.base.loader import RankLoader
    full = {"w": torch.arange(8 * 4, dtype=torch.float32).view(8, 4).bfloat16(), "s": torch.tensor(2.0)}
    def specs_of(r):
        return [Spec("w", (4, 4), torch.bfloat16, ("w",), lambda src, r, W: src["w"].narrow(0, r * 4, 4)),
                Spec("s", (), torch.float32, ("s",), lambda src, r, W: 1.0 / src["s"])]
    with tempfile.TemporaryDirectory() as d:
        paths = [Path(d) / f"rank{r}of2.safetensors" for r in range(2)]
        write_ranks([("t", ["w", "s"], specs_of)], paths, lambda keys: {k: full[k] for k in keys}, 2, log=lambda *_: None)
        for r in range(2):
            got = RankLoader(paths[r]).load(["w", "s"], device="cpu")
            assert torch.equal(got["w"], full["w"][r * 4:(r + 1) * 4]) and got["s"].item() == 0.5
        from safetensors import safe_open
        with safe_open(str(paths[1]), "pt") as f:                     # the reference reader agrees
            assert torch.equal(f.get_tensor("w"), full["w"][4:8])
    print("  preshard: streaming safetensors, two ranks, read back by RankLoader and safetensors OK")


if __name__ == "__main__":
    _selfcheck()
