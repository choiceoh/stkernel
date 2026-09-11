"""The box, declared -- not discovered (framework).

A budget is a list of claims on a fixed number; every claim carries where it
came from, and KV is what is left. Profiles supply the lines; this file
supplies Line, Budget, the provenance vocabulary and the box probes.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path


GIB = 1 << 30


MEASURED = "measured"


READ = "read"


LEDGER = "ledger"


DECLARED = "declared"


ESTIMATED = "estimated"


_ORDER = (MEASURED, READ, LEDGER, DECLARED, ESTIMATED)


@dataclass(frozen=True)
class Line:
    """One claim on the box."""

    name: str
    gib: float
    source: str
    evidence: str

    def __post_init__(self):
        if self.source not in _ORDER:
            raise ValueError(f"unknown provenance {self.source!r}")


class Budget:
    """A box, the claims on it, and whatever is left for KV."""

    def __init__(self, box_gib: float, lines: "list[Line]", label: str = ""):
        self.box_gib = box_gib
        self.lines = list(lines)
        self.label = label

    @property
    def claimed_gib(self) -> float:
        return sum(line.gib for line in self.lines)

    @property
    def kv_gib(self) -> float:
        return self.box_gib - self.claimed_gib

    @property
    def estimated(self) -> "list[Line]":
        return [line for line in self.lines if line.source == ESTIMATED]

    def is_gate(self) -> bool:
        """A budget can gate a boot only when nothing in it is a guess."""
        return not self.estimated and self.kv_gib > 0

    def table(self) -> str:
        width = max((len(line.name) for line in self.lines), default=0)
        width = max(width, len("box"), len("KV (remainder)"))
        out = [f"{self.label}" if self.label else "", ""]
        out.append(f"  {'box':<{width}}  {self.box_gib:>8.2f} GiB")
        out.append(f"  {'-' * width}  {'-' * 8}")
        for line in self.lines:
            out.append(
                f"  {line.name:<{width}}  {-line.gib:>8.2f} GiB   "
                f"[{line.source}] {line.evidence}"
            )
        out.append(f"  {'-' * width}  {'-' * 8}")
        out.append(f"  {'KV (remainder)':<{width}}  {self.kv_gib:>8.2f} GiB")
        return "\n".join(x for x in out if x is not None)

    def verdict(self) -> str:
        if self.kv_gib <= 0:
            return (
                f"DOES NOT FIT: claims exceed the box by {-self.kv_gib:.2f} GiB. "
                "Something must leave the box before KV gets anything."
            )
        guesses = self.estimated
        if guesses:
            names = ", ".join(line.name for line in guesses)
            total = sum(line.gib for line in guesses)
            return (
                f"NOT A GATE: {len(guesses)} estimated line(s) -- {names} -- "
                f"own {total:.2f} GiB, {total / self.box_gib:.0%} of the box and "
                f"{total / max(self.kv_gib, 1e-9):.1f}x the KV they leave. "
                "This budget's conclusion is theirs, not the measurements'."
            )
        return f"GATE: KV = {self.kv_gib:.2f} GiB, declared before load."


def probe_box() -> "tuple[float, float]":
    """(total device GiB, free GiB after our own context).

    Asking costs a CUDA context, so the difference is a real measurement of
    what a context costs on this node -- and the reason instruments.py asks
    ``is_initialized()`` first when it must NOT pay that cost.
    """
    import torch

    free, total = torch.cuda.mem_get_info()
    return total / GIB, free / GIB


def host_box() -> "tuple[float, float]":
    """(MemTotal GiB, MemAvailable GiB) -- what earlyoom actually counts."""
    fields = {}
    for raw in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = raw.partition(":")
        fields[key] = float(rest.strip().split()[0]) / (1 << 20)
    return fields["MemTotal"], fields["MemAvailable"]


def rank_weights(path: "str | Path") -> "tuple[float, int]":
    """(GiB, tensor count) read out of a safetensors header.

    The header is a constant of the checkpoint, so this is a `read`, not a
    measurement and not an estimate. It is also cheap: one 8-byte read plus
    the header, never the 84 GiB behind it.
    """
    path = Path(path)
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    end = max(entry["data_offsets"][1] for entry in header.values())
    return end / GIB, len(header)


def rank_files(checkpoint: "str | Path", world_size: int) -> "list[Path]":
    root = Path(checkpoint)
    files = [root / f"rank{i}of{world_size}.safetensors" for i in range(world_size)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"presharded ranks missing: {', '.join(missing)}")
    return files
