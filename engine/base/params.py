"""Declared parameters, bound to loader views (base). D1's other half.

A model does not allocate its weights. It DECLARES them -- name, rank-local
shape, dtype -- and the loader carves the bytes from the arena straight out
of the rank file (base/loader.RankLoader); `bind` then checks every declared
tensor against the view it got and hands back the mapping. No copy, no
`nn.Parameter`, no second allocation the arena does not know about (D16).

The same Spec drives the preshard (base/preshard.py): `build(sources, rank,
world)` says how this rank's tensor is made from the checkpoint's, so the
layout is written down once and the loader never repacks (D1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Spec:
    name: str
    shape: tuple
    dtype: object                       # torch.dtype
    sources: tuple = ()                 # checkpoint tensor names the build reads
    build: "Callable | None" = None     # build(sources: dict, rank, world) -> tensor of `shape`/`dtype`

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def nbytes(self) -> int:
        import torch
        return self.numel * torch.empty((), dtype=self.dtype).element_size()


def bind(specs: "list[Spec]", views: dict) -> dict:
    """{name: tensor} for every spec, or KeyError/ValueError -- never a silent
    reshape, never a cast (a checkpoint that loads and computes noise is the
    failure D3 exists for)."""
    out = {}
    for s in specs:
        if s.name not in views:
            raise KeyError(f"params: {s.name} is declared but the rank file has no such tensor")
        t = views[s.name]
        if tuple(t.shape) != tuple(s.shape) or t.dtype != s.dtype:
            raise ValueError(f"params: {s.name} declared {tuple(s.shape)} {s.dtype}, got {tuple(t.shape)} {t.dtype}")
        out[s.name] = t
    return out


def total_bytes(specs: "list[Spec]") -> int:
    return sum(s.nbytes() for s in specs)


def _selfcheck() -> None:
    import torch
    specs = [Spec("a", (2, 3), torch.bfloat16), Spec("b", (4,), torch.float32)]
    views = {"a": torch.zeros(2, 3, dtype=torch.bfloat16), "b": torch.zeros(4), "extra": torch.zeros(1)}
    p = bind(specs, views)
    assert set(p) == {"a", "b"} and total_bytes(specs) == 12 + 16
    for bad in ({"a": torch.zeros(3, 2, dtype=torch.bfloat16), "b": views["b"]},
                {"a": torch.zeros(2, 3), "b": views["b"]}, {"a": views["a"]}):
        try:
            bind(specs, bad); raise AssertionError("must refuse")
        except (KeyError, ValueError):
            pass
    print("  params: bind checks name, shape and dtype, never casts OK")


if __name__ == "__main__":
    _selfcheck()
