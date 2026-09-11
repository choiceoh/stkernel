"""One allocation for the box (base). Everything else is a view into it.

D16 in code: the arena is the ONLY device allocation the engine makes for
long-lived memory. Weights (the loader's blocks), KV block storage, state
slots and resident table scales are carved from it in order, each carve
named and accounted, so `table()` is the budget's ledger at runtime and
`remaining` is what KV gets.

Carving is a bump allocator on purpose: nothing long-lived is ever freed
(sequences free BLOCKS inside the KV region, not the region), so there is
no fragmentation to manage and no allocator to second-guess. The loader
measured what the caching allocator costs when it is allowed to think --
16.1% -- and what one arena costs -- 0.006 GiB.
"""
from __future__ import annotations

from dataclasses import dataclass

GIB = 1 << 30
ALIGN = 256                       # every carve starts on a 256 B boundary (TMA-friendly)


@dataclass(frozen=True)
class Region:
    name: str
    offset: int
    nbytes: int


class Arena:
    def __init__(self, nbytes: int, device: str = "cuda"):
        import torch

        self.nbytes = nbytes
        self.buf = torch.empty(nbytes, dtype=torch.uint8, device=device)   # the one allocation
        self.used = 0
        self.regions: "list[Region]" = []

    @property
    def remaining(self) -> int:
        return self.nbytes - self.used

    def carve(self, nbytes: int, name: str):
        """A uint8 view of `nbytes`, or MemoryError -- never a second allocation (D3)."""
        start = -(-self.used // ALIGN) * ALIGN
        if start + nbytes > self.nbytes:
            raise MemoryError(f"arena: {name} wants {nbytes / GIB:.3f} GiB, "
                              f"{(self.nbytes - start) / GIB:.3f} GiB left of {self.nbytes / GIB:.2f}")
        self.regions.append(Region(name, start, nbytes))
        self.used = start + nbytes
        return self.buf[start:start + nbytes]

    def table(self) -> str:
        width = max((len(r.name) for r in self.regions), default=4)
        out = [f"  arena {self.nbytes / GIB:.2f} GiB, used {self.used / GIB:.2f}, free {self.remaining / GIB:.2f}"]
        for r in self.regions:
            out.append(f"    {r.name:<{width}}  @{r.offset / GIB:8.3f}  {r.nbytes / GIB:8.3f} GiB")
        return "\n".join(out)


def _selfcheck() -> None:
    import re
    import torch
    from engine.base.loader import RankLoader

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    arena = Arena(3 * GIB)
    after_arena = torch.cuda.memory_allocated()
    assert after_arena - before == 3 * GIB, "the arena is one allocation of exactly its size"

    path = "/home/choiceoh/models/DeepSeek-V4.1-Flash-tp4/rank0of4.safetensors"
    loader = RankLoader(path)
    keys = [k for k in loader.keys() if re.match(r"layers\.0\.", k)]
    tensors = loader.load(keys, device="cuda", arena=arena)
    assert torch.cuda.memory_allocated() == after_arena, "loading into the arena allocates NOTHING else"
    assert [r.name.startswith("weights/") for r in arena.regions] and arena.used > 1.8 * GIB
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in keys[:40]:
            ref = f.get_tensor(k)
            assert torch.equal(ref.view(torch.uint8), tensors[k].cpu().view(torch.uint8)), k
    kv = arena.carve(512 << 20, "kv blocks")
    assert kv.numel() == 512 << 20 and arena.remaining < 3 * GIB - 2 * GIB
    try:
        arena.carve(2 * GIB, "too much"); raise AssertionError("overflow must raise")
    except MemoryError:
        pass
    print(arena.table())
    print(f"  arena: one allocation, loader carved {len(keys)} tensors into it, 40/40 byte-identical, overflow refused OK")


if __name__ == "__main__":
    _selfcheck()
