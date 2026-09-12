"""One allocation for persistent weights and caches (base).

D16 in code: weights (the loader's blocks), KV block storage, state
slots and resident table scales are carved from it in order, each carve
named and accounted, so `table()` is the budget's ledger at runtime and
`remaining` is uncarved space. Kernel workspaces and graph pools are separate
allocations governed by base/runtime_memory's byte ceiling and peak ledger.

Carving is a bump allocator on purpose: nothing long-lived is ever freed
(sequences free BLOCKS inside the KV region, not the region), so there is
no fragmentation to manage and no allocator to second-guess. The loader
measured what the caching allocator costs when it is allowed to think --
16.1% -- and what one arena costs -- 0.006 GiB.

One allocation is one VIRTUAL range. The physical side is the box's: on
GB10 the first 45-layer boot asked the driver for 55.4 GiB in a single
cudaMalloc and got CUDA_ERROR_OUT_OF_MEMORY before loading a byte (2026-09-11,
srv2), while the same box serves vLLM's 63 GiB every day -- mapped through
`expandable_segments`, 20 MiB physical chunks under one address range. So
the arena asks the caching allocator for exactly that: one contiguous
tensor, backed chunk by chunk (`torch._C._accelerator_setAllocatorSettings`).
Accounting is unchanged: `memory_allocated` still moves by exactly nbytes.
"""
from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
from pathlib import Path

GIB = 1 << 30
ALIGN = 256                       # every carve starts on a 256 B boundary (TMA-friendly)


def _meminfo() -> dict:
    return {key: int(value.split()[0]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            for key, value in [line.split(":", 1)]}


def touch_pages(nbytes: int) -> int:
    """Hold `nbytes` of anonymous memory for an instant, then give it back.

    MemAvailable counts clean page cache the kernel reclaims for an anonymous
    allocation but not, on this UMA box, for a large device one. Faulting the
    desired free extent as anonymous pages (MAP_POPULATE: one syscall, no
    Python loop) forces cache eviction once the already-free pages run out.
    Touching only the shortfall can use free pages without evicting anything.
    Releasing the allocation leaves its pages immediately free.
    """
    page = mmap.PAGESIZE
    n = -(-nbytes // page) * page
    if n <= 0:
        return 0
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_POPULATE", 0)
    region = mmap.mmap(-1, n, flags=flags)
    try:
        if not getattr(mmap, "MAP_POPULATE", 0):
            for off in range(0, n, page):                 # no MAP_POPULATE: fault every page by hand
                region[off] = 1
    finally:
        region.close()
    return n


def release_model_cache(roots) -> int:
    """Return clean model-download pages without allocating or changing files.

    The model volume also holds checkpoints being downloaded for preparation.
    Those pages can occupy UMA DRAM without belonging to a running process.
    DONTNEED returns only clean pages; active writes and file contents survive.
    Never follow symlinks or walk outside the explicitly supplied model roots.
    """
    released = 0
    for root in sorted({Path(p).resolve() for p in roots}):
        if root == Path(root.anchor):
            raise ValueError("a model cache root cannot be the filesystem root")
        for base, _, names in os.walk(root, followlinks=False):
            for name in names:
                if not name.endswith((".safetensors", ".incomplete")):
                    continue
                try:
                    fd = os.open(Path(base) / name, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                    released += 1
                except OSError:
                    # Downloads can rename their incomplete file during this
                    # walk. The final physical-memory check remains decisive.
                    continue
    return released


def prepare_allocation(nbytes: int, files, headroom: int, device_free, reclaim=touch_pages, *, cache_roots=()) -> dict:
    """Drop clean pages of the supplied weight files, reclaim the rest of the
    shortfall, then check physical headroom.

    MemAvailable includes reclaimable page cache. A large CUDA allocation on
    UMA can fail while that number still looks sufficient. This preflight
    counts immediately free pages: it drops the given files' cache, and when
    that is not enough but MemAvailable says the rest is reclaimable, it
    faults the desired free extent (`reclaim`, bounded so MemAvailable never dips under
    `headroom` -- earlyoom's floor is 5%, headroom is 16 GiB). It is a
    necessary admission check, not a guarantee against another process
    allocating after the check.
    """
    for path in files:
        with Path(path).open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    memory = _meminfo()
    need = nbytes + headroom
    free = min(memory["MemFree"], device_free())
    reclaimed = 0
    cache_files = 0
    if free < need and memory["MemFree"] < need and cache_roots:
        cache_files = release_model_cache(cache_roots)
        memory = _meminfo()
        free = min(memory["MemFree"], device_free())
    if free < need and memory["MemFree"] < need:
        if need > memory["MemAvailable"] - headroom:
            raise MemoryError(f"arena admission: allocation {nbytes/GIB:.2f} GiB plus headroom {headroom/GIB:.2f} GiB "
                              f"exceeds immediately free memory {free/GIB:.2f} GiB and cannot be reclaimed: MemAvailable "
                              f"{memory['MemAvailable']/GIB:.2f} GiB")
        reclaimed = reclaim(need) if reclaim is not None else 0
        memory = _meminfo()
        free = min(memory["MemFree"], device_free())
    if free < need:
        raise MemoryError(f"arena admission: allocation {nbytes/GIB:.2f} GiB plus "
                          f"headroom {headroom/GIB:.2f} GiB exceeds immediately free "
                          f"memory {free/GIB:.2f} GiB after reclaiming {reclaimed/GIB:.2f} GiB; MemAvailable "
                          f"{memory['MemAvailable']/GIB:.2f} GiB includes reclaimable pages")
    return dict(allocation=nbytes, headroom=headroom, immediately_free=free,
                available=memory["MemAvailable"], reclaimed=reclaimed, cache_files=cache_files)


def expandable_segments() -> bool:
    """Back every new caching-allocator segment with 20 MiB physical chunks
    under one virtual range. Safe to call after allocations exist: only new
    segments are affected. Returns whether the setting took."""
    import torch
    for setter in (getattr(getattr(torch, "_C", None), "_accelerator_setAllocatorSettings", None),
                   getattr(torch.cuda.memory, "_set_allocator_settings", None)):
        if setter is None:
            continue
        try:
            setter("expandable_segments:True")
            return True
        except Exception:                                   # noqa: BLE001 -- try the next entry point
            continue
    return False


@dataclass(frozen=True)
class Region:
    name: str
    offset: int
    nbytes: int


class Arena:
    def __init__(self, nbytes: int, device: str = "cuda", expandable: bool = True):
        import torch

        self.nbytes = nbytes
        self.expandable = expandable and str(device).startswith("cuda") and expandable_segments()
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
        out = [f"  arena {self.nbytes / GIB:.2f} GiB, used {self.used / GIB:.2f}, free {self.remaining / GIB:.2f}"
               f" ({'expandable segments' if self.expandable else 'one cudaMalloc'})"]
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
    assert arena.expandable, "GB10 maps the arena through expandable segments"
    segments = [s for s in torch.cuda.memory._snapshot()["segments"] if s["total_size"] >= 3 * GIB]
    assert segments and all(s.get("is_expandable") for s in segments), "the arena's segment must be expandable"
    touched = touch_pages(256 << 20)
    assert touched == 256 << 20

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
    print(f"  arena: one expandable allocation, loader carved {len(keys)} tensors into it, 40/40 byte-identical, overflow refused, 256 MiB reclaim touch OK")


if __name__ == "__main__":
    _selfcheck()
