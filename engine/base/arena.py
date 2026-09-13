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
import errno
import mmap
import os
from pathlib import Path
import time

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
    try:
        region = mmap.mmap(-1, n, flags=flags)
    except OSError as exc:
        if exc.errno != errno.ENOMEM:
            raise
        # MemAvailable is a physical estimate; strict overcommit separately
        # limits anonymous reservations before a single page can be faulted.
        # Retain both facts so another boot does not repeat the same request.
        try:
            memory = _meminfo()
        except OSError:
            memory = {}
        try:
            policy = Path("/proc/sys/vm/overcommit_memory").read_text().strip()
        except OSError:
            policy = "unknown"
        counters = ", ".join(f"{key}={memory[key]/GIB:.2f} GiB"
                             for key in ("MemFree", "MemAvailable", "Cached", "CommitLimit", "Committed_AS")
                             if key in memory)
        raise MemoryError(f"anonymous cache reclaim: mmap({n/GIB:.2f} GiB) refused; "
                          f"overcommit_memory={policy}, {counters}. "
                          "Return the file cache from the host before boot (launchers/st-return-file-cache.sh, "
                          "which start-st-glm53.sh runs unless ST_RECLAIM_FILE_CACHE=0); the arena budget is unchanged") from exc
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


RECLAIM_DIR_ENV = "ST_RECLAIM_DIR"
HOST_RECLAIM_GAP_S = 30.0
_asked_at: "dict[str, float]" = {}


def host_reclaim(directory=None, *, timeout_s: float = 150.0, fresh_s: float = 5.0, poll_s: float = 0.2,
                 min_gap_s: float = HOST_RECLAIM_GAP_S, clock=time.monotonic, sleep=time.sleep) -> "dict | None":
    """Ask this node's host to return its clean file cache now, and wait for it to say it did.

    The container cannot drop another workload's cache, and faulting the shortfall as anonymous memory needs
    commit room a strict-overcommit node does not have: srv2 (overcommit_memory=2, ratio 50, CommitLimit
    75.8 GiB) refused every such mapping on 2026-09-13. The host can drop it without allocating anything --
    launchers/st-reclaim-broker.sh, which the launcher starts for this rank and names in ST_RECLAIM_DIR.

    None when nobody serves: no directory, or a heartbeat older than `fresh_s` (a broker that ended, a boot
    without one), so a boot that has no broker does not wait. None too within `min_gap_s` of this rank's last
    question: warmup checkpoints run by the hundred, and a node that stays short would otherwise drop its
    cache at every one of them -- a slower boot, and every other workload on the box rereading its files.
    Otherwise {returned, rc, line}: returned is whether the host dropped it; line is its one-line report
    (MemFree, MemAvailable, Cached before and after).
    """
    directory = directory or os.environ.get(RECLAIM_DIR_ENV)
    if not directory:
        return None
    root = Path(directory)
    try:
        if time.time() - (root / "heartbeat").stat().st_mtime > fresh_s:
            return None
    except OSError:
        return None
    last = _asked_at.get(str(root))
    if last is not None and clock() - last < min_gap_s:
        return None
    _asked_at[str(root)] = clock()
    ident = f"{os.getpid()}-{time.time_ns()}"
    try:
        (root / "done").unlink()                        # an answer nobody read belongs to an earlier question
    except OSError:
        pass
    staged = root / f"request.{ident}.tmp"
    try:
        staged.write_text(ident + "\n")
        os.replace(staged, root / "request")
    except OSError as exc:
        return dict(returned=False, rc=None, line=f"could not ask the host: {exc}")
    deadline = clock() + timeout_s
    while clock() < deadline:
        try:
            lines = (root / "done").read_text().splitlines()
        except OSError:
            lines = []
        if len(lines) >= 2 and lines[0] == ident:
            try:
                (root / "done").unlink()
            except OSError:
                pass
            rc = int(lines[1]) if lines[1].strip().isdigit() else None
            return dict(returned=rc == 0, rc=rc, line=lines[2] if len(lines) > 2 else "")
        sleep(poll_s)
    return dict(returned=False, rc=None, line=f"the host did not answer in {timeout_s:.0f} s")


TRANSIENT_MARGIN = 2 << 30
"""How far above the box's SIGTERM line the momentary reclaim fault must stay.

This transient fault keeps two GiB even when a profile's steady reserve uses a smaller margin:
MemAvailable is an estimate and earlyoom samples rather than watches. What differs is which
line applies. The engine's full `headroom` -- workspace ceiling plus OS reserve -- is a POST-BOOT
requirement, and the check after the reclaim enforces it. During the fault nothing is allocated and
nothing is serving, so the only line with a consequence is the box's.
"""


def prepare_allocation(nbytes: int, files, headroom: int, device_free, reclaim=touch_pages, *, cache_roots=(),
                       host_reclaim=None) -> dict:
    """Drop clean pages of the supplied weight files, reclaim the rest of the
    shortfall, then check physical headroom.

    MemAvailable includes reclaimable page cache. A large CUDA allocation on
    UMA can fail while that number still looks sufficient. This preflight
    counts immediately free pages: it drops the given files' cache, and when
    that is not enough but MemAvailable says the rest is reclaimable, it
    faults the desired free extent (`reclaim`, bounded so the fault never takes MemAvailable under
    the box's own SIGTERM line -- base/runtime_memory.oom_floor, not the engine's full headroom,
    which is not allocated yet and has nothing serving behind it). It is a
    necessary admission check, not a guarantee against another process
    allocating after the check.

    `host_reclaim`: asked before the anonymous fault, when the model cache was not enough -- the host drops
    the cache without the commit charge a strict-overcommit node refuses (`host_reclaim` above). Its answer
    rides the report; when it is None or not enough, the fault below runs as it always did.
    """
    for path in files:
        with Path(path).open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    memory = _meminfo()
    file_cache = memory.get("Cached", 0)            # what was cache when admission began: whether the host's return reached it
    need = nbytes + headroom
    free = min(memory["MemFree"], device_free())
    reclaimed = 0
    cache_files = 0
    returned = None
    if free < need and memory["MemFree"] < need and cache_roots:
        cache_files = release_model_cache(cache_roots)
        memory = _meminfo()
        free = min(memory["MemFree"], device_free())
    if free < need and memory["MemFree"] < need and memory["MemAvailable"] >= need and host_reclaim is not None:
        returned = host_reclaim()                  # only when dropping what is reclaimable can cover the shortfall
        if returned is not None:
            memory = _meminfo()
            free = min(memory["MemFree"], device_free())
    said = "" if returned is None else f"; the host {'returned' if returned['returned'] else 'did not return'} file cache: {returned['line']}"
    if free < need and memory["MemFree"] < need:
        # What the momentary fault must not cross is the BOX's kill line, not the engine's whole
        # future headroom. The reclaim faults `need` for an instant -- MemFree first, so the cache
        # beyond it is evicted -- and then gives it back; the workspace and the OS reserve inside
        # `headroom` are not allocated yet and nothing is serving. Requiring `need + headroom` of
        # MemAvailable asked the node for the headroom twice, and once #710 took the OS reserve
        # from 4 GiB to the box's own floor + 2, that second copy grew past what the nodes have:
        # on 2026-09-12 a boot was refused at MemAvailable 92.74 GiB for an allocation of 60.99
        # that fits, because 60.99 + 20 + 20 does not. The post-reclaim check below is unchanged
        # and still enforces the full headroom, which is where it belongs.
        from engine.base.runtime_memory import oom_floor
        floor = oom_floor()[0] + TRANSIENT_MARGIN
        if need > memory["MemAvailable"] - floor:
            raise MemoryError(f"arena admission: allocation {nbytes/GIB:.2f} GiB plus headroom {headroom/GIB:.2f} GiB "
                              f"exceeds immediately free memory {free/GIB:.2f} GiB and cannot be reclaimed without "
                              f"crossing this box's SIGTERM line plus margin ({floor/GIB:.2f} GiB): MemAvailable "
                              f"{memory['MemAvailable']/GIB:.2f} GiB, file cache {file_cache/GIB:.2f} GiB when admission began{said}")
        reclaimed = reclaim(need) if reclaim is not None else 0
        memory = _meminfo()
        free = min(memory["MemFree"], device_free())
    if free < need:
        raise MemoryError(f"arena admission: allocation {nbytes/GIB:.2f} GiB plus "
                          f"headroom {headroom/GIB:.2f} GiB exceeds immediately free "
                          f"memory {free/GIB:.2f} GiB after reclaiming {reclaimed/GIB:.2f} GiB; MemAvailable "
                          f"{memory['MemAvailable']/GIB:.2f} GiB includes reclaimable pages "
                          f"(file cache {file_cache/GIB:.2f} GiB when admission began){said}")
    return dict(allocation=nbytes, headroom=headroom, immediately_free=free,
                available=memory["MemAvailable"], reclaimed=reclaimed, cache_files=cache_files, file_cache=file_cache,
                host_reclaim=returned)


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

    def release(self) -> int:
        """Free the one allocation now, whatever still points at it, and say how many bytes.

        Every carve is a VIEW, so dropping the arena's own reference frees nothing while a
        single weight tensor is alive -- and at shutdown there are always some: `build` handed
        `net` and `caches` back to a frame that is still on the stack. So this does not drop a
        reference, it resizes the storage to zero. The caching allocator takes the block back
        at once and `empty_cache` can hand the physical chunks to the driver; the views survive
        as zero-byte tensors, and touching one afterwards raises instead of reading memory that
        belongs to somebody else (D3).

        This is not a way to shrink a live arena. There is no such thing here on purpose --
        bump allocation, pinned NVRM pages, a footprint that is constant after boot
        (OOM_STUDY 2) -- and this is the end of the tenancy, not a trim in the middle of it.

        Idempotent: the second call has nothing to free and returns 0.
        """
        if self.buf is None:
            return 0
        given = self.nbytes
        self.buf.untyped_storage().resize_(0)
        self.buf = None
        self.regions = []
        self.used = self.nbytes
        return given

    def carve(self, nbytes: int, name: str):
        """A uint8 view of `nbytes`, or MemoryError -- never a second allocation (D3)."""
        if self.buf is None:
            raise MemoryError(f"arena: {name} wants {nbytes / GIB:.3f} GiB and the arena is released")
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
