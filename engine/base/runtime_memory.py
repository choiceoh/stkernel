"""A byte ceiling for arena + workspaces, with a measured boot ledger.

The allocator's fraction API only enforces this byte ceiling; it never sets
KV capacity. Direct CUDA allocations are outside that allocator, so boot
also checks node-wide immediately free memory. Other processes can consume it.

Each row also carries when it was taken and how long the step before it ran.
That costs nothing -- the row already synchronizes CUDA to read its peaks --
and it is what splits a boot's phases by time: the 45-layer boot's capture is
28.2 s over 115 graphs and two prefill warmups (boot-time study, 2026-09-11),
and until the rows had a clock nobody could say which of them it was.
"""
import json
import time
from pathlib import Path


def host_free_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemFree:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemFree is unavailable")


def reclaim_preparation_pages(need, headroom, *, cache_roots=()):
    """Make a boot's remaining byte budget available despite UMA file cache.

    Anonymous faults evict clean cache where CUDA allocation does not. The
    temporary allocation must itself leave the declared workspace/OS floor;
    insufficient MemAvailable leaves the existing admission failure intact.
    """
    from engine.base.arena import _meminfo, release_model_cache, touch_pages
    memory = _meminfo()
    if memory['MemFree'] < need and cache_roots:
        release_model_cache(cache_roots)
        memory = _meminfo()
    if memory['MemFree'] >= need or need > memory['MemAvailable'] - headroom:
        return 0
    return touch_pages(need)


class RuntimeMemory:
    def __init__(self, arena_bytes, workspace_bytes, os_reserve_bytes, *, comm=None,
                 cuda=None, host_free=host_free_bytes, reclaim=None):
        if min(arena_bytes, workspace_bytes, os_reserve_bytes) <= 0:
            raise ValueError("arena, workspace ceiling and OS reserve must be positive bytes")
        if cuda is None:
            import torch
            cuda = torch.cuda
        self.cuda, self.comm, self.host_free = cuda, comm, host_free
        self.reclaim = reclaim
        self.arena_bytes, self.workspace_bytes = arena_bytes, workspace_bytes
        self.os_reserve_bytes = os_reserve_bytes
        self.phases, self.ready, self.closed = [], False, False
        self.clock = time.monotonic                 # injectable for tests
        self.started = self.clock()
        self.last = self.started
        self.status = None
        if comm is not None:
            import torch
            self.status = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.baseline_reserved = cuda.memory_reserved()
        self.allocator_limit_bytes = self.baseline_reserved + arena_bytes + workspace_bytes
        free, total = cuda.mem_get_info()
        self.previous_fraction = cuda.get_per_process_memory_fraction()
        if self.allocator_limit_bytes > int(total * self.previous_fraction):
            raise MemoryError("runtime byte budget exceeds the existing allocator limit")
        if arena_bytes + workspace_bytes + os_reserve_bytes > min(free, host_free()):
            raise MemoryError("runtime byte budget plus OS reserve exceeds immediately free memory")
        if cuda.get_allocator_backend() != "native":
            raise RuntimeError("ST's byte ceiling requires the pinned native CUDA allocator")
        cuda.set_per_process_memory_fraction(self.allocator_limit_bytes / total)
        cuda.reset_peak_memory_stats()

    def checkpoint(self, phase):
        """Boot only: retain transient peaks and require every TP rank to pass."""
        cuda = self.cuda
        cuda.synchronize()                          # the row's peaks and its clock read the same instant
        reclaimed = 0
        host_free = self.host_free()
        need = self.os_reserve_bytes + max(0, self.allocator_limit_bytes-cuda.memory_reserved())
        if self.reclaim is not None and host_free < need:
            reclaimed = self.reclaim(need, self.workspace_bytes+self.os_reserve_bytes)
            host_free = self.host_free()
        now = self.clock()
        free, _ = cuda.mem_get_info()
        row = dict(phase=phase, at_seconds=round(now - self.started, 4),
                   seconds=round(now - self.last, 4), allocated_bytes=cuda.memory_allocated(),
                   reserved_bytes=cuda.memory_reserved(),
                   peak_allocated_bytes=cuda.max_memory_allocated(),
                   peak_reserved_bytes=cuda.max_memory_reserved(),
                   immediately_free_bytes=min(free, host_free), reclaimed_bytes=reclaimed)
        row["peak_workspace_bytes"] = max(0, row["peak_reserved_bytes"] - self.baseline_reserved - self.arena_bytes)
        error = None
        if row["peak_reserved_bytes"] > self.allocator_limit_bytes:
            error = "allocator peak exceeds the declared runtime byte ceiling"
        if row["immediately_free_bytes"] < self.os_reserve_bytes:
            error = "preparation consumed the OS memory reserve"
        if self.comm is not None:
            self.status.fill_(int(error is not None))
            if int(self.comm.all_reduce_max(self.status).item()) and error is None:
                error = "a TP peer failed runtime memory qualification"
        row["passed"] = error is None
        self.phases.append(row)
        self.last = now
        if error:
            self.ready = False
            raise MemoryError(f"{phase}: {error}")
        return row

    def spend(self, top: int = 8) -> "list[tuple[str, float]]":
        """Where a boot's checkpointed time went: the phase prefixes, most expensive first.
        `target/(4, 6, 4096)` counts under `target`, so the ladder, the samplers and
        the prefill warmups are separable without reading 351 rows."""
        totals = {}
        for row in self.phases:
            key = row["phase"].split("/")[0]
            totals[key] = totals.get(key, 0.0) + row.get("seconds", 0.0)
        return sorted(totals.items(), key=lambda kv: -kv[1])[:top]

    def weigh(self, top: int = 8) -> "list[tuple[str, int]]":
        """Where a boot's memory went: what each phase prefix ADDED to the allocator's reservation, most first.
        `spend()` answers this for time and has since the boot-time study; the bytes were in every row all along
        and nobody summed them, so "which capture is worth its footprint" meant reading 351 rows by hand.
        The reservation is what the box loses (GB10: reserved minus allocated is memory nobody gets back), so
        that is the column, and a phase that gave memory back counts negative rather than being clipped to zero."""
        totals, last = {}, self.baseline_reserved
        for row in self.phases:
            key = row["phase"].split("/")[0]
            totals[key] = totals.get(key, 0) + (row["reserved_bytes"] - last)
            last = row["reserved_bytes"]
        return sorted(totals.items(), key=lambda kv: -kv[1])[:top]

    def report(self):
        return dict(ready=self.ready, arena_bytes=self.arena_bytes,
                    workspace_limit_bytes=self.workspace_bytes,
                    os_reserve_bytes=self.os_reserve_bytes,
                    baseline_reserved_bytes=self.baseline_reserved,
                    allocator_limit_bytes=self.allocator_limit_bytes,
                    seconds=round(self.last - self.started, 4),
                    spend=dict(self.spend()), weigh=dict(self.weigh()),
                    phases=self.phases)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2) + "\n")

    def close(self):
        if not self.closed:
            self.cuda.set_per_process_memory_fraction(self.previous_fraction)
            self.closed = True
