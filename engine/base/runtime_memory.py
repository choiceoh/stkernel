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


def host_available_bytes():
    """MemAvailable, which is the number earlyoom decides on -- not MemFree.

    `host_free_bytes` above reads MemFree, which is what the boot gate needs: a driver
    allocation cannot take a clean page back the way an anonymous fault can. Anyone asking
    "how close is this box to being killed" wants the other line.
    """
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def live_device_blocks(top: int = 6) -> "list[int]":
    """The sizes of device blocks still ALLOCATED, largest first.

    After a release, `memory_reserved` says what the allocator still holds and this says what
    a live tensor still holds -- the part no amount of `empty_cache` can give back, because
    something in the process is still pointing at it. That is the number that decides whether
    a shutdown was clean.

    The blocks have no names unless `_record_memory_history` was on, so this is a lead rather
    than an answer: one 268 MiB survivor is a different bug from forty 2 MiB ones, and the
    shape of the list says which without paying for recorded history on every boot.
    """
    import torch
    if not torch.cuda.is_available():
        return []
    sizes = [block["size"]
             for segment in torch.cuda.memory_snapshot()
             for block in segment.get("blocks", ())
             if block.get("state") == "active_allocated"]
    return sorted(sizes, reverse=True)[:top]


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
        # What the box is already down before the arena exists and outside the caching
        # allocator: the CUDA context, NCCL's channel buffers and the one-shot transport.
        # It is armed after NCCL is up and before anything large is asked for, so on a node
        # serving nothing else this is that floor -- and it is a number from THIS stack, where
        # the budget's 5.54 GiB was carried over from vLLM's 40th-boot table. Another tenant
        # allocating in the same instant inflates it; the ledger says where it was taken.
        self.floor_bytes = max(0, (total - free) - self.baseline_reserved)
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

    def measured(self) -> dict:
        """The two numbers vLLM gets from `profile_run` and `profile_cudagraph_memory`.

        Both were already in the rows and nobody split them, so `budget.py` kept declaring a
        12 GiB workspace ceiling and quoting vLLM's slope for the activations underneath it
        (45차 §50). The split is structural, not a new measurement:

        `prefill` -- the largest legal chunk runs before any graph is captured
        (`_warmup_prefill_memory`), so the peak workspace standing at the last `prefill/` row
        IS the activation peak: load is arena, and nothing bigger has happened yet.

        `graphs` -- capture only ever ADDS reservation and keeps it, so what the reservation
        gained after that last prefill row is the pools plus persistent scratch, i.e. what the
        box loses for as long as this engine serves.

        `peak` is the whole transient the ceiling has to cover, and `retained` is what remains
        at the end -- reserved minus allocated is memory nobody gets back on this box.
        """
        if not self.phases:
            return {}
        outside = lambda row: row["reserved_bytes"] - self.baseline_reserved - self.arena_bytes   # noqa: E731
        prefill = [row for row in self.phases if row["phase"].startswith("prefill/")]
        last_prefill = prefill[-1] if prefill else None
        final = self.phases[-1]
        return dict(
            floor_bytes=self.floor_bytes,
            prefill_peak_bytes=max((row["peak_workspace_bytes"] for row in prefill), default=0),
            prefill_shapes=[row["phase"] for row in prefill if row["phase"].endswith("/prepared")],
            graph_bytes=max(0, outside(final) - (outside(last_prefill) if last_prefill else 0)),
            peak_workspace_bytes=max(row["peak_workspace_bytes"] for row in self.phases),
            retained_workspace_bytes=max(0, outside(final)),
            workspace_limit_bytes=self.workspace_bytes,
            at_phase=final["phase"],
        )

    def report(self):
        return dict(ready=self.ready, arena_bytes=self.arena_bytes,
                    floor_bytes=self.floor_bytes, measured=self.measured(),
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
