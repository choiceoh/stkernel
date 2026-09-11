"""A byte ceiling for arena + workspaces, with a measured boot ledger.

The allocator's fraction API only enforces this byte ceiling; it never sets
KV capacity. Direct CUDA allocations are outside that allocator, so boot
also checks node-wide immediately free memory. Other processes can consume it.
"""
import json
from pathlib import Path


def host_free_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemFree:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemFree is unavailable")


class RuntimeMemory:
    def __init__(self, arena_bytes, workspace_bytes, os_reserve_bytes, *, comm=None,
                 cuda=None, host_free=host_free_bytes):
        if min(arena_bytes, workspace_bytes, os_reserve_bytes) <= 0:
            raise ValueError("arena, workspace ceiling and OS reserve must be positive bytes")
        if cuda is None:
            import torch
            cuda = torch.cuda
        self.cuda, self.comm, self.host_free = cuda, comm, host_free
        self.arena_bytes, self.workspace_bytes = arena_bytes, workspace_bytes
        self.os_reserve_bytes = os_reserve_bytes
        self.phases, self.ready, self.closed = [], False, False
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
        cuda.synchronize()
        free, _ = cuda.mem_get_info()
        row = dict(phase=phase, allocated_bytes=cuda.memory_allocated(),
                   reserved_bytes=cuda.memory_reserved(),
                   peak_allocated_bytes=cuda.max_memory_allocated(),
                   peak_reserved_bytes=cuda.max_memory_reserved(),
                   immediately_free_bytes=min(free, self.host_free()))
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
        if error:
            self.ready = False
            raise MemoryError(f"{phase}: {error}")
        return row

    def report(self):
        return dict(ready=self.ready, arena_bytes=self.arena_bytes,
                    workspace_limit_bytes=self.workspace_bytes,
                    os_reserve_bytes=self.os_reserve_bytes,
                    baseline_reserved_bytes=self.baseline_reserved,
                    allocator_limit_bytes=self.allocator_limit_bytes,
                    phases=self.phases)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2) + "\n")

    def close(self):
        if not self.closed:
            self.cuda.set_per_process_memory_fraction(self.previous_fraction)
            self.closed = True
