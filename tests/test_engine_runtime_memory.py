"""Workspace peaks, physical reserves and allocator limits are separate contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from engine.base.runtime_memory import RuntimeMemory


class Cuda:
    def __init__(self):
        self.fraction = .9
        self.reserved = self.allocated = self.peak_reserved = self.peak_allocated = 10
        self.free, self.total = 900, 1000
        self.backend = "native"
    def memory_reserved(self): return self.reserved
    def memory_allocated(self): return self.allocated
    def max_memory_reserved(self): return self.peak_reserved
    def max_memory_allocated(self): return self.peak_allocated
    def mem_get_info(self): return self.free, self.total
    def get_per_process_memory_fraction(self): return self.fraction
    def set_per_process_memory_fraction(self, value): self.fraction = value
    def get_allocator_backend(self): return self.backend
    def reset_peak_memory_stats(self): pass
    def synchronize(self): pass


class RuntimeMemoryTests(unittest.TestCase):
    def budget(self, cuda=None, host_free=lambda: 800):
        return RuntimeMemory(400, 200, 100, cuda=cuda or Cuda(), host_free=host_free)

    def test_byte_ceiling_includes_existing_allocator_and_restores_previous_limit(self):
        cuda = Cuda()
        memory = self.budget(cuda)
        self.assertEqual(memory.allocator_limit_bytes, 610)
        self.assertEqual(cuda.fraction, .61)
        memory.close(); memory.close()
        self.assertEqual(cuda.fraction, .9)

    def test_peak_cannot_be_hidden_by_freed_intermediates(self):
        cuda = Cuda()
        memory = self.budget(cuda)
        cuda.allocated = cuda.reserved = 420
        cuda.peak_allocated = 570
        cuda.peak_reserved = 600
        row = memory.checkpoint("target/4/6")
        self.assertEqual(row['peak_workspace_bytes'], 190)
        self.assertEqual(row['allocated_bytes'], 420)
        cuda.peak_reserved = 620
        with self.assertRaisesRegex(MemoryError, "byte ceiling"):
            memory.checkpoint("sampling/stochastic")
        self.assertFalse(memory.phases[-1]['passed'])
        self.assertFalse(memory.ready)
        memory.close()

    def test_non_allocator_or_peer_usage_must_leave_the_os_reserve(self):
        free = [800]
        memory = self.budget(host_free=lambda: free[0])
        free[0] = 99
        with self.assertRaisesRegex(MemoryError, "OS memory reserve"):
            memory.checkpoint("external CUDA allocation")
        memory.close()

    def test_invalid_admission_does_not_change_allocator_configuration(self):
        for free, previous, backend in [(600, .9, "native"), (800, .5, "native"), (800, .9, "cudaMallocAsync")]:
            cuda = Cuda(); cuda.fraction = previous; cuda.backend = backend
            with self.assertRaises((MemoryError, RuntimeError)):
                self.budget(cuda, lambda: free)
            self.assertEqual(cuda.fraction, previous)

    def test_rows_carry_a_clock_and_the_report_splits_the_spend_by_phase(self):
        """The boot-time study's missing field: which phases the seconds went to."""
        ticks = iter([101.5, 103.5, 103.6, 110.6, 111.0])        # the clock reads once per checkpoint
        memory = self.budget()
        memory.clock = lambda: next(ticks)
        memory.started = memory.last = 100.0
        for phase in ("loaded", "prefill/6912/0/prepared", "target/(1, 6, 4096)/warmup",
                      "target/(1, 6, 4096)/captured", "sampling/greedy/(1, 6)/captured"):
            memory.checkpoint(phase)
        rows = memory.phases
        self.assertEqual([r["at_seconds"] for r in rows], [1.5, 3.5, 3.6, 10.6, 11.0])
        self.assertEqual([r["seconds"] for r in rows], [1.5, 2.0, 0.1, 7.0, 0.4])
        self.assertEqual(memory.spend(), [("target", 7.1), ("prefill", 2.0), ("loaded", 1.5), ("sampling", 0.4)])
        report = memory.report()
        self.assertEqual(report["seconds"], 11.0)
        self.assertEqual(report["spend"]["target"], 7.1)
        memory.close()

    def test_report_preserves_declared_limits_and_measured_phases(self):
        memory = self.budget()
        memory.checkpoint("loaded")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0" / "memory.json"
            memory.write(path)
            report = json.loads(path.read_text())
            self.assertFalse(report['ready'])
            self.assertEqual(report['workspace_limit_bytes'], 200)
            self.assertEqual(report['phases'][0]['phase'], 'loaded')
        memory.close()


torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class CudaMemoryTests(unittest.TestCase):
    def test_small_allocator_ceiling_rejects_large_allocation_without_using_host_memory(self):
        # A separate process owns its allocator; never change other tests' pools.
        import subprocess
        import sys
        code = '''
import torch
from engine.base.runtime_memory import RuntimeMemory
MiB = 1 << 20
torch.cuda.init()
before = torch.cuda.get_per_process_memory_fraction()
# Small allocations reserve a 20 MiB allocator slab in the pinned runtime.
budget = RuntimeMemory(4*MiB, 28*MiB, 64*MiB)
try:
    x = torch.empty(4*MiB, device="cuda", dtype=torch.uint8)
    budget.checkpoint("small arena")
    try:
        torch.empty(64*MiB, device="cuda", dtype=torch.uint8)
    except torch.OutOfMemoryError:
        pass
    else:
        raise AssertionError("allocator ignored byte ceiling")
finally:
    budget.close()
assert torch.cuda.get_per_process_memory_fraction() == before
print("BYTE_CEILING_PASS")
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BYTE_CEILING_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
