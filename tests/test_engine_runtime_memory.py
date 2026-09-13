"""Workspace peaks, physical reserves and allocator limits are separate contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


class OomFloorTests(unittest.TestCase):
    """The box's kill line, read from the box.

    A reserve the engine picks for itself is a number about nothing. earlyoom on this fleet
    SIGTERMs at 6 GiB and SIGKILLs at 4.5, `--prefer python3`, the engine first on purpose --
    and the engine kept 4 GiB, below both, so its own "preparation consumed the OS memory
    reserve" check could never fire first (2026-09-12).
    """

    def write(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "earlyoom"
        path.write_text(text)
        return path

    def test_compressed_host_budget_counts_at_admission_but_does_not_expand_cuda(self):
        kwargs = dict(arena_bytes=400, workspace_bytes=200, os_reserve_bytes=100,
                      cuda=Cuda(), host_free=lambda: 750, host_available=lambda: 750, floor=(60, 45))
        with self.assertRaisesRegex(MemoryError, 'immediately free'):
            RuntimeMemory(**kwargs, host_budget_bytes=51)
        memory = RuntimeMemory(**kwargs, host_budget_bytes=50)
        self.assertEqual(memory.allocator_limit_bytes, 610)
        self.assertEqual(memory.report()['host_budget_bytes'], 50)
        memory.close()

    def read(self, path, default=(1, 2)):
        from engine.base.runtime_memory import oom_floor
        return oom_floor(default, sources=(path,))

    def test_it_reads_both_stages_in_kib_from_the_running_configuration(self):
        path = self.write('# comment\nEARLYOOM_ARGS="-M 6291456,4718592 -s 100,100 -r 60 -n"\n')
        self.assertEqual(self.read(path), (6 << 30, 9 << 29))          # 6.0 GiB and 4.5 GiB

    def test_one_stage_means_the_kill_line_is_half_like_earlyooms_own_default(self):
        self.assertEqual(self.read(self.write('EARLYOOM_ARGS="-M 6291456"\n')), (6 << 30, 3 << 30))

    def test_a_configuration_it_cannot_read_falls_back_declared_not_optimistic(self):
        from engine.base.runtime_memory import DECLARED_OOM_FLOOR, oom_floor
        self.assertEqual(self.read(self.write('EARLYOOM_ARGS="-m 2 -s 2"\n')), (1, 2))
        self.assertEqual(self.read(self.write("nothing here\n")), (1, 2))
        self.assertEqual(oom_floor(sources=(Path("/nonexistent/earlyoom"),)), DECLARED_OOM_FLOOR)

    def test_the_profile_keeps_more_than_the_box_kills_at(self):
        """The regression itself: 4.0 GiB reserved against a 6.0 GiB SIGTERM."""
        from engine.base.runtime_memory import oom_floor
        from engine.profiles.glm53.budget import os_reserve_gib
        sigterm, _ = oom_floor()
        self.assertGreater(os_reserve_gib(), sigterm / (1 << 30))


class RuntimeMemoryTests(unittest.TestCase):
    def budget(self, cuda=None, host_free=lambda: 800):
        return RuntimeMemory(400, 200, 100, cuda=cuda or Cuda(), host_free=host_free)

    def test_a_reading_that_fails_still_reaches_the_vote_and_says_why(self):
        """A rank that raised between `synchronize` and the MAX used to leave its peers at the
        collective until NCCL's deadline; the reading's failure is now the flag the vote carries."""
        cuda = Cuda()
        memory = self.budget(cuda)
        calls = []
        cuda.synchronize = lambda: (calls.append(1), (_ for _ in ()).throw(RuntimeError("CUDA error: an illegal memory access")))[1]
        with self.assertRaises(MemoryError) as caught:
            memory.checkpoint("target/4/6")
        self.assertIn("RuntimeError: CUDA error: an illegal memory access", str(caught.exception))
        self.assertEqual(memory.phases[-1]["phase"], "target/4/6")
        self.assertIn("illegal memory access", memory.phases[-1]["read_error"])
        self.assertFalse(memory.phases[-1]["passed"])
        self.assertFalse(memory.ready)
        with self.assertRaises(MemoryError) as caught:
            self.budget(Cuda()).checkpoint("capture_decode/failed", failed="ValueError: no dense rows")
        self.assertEqual(str(caught.exception), "capture_decode/failed: ValueError: no dense rows")

    def test_byte_ceiling_includes_existing_allocator_and_restores_previous_limit(self):
        cuda = Cuda()
        memory = self.budget(cuda)
        self.assertEqual(memory.allocator_limit_bytes, 610)
        self.assertEqual(cuda.fraction, .61)
        memory.close(); memory.close()
        self.assertEqual(cuda.fraction, .9)

    def test_every_row_records_how_close_the_box_came_to_killing_it(self):
        """A ledger that says `ready` and nothing else cannot tell a boot that had room from
        one that was about to be SIGKILLed: fourteen recorded boots reached 1.48-15.74 GiB of
        free memory and every one of them read healthy (2026-09-12)."""
        cuda = Cuda()
        memory = RuntimeMemory(400, 200, 100, cuda=cuda, host_free=lambda: 800,
                               host_available=lambda: 700, floor=(600, 450))
        row = memory.checkpoint("target/4/6")
        self.assertEqual((row["available_bytes"], row["oom_margin_bytes"]), (700, 100))
        self.assertFalse(row["oom_close"])

        memory.host_available = lambda: 520                     # under SIGTERM, over SIGKILL
        close = memory.checkpoint("target/2/6")
        self.assertTrue(close["oom_close"] and close["passed"], "a warning, not a refusal")
        self.assertEqual(close["oom_margin_bytes"], -80)

        report = memory.measured()
        self.assertEqual(report["oom_margin_bytes"], -80)
        self.assertEqual(report["oom_margin_phase"], "target/2/6")
        self.assertEqual(report["oom_close_phases"], ["target/2/6"])
        self.assertEqual((report["oom_sigterm_bytes"], report["oom_sigkill_bytes"]), (600, 450))

    def test_below_the_boxs_kill_line_the_boot_stops_instead_of_being_killed(self):
        """Carrying on past it only means dying with nothing in the ledger to say why."""
        cuda = Cuda()
        memory = RuntimeMemory(400, 200, 100, cuda=cuda, host_free=lambda: 800,
                               host_available=lambda: 400, floor=(600, 450))
        with self.assertRaisesRegex(MemoryError, "SIGKILL line"):
            memory.checkpoint("target/1/6")
        self.assertFalse(memory.phases[-1]["passed"])
        self.assertFalse(memory.ready)

    def test_an_unreadable_meminfo_does_not_take_the_boot_down(self):
        def refuse():
            raise RuntimeError("MemAvailable is unavailable")
        memory = RuntimeMemory(400, 200, 100, cuda=Cuda(), host_free=lambda: 800,
                               host_available=refuse, floor=(600, 450))
        row = memory.checkpoint("target/4/6")
        self.assertEqual(row["available_bytes"], row["immediately_free_bytes"])

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

    def test_the_floor_is_what_the_box_is_down_before_the_arena_and_outside_the_allocator(self):
        # 1000 total, 900 free, 10 of it the allocator's: 90 is the CUDA context plus NCCL.
        # The budget's floor line quoted vLLM's 40th boot until this number existed.
        cuda = Cuda()
        memory = self.budget(cuda)
        self.assertEqual(memory.floor_bytes, 90)
        self.assertEqual(memory.report()["floor_bytes"], 90)
        memory.close()

    def test_measured_splits_the_prefill_activation_peak_from_what_the_graphs_keep(self):
        # ST's `profile_run` equivalent: the largest legal prefill runs before any graph is
        # captured, so the peak standing at the last prefill row IS the activation peak, and
        # everything capture adds to the reservation afterwards is the pools.
        cuda = Cuda()
        memory = self.budget(cuda)
        cuda.allocated = cuda.reserved = cuda.peak_reserved = 410          # the arena, nothing else
        memory.checkpoint("prefill/6912/0/before")
        cuda.peak_reserved = 470                                           # activations, given back after
        memory.checkpoint("prefill/6912/0/prepared")
        cuda.reserved = 430                                                # graph pools, kept
        memory.checkpoint("target/(4, 6, 4096)/captured")

        m = memory.measured()
        self.assertEqual(m["prefill_peak_bytes"], 60)
        self.assertEqual(m["prefill_shapes"], ["prefill/6912/0/prepared"])
        self.assertEqual(m["graph_bytes"], 20)
        self.assertEqual(m["peak_workspace_bytes"], 60)
        self.assertEqual(m["retained_workspace_bytes"], 20)
        self.assertEqual(m["at_phase"], "target/(4, 6, 4096)/captured")
        self.assertEqual(memory.report()["measured"], m)
        memory.close()

    def test_measured_says_nothing_rather_than_guessing_before_a_phase_exists(self):
        memory = self.budget()
        self.assertEqual(memory.measured(), {})
        memory.close()

    def test_non_allocator_or_peer_usage_must_leave_the_os_reserve(self):
        free = [800]
        memory = self.budget(host_free=lambda: free[0])
        free[0] = 99
        with self.assertRaisesRegex(MemoryError, "OS memory reserve"):
            memory.checkpoint("external CUDA allocation")
        memory.close()

    def test_reclaim_covers_remaining_workspace_without_relaxing_either_limit(self):
        free = [800]
        calls = []
        cuda = Cuda()
        def reclaim(need, headroom):
            calls.append((need, headroom))
            free[0] = need
            return need
        memory = RuntimeMemory(400, 200, 100, cuda=cuda, host_free=lambda: free[0], reclaim=reclaim)
        cuda.reserved = cuda.peak_reserved = 510
        free[0] = 99
        row = memory.checkpoint('cache refilled during preparation')
        self.assertEqual(calls, [(200, 300)])
        self.assertEqual(row['reclaimed_bytes'], 200)
        self.assertTrue(row['passed'])
        cuda.peak_reserved = 611
        with self.assertRaisesRegex(MemoryError, 'byte ceiling'):
            memory.checkpoint('workspace overflow')
        memory.close()

    def test_reclaim_failure_still_refuses_readiness(self):
        free = [800]
        memory = RuntimeMemory(400, 200, 100, cuda=Cuda(), host_free=lambda: free[0],
                               reclaim=lambda *_: 0)
        free[0] = 99
        with self.assertRaisesRegex(MemoryError, 'OS memory reserve'):
            memory.checkpoint('unreclaimable allocation')
        memory.close()

    def test_reclaim_faults_the_free_extent_and_preserves_available_headroom(self):
        from engine.base.runtime_memory import reclaim_preparation_pages
        with patch('engine.base.arena._meminfo', return_value={'MemFree': 50, 'MemAvailable': 700}), \
             patch('engine.base.arena.touch_pages', return_value=200) as touch:
            self.assertEqual(reclaim_preparation_pages(200, 300), 200)
            touch.assert_called_once_with(200)
        with patch('engine.base.arena._meminfo', return_value={'MemFree': 50, 'MemAvailable': 499}), \
             patch('engine.base.arena.touch_pages') as touch:
            self.assertEqual(reclaim_preparation_pages(200, 300), 0)
            touch.assert_not_called()

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

    def test_the_report_splits_the_memory_by_phase_the_way_it_splits_the_seconds(self):
        """Every row carried the bytes all along; nothing summed them, so "is this capture worth its footprint"
        meant reading every row by hand. `weigh()` is `spend()` for the reservation, which is what the box loses."""
        memory = self.budget()
        memory.clock = lambda: 0.0
        base = memory.baseline_reserved
        for phase, reserved in [("loaded", base + 100), ("target/(1, 6, 4096)/captured", base + 340),
                                ("target/(4, 6, 4096)/captured", base + 500), ("sampling/greedy/captured", base + 520),
                                ("freed", base + 440)]:
            memory.cuda.reserved = reserved
            memory.cuda.peak_reserved = max(memory.cuda.peak_reserved, reserved)
            memory.checkpoint(phase)
        self.assertEqual(memory.weigh(), [("target", 400), ("loaded", 100), ("sampling", 20), ("freed", -80)])
        self.assertEqual(memory.report()["weigh"]["target"], 400)
        self.assertEqual(sum(dict(memory.weigh()).values()), memory.cuda.reserved - base, "the parts are the whole")
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
