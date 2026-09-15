"""What the memory gate's 37 seconds are made of: the row's stamps and the windows walked once at the end.

The production boot of main `3acae017` spent 84.8 s in the prefill memory gate, and the ledger's four rows
could not say why the 128-token pass (15.41 s) cost eight times the 1,024-token one (2.07 s): a row's
`seconds` is forward, fleet vote and reclaim in one number. These tests pin the split and the attribution
that answer it -- not the values, which only a boot can give.
"""
import importlib.util
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch


class Clock:
    """perf_counter ticks a second per call, time() ticks ten -- so a stamp is readable in an assertion."""

    def __init__(self):
        self.perf, self.wall = itertools.count(0.0, 1.0), itertools.count(1000.0, 10.0)

    def perf_counter(self):
        return next(self.perf)

    def time(self):
        return next(self.wall)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class GateStampTests(unittest.TestCase):
    def engine(self, *, aux=False):
        import torch
        from engine.base import jit_writes
        from engine.profiles.glm53.adapter import Glm53Engine
        caches = MagicMock(device="cpu", snapshots=0)
        caches.pool.rows_in_use, caches.pool.num_blocks = 0, 8
        caches.slots.owner = [-1, -1]
        caches.slots.take.return_value = 1
        engine = Glm53Engine.__new__(Glm53Engine)
        engine.caches, engine.F, engine.prefill_chunk = caches, NS(block=64), 256
        engine.max_context, engine.memory = 512, MagicMock()
        engine.memory.checkpoint.return_value = {"allocator_reclaimed_bytes": 0}
        engine.net = NS(comm=NS(rank=2, all_reduce_max=lambda x: x), head=lambda x: x)
        engine._prefill_forward = MagicMock(
            return_value=(torch.ones(1, 8), torch.ones(1, 8) if aux else None))
        engine._forward = MagicMock(return_value=(torch.ones(1, 8), None))
        engine._observe_prefill = MagicMock()
        engine.jit_windows = jit_writes.Windows([])
        return engine

    def stamps(self, engine):
        return [call.kwargs["stamps"] for call in engine.memory.checkpoint.call_args_list
                if call.kwargs.get("stamps") is not None]

    def test_each_gate_row_carries_its_forward_its_vote_and_its_observation(self):
        from engine.profiles.glm53 import adapter
        engine = self.engine()
        with patch.object(adapter, "time", Clock()):
            engine._warmup_prefill_memory()
        # perf_counter: began, after the forward, after the vote, after the (absent) observation
        self.assertEqual(self.stamps(engine),
                         [{"forward_seconds": 1.0, "vote_seconds": 1.0, "observe_seconds": 1.0}] * 2)

    def test_the_vote_is_stamped_apart_from_the_forward(self):
        """A slow peer lands in `vote_seconds`, never in `forward_seconds`: that is the whole point of it."""
        from engine.profiles.glm53 import adapter
        engine = self.engine()
        clock = Clock()
        slow = [0.0, 1.0, 9.0, 10.0,                      # the collective took eight of these ten seconds
                20.0, 21.0, 22.0, 23.0]                   # and the far pass runs on after it
        clock.perf_counter = lambda: slow.pop(0)
        with patch.object(adapter, "time", clock):
            engine._warmup_prefill_memory()
        first = self.stamps(engine)[0]
        self.assertEqual((first["forward_seconds"], first["vote_seconds"]), (1.0, 8.0))

    def test_the_sync_before_the_vote_is_the_devices_and_only_where_there_is_one(self):
        import torch
        engine = self.engine()
        with patch.object(torch.cuda, "is_initialized", return_value=False),              patch.object(torch.cuda, "synchronize") as sync:
            engine._warmup_prefill_memory()
        sync.assert_not_called()                          # no context: there is nothing to wait for
        with patch.object(torch.cuda, "is_initialized", return_value=True),              patch.object(torch.cuda, "synchronize") as sync:
            engine._warmup_prefill_memory()
        self.assertEqual(sync.call_count, 2)              # one per pass, before its vote

    def test_an_observed_pass_stamps_its_observation_after_the_vote(self):
        from engine.profiles.glm53 import adapter
        engine = self.engine(aux=True)
        with patch.object(adapter, "time", Clock()):
            engine._warmup_prefill_memory()
        self.assertTrue(all(s["observe_seconds"] == 1.0 for s in self.stamps(engine)))
        self.assertEqual(engine._observe_prefill.call_count, 2)

    def test_every_pass_and_the_kernel_warmup_leave_a_window_for_the_walk(self):
        from engine.profiles.glm53 import adapter
        engine = self.engine()
        with patch.object(adapter, "time", Clock()):
            engine._warmup_prefill_memory()
            with patch("builtins.print"):
                engine._warmup_serving_kernels()
        names = [name for name, _, _ in engine.jit_windows.windows]
        self.assertEqual(names, ["prefill/256/0", "prefill/256/256", "warm kernels"])
        for _, start, end in engine.jit_windows.windows:
            self.assertLess(start, end)

    def test_a_partially_built_engine_still_runs_the_gate(self):
        """The windows are `capture_decode`'s; a test that calls the warmup alone must not need them."""
        engine = self.engine()
        del engine.jit_windows
        engine._warmup_prefill_memory()
        self.assertEqual(engine._prefill_forward.call_count, 2)


class CaptureWiringTests(unittest.TestCase):
    """`capture_decode` owns the windows: it opens them, and it walks them once, after the fleet's vote."""

    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / "engine/profiles/glm53/adapter.py").read_text(encoding="utf-8")
        self.capture = source[source.index("    def capture_decode(self, max_seqs: int)"):
                              source.index("    def _check_graph_pools(self)")]

    def test_the_windows_open_before_the_first_warmup(self):
        self.assertLess(self.capture.index("self.jit_windows = jit_writes.Windows()"),
                        self.capture.index("self._warmup_prefill_memory()"))

    def test_the_capture_window_closes_and_the_walk_runs_after_the_ready_vote(self):
        closed = self.capture.index('self.jit_windows.mark("capture", graphs_began, time.time())')
        vote = self.capture.index('self.memory.checkpoint("ready")')
        walk = self.capture.index("self.jit_windows.scan()")
        self.assertLess(closed, vote)
        self.assertLess(vote, walk)


class LedgerRowTests(unittest.TestCase):
    """`RuntimeMemory.checkpoint(stamps=...)`: the caller's boundaries, in the row, boot to boot."""

    def memory(self):
        from engine.base.runtime_memory import RuntimeMemory
        cuda = MagicMock()
        cuda.memory_reserved.return_value = 0
        cuda.memory_allocated.return_value = 0
        cuda.max_memory_allocated.return_value = 0
        cuda.max_memory_reserved.return_value = 0
        cuda.mem_get_info.return_value = (1 << 40, 1 << 40)
        cuda.get_per_process_memory_fraction.return_value = 1.0
        cuda.get_allocator_backend.return_value = "native"
        return RuntimeMemory(1 << 30, 1 << 30, 1 << 30, cuda=cuda, host_free=lambda: 1 << 40,
                             host_available=lambda: 1 << 40, floor=(1 << 20, 1 << 19))

    def test_stamps_join_the_row_and_the_rows_own_columns_win(self):
        memory = self.memory()
        row = memory.checkpoint("prefill/128/0/prepared", stamps={"forward_seconds": 13.0, "seconds": -1})
        self.assertEqual(row["forward_seconds"], 13.0)
        self.assertNotEqual(row["seconds"], -1)           # what the row measures, the row writes
        self.assertEqual(memory.phases[-1]["forward_seconds"], 13.0)

    def test_a_row_without_stamps_is_unchanged(self):
        row = self.memory().checkpoint("loaded")
        self.assertNotIn("forward_seconds", row)


if __name__ == "__main__":
    unittest.main()
