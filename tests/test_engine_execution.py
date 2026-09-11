"""Run ownership and failure isolation; the control-plane tests need no torch."""
import concurrent.futures
import gc
import importlib.util
import threading
import unittest
import weakref
from unittest.mock import patch

from engine.base.comm import LocalTP, RankLeft


class ExecutionOwnershipTests(unittest.TestCase):
    def test_invalid_world_timeout_and_rank_are_rejected(self):
        for world, timeout in ((0, 1), (-1, 1), (True, 1), (4, 0), (4, -1), (4, float("inf")), (4, float("nan"))):
            with self.subTest(world=world, timeout=timeout), self.assertRaises(ValueError):
                LocalTP(world, timeout)
        for rank in (-1, 4, True):
            with self.assertRaises(ValueError):
                LocalTP(4).rank(rank)

    def test_runs_exchange_objects_and_dispatch_to_their_owner_repeatedly(self):
        tp = LocalTP(4, timeout_s=1)
        owner = threading.current_thread()
        def work(comm, tick):
            where = comm.on_main(threading.current_thread)
            value = comm.broadcast_object((tick, comm.rank))
            comm.barrier()
            return where is owner, value
        for tick in range(4):
            self.assertEqual(tp.run(work, tick), [(True, (tick, 0))] * 4)

    def test_two_concurrent_groups_keep_distinct_dispatch_and_collectives(self):
        rendezvous = threading.Barrier(2, timeout=2)
        def group(tag):
            tp = LocalTP(4, timeout_s=2)
            owner = threading.current_thread()
            def work(comm):
                if comm.rank == 0:
                    rendezvous.wait()
                comm.barrier()
                value = comm.broadcast_object(tag if comm.rank == 0 else None)
                where = tp.on_main(threading.current_thread)
                return value, where is owner
            return tp.run(work)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(group, tag) for tag in ("first", "second")]
            self.assertEqual([f.result(timeout=5) for f in futures],
                             [[("first", True)] * 4, [("second", True)] * 4])

    def test_idle_and_stale_rank_handles_cannot_dispatch_or_join_a_later_run(self):
        tp = LocalTP(4, timeout_s=1)
        with self.assertRaisesRegex(RuntimeError, "active run"):
            tp.on_main(lambda: self.fail("idle dispatch executed"))
        unstarted = tp.rank(0)
        with self.assertRaisesRegex(RuntimeError, "inactive or completed"):
            unstarted.barrier()
        old = tp.run(lambda comm: comm)
        with self.assertRaisesRegex(RuntimeError, "inactive or completed"):
            old[0].on_main(lambda: None)
        def next_run(comm):
            with self.assertRaisesRegex(RuntimeError, "inactive or completed"):
                old[comm.rank].broadcast_object("stale")
            return comm.broadcast_object("new")
        self.assertEqual(tp.run(next_run), ["new"] * 4)

    def test_overlapping_and_foreign_calls_are_rejected_without_disturbing_owner(self):
        tp = LocalTP(4, timeout_s=2)
        entered, release = threading.Event(), threading.Event()
        def work(comm):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test release was not signalled")
            return comm.broadcast_object("owned")
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(tp.run, work)
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    tp.run(lambda comm: self.fail("overlapping rank executed"))
                with self.assertRaisesRegex(RuntimeError, "foreign thread"):
                    tp.on_main(lambda: self.fail("foreign kernel executed"))
                with self.assertRaisesRegex(RuntimeError, "foreign thread"):
                    tp.rank(0).barrier()
            finally:
                release.set()
            self.assertEqual(future.result(timeout=3), ["owned"] * 4)

    def test_nested_run_is_rejected_without_breaking_outer_collectives(self):
        tp = LocalTP(4, timeout_s=1)
        def work(comm):
            with self.assertRaisesRegex(RuntimeError, "nested"):
                tp.run(lambda rank: None)
            return comm.broadcast_object("outer")
        self.assertEqual(tp.run(work), ["outer"] * 4)

    def test_rank_failure_preserves_cause_and_next_run_has_fresh_collectives(self):
        tp = LocalTP(4, timeout_s=1)
        cause = ValueError("rank 2 deliberate failure")
        def work(comm):
            if comm.rank == 2:
                raise cause
            comm.barrier()
        with self.assertRaisesRegex(RuntimeError, "rank 2 failed") as raised:
            tp.run(work)
        self.assertIs(raised.exception.__cause__, cause)
        self.assertEqual(tp.run(lambda comm: comm.broadcast_object("recovered")), ["recovered"] * 4)

    def test_kernel_failure_cancels_remaining_queued_work_and_allows_fresh_run(self):
        tp = LocalTP(4, timeout_s=1)
        calls = []
        cause = ValueError("kernel deliberate failure")
        def kernel():
            calls.append(threading.current_thread())
            raise cause
        with self.assertRaises(RuntimeError) as raised:
            tp.run(lambda comm: comm.on_main(kernel))
        self.assertIs(raised.exception.__cause__, cause)
        self.assertEqual(calls, [threading.current_thread()])
        self.assertEqual(tp.run(lambda comm: comm.on_main(lambda: 13)), [13] * 4)

    def test_partial_thread_start_failure_joins_started_workers(self):
        tp = LocalTP(4, timeout_s=1)
        start = threading.Thread.start
        started = []
        def limited(thread):
            if len(started) == 2:
                raise RuntimeError("thread start deliberate failure")
            start(thread)
            started.append(thread)
        with patch.object(threading.Thread, "start", limited):
            with self.assertRaisesRegex(RuntimeError, "thread start deliberate failure"):
                tp.run(lambda comm: comm.barrier())
        self.assertTrue(all(not t.is_alive() for t in started))
        self.assertEqual(tp.run(lambda comm: comm.rank), [0, 1, 2, 3])

    def test_missing_collective_participant_times_out_and_next_run_is_clean(self):
        tp = LocalTP(4, timeout_s=.05)
        def work(comm):
            if comm.rank:
                comm.barrier()
        with self.assertRaises(RuntimeError) as raised:
            tp.run(work)
        self.assertIsInstance(raised.exception.__cause__, RankLeft)
        self.assertEqual(tp.run(lambda comm: comm.broadcast_object(29)), [29] * 4)

    def test_completed_runs_do_not_retain_exchange_or_result_objects(self):
        class Payload:
            pass
        tp = LocalTP(4, timeout_s=1)
        def work(comm):
            shared = comm.broadcast_object(Payload() if comm.rank == 0 else None)
            return comm, shared
        result = tp.run(work)
        old_rank, value = result[0]
        reference = weakref.ref(value)
        del result, value
        gc.collect()
        self.assertIsNone(reference())
        with self.assertRaises(RuntimeError):
            old_rank.barrier()


torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA PyTorch")
class KernelBindingTests(unittest.TestCase):
    def test_tensor_collectives_remain_exact_across_runs_and_a_rank_failure(self):
        tp = LocalTP(4, timeout_s=2)
        def work(comm, tick):
            local = torch.full((128,), tick + comm.rank, device="cuda")
            reduced = comm.all_reduce(local)
            gathered = comm.all_gather(torch.full((1,), comm.rank, device="cuda", dtype=torch.int32))
            comm.barrier()
            return reduced, gathered
        for tick in range(3):
            if tick == 1:
                def fail(comm):
                    if comm.rank == 1:
                        raise ValueError("rank failure between tensor runs")
                    comm.barrier()
                with self.assertRaises(RuntimeError):
                    tp.run(fail)
            for reduced, gathered in tp.run(work, tick):
                self.assertTrue(torch.equal(reduced, torch.full_like(reduced, 4 * tick + 6)))
                self.assertEqual(gathered.tolist(), [0, 1, 2, 3])

    def test_lane_tables_capture_their_executor_and_direct_table_stays_direct(self):
        from engine.profiles.glm53 import lanes
        from engine.kernels.kpool import fwht128_quant_fp8
        from dataclasses import fields
        from functools import wraps
        rows = torch.arange(32 * 128, device="cuda").reshape(32, 128).bfloat16()
        direct = lanes.served()
        expected = direct.indexer_quant(rows)  # JIT before concurrent callers
        owners, tables, traces = [LocalTP(4, 2) for _ in range(2)], [], [[], []]
        for i, owner in enumerate(owners):
            @wraps(fwht128_quant_fp8)
            def traced(x, index=i):
                traces[index].append(threading.current_thread())
                return fwht128_quant_fp8(x)
            with patch("engine.kernels.kpool.fwht128_quant_fp8", traced):
                tables.append(lanes.served(tp=owner))
        rendezvous = threading.Barrier(2, timeout=5)
        def group(i):
            thread = threading.current_thread()
            def work(comm):
                if comm.rank == 0:
                    rendezvous.wait()
                comm.barrier()
                return tables[i].indexer_quant(rows)
            output = owners[i].run(work)
            self.assertEqual(traces[i], [thread] * 4)
            for quant, scale in output:
                self.assertTrue(torch.equal(quant.view(torch.uint8), expected[0].view(torch.uint8)))
                self.assertTrue(torch.equal(scale, expected[1]))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(group, i) for i in range(2)]
            for future in futures:
                future.result(timeout=10)
        for table in tables:
            # Every lane is bound, including reference_for overrides: an idle
            # owner must reject before executing a kernel or inspecting args.
            for field in fields(table):
                if field.name != "name":
                    with self.subTest(lane=field.name), self.assertRaisesRegex(RuntimeError, "active run"):
                        getattr(table, field.name)()
        actual = direct.indexer_quant(rows)
        self.assertTrue(torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8)))
        self.assertTrue(torch.equal(actual[1], expected[1]))


if __name__ == "__main__":
    unittest.main()
