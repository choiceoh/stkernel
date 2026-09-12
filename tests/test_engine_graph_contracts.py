"""CPU checks for the CUDA-graph contracts (2026-09-12 review).

Four invariants that no numerical test can see: which comm may be captured, how
far the capacity ladder goes, that the drafter's graphs do not share the target
graphs' memory pool, and that the replay path stages its small arrays through
pinned memory instead of building a CPU tensor per step.
"""
import ast
import contextlib
import sys
import types
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DECODE_GRAPHS = ROOT / "engine/profiles/glm53/decode_graphs.py"


def _pure(name, path=DECODE_GRAPHS):
    """One module-level function, compiled alone: the file itself imports triton."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = types.ModuleType("pure")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), module.__dict__)
    return getattr(module, name)


class CaptureSafetyTests(unittest.TestCase):
    def test_device_step_runs_the_same_embedding_prologue_as_eager_decode(self):
        from engine.base.comm import Comm
        from engine.profiles.glm53.decode_graphs import DeviceStep
        from engine.profiles.glm53.net import Glm53Net, Step
        ids = torch.arange(12)
        device_step = DeviceStep(ids, torch.tensor([0, 9]), 6)
        eager_step = Step.decode([(ids[:6], 0, 0, 1), (ids[6:], 9, 1, 2)])
        net = Glm53Net.__new__(Glm53Net)
        net.F = SimpleNamespace(hidden=8, hc=2)
        net.layers, net.prefill_transport, net.probe = (), None, None
        net.comm = Comm()
        net.embed = lambda tokens: tokens.float()[:, None].expand(-1, 8)
        actual = net.forward(device_step, None, finish=False)
        expected = net.forward(eager_step, None, finish=False)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[3], expected[3]))

    def test_localtp_is_marked_unsafe_and_the_fleet_comm_safe(self):
        from engine.base.comm import Comm, LocalTP
        self.assertTrue(Comm.graph_capture_safe)
        self.assertFalse(LocalTP.graph_capture_safe)

    def test_capture_refuses_a_host_barrier_comm_before_touching_the_device(self):
        from engine.base.comm import LocalTP
        from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs
        net = SimpleNamespace(comm=LocalTP(4), F=SimpleNamespace(spec_k=5))
        caches = SimpleNamespace(slots=SimpleNamespace(owner=[-1] * 5))
        with self.assertRaisesRegex(ValueError, "cannot be captured"):
            Glm53DecodeGraphs(net, caches, max_seqs=4, tokens=6)
        # and the earlier contract checks still come first
        with self.assertRaisesRegex(ValueError, "target or verify width"):
            Glm53DecodeGraphs(net, caches, max_seqs=4, tokens=3)
        busy = SimpleNamespace(slots=SimpleNamespace(owner=[-1, 0, -1, -1, -1]))
        with self.assertRaisesRegex(ValueError, "no live state slots"):
            Glm53DecodeGraphs(net, busy, max_seqs=4, tokens=6)


class CapacityLadderTests(unittest.TestCase):
    def setUp(self):
        self.ladder = _pure("capacity_ladder")

    def test_the_ladder_stops_at_the_served_ceiling(self):
        pool, trained = 1_370_880, 1_048_576          # GLM-5.3 at KV 8.73 GiB
        self.assertEqual(len(self.ladder(pool, trained, None)), 9)
        self.assertEqual(self.ladder(pool, trained, 131072),
                         [4096, 8192, 16384, 32768, 65536, 131072])
        self.assertEqual(self.ladder(pool, trained, 4096), [4096])

    def test_every_admitted_context_lands_in_a_bucket(self):
        for ceiling in (4096, 5000, 131072, 1_048_576):
            buckets = self.ladder(1_370_880, 1_048_576, ceiling)
            self.assertEqual(buckets[-1], ceiling)          # the door's longest horizon is covered
            self.assertEqual(sorted(buckets), buckets)
            for context in (1, 4095, 4096, ceiling - 1, ceiling):
                self.assertTrue(any(context <= b for b in buckets), (ceiling, context))

    def test_the_pool_and_the_trained_positions_also_bound_it(self):
        self.assertEqual(self.ladder(100_000, 1_048_576, None)[-1], 100_000)
        self.assertEqual(self.ladder(1_370_880, 8192, None), [4096, 8192])
        self.assertEqual(self.ladder(1_370_880, 8192, 1_048_576)[-1], 8192)
        for bad in ((0, 10, None), (10, 0, None), (10, 10, 0), (10, 10, -1)):
            with self.assertRaises(ValueError):
                self.ladder(*bad)


class GraphPoolSeparationTests(unittest.TestCase):
    """The decode loop reads the target step's aux hidden states across segments while
    the drafter's observation graph replays between them: one pool would corrupt them."""

    def _engine(self, target_pool, drafter_pools):
        from engine.profiles.glm53.adapter import Glm53Engine
        engine = Glm53Engine.__new__(Glm53Engine)
        engine.decode_graphs = SimpleNamespace(graphs=SimpleNamespace(pool=target_pool))
        engine.drafter = SimpleNamespace(
            decode_graphs=SimpleNamespace(proposals=SimpleNamespace(pool=drafter_pools[0]),
                                          observations=SimpleNamespace(pool=drafter_pools[1])))
        return engine

    def test_distinct_pools_pass_and_a_shared_pool_is_refused(self):
        self._engine("target", ("a", "b"))._check_graph_pools()
        for shared in (("target", "b"), ("a", "target")):
            with self.assertRaisesRegex(ValueError, "share the target graphs"):
                self._engine("target", shared)._check_graph_pools()

    def test_the_base_class_gives_every_instance_its_own_pool(self):
        source = (ROOT / "engine/base/graphs.py").read_text()
        self.assertIn("self.pool = pool = torch.cuda.graph_pool_handle()", source)
        # and publishes a graph only once its capture returned
        self.assertIn("except BaseException:\n                        g.reset()\n                        raise", source)


class ReplayStagingTests(unittest.TestCase):
    def test_the_replay_path_stages_through_pinned_memory(self):
        source = DECODE_GRAPHS.read_text()
        self.assertIn("pin_memory=True", source)
        self.assertNotIn("torch.tensor([s.ctx for s in step.segments]", source)
        self.assertNotIn("inputs[1].copy_(torch.tensor(temperatures))", source)
        for line in ("target.contexts.copy_(self.staging[0, :n], non_blocking=True)",
                     "seqs.copy_(self.staging[1, :n], non_blocking=True)",
                     "slots.copy_(self.staging[2, :n], non_blocking=True)",
                     "inputs[1].copy_(self.temps[:rows], non_blocking=True)"):
            self.assertIn(line, source)

    def test_pinned_staging_round_trips_through_its_numpy_view(self):
        """The replay path writes the staging block through numpy and copies the torch
        view: the two must address the same bytes, or a replay reads stale ids."""
        pinned = torch.cuda.is_available()
        staging = torch.empty(3, 4, dtype=torch.int64, pin_memory=pinned)
        if pinned:
            self.assertTrue(staging.is_pinned())
        staged = staging.numpy()
        segments = [SimpleNamespace(ctx=c, seq=s, slot=l) for c, s, l in ((7, 0, 1), (9, 2, 3))]
        for i, s in enumerate(segments):
            staged[0, i], staged[1, i], staged[2, i] = s.ctx, s.seq, s.slot
        self.assertEqual(staging[:, :2].tolist(), [[7, 9], [0, 2], [1, 3]])
        temps = torch.empty(6, dtype=torch.float32, pin_memory=pinned)
        temps.numpy()[:3] = [0.0, 0.7, 1.2]                       # the door hands floats, and ints
        self.assertEqual([round(v, 4) for v in temps[:3].tolist()], [0.0, 0.7, 1.2])
        temps.numpy()[:3] = [0, 1, 2]
        self.assertEqual(temps[:3].tolist(), [0.0, 1.0, 2.0])

    def test_the_decode_step_asks_for_its_shape_once(self):
        source = (ROOT / "engine/profiles/glm53/adapter.py").read_text()
        self.assertIn("shape = self.decode_graphs.shape(step)", source)
        self.assertIn("self.decode_graphs.run(step, shape)", source)
        self.assertNotIn("self.sampling_graphs.run(self.decode_graphs.shape(step)", source)
        self.assertIn("def run(self, step, shape=None):", DECODE_GRAPHS.read_text())


class CaptureOrderTests(unittest.TestCase):
    """The first capture sizes the pool every later one shares, so it is the largest."""

    def setUp(self):
        self.text = DECODE_GRAPHS.read_text()

    def test_the_target_shapes_run_largest_first(self):
        self.assertIn("for n in range(max_seqs, 0, -1)", self.text)
        seqs = self.text.index("for n in range(max_seqs, 0, -1)")
        self.assertIn("for capacity in reversed(self.capacities)",
                      self.text[seqs:seqs + 200])

    def test_the_bucket_lookup_still_reads_the_ladder_upwards(self):
        # shape() must return the FIRST capacity that covers the context, so the
        # ladder it walks stays ascending however the captures are ordered.
        body = self.text[self.text.index("    def shape(self, step):"):]
        body = body[:body.index("\n    def ", 10)]
        self.assertIn("for capacity in self.capacities:", body)
        self.assertIn("if end <= capacity:", body)

    def test_one_warmup_pass_is_the_declared_default(self):
        source = (ROOT / "engine/base/graphs.py").read_text()
        self.assertIn("warmup=1, generators=()", source)
        self.assertNotIn("warmup_for", self.text)


class FrozenCollectorTests(unittest.TestCase):
    """A Triton kernel collected mid-capture unloads its module and voids the graph."""

    def test_the_capture_runs_with_collection_off(self):
        import gc
        from engine.base.graphs import frozen_gc
        was = gc.isenabled()
        with frozen_gc():
            self.assertFalse(gc.isenabled())
            with frozen_gc():                       # nested guards must not hand it back
                self.assertFalse(gc.isenabled())
            self.assertFalse(gc.isenabled())
        self.assertEqual(gc.isenabled(), was)

    def test_the_collector_comes_back_after_a_failed_capture(self):
        import gc
        from engine.base.graphs import frozen_gc
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            with frozen_gc():
                raise RuntimeError("capture failed")
        self.assertTrue(gc.isenabled())

    def test_the_capture_loop_is_inside_the_guard(self):
        source = (ROOT / "engine/base/graphs.py").read_text()
        guard = source.index("with frozen_gc():")
        self.assertLess(guard, source.index("for shape in shapes:"))
        self.assertLess(guard, source.index("g.capture_begin("))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class WarmupPolicyTests(unittest.TestCase):
    """The mechanism, not the profile's policy: a caller may declare a count per shape.

    The GLM-5.3 profile declares one pass for every shape (the kernels are compiled
    before capture starts). A caller who knows better may still ask for more, and a
    caller who asks for none is refused.
    """

    def capture(self, warmup):
        from engine.base.graphs import DecodeGraphs
        weight = torch.randn(32, 32, device="cuda")
        runs = {}

        def make_inputs(n, t, capacity):
            return {"x": torch.zeros(n * t, 32, device="cuda"), "cap": float(capacity)}

        def step(inputs):
            runs[inputs["cap"]] = runs.get(inputs["cap"], 0) + 1
            h = inputs["x"]
            for _ in range(3):
                h = torch.tanh(h @ weight) + inputs["cap"] * 1e-6
            return h

        shapes = [(n, 2, cap) for n in (1, 2) for cap in (4, 8, 16)]
        kwargs = {} if warmup is None else {"warmup": warmup}
        return DecodeGraphs(step, make_inputs, shapes, **kwargs), step, runs, weight

    def test_the_default_is_one_pass_per_shape(self):
        graphs, step, runs, weight = self.capture(None)
        try:
            self.assertEqual(len(graphs.graphs), 6)
            for cap in (4.0, 8.0, 16.0):
                self.assertEqual(runs[cap], 2 * (1 + 1))   # two families, one warmup plus the capture
        finally:
            graphs.close()

    def test_a_declared_count_per_shape_is_honoured_and_replays_what_eager_computes(self):
        warmed = set()

        def policy(shape):
            first = shape[:2] not in warmed
            warmed.add(shape[:2])
            return 2 if first else 1

        graphs, step, runs, weight = self.capture(policy)
        try:
            self.assertEqual(len(graphs.graphs), 6)
            # cap 4 is each family's first: two warmups plus the capture, twice over; the rest one plus one
            self.assertEqual(runs[4.0], 2 * (2 + 1))
            self.assertEqual(runs[8.0], 2 * (1 + 1))
            self.assertEqual(runs[16.0], 2 * (1 + 1))
            for shape in graphs.graphs:
                x = torch.randn(shape[0] * shape[1], 32, device="cuda")
                replay = graphs.run(shape, lambda inputs: inputs["x"].copy_(x))
                eager = step({"x": x, "cap": float(shape[2])})
                self.assertTrue(torch.allclose(replay, eager, atol=1e-5), shape)
        finally:
            graphs.close()

    def test_a_policy_that_asks_for_no_warmup_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one warmup pass"):
            self.capture(lambda shape: 0)


class _Stream:
    def wait_stream(self, other): pass


class _Graph:
    def reset(self): pass
    def register_generator_state(self, generator): pass
    def capture_begin(self, pool, capture_error_mode="global"): pass
    def capture_end(self): pass


class _Cuda:
    """Enough of torch.cuda to run the capture loop's bookkeeping on a CPU host."""
    def __init__(self): self.flushes = 0
    def empty_cache(self): self.flushes += 1
    def graph_pool_handle(self): return object()
    def Stream(self): return _Stream()
    def current_stream(self): return _Stream()
    def stream(self, s): return contextlib.nullcontext()
    def CUDAGraph(self): return _Graph()
    def synchronize(self): pass


class _Ledger:
    def __init__(self): self.rows = []
    def checkpoint(self, phase): self.rows.append(phase)


class LedgerRowTests(unittest.TestCase):
    """A ledger row is a fleet-wide barrier, so the capture writes one per shape.

    Every row synchronizes the device and all-reduces a qualification flag across
    the TP ranks. A measured boot paid 21.9 ms per row; three rows over 51 shapes
    was 3.4 s, 12% of that boot's graph work.
    """

    def capture(self, **kw):
        from engine.base import graphs as module
        ledger = _Ledger()
        with unittest.mock.patch.object(module, "torch", SimpleNamespace(cuda=_Cuda())):
            module.DecodeGraphs(lambda inp: inp, lambda *shape: shape,
                                [(1, 6), (2, 6), (4, 6)], warmup=1,
                                memory=ledger, label="target", **kw)
        return ledger.rows

    def test_one_row_per_shape_by_default(self):
        self.assertEqual(self.capture(), ["target/(1, 6)", "target/(2, 6)", "target/(4, 6)"])

    def test_detail_restores_the_split_that_priced_the_warmup_policy(self):
        rows = self.capture(detail=True)
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[:3], ["target/(1, 6)/before", "target/(1, 6)/warmup",
                                    "target/(1, 6)/captured"])

    def test_a_capture_without_a_ledger_asks_for_nothing(self):
        from engine.base import graphs as module
        with unittest.mock.patch.object(module, "torch", SimpleNamespace(cuda=_Cuda())):
            module.DecodeGraphs(lambda inp: inp, lambda *shape: shape, [(1, 6)], warmup=1)


class AllocatorFlushTests(unittest.TestCase):
    """torch.cuda.graph() flushes the caching allocator before every capture.

    Per shape that returns exactly what the shape's own warmup just allocated, and
    on unified memory the next shape has to map those pages again. The capture
    drives capture_begin/capture_end itself so the flush happens once.
    """

    def test_the_allocator_is_flushed_once_for_the_instance_not_once_per_shape(self):
        from engine.base import graphs as module
        cuda = _Cuda()
        with unittest.mock.patch.object(module, "torch", SimpleNamespace(cuda=cuda)):
            module.DecodeGraphs(lambda inp: inp, lambda *shape: shape,
                                [(1, 6), (2, 6), (4, 6), (8, 6)], warmup=1)
        self.assertEqual(cuda.flushes, 1)

    def test_a_capture_that_raises_still_ends_its_capture(self):
        from engine.base import graphs as module
        ended = []

        class _Watched(_Graph):
            def capture_end(self): ended.append(self)

        cuda = _Cuda()
        cuda.CUDAGraph = _Watched
        calls = []

        def boom(inp):
            calls.append(inp)
            if len(calls) > 1:                 # the warmup pass runs; the recorded one fails
                raise RuntimeError("kernel refused")
            return inp

        with unittest.mock.patch.object(module, "torch", SimpleNamespace(cuda=cuda)):
            with self.assertRaisesRegex(RuntimeError, "kernel refused"):
                module.DecodeGraphs(boom, lambda *shape: shape, [(1, 6)], warmup=1)
        # the stream must not be left recording, or every later capture in the process fails
        self.assertEqual(len(ended), 1)


class BatchedPoolTests(unittest.TestCase):
    """One pass over [segments, pools] must produce what the per-segment loop produced."""

    KP, D, TAIL, LENGTH = 4, 8, 8, 6

    def graph_caches(self, n, capacity):
        from engine.profiles.glm53.decode_graphs import GraphCaches
        F = SimpleNamespace(kpool=self.KP, idx_dim=self.D, block=16)
        layout = SimpleNamespace(block_bytes=4096, pool_offsets={0: 512})
        real = SimpleNamespace(F=F, layout=layout,
                               block_table=torch.arange(1, 1 + 5 * 6, dtype=torch.int32).view(5, 6))
        caches = GraphCaches(real, torch.arange(n), torch.arange(n), capacity)
        caches.gather()
        return caches

    def test_pool_rows_is_pool_slots_run_once_per_segment(self):
        caches = self.graph_caches(n=3, capacity=256)
        pool_ids = torch.stack([torch.arange(2) + 3 * i for i in range(3)])
        rows = caches.pool_rows(0, pool_ids)
        for i in range(3):
            self.assertTrue(torch.equal(rows[i], caches.pool_slots(0, i, pool_ids[i])), i)

    def windows(self, contexts, tails, k, gate):
        """The window each segment pools: the tail ring's earlier tokens, then this step's."""
        kp, length, width = self.KP, self.LENGTH, tails.shape[1]
        pools = (kp - 1 + length) // kp
        out = []
        for i, ctx in enumerate(contexts.tolist()):
            relative = torch.arange(pools * kp) - ctx % kp
            earlier = (relative < 0)[:, None]
            kw = torch.where(earlier, tails[i][(ctx + relative) % width, 0],
                             k[i][relative.clamp(0, length - 1)])
            gw = torch.where(earlier, tails[i][(ctx + relative) % width, 1],
                             gate[i][relative.clamp(0, length - 1)])
            out.append((kw.view(pools, kp, self.D), gw.view(pools, kp, self.D),
                        (ctx % kp + length) // kp))
        return out

    def test_the_batched_pass_pools_the_same_windows_and_writes_the_same_rows(self):
        from engine.profiles.glm53 import decode_graphs as module
        n, capacity = 3, 256
        torch.manual_seed(72)
        contexts = torch.tensor([0, 7, 30])
        tails = torch.randn(n, self.TAIL, 2, self.D)
        k, gate = torch.randn(n, self.LENGTH, self.D), torch.randn(n, self.LENGTH, self.D)
        caches = self.graph_caches(n, capacity)
        pools = (self.KP - 1 + self.LENGTH) // self.KP
        seen, written = {}, []
        caches.pool_keys = lambda layer: torch.zeros(capacity, self.D, dtype=torch.uint8)
        caches.pool_scales = lambda layer: torch.zeros(capacity)
        caches.write_tail = lambda layer, i, ctx, keys, gates: written.append((i, int(ctx)))
        net = SimpleNamespace(
            F=SimpleNamespace(kpool=self.KP, idx_dim=self.D), p={"L0.idx.ape": None},
            lanes=SimpleNamespace(kpool_compress=lambda kw, gw, ape: (
                seen.update(kw=kw.clone(), gw=gw.clone()),
                (torch.zeros(kw.shape[0], self.D, dtype=torch.uint8), torch.zeros(kw.shape[0], 1)))[1]))
        calls = []
        with unittest.mock.patch.object(module, "scatter_rows",
                                        lambda src, dst, idx, valid: calls.append((idx.clone(), int(valid)))):
            got = module.complete_pools(net, 0, contexts, self.LENGTH, tails, k, gate, caches)
        self.assertEqual(got, caches.candidate_capacity)
        want = self.windows(contexts, tails, k, gate)
        self.assertTrue(torch.equal(seen["kw"], torch.cat([w[0] for w in want])), "pooled keys")
        self.assertTrue(torch.equal(seen["gw"], torch.cat([w[1] for w in want])), "pooled scores")
        # two scatters per segment, each with that segment's rows and its own count
        self.assertEqual([c[1] for c in calls], [w[2] for w in want for _ in range(2)])
        for i, ctx in enumerate(contexts.tolist()):
            ids = (ctx // self.KP + torch.arange(pools)).clamp_max(caches.candidate_capacity - 1)
            self.assertTrue(torch.equal(calls[2 * i][0], caches.pool_slots(0, i, ids).long()), i)
        self.assertEqual(written, [(0, 0), (1, 7), (2, 30)])


class KeptConstantTests(unittest.TestCase):
    """The decode path's index constants are built once, and never inside a capture."""

    def setUp(self):
        from engine.base import constants
        self.constants = constants
        constants.forget()

    def tearDown(self):
        self.constants.forget()

    def test_the_same_length_comes_back_as_the_same_tensor(self):
        kept = self.constants.iota(6, "cpu")
        self.assertIs(self.constants.iota(6, "cpu"), kept)
        self.assertEqual(kept.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertIsNot(self.constants.fresh(6, "cpu"), self.constants.fresh(6, "cpu"))

    def test_a_new_constant_is_refused_while_a_graph_is_recording(self):
        # Memory taken during capture belongs to that graph's pool, which the next
        # capture takes back, so a constant built there would alias it later.
        with unittest.mock.patch.object(torch.cuda, "is_current_stream_capturing",
                                        return_value=True):
            with self.assertRaisesRegex(RuntimeError, "while a graph was recording"):
                self.constants.iota(12, "cuda")

    def test_one_already_built_is_handed_out_during_a_capture(self):
        kept = self.constants.iota(12, "cpu")               # as the warmup pass would
        with unittest.mock.patch.object(torch.cuda, "is_current_stream_capturing",
                                        return_value=True):
            self.assertIs(self.constants.iota(12, "cpu"), kept)

    def test_the_captured_decode_path_builds_no_index_of_its_own(self):
        # complete_pools runs only under capture, so both of its indices are bounded.
        source = DECODE_GRAPHS.read_text()
        body = source[source.index("def complete_pools"):source.index("@dataclass")]
        self.assertNotIn("torch.arange", body)
        self.assertEqual(body.count("iota("), 3)      # the window, the pool ids, the segment rows
        net = (ROOT / "engine/profiles/glm53/net.py").read_text()
        for loop in ("_indexer", "_dsa"):
            chunk = net[net.index(f"def {loop}("):]
            chunk = chunk[:chunk.index("\n    def ", 10)]
            self.assertIn("index = iota if", chunk, loop)
            self.assertNotIn("torch.arange(s.length", chunk, loop)


class DeviceStepTests(unittest.TestCase):
    """The step's segment tuple is asked for once per layer, so it is built once."""

    def step(self, n=4, tokens=6):
        from engine.profiles.glm53.decode_graphs import DeviceStep
        contexts = torch.zeros(n, dtype=torch.int64)
        return DeviceStep(torch.zeros(n * tokens, dtype=torch.int64), contexts, tokens), contexts

    def test_the_same_tuple_comes_back_every_time(self):
        step, _ = self.step()
        self.assertIs(step.segments, step.segments)
        self.assertEqual([(s.seq, s.slot, s.start, s.length) for s in step.segments],
                         [(0, 0, 0, 6), (1, 1, 6, 6), (2, 2, 12, 6), (3, 3, 18, 6)])

    def test_a_cached_segment_still_reads_what_replay_wrote(self):
        # The contexts buffer is overwritten in place, never rebound: that is what lets
        # the captured graph record its address, and what keeps these views correct.
        step, contexts = self.step()
        held = step.segments
        contexts.copy_(torch.tensor([10, 20, 30, 40]))
        self.assertEqual([int(s.ctx) for s in held], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
