"""CPU checks for the CUDA-graph contracts (2026-09-12 review).

Four invariants that no numerical test can see: which comm may be captured, how
far the capacity ladder goes, that the drafter's graphs do not share the target
graphs' memory pool, and that the replay path stages its small arrays through
pinned memory instead of building a CPU tensor per step.
"""
import ast
import sys
import types
import unittest
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
        self.assertIn("except BaseException:\n                    g.reset()\n                    raise", source)


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


class WarmupDeclarationTests(unittest.TestCase):
    """The family-aware warmup is only correct if the shapes arrive family by family."""

    def setUp(self):
        self.text = DECODE_GRAPHS.read_text()

    def test_the_target_shapes_are_ordered_by_family_and_declare_their_warmup(self):
        self.assertIn("warmup=warmup_for", self.text)
        seqs = self.text.index("for n in range(1, max_seqs + 1)")
        caps = self.text.index("for capacity in self.capacities", seqs)
        # capacity-major order would give two passes to the first four shapes and one to every
        # family after them, which is not what "the first shape of a family" means
        self.assertLess(seqs, caps)

    def test_the_policy_gives_the_first_shape_of_each_family_two_passes(self):
        body = self.text[self.text.index("def warmup_for"):self.text.index("self.graphs = DecodeGraphs")]
        self.assertIn("key = shape[:2]", body)
        self.assertIn("return 2 if first else 1", body)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class WarmupPolicyTests(unittest.TestCase):
    """A declared warmup count per shape, and a graph that still replays what eager computes."""

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
        return DecodeGraphs(step, make_inputs, shapes, warmup=warmup), step, runs, weight

    def test_a_family_warms_twice_then_once_and_replays_what_eager_computes(self):
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


if __name__ == "__main__":
    unittest.main()
