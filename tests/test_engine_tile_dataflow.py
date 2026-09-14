"""Tile dependency, ownership and numerical contracts; no GPU performance claims."""
import unittest

import torch

from engine.modules.tile_dataflow import MLPPlan, PersistentDense, reference
from engine.modules.speculative_tree import Tree
from engine.profiles.glm53.lanes import swiglu_clamped
from engine.profiles.glm53.tree_decode import Verification
from tests.test_engine_execution_plans import model


class TileDataflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(349)

    def test_random_ready_schedules_match_full_mlp_and_cover_each_writer_once(self):
        for rows, hidden, intermediate in ((1, 63, 71), (4, 128, 96), (17, 95, 33)):
            plan = MLPPlan(rows, hidden, intermediate)
            x = torch.randn(rows, hidden).bfloat16()
            w = (torch.randn(2*intermediate, hidden)*.03).bfloat16()
            down = (torch.randn(hidden, intermediate)*.03).bfloat16()
            g, u = torch.nn.functional.linear(x, w).chunk(2, -1)
            expected = torch.nn.functional.linear(swiglu_clamped(g, u, 10.), down)
            previous = None
            for seed in range(4):
                actual, order = reference(plan, x, w, down, 10., seed=seed)
                torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-4)
                if previous is not None:
                    torch.testing.assert_close(actual, previous, rtol=0, atol=0)
                previous = actual
                self.assertEqual(sorted(order), list(range(len(plan.tasks))))
                positions = {task: at for at, task in enumerate(order)}
                for task, (_, _, deps) in enumerate(plan.tasks):
                    self.assertTrue(all(positions[d] < positions[task] for d in deps))

    def test_shapes_dtype_and_memory_budget_refuse(self):
        for args in ((0, 64, 64), (33, 64, 64), (True, 64, 64)):
            with self.assertRaises(ValueError):
                MLPPlan(*args)
        with self.assertRaisesRegex(ValueError, "scratch"):
            MLPPlan(32, 4096, 8192, max_scratch_bytes=1024)
        plan = MLPPlan(1, 32, 32)
        with self.assertRaisesRegex(ValueError, "BF16"):
            reference(plan, torch.zeros(1, 32), torch.zeros(64, 32), torch.zeros(32, 32), 10.)

    def test_persistent_binding_executes_all_dense_layers_in_target_tree(self):
        net, cache = model(("kda", "dsa", "kda"))
        slot = cache.slots.take(0)
        cache.pool.reserve(0, 8)
        tree = Tree((1, 2, 3), (-1, 0, 0))
        with Verification(net, cache, tree, seq=0, slot=slot, context=0) as run:
            expected = run.verify()
        binding = PersistentDense(backend="reference")
        with Verification(net, cache, tree, seq=0, slot=slot, context=0, persistent_mlp=binding) as run:
            actual = run.verify()
            torch.testing.assert_close(actual, expected, rtol=.008, atol=.008)
            self.assertEqual(binding.executed, set(net.layers))
            result = run.commit(budget=1)
            self.assertEqual(result["path"], (0,))

    def test_binding_refuses_prepared_or_retired_weights_instead_of_changing_precision(self):
        net, _ = model()
        name = "L0.mlp.gate_up"
        net.dense[name] = object()
        with self.assertRaisesRegex(ValueError, "packed weights"):
            PersistentDense(backend="reference").validate(net, 8)
        net.dense.clear(); net.p[name] = None
        with self.assertRaisesRegex(ValueError, "packed weights"):
            PersistentDense(backend="reference").validate(net, 8)


if __name__ == "__main__":
    unittest.main()
