"""Tile order, two-slot lifetime, rank placement and KDA state invariance."""
from types import SimpleNamespace as NS
import unittest
import torch

from engine.base.comm import LocalTP
from engine.kernels.prefill_collectives.tiles import tile_pipeline
from engine.profiles.glm53.execution import ExecutionPlan, prefill_layer_major
from engine.profiles.glm53.net import Step
from tests.test_engine_execution_plans import model


class OraclePrefill:
    def __init__(self, comm, enabled):
        self.comm, self.project_tiles = comm, enabled
        self.calls = 0

    def all_gather(self, x):
        return self.comm.all_gather(x, dim=0)

    def reduce_scatter(self, x):
        return self.comm.all_reduce(x).chunk(4, dim=0)[self.comm.rank].contiguous()

    def gather_project(self, x, project):
        self.calls += 1
        result = None
        def launch(start, end, slot):
            return self.all_gather(x[start:end].contiguous())
        def consume(start, end, value):
            nonlocal result
            y = project(value)
            if result is None:
                result = torch.empty(4, x.shape[0], y.shape[-1], dtype=y.dtype)
            result[:, start:end].copy_(y.view(4, end-start, -1))
        tile_pipeline(x.shape[0], 16, launch, consume)
        return result.flatten(0, 1)


class PrefillTileTests(unittest.TestCase):
    def test_packet_projection_requires_the_unobserved_fp8_lane(self):
        from engine.kernels.dense import DenseLinear
        layer = DenseLinear.__new__(DenseLinear)
        layer.cols, layer.observer, layer.fp8 = 4096, None, NS(observer=None)
        self.assertTrue(callable(layer.packet_projector()))
        layer.observer = lambda x: None
        self.assertIsNone(layer.packet_projector())
        layer.observer = None
        layer.fp8.observer = lambda x: None
        self.assertIsNone(layer.packet_projector())
        layer.fp8 = None
        self.assertIsNone(layer.packet_projector())
        layer.fp8, layer.cols = NS(observer=None), 128
        self.assertIsNone(layer.packet_projector())

    def test_slots_are_released_before_reuse_and_tail_order_is_exact(self):
        for rows in (1, 255, 256, 257, 576, 2304):
            slots, log = [None, None], []
            actual = []
            def launch(a, b, slot):
                self.assertIsNone(slots[slot])
                slots[slot] = (a, b)
                log.append(("launch", a))
                return slot
            def consume(a, b, slot):
                self.assertEqual(slots[slot], (a, b))
                slots[slot] = None
                actual.extend(range(a, b))
                log.append(("consume", a))
            tile_pipeline(rows, 256, launch, consume)
            self.assertEqual(actual, list(range(rows)))
            self.assertEqual(slots, [None, None])
            if rows > 256:
                self.assertEqual([s[0] for s in log[:3]], ["launch", "launch", "consume"])

    def test_real_tp4_prefill_matches_output_and_all_cache_state(self):
        torch.set_num_threads(1)
        def rank(comm):
            net, caches = model(("kda", "dsa", "kda"), comm=comm)
            transport = OraclePrefill(comm, False)
            net.prefill_transport = transport
            slot = caches.slots.take(2)
            caches.pool.reserve(2, 192)
            step = Step.prefill(torch.arange(192) % net.vp, 0, 2, slot)
            caches.prepare(step)
            state, paged = caches.state.clone(), caches.paged.clone()
            expected = net.forward(step, caches, aux_layers=[0, 2])
            after, paged_after = caches.state.clone(), caches.paged.clone()
            for layer_major in (False, True):
                caches.state.copy_(state); caches.paged.copy_(paged)
                transport.project_tiles = True
                if layer_major:
                    actual = prefill_layer_major(net, step, caches, NS(tile_rows=192, prefill_tiles=1), [0, 2])
                else:
                    actual = net.forward(step, caches, aux_layers=[0, 2])
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(caches.state, after, rtol=0, atol=0)
                torch.testing.assert_close(caches.paged, paged_after, rtol=0, atol=0)
            self.assertEqual(transport.calls, 4)
            return actual[0]
        results = LocalTP(4, timeout_s=30).run(rank)
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=0)

    def test_explicit_plan_and_invalid_tile_sizes(self):
        self.assertFalse(ExecutionPlan().prefill_project_tiles)
        self.assertTrue(ExecutionPlan(prefill_project_tiles=True).active)
        with self.assertRaises(ValueError):
            ExecutionPlan(prefill_project_tiles=1)
        for rows, tile in ((0, 1), (1, 0), (True, 1)):
            with self.assertRaises(ValueError):
                tile_pipeline(rows, tile, None, None)


if __name__ == "__main__":
    unittest.main()
