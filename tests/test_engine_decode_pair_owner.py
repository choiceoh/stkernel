"""Serving projection copies must follow smoothing and live in the arena."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class PairOwnerTests(unittest.TestCase):
    def test_arena_copy_uses_smoothed_weights_and_dispatch_stays_decode_only(self):
        from engine.base.arena import Arena
        from engine.profiles.glm53.net import Glm53Net
        from tests.test_engine_glm53 import tiny_facts
        f = replace(tiny_facts(), hidden=4096, idx_dim=128)
        layer = next(L for L in range(f.layers) if f.is_dsa(L))
        net = Glm53Net(f, SimpleNamespace(rank=0, world_size=4), SimpleNamespace(rmsnorm=None, swiglu=None), [layer])
        with self.assertRaises(RuntimeError):
            net.prepare_decode_projections(None)
        weights = [torch.randn(128, 4096).bfloat16() for _ in range(2)]
        net.p = {f'L{layer}.idx.wk': weights[0], f'L{layer}.idx.gate': weights[1]}
        net.dense['prepared'] = True
        self.assertEqual(net.decode_projection_nbytes(), 2 << 20)
        arena = Arena(net.decode_projection_nbytes(), device='cpu', expandable=False)
        def pair(wk, gate, *, storage):
            storage[:128].copy_(wk); storage[128:].copy_(gate)
            return SimpleNamespace(weight=storage)
        # Simulate the BF16 readers changed by smooth_inputs, before preparation.
        weights[0].mul_(2); weights[1].mul_(.5)
        module = SimpleNamespace(IndexerPair=pair, KdaPair=None, DECODE_ROWS=(1, 6, 7, 14, 21, 28))
        with patch.dict('sys.modules', {'engine.kernels.decode_projection': module}):
            net.prepare_decode_projections(arena)
            owner = net._decode_pair(layer, SimpleNamespace(captured=True), 28)
            torch.testing.assert_close(owner.weight[:128], weights[0], rtol=0, atol=0)
            torch.testing.assert_close(owner.weight[128:], weights[1], rtol=0, atol=0)
            self.assertIsNone(net._decode_pair(layer, SimpleNamespace(captured=False), 28))
            self.assertIsNone(net._decode_pair(layer, SimpleNamespace(captured=True), 8))
            self.assertNotEqual(owner.weight.data_ptr(), weights[0].data_ptr())
            weights[0].zero_()
            self.assertTrue(owner.weight[:128].ne(0).any().item())
            with self.assertRaises(RuntimeError):
                net.prepare_decode_projections(arena)


if __name__ == '__main__':
    unittest.main()
