"""The drafter can prepare a valid inverse without invoking GPU linear algebra."""
import tempfile
import unittest
from unittest.mock import patch

import torch

from engine.kernels.dense.packing import gptq_factor
from engine.kernels.dense.store import PackStore


class DrafterFactorDeviceTest(unittest.TestCase):
    def test_cuda_weight_uses_valid_cpu_factor_and_reuses_it(self):
        H = torch.tensor([[4., 1.], [1., 2.]])
        name = 'DFlash2Qwen3ForCausalLM/outputs-5-14-24-33-42/model.layers.0.self_attn.qkv_proj'
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 1)
            with patch('engine.kernels.dense.packing.gptq_factor', wraps=gptq_factor) as build:
                factor = store._factor(name, H, 'smooth', torch.device('cuda:0'))
                reused = store._factor(name, H, 'smooth', torch.device('cuda:0'))
            self.assertIs(factor, reused)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(build.call_args.kwargs['factor_device'], 'cpu')
            self.assertTrue(torch.isfinite(factor[1]).all())
            self.assertEqual(factor[1].device.type, 'cpu')
            self.assertEqual(store.stats['factor_cpu'], 1)
            self.assertEqual(store.stats['factor_reused'], 1)
            expected = gptq_factor(H, factor_device='cpu')
            torch.testing.assert_close(factor[1], expected[1], rtol=0, atol=0)

    def test_indefinite_data_still_fails_without_a_cached_factor(self):
        H = torch.tensor([[1., 100.], [100., 1.]])
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 1)
            with self.assertRaisesRegex(RuntimeError, 'Hessian not positive-definite'):
                store._factor('DFlash2Qwen3ForCausalLM/model.test', H, 'none', 'cuda:0')
            self.assertIsNone(store._factor_entry)
            self.assertEqual(store.stats['factor_built'], 0)

    def test_target_factor_keeps_its_requested_device(self):
        expected = (None, torch.eye(2), torch.zeros(2, dtype=torch.bool))
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 1)
            with patch('engine.kernels.dense.packing.gptq_factor', return_value=expected) as build:
                store._factor('L0.kda.in_proj', torch.eye(2), 'none', 'cuda:0')
            self.assertEqual(build.call_args.kwargs['factor_device'], 'cuda:0')


if __name__ == '__main__':
    unittest.main()
