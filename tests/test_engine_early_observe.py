"""Tentative DFlash context must not calibrate or commit rejected rows."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from engine.profiles.glm53.drafter import Drafter


class Projection:
    cols = 16

    def __init__(self):
        self.weight = torch.randn(8, self.cols).bfloat16() * .1
        self.observer = Mock()

    def __call__(self, x, rows_ok=None, *, observe=True):
        if observe:
            self.observer(x, rows_ok)
        return torch.nn.functional.linear(x, self.weight)


class EarlyObserveTests(unittest.TestCase):
    def test_projection_is_identical_and_only_final_retained_rows_calibrate(self):
        torch.manual_seed(41)
        d = Drafter.__new__(Drafter)
        d.F = NS(layers=2, head_dim=4, rms_eps=1e-6, rope_theta=10000.)
        d.local_kv_heads = 2
        projection = Projection()
        d.dense = {"fc.weight": projection}
        d.p = {"hidden_norm.weight": torch.ones(8).bfloat16()}
        d.context_kv = torch.randn(32, 8).bfloat16() * .1
        d.context_norm = torch.ones(2, 4).bfloat16()
        positions = torch.arange(7)[None, :] + torch.tensor([0, 765, 8192])[:, None]
        aux = torch.randn(21, 16).bfloat16()
        retained = torch.tensor([0, 1, 7])
        expected = d._project_context(positions, aux, retained)
        projection.observer.reset_mock()
        prepared = d._project_context(positions, aux, torch.full((3,), 7), observe=False)
        torch.testing.assert_close(prepared, expected, rtol=0, atol=0)
        projection.observer.assert_not_called()
        field, slots = object(), torch.tensor([2, 1, 3])
        with patch("engine.kernels.draft_observe.write_context") as write:
            d.observe_prepared(field, slots, positions, prepared, retained, aux)
        projection.observer.assert_called_once()
        observed, mask = projection.observer.call_args.args
        self.assertEqual(observed.data_ptr(), aux.data_ptr())
        torch.testing.assert_close(observed, aux, rtol=0, atol=0)
        self.assertEqual(mask.tolist(), [False]*7 + [True]+[False]*6 + [True]*7)
        write.assert_called_once()
        self.assertIs(write.call_args.args[4], d.context_norm)
        self.assertTrue(torch.equal(write.call_args.args[5], retained))


if __name__ == "__main__":
    unittest.main()
