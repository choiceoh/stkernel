"""The native observation call that boot captures, through the real KV writer guard."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from engine.kernels.dense import DenseLinear
from engine.kernels.dense.calibration import Calibration
from engine.profiles.glm53.drafter import Drafter, DrafterFacts


class DraftCommitTests(unittest.TestCase):
    def drafter(self, calibration):
        facts = DrafterFacts(layers=2, hidden=4, heads=1, kv_heads=1, head_dim=4,
            inter=4, rms_eps=1e-6, rope_theta=10000., window=16, block=8, mask_id=7,
            conv_taps=2, conv_group=4, sel_rank=4, sel_top_k=2, target_layers=(1,), k=7)
        d = Drafter(facts, SimpleNamespace(), 8)
        d.decode_calibration = calibration
        d.p = {'hidden_norm.weight': torch.ones(4).bfloat16(), **{
            f'layers.{layer}.self_attn.k_norm.weight': torch.ones(4).bfloat16()
            for layer in range(facts.layers)}}
        d.context_kv = torch.cat([torch.eye(4)] * (2 * facts.layers)).bfloat16()
        layer = DenseLinear.__new__(DenseLinear)
        layer.rows = layer.cols = 4
        layer.decode_precision, layer.decode_fp8 = 'fp8', None
        layer.fp8, layer.executed = lambda x: x, 0
        d.dense = {'fc.weight': layer}
        stats = Calibration('cpu', max_decode_rows=8)
        stats.attach('fc', layer, [('fc', 0, 4)], small_rows=True, decode_only=True)
        stats.arm()
        return d, stats

    @staticmethod
    def write_cpu(k, v, field, slot, positions, valid, has_valid, *layout):
        # Only the Triton launch is replaced. write_draft_kv still validates
        # the actual native-ring arguments before this callback is reached.
        layer = layout[-2] // field.stride(1)
        count = int(valid) if has_valid else len(positions)
        indices = positions[:count] % field.shape[3]
        field[int(slot[0]), layer, 0, indices, :k.shape[1]] = k[:count]
        field[int(slot[0]), layer, 1, indices, :v.shape[1]] = v[:count]

    def test_committed_native_ring_and_calibration_cover_every_k7_capture_length(self):
        for calibration in (False, True):
            for tokens in range(1, 9):
                with self.subTest(calibration=calibration, tokens=tokens):
                    d, stats = self.drafter(calibration)
                    field = torch.full((3, 2, 2, 16, 1, 4), -9., dtype=torch.bfloat16)
                    expected = field.clone()
                    slot = torch.tensor([1])
                    positions = torch.arange(14, 14 + tokens)
                    aux = torch.arange(tokens * 4).reshape(tokens, 4).bfloat16()
                    reference, _ = self.drafter(calibration)
                    reference.observe_committed(expected[1], positions, aux)
                    with patch('engine.kernels.draft_attention._write_kv') as kernel:
                        kernel.__getitem__.return_value.side_effect = self.write_cpu
                        d.observe_committed((field, slot), positions, aux)
                    self.assertTrue(torch.equal(field, expected))
                    self.assertEqual(kernel.__getitem__.return_value.call_count, d.F.layers)
                    self.assertEqual(stats.rows['fc'].item(), tokens if calibration else 0)
                    values = aux.float() if calibration else torch.zeros_like(aux).float()
                    torch.testing.assert_close(stats.H['fc'], values.T @ values)

    def test_masked_native_ring_keeps_the_device_count_and_calibration_prefix(self):
        for count in (0, 1, 7, 8):
            with self.subTest(count=count):
                d, stats = self.drafter(True)
                field = torch.full((3, 2, 2, 16, 1, 4), -9., dtype=torch.bfloat16)
                expected = field.clone()
                slot, valid = torch.tensor([2]), torch.tensor(count)
                positions = torch.arange(14, 22)
                aux = torch.arange(32).reshape(8, 4).bfloat16()
                reference, _ = self.drafter(True)
                reference.observe_masked(expected[2], positions, aux, valid)
                with patch('engine.kernels.draft_attention._write_kv') as kernel:
                    kernel.__getitem__.return_value.side_effect = self.write_cpu
                    d.observe_masked((field, slot), positions, aux, valid)
                self.assertTrue(torch.equal(field, expected))
                for call in kernel.__getitem__.return_value.call_args_list:
                    self.assertIs(call.args[5], valid)
                    self.assertTrue(call.args[6])
                self.assertEqual(stats.rows['fc'].item(), count)
                values = aux[:count].float()
                torch.testing.assert_close(stats.H['fc'], values.T @ values)


if __name__ == '__main__':
    unittest.main()
