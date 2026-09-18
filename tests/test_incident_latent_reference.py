"""CPU gates for the private native-FP8 / BF16 latent reference pair."""
from types import SimpleNamespace as NS
import unittest

import torch

from engine.profiles.glm53.incident_latent_reference import LatentReference, control


def step(start, count):
    return NS(segments=[NS(seq=0, ctx=start, start=0, length=count)])


class LatentReferenceTests(unittest.TestCase):
    def fixture(self, bf16):
        torch.manual_seed(17)
        values = torch.randn(8, 16).to(torch.bfloat16)
        cache = torch.full((48, 16), float('nan')).to(torch.float8_e4m3fn)
        # Non-identity physical blocks and an offset within a multi-layer block.
        row = torch.tensor([2, 0, -1, -1], dtype=torch.int32)
        physical = torch.tensor([28, 29, 30, 31, 4, 5, 6, 7])
        cache[physical] = values.to(torch.float8_e4m3fn)
        caches = NS(token_map=lambda layer, seq: (row, 4, 12, 4), latent=lambda layer: cache)
        reference = LatentReference((1, 0, bf16), bf16=bf16, capacity=16, chunk=2)
        reference.write(3, values[:5], step(0, 5))
        reference.write(3, values[5:], step(5, 3))
        return values, cache, physical, caches, reference

    def test_fp8_and_bf16_use_same_attention_over_their_actual_values(self):
        for bf16 in (False, True):
            with self.subTest(bf16=bf16):
                values, cache, physical, caches, reference = self.fixture(bf16)
                q = torch.randn(5, 3, 16).to(torch.bfloat16)
                positions = torch.tensor([[0, 1, 2], [3, 4, 0], [5, 0, 0], [6, 7, 3], [0, 0, 0]])
                valid = torch.tensor([3, 2, 1, 3, 0], dtype=torch.int32)
                slots = physical[positions].to(torch.int32)
                for row, n in enumerate(valid.tolist()):
                    slots[row, n:] = -1
                actual = reference.context(3, q, cache, slots, valid, step(5, 3), caches, 0.25)
                for row, n in enumerate(valid.tolist()):
                    if not n:
                        self.assertTrue(torch.equal(actual[row], torch.zeros_like(actual[row])))
                        continue
                    selected = values[positions[row, :n]].float() if bf16 else cache[slots[row, :n].long()].float()
                    expected = (torch.softmax((q[row].float() @ selected.T) * 0.25, -1) @ selected).to(q.dtype)
                    self.assertTrue(torch.equal(actual[row], expected))

    def test_paged_indices_recover_logical_positions_and_reject_foreign_layer(self):
        _, _, physical, caches, reference = self.fixture(True)
        slots = physical[None, :].to(torch.int32)
        logical, active = reference.logical_slots(3, slots, torch.tensor([8]), step(5, 3), caches)
        self.assertEqual(logical.tolist(), [list(range(8))])
        self.assertTrue(bool(active.all()))
        slots[0, 0] = 32  # same physical block, next layer's row
        with self.assertRaisesRegex(ValueError, 'different layer'):
            reference.logical_slots(3, slots, torch.tensor([8]), step(5, 3), caches)

    def test_sidecar_refuses_missing_prefix_and_capacity_overflow(self):
        reference = LatentReference((1, 0, 27), bf16=True, capacity=4)
        with self.assertRaisesRegex(ValueError, 'uncached contiguous'):
            reference.write(3, torch.zeros(1, 16, dtype=torch.bfloat16), step(4, 1))
        with self.assertRaisesRegex(ValueError, 'uncached contiguous'):
            reference.write(3, torch.zeros(5, 16, dtype=torch.bfloat16), step(0, 5))

    def test_scope_restores_hook_and_resets_values_for_next_admission(self):
        net = NS(incident_latent_reference='previous')
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with control(net, (1, 0, 26), bf16=False):
                old = net.incident_latent_reference
                old.ends[3] = 10
                raise RuntimeError('stop')
        self.assertEqual(net.incident_latent_reference, 'previous')
        with control(net, (1, 0, 26), bf16=False):
            self.assertIs(net.incident_latent_reference, old)
        with control(net, (2, 0, 27), bf16=True):
            self.assertEqual(net.incident_latent_reference.ends, {})
            self.assertTrue(net.incident_latent_reference.bf16)


if __name__ == '__main__':
    unittest.main()
