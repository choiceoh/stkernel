import unittest
from types import SimpleNamespace

import torch

from engine.modules.sparse_indexer import select_with_tail, topk_positions
from engine.profiles.glm53 import net as module
from engine.profiles.glm53.incident_tail_reference import unpinned_complete_pools


class UpstreamTailReferenceTests(unittest.TestCase):
    def test_completed_pool_competes_but_incomplete_tail_is_appended(self):
        # Complete pool 2 has the lowest score. At 12 tokens upstream drops it;
        # at 13 tokens token 12 is appended independently of all pool scores.
        scores = torch.tensor([[3., 2., -10.], [3., 2., -10.]])
        lengths = torch.tensor([12, 13], dtype=torch.int32)
        pin = module.tail_pin_pools(lengths, 4)
        previous = module.pin_pools_in_logits
        net = SimpleNamespace(rank=0)
        with unpinned_complete_pools(net):
            module.pin_pools_in_logits(scores, pin, k=2)
        self.assertIs(module.pin_pools_in_logits, previous)
        pools = topk_positions(scores, 2)
        self.assertEqual([set(row) for row in pools.tolist()], [{0, 1}, {0, 1}])
        tokens = select_with_tail(pools, lengths, 4)
        self.assertEqual(set(tokens[0].tolist()), set(range(8)) | {-1})
        self.assertEqual(set(tokens[1].tolist()), set(range(8)) | {12, -1})
        previous(scores, pin, k=2)
        self.assertIn(2, topk_positions(scores, 2)[0].tolist())

    def test_restore_after_error(self):
        previous = module.pin_pools_in_logits
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with unpinned_complete_pools(SimpleNamespace(rank=0)):
                raise RuntimeError('sentinel')
        self.assertIs(module.pin_pools_in_logits, previous)


if __name__ == '__main__':
    unittest.main()
