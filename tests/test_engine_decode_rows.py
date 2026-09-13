"""A captured decode step folds its rows: which steps fold, and that the folded addressing is the per-row one.

The kernels themselves are pinned on the GPU (tests/test_engine_kda_ring.py, tests/test_engine_conv_ring.py,
rows tests; probes/engine_decode_graph_check.py end to end). What the CPU can pin is the decision
(`Glm53Net._ring_rows`) and the gathered block-table arithmetic (`GraphCaches.token_rows`) against the
one-row functions it replaces.
"""
import unittest
from types import SimpleNamespace as NS

import torch

from engine.profiles.glm53.caches import Glm53Caches
from engine.profiles.glm53.decode_graphs import GraphCaches
from engine.profiles.glm53.net import Glm53Net, Segment


def step(lengths, captured=True, contexts=True):
    segs, start = [], 0
    for i, n in enumerate(lengths):
        segs.append(Segment(i, i + 1, 10 * i, start, n)); start += n
    fields = dict(segments=tuple(segs), captured=captured)
    if contexts:
        fields["contexts"] = torch.tensor([10 * i for i in range(len(lengths))])
    return NS(**fields)


class RowFoldDecisionTests(unittest.TestCase):
    def net(self, rows=True):
        lanes = NS(kda_recurrent_ring_rows=object() if rows else None, conv_ring_rows=object() if rows else None)
        return NS(lanes=lanes)

    def test_a_captured_decode_step_folds_every_row(self):
        for lengths in ((7,), (7, 7), (1, 1, 1, 1), (7, 7, 7, 7)):
            self.assertEqual(Glm53Net._ring_rows(self.net(), step(lengths), 9, 7), len(lengths))

    def test_what_keeps_the_per_segment_loop(self):
        net = self.net()
        self.assertEqual(Glm53Net._ring_rows(net, step((7, 7), captured=False), 9, 7), 0)     # eager
        self.assertEqual(Glm53Net._ring_rows(net, step((7, 7), contexts=False), 9, 7), 0)     # no device contexts
        self.assertEqual(Glm53Net._ring_rows(net, step((2304,)), 9, 7), 0)                    # a prefill chunk
        self.assertEqual(Glm53Net._ring_rows(net, step((7, 6)), 9, 7), 0)                     # uneven rows
        self.assertEqual(Glm53Net._ring_rows(net, step((8, 8)), 9, 7), 0)                     # past the recurrent ring
        self.assertEqual(Glm53Net._ring_rows(net, step((9, 9)), 9, 9), 0)                     # past the conv ring's 8
        self.assertEqual(Glm53Net._ring_rows(self.net(rows=False), step((7, 7)), 9, 7), 0)   # a reference table


class TokenRowsTests(unittest.TestCase):
    def test_token_rows_equals_token_slots_row_by_row(self):
        torch.manual_seed(5)
        F = NS(block=768, kv_lora=512, kpool=4)
        layout = NS(block_bytes=768 * 512 + 2048, token_offsets={3: 0, 7: 768 * 512})
        real = NS(F=F, layout=layout, block_table=torch.randint(0, 40, (6, 12), dtype=torch.int32))
        rows = 4
        caches = GraphCaches(real, torch.tensor([5, 2, 0, 3]), torch.tensor([1, 2, 3, 4]), 4096)
        caches.gather()
        positions = torch.tensor([[2000 + i for i in range(7)], [0, 1, 2, 3, 4, 5, 6],
                                  [767 + i for i in range(7)], [5000 + i for i in range(7)]])
        for layer in (3, 7):
            folded = caches.token_rows(layer, positions)
            self.assertEqual(folded.dtype, torch.int32)
            for i in range(rows):
                one = Glm53Caches.token_slots(caches, layer, i, positions[i])
                torch.testing.assert_close(folded[i], one, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
