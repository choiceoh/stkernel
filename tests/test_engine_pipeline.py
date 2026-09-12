"""The decode pipeline's pure parts (45차 §23 B3): the batch distribution and the host readback bookkeeping."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from engine.profiles.glm53.pipeline import AsyncDecode, Pending, distribution_batch  # noqa: E402


class DistributionBatchTests(unittest.TestCase):
    def test_greedy_rows_are_one_hot_and_nucleus_rows_keep_the_smallest_prefix(self):
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.0], [1.0, 3.0, 2.0, 0.0], [1.0, 3.0, 2.0, 0.0]])
        temps = torch.tensor([0.0, 1.0, 1.0])
        top_p = torch.tensor([1.0, 1.0, 0.5])
        probs = distribution_batch(logits, temps, top_p, nucleus=True)
        self.assertEqual(probs[0].tolist(), [0.0, 1.0, 0.0, 0.0])
        torch.testing.assert_close(probs[1], torch.softmax(logits[1], -1))
        self.assertEqual(probs[2].tolist(), [0.0, 1.0, 0.0, 0.0])                     # the top token alone reaches 0.5
        torch.testing.assert_close(probs.sum(1), torch.ones(3))
        plain = distribution_batch(logits, temps, torch.ones(3), nucleus=False)
        torch.testing.assert_close(plain[2], torch.softmax(logits[2], -1))

    def test_only_the_listed_rows_are_sorted_and_truncated(self):
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]] * 3)
        temps, top_p = torch.ones(3), torch.tensor([1.0, 0.5, 0.5])
        probs = distribution_batch(logits, temps, top_p, nucleus=torch.tensor([2]))
        torch.testing.assert_close(probs[1], torch.softmax(logits[1], -1))         # not listed: untouched
        self.assertEqual(probs[2].tolist(), [0.0, 1.0, 0.0, 0.0])
        none = distribution_batch(logits, temps, top_p, nucleus=torch.zeros(0, dtype=torch.int64))
        torch.testing.assert_close(none, torch.softmax(logits, -1))
        torch.testing.assert_close(distribution_batch(logits, temps, top_p, nucleus=None), none)


class ShrinkTests(unittest.TestCase):
    def test_shrink_remaps_the_nucleus_positions_to_the_surviving_rows(self):
        e = SimpleNamespace(drafter=SimpleNamespace(k=2), caches=SimpleNamespace(pool=SimpleNamespace(max_seqs=4), device=torch.device("cpu")),
                            F=SimpleNamespace(block=2))
        p = AsyncDecode(e)
        t, n = p.t, 3
        b = {name: torch.arange(n) for name in ("seqs", "real_slot", "slot", "ctx", "generated", "limit", "temps", "top_p", "anchor")}
        b.update(ends=torch.zeros(n, 1, dtype=torch.int64), alive=torch.ones(n, dtype=torch.bool), drafts=torch.zeros(n, 2, dtype=torch.int64),
                 ids=torch.arange(n * t), dists=None, nucleus_rows=[j for i in (0, 2) for j in range(i * t, i * t + t)])
        b["nucleus"] = torch.tensor(b["nucleus_rows"])
        p.buf, p.batch = b, (1, 2, 3)
        p._shrink([3, 1])                                                             # old row 2 -> new 0, old row 0 -> new 1
        self.assertEqual(b["nucleus_rows"], [t + j for j in range(t)] + list(range(t)))
        self.assertEqual(b["nucleus"].tolist(), b["nucleus_rows"])
        self.assertEqual(b["seqs"].tolist(), [2, 0])
        p._shrink([3])                                                                # the row without a nucleus is gone
        self.assertEqual(b["nucleus_rows"], list(range(t)))
        p._shrink([])
        self.assertIsNone(b["nucleus"])
        self.assertEqual(b["nucleus_rows"], [])


class ResolveTests(unittest.TestCase):
    def test_resolve_applies_counts_in_launch_order_and_ignores_released_rows(self):
        e = SimpleNamespace(drafter=SimpleNamespace(k=2), caches=SimpleNamespace(pool=SimpleNamespace(max_seqs=4), device=torch.device("cpu")),
                            tokens={1: [5], 2: [6]}, ctx={1: 1, 2: 1}, inflight={1: 1, 2: 1}, accepted_total=0, drafted_total=0, steps=0,
                            F=SimpleNamespace(block=2), staged={})
        p = AsyncDecode(e)
        lane = p.free.pop(0)
        p.host[lane]["tokens"][:2] = torch.tensor([[7, 8, 9], [1, 2, 3]])
        p.host[lane]["count"][:2] = torch.tensor([2, 3])
        p.host[lane]["done"][:2] = torch.tensor([False, True])
        p.host[lane]["accepted"][:2] = torch.tensor([1, 2])
        first = Pending(p, lane, [1, 2], None)
        p.pending.append(first)
        second = Pending(p, 0, [1], None)
        p.pending.append(second)
        with self.assertRaises(RuntimeError):
            p.resolve(second)                                                         # launch order
        del e.tokens[2]                                                               # released meanwhile: nothing applied
        self.assertEqual(first.resolve(), [False, True])
        self.assertEqual(e.tokens[1], [5, 7, 8])
        self.assertEqual((e.ctx[1], e.inflight[1], e.inflight[2]), (3, 0, 0))
        self.assertEqual(e.staged, {1: 2})                                            # 1 -> 3 crossed the block boundary at 2
        self.assertEqual((e.accepted_total, e.drafted_total, e.steps), (1, 2, 1))
        self.assertIn(lane, p.free)


if __name__ == "__main__":
    unittest.main()
