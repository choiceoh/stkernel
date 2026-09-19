"""engine/kernels/common/row_tap: rows and labels into a device ring with no host in the loop, and drained in order --
the tap fleet --tap-draft-queries puts in net.draft_tokens.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_row_tap
"""
import importlib.util
import os
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
READY = torch is not None and importlib.util.find_spec("triton") is not None
DEVICE = "cpu" if INTERPRET else "cuda"


@unittest.skipUnless(READY and (INTERPRET or (torch is not None and torch.cuda.is_available())),
                     "CUDA or Triton interpreter required")
class RowTapTests(unittest.TestCase):
    def test_rows_come_back_in_the_order_they_went_in(self):
        from engine.kernels.common.row_tap import RowTap
        tap = RowTap(8, 5, DEVICE, slack=0)
        a = torch.arange(15, dtype=torch.float32).view(3, 5).bfloat16().to(DEVICE)
        tap(a, torch.tensor([7, 8, 9], device=DEVICE))
        rows, ids, count = tap.drain()
        self.assertEqual((count, sorted(ids.tolist())), (3, [7, 8, 9]))
        self.assertEqual(sorted(map(tuple, rows.float().tolist())), sorted(map(tuple, a.float().cpu().tolist())))
        self.assertEqual(tap.drain()[0].shape[0], 0)                       # nothing new

    def test_the_ring_keeps_the_last_capacity_rows(self):
        from engine.kernels.common.row_tap import RowTap
        tap = RowTap(4, 2, DEVICE, slack=0)
        for i in range(3):
            x = torch.full((2, 2), float(i), device=DEVICE).bfloat16()
            tap(x, torch.tensor([i, i], device=DEVICE))
        rows, ids, count = tap.drain()
        self.assertEqual(count, 6)
        self.assertEqual(sorted(ids.tolist()), [1, 1, 2, 2])

    def test_slack_leaves_the_newest_rows_for_the_next_drain(self):
        from engine.kernels.common.row_tap import RowTap
        tap = RowTap(16, 2, DEVICE, slack=2)
        tap(torch.zeros(5, 2, device=DEVICE).bfloat16(), torch.arange(5, device=DEVICE))
        self.assertEqual(tap.drain()[1].shape[0], 3)
        self.assertEqual(tap.drain(final=True)[1].shape[0], 2)

    def test_the_net_taps_what_it_drafts(self):
        from unittest import mock
        from engine.profiles.qwen38.net import Qwen38Net
        net = object.__new__(Qwen38Net)
        net.draft_index, seen = None, []
        net.draft_tap = lambda h, picks: seen.append((h.shape[0], picks.tolist()))
        with mock.patch.object(Qwen38Net, "head_tokens", return_value=torch.tensor([3, 4])):
            self.assertEqual(net.draft_tokens(torch.zeros(2, 4)).tolist(), [3, 4])
        self.assertEqual(seen, [(2, [3, 4])])


if __name__ == "__main__":
    unittest.main()
