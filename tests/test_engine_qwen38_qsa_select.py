"""A decode step's QSA block selection in one launch (engine/QWEN38_CARRY.md Q7, the Triton form).

`engine/kernels/qsa.select_blocks` took torch.topk for every captured step, inside a dozen launches of masking and
unpacking a QSA layer. `engine/kernels/qsa_select.select` is one program a row: the k-th value by 32 counting passes
over the floats' bits, ties to the lower block (prefill_topk's rule and GLM's st_dsa_select's), the ids written
ascending by a prefix sum, -1 after a row's picks.

Held here to the rule itself -- a stable descending sort's first k, as a set, in ascending order -- and to torch.topk
wherever torch.topk had no choice to make.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_qsa_select
"""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
RUNS = torch is not None and TRITON and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET else "cuda"


def rule(logits, visible, k):
    """The selection's statement: of a row's first `visible` columns, the k largest, equal scores to the lower block,
    ascending; -1 after them."""
    rows, columns = logits.shape
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        v = min(int(visible[r]), columns)
        if v:
            order = torch.sort(logits[r, :v].cpu(), descending=True, stable=True).indices[:k]
            picks = torch.sort(order).values
            out[r, :picks.numel()] = picks.to(torch.int32)
    return out


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class SelectTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1010)
        if INTERPRET:
            patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
            patch.start()
            self.addCleanup(patch.stop)

    def run_select(self, logits, visible, k):
        from engine.kernels import qsa_select
        out = torch.full((logits.shape[0], k), -7, dtype=torch.int32, device=DEVICE)      # every slot must be written
        qsa_select.select(logits, visible, k, out)
        return out.cpu()

    def check(self, logits, visible, k):
        got = self.run_select(logits, visible, k)
        self.assertTrue(torch.equal(got, rule(logits, visible, k)))
        return got

    def scores(self, rows, columns):
        """Scores as the scorer leaves them: relu sums, so a good share of exact zeros and many equal values."""
        raw = torch.randn(rows, columns, device=DEVICE)
        return torch.relu(raw).mul(4).round().div(4)                     # quarter steps: ties everywhere

    def test_it_is_the_rule_at_every_visible_count(self):
        k = 16 if INTERPRET else 512
        for columns in ((40, 64) if INTERPRET else (700, 1024, 5000, 32768)):
            with self.subTest(columns=columns):
                rows = 5
                logits = self.scores(rows, columns)
                visible = torch.tensor([0, 1, k - 1, k, columns], dtype=torch.int32, device=DEVICE).clamp_max(columns)
                self.check(logits, visible, k)

    def test_ties_across_the_kth_place_go_to_the_lower_blocks(self):
        k, columns = 8, 64
        logits = torch.zeros(2, columns, device=DEVICE)
        logits[0, [50, 3, 40]] = 2.0                                     # three leaders, then sixty-one equal zeros
        logits[1, :] = 1.5                                               # every block equal
        visible = torch.tensor([columns, 20], dtype=torch.int32, device=DEVICE)
        got = self.check(logits, visible, k)
        self.assertEqual(got[0].tolist(), [0, 1, 2, 3, 4, 5, 40, 50])    # 3, 40, 50 and the five lowest zeros
        self.assertEqual(got[1].tolist(), list(range(k)))

    def test_columns_past_the_visible_blocks_are_never_chosen_whatever_they_hold(self):
        k, columns = 8, 64
        logits = self.scores(3, columns)
        logits[:, 10:] = float("inf")                                    # the scorer never writes these: any bytes
        logits[1, 12] = float("nan")
        visible = torch.tensor([10, 4, 0], dtype=torch.int32, device=DEVICE)
        got = self.check(logits[:, :].contiguous(), visible, k)
        self.assertTrue(bool((got[0][:8] < 10).all()) and got[1].tolist() == [0, 1, 2, 3, -1, -1, -1, -1])
        self.assertEqual(got[2].tolist(), [-1] * k)

    def test_negative_scores_order_as_floats_do(self):
        k, columns = 4, 32
        logits = torch.linspace(-3, 3, columns, device=DEVICE).flip(0).contiguous()[None].repeat(2, 1)
        logits[1] = -logits[1] - 10                                      # all negative, ascending
        visible = torch.tensor([columns, columns], dtype=torch.int32, device=DEVICE)
        got = self.check(logits, visible, k)
        self.assertEqual(got[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(got[1].tolist(), [columns - 4, columns - 3, columns - 2, columns - 1])

    def test_where_torch_topk_had_no_choice_the_set_is_torch_topks(self):
        k = 16 if INTERPRET else 512
        columns = 256 if INTERPRET else 4096
        # no ties: every row a permutation of distinct values
        logits = torch.stack([torch.randperm(columns, device=DEVICE) for _ in range(4)]).float().div(8).sub(100)
        self.assertTrue(all(len(set(row)) == columns for row in logits.tolist()))
        visible = torch.full((4,), columns, dtype=torch.int32, device=DEVICE)
        got = self.run_select(logits, visible, k)
        want = torch.topk(logits, k, dim=1).indices.sort(dim=1).values.to(torch.int32).cpu()
        self.assertTrue(torch.equal(got, want))

    def test_a_row_view_and_what_it_refuses(self):
        from engine.kernels import qsa_select
        k, columns = 8, 64
        logits = self.scores(6, columns)
        visible = torch.full((3,), columns, dtype=torch.int32, device=DEVICE)
        self.check(logits[::2], visible, k)                              # strided rows, packed columns
        out = torch.empty(3, k, dtype=torch.int32, device=DEVICE)
        cases = {
            "half precision": lambda: qsa_select.select(logits[:3].half(), visible, k, out),
            "strided columns": lambda: qsa_select.select(logits[:3].t().contiguous().t(), visible, k, out),
            "int64 counts": lambda: qsa_select.select(logits[:3], visible.long(), k, out),
            "another width of out": lambda: qsa_select.select(logits[:3], visible, k, out[:, :4]),
        }
        for name, call in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                call()


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class SelectBlocksTests(unittest.TestCase):
    """qsa.select_blocks hands a step the radix select does not admit to the launch, and keeps torch.topk for the CPU."""

    def setUp(self):
        torch.manual_seed(926)

    def test_a_decode_steps_rows_take_the_launch(self):
        from engine.kernels import qsa, qsa_select
        k, columns = 8, 64
        logits = torch.relu(torch.randn(2, columns, device=DEVICE))
        visible = torch.tensor([columns, 5], dtype=torch.int32, device=DEVICE)
        out = torch.empty(2, k, dtype=torch.int32, device=DEVICE)
        with mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)) if INTERPRET else mock.patch.object(
                qsa_select, "admits", wraps=qsa_select.admits):
            with mock.patch.object(torch, "topk", side_effect=AssertionError("torch.topk served a device step")):
                qsa.select_blocks(logits, visible, k, out)
        self.assertTrue(torch.equal(out.cpu(), rule(logits, visible, k)))

    def test_a_table_wider_than_a_program_pays_for_keeps_the_torch_form(self):
        from engine.kernels import qsa, qsa_select
        k = 8
        logits = torch.rand(1, qsa_select.WIDEST + 1, device=DEVICE)
        self.assertFalse(qsa_select.admits(logits, k) if not INTERPRET else False)
        visible = torch.tensor([logits.shape[1]], dtype=torch.int32, device=DEVICE)
        out = torch.empty(1, k, dtype=torch.int32, device=DEVICE)
        with mock.patch.object(qsa_select, "select", side_effect=AssertionError("the launch took a table past WIDEST")):
            qsa.select_blocks(logits, visible, k, out)
        self.assertEqual(sorted(out[0].tolist()), torch.topk(logits[0], k).indices.sort().values.tolist())

    @unittest.skipUnless(INTERPRET, "the CPU form")
    def test_without_a_device_the_torch_form_still_serves(self):
        from engine.kernels import qsa
        k, columns = 8, 64
        logits = torch.rand(2, columns)                                  # no ties: both forms choose one set
        visible = torch.tensor([columns, 5], dtype=torch.int32)
        out = torch.empty(2, k, dtype=torch.int32)
        qsa.select_blocks(logits, visible, k, out)
        picked = [sorted(i for i in row if i >= 0) for row in out.tolist()]
        self.assertEqual(picked, [[i for i in row if i >= 0] for row in rule(logits, visible, k).tolist()])


if __name__ == "__main__":
    unittest.main()
