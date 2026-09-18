"""A run of rows of one request scores from one read of each key tile, and the scores are the row launch's bytes
(engine/QWEN38_CARRY.md Q8).

`_qsa_mqa_paged_kernel` is a program a query row: a verify step's two (K=1) to four (K=3) rows of a sequence, and every
row of a prefill segment, each read the same paged index keys, tile by tile. `_qsa_mqa_paged_group_kernel` takes up to
four consecutive rows of one request a program, reads a tile once -- as far as the run's furthest row sees -- and each
row takes its scores with the row kernel's own dot and stores them under its own horizon. At 128K a K=1 verify step
read the index keys twice in each of 13 QSA layers; it reads them once.

Held on the served kernels -- a GPU, or TRITON_INTERPRET=1 -- byte for byte against the row launch: a captured step's
shape (requests of 2, 3 and 4 tokens, their horizons apart by a group boundary inside a run), a prefill segment whose
last run is short, rows past the 32-row launch geometry, the selection through both entries with the scorer's calls
cut to whole runs, and the wrapper's refusals. Only what a row may read is compared: past its visible blocks neither
launch writes.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_qsa_group_scores
"""
import unittest
from unittest import mock

from tests import test_engine_qwen38_kernels as harness                  # the module, so its tests are not collected here
from tests.test_engine_qwen38_kernels import RUNS, RUNS_REASON, TRITON, W, served_kernels, torch


def fixture(requests, seed):
    selection = harness.SelectionTests()
    selection.requests = lambda: requests                                  # (seq, slot, ctx, length)
    return selection.fixture(seed)


def scores(f, group):
    from engine.kernels import qsa
    m = f.meta
    return qsa.qsa_mqa_paged(f.queries, f.key_cache, m.page_table, m.rows_req, m.positions32, m.lengths, W.ratio,
                             group=group)


@unittest.skipUnless(RUNS, RUNS_REASON)
class GroupScoreTests(unittest.TestCase):
    def assertRowBytes(self, f, group):
        """Every row's visible count, and its logits over the columns it may read, against the row launch."""
        with served_kernels():
            want, want_visible = scores(f, 1)
            got, got_visible = scores(f, group)
        self.assertTrue(torch.equal(got_visible, want_visible))
        self.assertEqual((got.dtype, tuple(got.shape)), (want.dtype, tuple(want.shape)))
        seen = 0
        for row, visible in enumerate(want_visible.tolist()):
            self.assertTrue(torch.equal(got[row, :visible], want[row, :visible]), f"row {row} of group {group}")
            seen += visible
        self.assertGreater(seen, 0)

    def test_a_captured_steps_rows_score_as_they_did_alone(self):
        r, blocks = W.ratio, W.budget // W.ratio
        for tokens in (2, 3, 4):
            # each request `tokens` rows, as a verify step lays them: from the first position, across a group
            # boundary inside the run (the rows' horizons differ), past the budget, and past a 64-block score tile
            requests = tuple((seq, seq + 1, ctx, tokens) for seq, ctx in enumerate(
                (0, r - 1, blocks * r - 2, 3 * blocks * r + 1, (64 + blocks) * r + r - 1)))
            with self.subTest(tokens=tokens):
                self.assertRowBytes(fixture(requests, 800 + tokens), tokens)

    def test_a_prefill_segment_with_a_short_last_run(self):
        for rows, group in ((37, 4), (37, 3), (9, 2), (2, 4)):             # 37 rows: past the 32-row launch geometry
            with self.subTest(rows=rows, group=group):
                self.assertRowBytes(fixture(((0, 1, 5, rows),), 900 + rows + group), group)

    def test_one_row_is_the_row_launch(self):
        from engine.kernels import qsa
        f = fixture(((0, 1, 3 * W.budget, 1),), 77)
        with served_kernels(), mock.patch.object(qsa, "_qsa_mqa_paged_group_kernel") as grouped:
            scores(f, 4)
        grouped.__getitem__.assert_not_called()                           # nothing to share: the kernel it always ran

    def test_the_selection_is_the_row_launchs_through_both_entries(self):
        from engine.kernels import qsa
        f = fixture(((0, 1, 2 * W.budget + 1, 23),), 1001)
        m = f.meta
        args = (f.queries, f.key_cache, m.page_table, m.rows_req, m.positions32, m.lengths, W.budget, W.ratio)
        columns = m.page_table.shape[1] * f.key_cache.shape[1]
        with served_kernels():
            blocks, tokens = qsa.qsa_select_paged_blocks(*args), qsa.qsa_select_paged_tokens(*args)
            for group in (2, 3, 4):
                with self.subTest(group=group):
                    self.assertTrue(torch.equal(qsa.qsa_select_paged_blocks(*args, group=group), blocks))
                    self.assertTrue(torch.equal(qsa.qsa_select_paged_tokens(*args, group=group), tokens))
                    # a workspace of two rows a call: the scorer's calls are cut to whole runs (4, 3 -> 3, 2 -> 2)
                    with mock.patch.object(qsa, "_LOGITS_WORKSPACE_BYTES", 2 * columns * 4):
                        self.assertEqual(qsa._rows_a_scoring_call(columns, group), group)
                        self.assertTrue(torch.equal(qsa.qsa_select_paged_blocks(*args, group=group), blocks))


@unittest.skipUnless(torch is not None and TRITON, "engine/kernels/qsa imports Triton")
class WrapperTests(unittest.TestCase):
    def test_a_group_is_one_to_four_rows(self):
        from engine.kernels import qsa
        q = torch.zeros(2, 4, 16, dtype=torch.bfloat16)
        cache, table = torch.zeros(4, 8, 1, 16, dtype=torch.bfloat16), torch.zeros(1, 4, dtype=torch.int32)
        rows, lengths = torch.zeros(2, dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
        for group in (0, 5, -1, True, 2.0):
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "groups 1..4 rows"):
                qsa.qsa_mqa_paged(q, cache, table, rows, rows, lengths, 4, group=group)

    def test_the_scorers_calls_are_whole_runs(self):
        from engine.kernels import qsa
        with mock.patch.object(qsa, "_LOGITS_WORKSPACE_BYTES", 513 * 1024 * 4):            # 513 rows of 1024 columns
            self.assertEqual([qsa._rows_a_scoring_call(1024, g) for g in (1, 2, 3, 4)], [513, 512, 513, 512])
            self.assertEqual(qsa._rows_a_scoring_call(1024 * 1024, 4), 4)                  # never less than a run
            # the split rule (carry Q11) cuts its calls where the scorer does
            self.assertTrue(qsa.shards_select_alike(2048, [512] * 4, 1024, 512, 4))
            self.assertFalse(qsa.shards_select_alike(2048 + 40, [522] * 4, 1024, 512, 4))   # 512 + 10


if __name__ == "__main__":
    unittest.main()
