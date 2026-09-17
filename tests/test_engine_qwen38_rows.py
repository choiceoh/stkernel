"""A captured Qwen3.8 step's row mapping is packed at every width, and the QSA kernels refuse one that is not.

`Qwen38Net.step_meta` built a captured step's `rows_req` as `iota(n)[:, None].expand(n, t).reshape(-1)`. For n > 1 the
reshape copies; for one row it returns a stride-0 view over a single int32. The QSA compression, scoring and expansion
kernels load that mapping at `ptr + row` with no stride, so the second token of a C=1 step read four bytes past the
tensor's storage, and the sparse attention -- the one wrapper that checked the stride -- then refused the step. C=1 with
the MTP drafter (two tokens a row) is the width that hit it. No test had built a captured step's metadata.
"""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

TRITON = importlib.util.find_spec("triton") is not None


@unittest.skipUnless(torch is not None, "requires torch")
class CapturedRowsTests(unittest.TestCase):
    def meta(self, rows: int, tokens: int):
        from engine.profiles.qwen38.facts import BLOCK
        from engine.profiles.qwen38.net import Qwen38Net
        blocks = 4
        caches = SimpleNamespace(block_table=torch.arange(rows * blocks, dtype=torch.int32).reshape(rows, blocks))
        step = SimpleNamespace(captured=True, rows=rows, tokens=tokens, blocks=blocks,
                               ids=torch.zeros(rows * tokens, dtype=torch.int64),
                               contexts=torch.arange(rows, dtype=torch.int64) * 7 + 5,
                               seqs=torch.arange(rows, dtype=torch.long), slots=torch.arange(rows, dtype=torch.int64))
        net = SimpleNamespace(F=SimpleNamespace(block=BLOCK, idx_ratio=4))   # the checkpoint's indexer_compress_ratio
        return Qwen38Net.step_meta(net, step, caches)

    def test_every_captured_width_packs_its_row_mapping(self):
        for rows in (1, 2, 4):
            for tokens in (1, 2, 4):
                with self.subTest(rows=rows, tokens=tokens):
                    meta = self.meta(rows, tokens)
                    expected = torch.arange(rows, dtype=torch.int32).repeat_interleave(tokens)
                    self.assertTrue(torch.equal(meta.rows_req, expected))
                    self.assertTrue(meta.rows_req.is_contiguous())
                    if rows * tokens > 1:
                        self.assertEqual(meta.rows_req.stride(), (1,))

    def test_the_one_row_step_is_the_width_that_needed_it(self):
        # the expression the step used to return: a view whose second element aliases the first
        view = torch.arange(1, dtype=torch.int32)[:, None].expand(1, 2).reshape(-1)
        self.assertEqual(view.stride(), (0,))
        self.assertEqual(self.meta(1, 2).rows_req.stride(), (1,))


@unittest.skipUnless(torch is not None and TRITON, "the QSA module imports Triton")
class PackedRowMetadataTests(unittest.TestCase):
    """Every wrapper whose kernel loads one-dimensional row metadata at `ptr + row` refuses a strided mapping before
    any launch. `is_cuda` is patched so the CPU reaches the checks; nothing here launches a kernel."""

    def setUp(self):
        patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
        patch.start()
        self.addCleanup(patch.stop)

    @staticmethod
    def one_row_view(dtype=torch.int32):
        return torch.zeros(1, dtype=dtype)[:, None].expand(1, 2).reshape(-1)

    def test_scoring_refuses_a_strided_row_mapping(self):
        from engine.kernels import qsa
        q = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
        cache, table = torch.zeros(4, 768, 1, 128, dtype=torch.bfloat16), torch.zeros(1, 4, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "packed row metadata"):
            qsa.qsa_mqa_paged(q, cache, table, self.one_row_view(), torch.arange(2, dtype=torch.int32),
                              torch.tensor([2], dtype=torch.int32), 4)

    def test_expansion_refuses_a_strided_row_mapping(self):
        from engine.kernels import qsa
        with self.assertRaisesRegex(ValueError, "packed row metadata"):
            qsa.expand_qsa_block_indices_cuda(torch.zeros(2, 512, dtype=torch.int32), torch.arange(2, dtype=torch.int32),
                                              torch.tensor([2], dtype=torch.int32), self.one_row_view(), 4, 2048)

    def test_compression_refuses_a_strided_row_mapping(self):
        from engine.kernels import qsa
        with self.assertRaisesRegex(ValueError, "packed row metadata"):
            qsa.qsa_compress_groups_with_ratio(torch.zeros(2, 1, 128, dtype=torch.bfloat16),
                                               torch.zeros(2, 1, 3, dtype=torch.int64),
                                               torch.zeros(4, 8, 1, 128, dtype=torch.bfloat16),
                                               torch.zeros(1, 1, dtype=torch.int32), self.one_row_view(),
                                               torch.tensor([0, 2], dtype=torch.int32),
                                               torch.arange(2, dtype=torch.int32), torch.full((2,), -1, dtype=torch.int32), 4)

    def test_stores_refuse_a_strided_slot_mapping(self):
        from engine.kernels import qsa
        with self.assertRaisesRegex(ValueError, "packed row metadata"):
            qsa.qsa_store_cache_rows(torch.zeros(4, 768, 1, 128, dtype=torch.bfloat16), self.one_row_view(),
                                     torch.zeros(2, 128, dtype=torch.bfloat16))

    def test_packed_and_single_row_mappings_pass_the_check(self):
        from engine.kernels import qsa
        single = torch.zeros(2, dtype=torch.int32)[:1]
        qsa._packed_rows("test", torch.arange(4, dtype=torch.int32), single, torch.zeros(0, dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
