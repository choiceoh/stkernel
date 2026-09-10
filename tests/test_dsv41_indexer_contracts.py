"""Boundary tests for externally supplied compact-indexer tensors/collectives."""
import ast
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay/modules/dsv41_model"))
from dsv41_indexer import compact_index_scores, compact_topk, select_candidate_ids


class CompactIndexerContracts(unittest.TestCase):
    def inputs(self):
        # The visible cache is a strided view of a larger allocation, as in
        # the reference. Distinct batches catch ignoring the physical stride.
        q = torch.ones(2, 1, 2, 4, dtype=torch.bfloat16)
        cache = torch.arange(2 * 16 * 4).reshape(2, 16, 4).to(torch.bfloat16)
        keys = cache[:, :9]
        weights = torch.tensor([[[1, -.5]], [[.25, .5]]], dtype=torch.bfloat16)
        ids = torch.tensor([[[0, 8, -1]], [[2, 7, -1]]], dtype=torch.int32)
        return q, keys, weights, ids

    def test_strided_cache_and_rounding_match_dense_gather(self):
        q, keys, weights, ids = self.inputs()
        self.assertFalse(keys.is_contiguous())
        full = (torch.einsum("bqhd,bsd->bqhs", q, keys).relu()
                * weights.unsqueeze(-1)).sum(dim=2)
        expected = full.gather(-1, ids.clamp_min(0).long()).masked_fill(ids < 0, 0)
        original = keys.clone()
        self.assertTrue(torch.equal(compact_index_scores(q, keys, weights, ids), expected))
        self.assertTrue(torch.equal(keys, original))

    def test_invalid_ids_do_not_read_or_enter_topk(self):
        q, keys, weights, ids = self.inputs()
        ids = ids.clone()
        ids[..., 1] = 2**31 - 1
        values = compact_index_scores(q, keys, weights, ids)
        self.assertTrue(torch.equal(values[..., 1:], torch.zeros_like(values[..., 1:])))
        chosen = compact_topk(values, ids, 9, 10, 1, full_width=9)
        self.assertTrue(torch.equal(chosen, ids[..., :1] + 10))

    def test_collective_called_once_on_contiguous_bf16_payload(self):
        args = self.inputs()
        original = compact_index_scores(*args)
        calls = []

        def reduce(value):
            calls.append((value.shape, value.dtype, value.is_contiguous()))
            value.mul_(2)
            return value

        actual = compact_index_scores(*args, reduce_fn=reduce)
        self.assertEqual(calls, [(torch.Size([2, 1, 3]), torch.bfloat16, True)])
        self.assertTrue(torch.equal(actual, original * 2))

    def test_async_or_replacement_collective_return_rejected(self):
        for bad in (lambda value: object(), lambda value: value.clone()):
            with self.subTest(callback=bad), self.assertRaisesRegex(TypeError, "in place"):
                compact_index_scores(*self.inputs(), reduce_fn=bad)

    def test_collective_cannot_resize_payload(self):
        def bad(value):
            value.resize_(1)
        with self.assertRaisesRegex(ValueError, "metadata"):
            compact_index_scores(*self.inputs(), reduce_fn=bad)

    def test_bad_dtype_shape_and_chunk_refused(self):
        q, keys, weights, ids = self.inputs()
        for args in ((q.float(), keys, weights, ids),
                     (q, keys, weights.float(), ids),
                     (q, keys, weights, ids.long()),
                     (q, keys, weights[:, :, :1], ids)):
            with self.subTest(shapes=[tuple(x.shape) for x in args]), \
                    self.assertRaises((ValueError, TypeError)):
                compact_index_scores(*args)
        for chunk in (0, -1, True, 1.5):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                compact_index_scores(q, keys, weights, ids, query_chunk_size=chunk)
        with self.assertRaises(ValueError):
            compact_index_scores(q, keys, weights, ids, backend="automatic")

    def test_empty_keyspace_preserves_collective_shape(self):
        q, keys, weights, ids = self.inputs()
        calls = []
        values = compact_index_scores(q, keys[:, :0], weights, ids[..., :0],
                                      reduce_fn=lambda value: calls.append(tuple(value.shape)))
        self.assertEqual(calls, [(2, 1, 0)])
        self.assertEqual(tuple(compact_topk(values, ids[..., :0], 0, 0, 512,
                                            full_width=0).shape), (2, 1, 0))

    def test_offset_must_fit_int32(self):
        q, keys, weights, ids = self.inputs()
        scores = compact_index_scores(q, keys, weights, ids)
        for offset in (-1, 2**31 - 1):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                compact_topk(scores, ids, 9, offset, 1, full_width=9)

    def test_width_is_runtime_and_does_not_specialize_every_decode_token(self):
        path = ROOT / "overlay/modules/dsv41_model/dsv41_indexer_triton.py"
        tree = ast.parse(path.read_text())
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_compact_scores_kernel")
        width = next(arg for arg in kernel.args.args if arg.arg == "WIDTH")
        self.assertIsNone(width.annotation)
        decorator = kernel.decorator_list[0]
        no_specialize = next(keyword.value for keyword in decorator.keywords
                             if keyword.arg == "do_not_specialize")
        self.assertIn("WIDTH", ast.literal_eval(no_specialize))


if __name__ == "__main__":
    unittest.main()
