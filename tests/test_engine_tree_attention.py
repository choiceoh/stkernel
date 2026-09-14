"""Independent tree address oracles, poisoned padding, and native CPU interpretation."""
import os
import unittest

import torch

from engine.modules.speculative_tree import Tree
from engine.modules.tree_attention import pool_slots
from engine.modules.sparse_attention import mla_sparse_mqa


def slot_cases(groups=9):
    tree = Tree(tuple(range(8)), (-1, 0, 0, 1, 3, 2, 5, 6))
    paths = torch.tensor([tree.path(i)+(i,)*(5-len(tree.path(i))) for i in range(8)])
    for pool in (1, 4, 8):
        for context in (0, 1, 3, 4, 7, 63, 64, 65, 129, 32767, 131069):
            block, stride, offset = 64, 32768, 1024
            table = torch.arange((context+7)//block+1, dtype=torch.int32).flip(0)*3
            lengths = torch.tensor([context+d+1 for d in tree.depths], dtype=torch.int32)
            # Duplicate pools, future pools, negative/overflow ids and a last
            # complete pool. Large invalid int64 values must be rejected before narrowing.
            ids = torch.tensor([[0, 1, 0, n//pool-1, n//pool, -1, 2**32, -2**32, 5] for n in lengths.tolist()])
            ids = ids.repeat(1, (groups+8)//9)[:, :groups].contiguous()
            expected = torch.zeros((8, ids.shape[1]*pool+pool-1), dtype=torch.int32)
            counts = torch.zeros(8, dtype=torch.int32)
            for row, length in enumerate(lengths.tolist()):
                positions = [i for p in ids[row].tolist() if 0 <= p < length//pool
                             for i in range(p*pool, (p+1)*pool)]
                positions += list(range(length-length % pool, length))
                positions.sort(reverse=True)
                values = [int(table[pos//block])*stride+offset+pos % block if pos < context
                          else -tree.path(row)[pos-context]-1 for pos in positions]
                expected[row, :len(values)] = torch.tensor(values, dtype=torch.int32)
                counts[row] = len(values)
            yield (ids, lengths, pool, table, block, stride, offset, paths, context), expected, counts


class TreeAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def check_mapper(self, mapper):
        for args, expected, lengths in (*slot_cases(), *slot_cases(512)):
            with self.subTest(pool=args[2], context=args[-1]):
                out, counts = torch.full_like(expected, 12345), torch.full_like(lengths, 12345)
                mapper(*args, out, counts)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                torch.testing.assert_close(counts, lengths, rtol=0, atol=0)

    def test_reference_addresses_match_independent_ancestry_and_duplicate_sort(self):
        self.check_mapper(pool_slots)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU Triton interpreter check')
    def test_native_addresses_match_independent_oracle(self):
        from engine.kernels.indexer import tree_pool_slots
        self.check_mapper(tree_pool_slots)

    def test_two_banks_equal_copied_cache_and_ignore_poisoned_padding(self):
        torch.manual_seed(143)
        canonical = torch.randn(15, 32).to(torch.float8_e4m3fn)
        private = torch.randn(8, 32).to(torch.float8_e4m3fn)
        q = torch.randn(8, 2, 32).bfloat16()
        ids = torch.tensor([[2, -i-1, -1, 5, -i-1, -2**31, 2**31-1] for i in range(8)], dtype=torch.int32)
        counts = torch.full((8,), 5, dtype=torch.int32)
        # Siblings and padding contain NaNs, so masking must precede math.
        private[7].fill_(float('nan'))
        counts[7] = 1
        copied = torch.zeros(8, 7, 32, dtype=torch.float32)
        for row in range(8):
            for col in range(int(counts[row])):
                slot = int(ids[row, col])
                copied[row, col] = canonical[slot].float() if slot >= 0 else private[-slot-1].float()
        copied = copied.to(torch.float8_e4m3fn).view(-1, 32)
        linear_ids = torch.arange(56, dtype=torch.int32).view(8, 7)
        for scale in (1., .75):
            got = mla_sparse_mqa(q, canonical, ids, counts, .125, scale, branch=private)
            want = mla_sparse_mqa(q, copied, linear_ids, counts, .125, scale)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(got).all())

    def test_contract_refuses_unsafe_geometry_before_writing(self):
        args, expected, lengths = next(slot_cases())
        for index, value in ((2, 3), (4, 0), (5, 1024), (6, -1), (8, -1), (7, args[7].int())):
            changed = list(args)
            changed[index] = value
            out, counts = torch.full_like(expected, 777), torch.full_like(lengths, 777)
            with self.assertRaises(ValueError):
                pool_slots(*changed, out, counts)
            self.assertTrue(torch.all(out == 777) and torch.all(counts == 777))


if __name__ == '__main__':
    unittest.main()
