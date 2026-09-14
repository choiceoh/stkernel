"""Exact paged prefix copies and convolution reads from the canonical ring."""
import importlib.util
import os
import unittest

import torch

from engine.modules.tree_attention import key_bank
from engine.modules.tree_kda import Topology, conv
from engine.modules.speculative_tree import Tree

INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
TRITON = importlib.util.find_spec('triton') is not None

if TRITON:
    import triton
    import triton.language as tl
    from engine.kernels.kda.tree import _conv_sum

    @triton.jit
    def ring_sum(X, W, HISTORY, PATH, OUT, context, C: tl.constexpr, RING: tl.constexpr):
        node = tl.program_id(0)
        c = tl.arange(0, 256)
        value = _conv_sum(X, W, HISTORY, PATH, node, c, C, 4, context, RING)
        tl.store(OUT+node*C+c, value, c < C)


def bank_cases():
    gen = torch.Generator().manual_seed(316)
    for prefix, private_count in ((0, 0), (0, 8), (1, 0), (15, 7), (16, 1), (17, 32), (65, 9), (8192, 8), (32768, 8)):
        per, stride, offset, width = 16, 53, 17, 128
        pages = (prefix+per-1)//per
        table_store = torch.full((2*(pages+1),), -1, dtype=torch.int32)
        table = table_store[::2]
        table[:pages] = torch.arange(pages, dtype=torch.int32).flip(0)*2+1
        records = (2*pages+2)*stride
        # The cache views and private records are intentionally strided. Use
        # all byte patterns, including NaN scale payloads and negative zero.
        bytes_ = torch.randint(0, 256, (records, width+4), dtype=torch.uint8, generator=gen)
        keys = bytes_[:, :width].view(torch.float8_e4m3fn)
        scales = torch.randint(-2**31, 2**31-1, (records*3,), dtype=torch.int32, generator=gen)[::3].view(torch.float32)
        private = torch.randint(0, 256, (private_count, width+4), dtype=torch.uint8, generator=gen)[:, :width].view(keys.dtype)
        ps = torch.randint(-2**31, 2**31-1, (private_count*2,), dtype=torch.int32, generator=gen)[::2].view(torch.float32)
        ids = torch.tensor([int(table[i//per])*stride+offset+i % per for i in range(prefix)], dtype=torch.int64)
        expected = torch.cat((keys.view(torch.uint8)[ids], private.view(torch.uint8)))
        expected_scales = torch.cat((scales.view(torch.int32)[ids], ps.view(torch.int32)))
        yield (keys, scales, private, ps, table, per, stride, offset, prefix), expected, expected_scales


class TreeBankTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def check_bank(self, fn):
        for args, expected, expected_scales in bank_cases():
            before = (args[0].view(torch.uint8).clone(), args[1].view(torch.int32).clone())
            actual, scale = fn(*args)
            self.assertTrue(torch.equal(actual.view(torch.uint8), expected))
            self.assertTrue(torch.equal(scale.view(torch.int32), expected_scales))
            self.assertTrue(actual.is_contiguous() and scale.is_contiguous())
            self.assertTrue(torch.equal(args[0].view(torch.uint8), before[0]))
            self.assertTrue(torch.equal(args[1].view(torch.int32), before[1]))

    def test_reference_copies_exact_paged_and_private_record_bits(self):
        self.check_bank(key_bank)

    @unittest.skipUnless(INTERPRET and TRITON, 'explicit CPU Triton interpreter check')
    def test_native_single_bank_copy_matches_the_independent_address_oracle(self):
        from engine.kernels.indexer import tree_key_bank
        self.check_bank(tree_key_bank)
        self.assertFalse(torch.cuda.is_initialized())

    def test_rejects_invalid_map_and_record_contracts(self):
        args, _, _ = next(bank_cases())
        for index, bad in ((5, 0), (6, 20), (7, -1), (8, -1), (8, 1000), (0, args[0].float()),
                           (1, args[1].bfloat16()), (4, args[4].long())):
            changed = list(args)
            changed[index] = bad
            with self.assertRaisesRegex(ValueError, 'key bank'):
                key_bank(*changed)

    @unittest.skipUnless(INTERPRET and TRITON, 'explicit CPU Triton interpreter check')
    def test_ring_gather_matches_materialized_history_and_masks_poison_before_math(self):
        torch.manual_seed(493)
        tree = Tree(tuple(range(8)), (-1, 0, 0, 1, 3, 2, 5, 6))
        topology = Topology(tree, 'cpu')
        channels, size = 193, 11
        raw = torch.randn(8, channels).bfloat16()
        for context in (0, 1, 2, 3, 10, 11, 12, 131069):
            positions = torch.arange(context-3, context)
            history = torch.full((channels, size), float('nan'), dtype=raw.dtype)
            for pos in positions.tolist():
                if pos >= 0:
                    history[:, pos % size] = torch.randn(channels).bfloat16()
            prefix = history[:, positions.clamp_min(0) % size].masked_fill((positions < 0)[None, :], 0)
            for dtype in (torch.bfloat16, torch.float32):
                weight = torch.randn(channels, 4).to(dtype)
                actual = torch.empty(8, channels)
                ring_sum[(8,)](raw, weight, history, topology.conv, actual, context, channels, size)
                expected = torch.zeros_like(actual)
                for node in range(8):
                    path = tree.path(node)
                    for tap, j in enumerate(range(len(path)-4, len(path))):
                        value = raw[path[j]] if j >= 0 else prefix[:, 3+j]
                        expected[node] += value.float()*weight[:, tap].float()
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(conv(tree, raw, weight, history, topology=topology, context=context),
                                           conv(tree, raw, weight, prefix, topology=topology), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
