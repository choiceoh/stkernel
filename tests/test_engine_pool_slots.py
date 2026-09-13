"""Pool-level sorting must exactly preserve the expanded-token oracle."""
import importlib.util
import os
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"


@unittest.skipUnless(torch is not None and (torch.cuda.is_available() or INTERPRET)
                     and importlib.util.find_spec("triton") is not None, "requires CUDA PyTorch and Triton")
class PoolSlotTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.indexer import pool_slots
        from engine.modules.sparse_indexer import pool_slots as reference   # the expanded-token oracle, 2-D tables included
        self.fused, self.reference = pool_slots, reference

    def compare(self, ids, lengths, pool, table=None, block=16, stride=3072, offset=512):
        rows, groups = ids.shape
        width = groups * pool + pool - 1
        originals = ids.clone(), lengths.clone()
        buffers = [torch.full((rows + 2, width * 2 + 7), -777, device=DEVICE, dtype=torch.int32) for _ in range(2)]
        counts = [torch.full((2 * rows + 3,), -777, device=DEVICE, dtype=torch.int32) for _ in range(2)]
        for fn, buf, count in zip((self.reference, self.fused), buffers, counts):
            fn(ids, lengths, pool, table, block, stride, offset,
               buf[1:rows + 1, 2:2 * width + 2:2], count[1:2 * rows + 1:2])
        self.assertTrue(torch.equal(buffers[0], buffers[1]))
        self.assertTrue(torch.equal(counts[0], counts[1]))
        self.assertTrue(torch.equal(ids, originals[0]) and torch.equal(lengths, originals[1]))

    def test_exact_duplicates_invalid_pools_tails_and_strides(self):
        g = torch.Generator(device=DEVICE).manual_seed(930)
        lengths = torch.tensor([0, 1, 3, 4, 5, 7, 8, 9, 2047, 2048, 2049, 65535], device=DEVICE, dtype=torch.int32)
        strided = torch.zeros(24, device=DEVICE, dtype=torch.int32)
        strided[::2] = lengths
        lengths = strided[::2]
        table = torch.randperm(8192, device=DEVICE, generator=g).int()[::2]
        for groups in (1, 2, 31, 129, 512, 513):
            for pool in (1, 2, 4, 8):
                with self.subTest(groups=groups, pool=pool):
                    ids = torch.randint(-8, 20000, (12, groups * 2 + 1), device=DEVICE, dtype=torch.int32, generator=g)[:, 1::2]
                    ids[:, ::3] = 0
                    ids[:, 1::3] = lengths[:, None] // pool  # future/incomplete pools
                    ids[-1, :groups // 2] = 2               # a long duplicate run
                    self.compare(ids, lengths, pool, table)
                    self.compare(ids, lengths, pool)

    def test_a_block_table_per_group_of_rows_matches_one_group_launches(self):
        """A captured decode step finalizes every row in one launch over the rows' own block rows (2-D table,
        `tokens` rows to a row): byte-equal to one launch per group with that group's row, kernel and oracle."""
        g = torch.Generator(device=DEVICE).manual_seed(931)
        groups, tokens, pool, blocks = 3, 7, 4, 512
        rows = groups * tokens
        ids = torch.randint(-8, 2100, (rows, 33), device=DEVICE, dtype=torch.int32, generator=g)
        lengths = torch.randint(0, blocks * 16, (rows,), device=DEVICE, dtype=torch.int32, generator=g)
        lengths[::5] = 0
        width = 33 * pool + pool - 1
        for strided in (False, True):
            table = torch.randperm(8192, device=DEVICE, generator=g).int()[:groups * blocks * 2].view(groups, blocks * 2)
            table = table[:, ::2] if strided else table[:, :blocks].contiguous()
            with self.subTest(strided=strided):
                outs = [torch.full((rows, width), -777, device=DEVICE, dtype=torch.int32) for _ in range(3)]
                counts = [torch.full((rows,), -777, device=DEVICE, dtype=torch.int32) for _ in range(3)]
                self.fused(ids, lengths, pool, table, 16, 3072, 512, outs[0], counts[0], tokens=tokens)
                self.reference(ids, lengths, pool, table, 16, 3072, 512, outs[1], counts[1], tokens=tokens)
                for i in range(groups):
                    sl = slice(i * tokens, (i + 1) * tokens)
                    self.fused(ids[sl], lengths[sl], pool, table[i], 16, 3072, 512, outs[2][sl], counts[2][sl])
                for out, count in zip(outs[1:], counts[1:]):
                    self.assertTrue(torch.equal(outs[0], out))
                    self.assertTrue(torch.equal(counts[0], count))

    def test_empty_shapes_and_all_padding_overwrite(self):
        for rows, groups in ((0, 512), (3, 0), (6, 512)):
            self.compare(torch.full((rows, groups), -1, device=DEVICE, dtype=torch.int32),
                         torch.zeros(rows, device=DEVICE, dtype=torch.int32), 4,
                         torch.tensor([7, 2], device=DEVICE, dtype=torch.int32))

    def test_large_integer_positions(self):
        ids = torch.tensor([[2**27 + 1, 2**27, -1, 2**27 + 1]], device=DEVICE, dtype=torch.int32)
        lengths = torch.tensor([2**29 + 11], device=DEVICE, dtype=torch.int32)
        self.compare(ids, lengths, 4)
        self.compare(ids, lengths, 4, torch.arange(1024, device=DEVICE, dtype=torch.int32).flip(0), 2**20, 2**21, 73)

    @unittest.skipIf(INTERPRET, "graph replay needs a real GPU")
    def test_graph_replay_reads_pools_lengths_and_recycled_blocks(self):
        ids = torch.tensor([[2, 0, -1, 2], [0, 1, 2, -1]], device=DEVICE, dtype=torch.int32)
        lengths = torch.tensor([15, 8], device=DEVICE, dtype=torch.int32)
        table = torch.tensor([7, 2], device=DEVICE, dtype=torch.int32)
        out = torch.empty((2, 19), device=DEVICE, dtype=torch.int32)
        counts = torch.empty(2, device=DEVICE, dtype=torch.int32)
        args = ids, lengths, 4, table, 16, 512, 32, out, counts
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.fused(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.fused(*args)
        expected, valid = torch.empty_like(out), torch.empty_like(counts)
        for change in range(3):
            if change:
                ids.fill_(change - 1)
                lengths.copy_(torch.tensor([3, 21] if change == 1 else [0, 0], device=DEVICE, dtype=torch.int32))
                table.copy_(torch.tensor([3, 9], device=DEVICE, dtype=torch.int32))
            graph.replay()
            self.reference(*args[:-2], expected, valid)
            self.assertTrue(torch.equal(out, expected) and torch.equal(counts, valid))


if __name__ == "__main__":
    unittest.main()
