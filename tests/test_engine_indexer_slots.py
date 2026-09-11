"""Exact integer contracts for the fused sparse-indexer slot finalizer."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and importlib.util.find_spec("triton") is not None, "requires CUDA PyTorch and Triton")
class IndexerSlotTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.indexer import indexer_slots
        from engine.modules.sparse_indexer import indexer_slots as reference
        self.fused, self.reference = indexer_slots, reference

    def compare(self, tokens, table, block_size, block_stride, offset):
        rows, width = tokens.shape
        original = tokens.clone()
        # Noncontiguous output views inside sentinel buffers expose row/column
        # stride bugs, output overrun, and unwritten padding on repeat calls.
        buffers = [torch.full((rows + 2, 2 * width + 7), -777, dtype=torch.int32, device="cuda") for _ in range(2)]
        counts = [torch.full((2 * rows + 3,), -777, dtype=torch.int32, device="cuda") for _ in range(2)]
        for fn, buf, count in zip((self.reference, self.fused), buffers, counts):
            fn(tokens, table, block_size, block_stride, offset,
               buf[1:rows + 1, 2:2 * width + 2:2], count[1:2 * rows + 1:2])
        self.assertTrue(torch.equal(buffers[0], buffers[1]))
        self.assertTrue(torch.equal(counts[0], counts[1]))
        self.assertTrue(torch.equal(tokens, original))

    def test_exact_sort_count_and_mapping_for_non_power_of_two_widths(self):
        g = torch.Generator(device="cuda").manual_seed(402)
        table = torch.randperm(80, device="cuda", generator=g).to(torch.int32)[::2]
        for rows, width in ((1, 1), (6, 11), (24, 129), (6, 2048), (6, 2051), (256, 2051), (2, 4099)):
            with self.subTest(rows=rows, width=width):
                tokens = torch.randint(-8, 40 * 16, (rows, width * 2 + 3), device="cuda", dtype=torch.int32, generator=g)[:, 1:2 * width + 1:2]
                tokens[0] = -1
                if rows > 1:
                    tokens[1, ::3] = 17  # duplicate positions must retain exact ordering
                self.compare(tokens, table, 16, 3072, 1024)
                self.compare(tokens, None, 1, 1, 0)

    def test_identity_and_large_integer_positions_are_exact(self):
        tokens = torch.tensor([[2**25 + 7, 0, -1, 2**25 + 6, 2**25 + 7],
                               [-1, -1, -1, -1, -1]], device="cuda", dtype=torch.int32)
        table = torch.arange(64, device="cuda", dtype=torch.int32).flip(0)
        self.compare(tokens, None, 1, 1, 0)
        self.compare(tokens, table, 2**20, 2**21, 73)

    def test_empty_shapes_and_all_padding_overwrite_outputs(self):
        table = torch.tensor([7, 2], device="cuda", dtype=torch.int32)
        for rows, width in ((0, 11), (3, 0), (6, 2051)):
            self.compare(torch.full((rows, width), -1, dtype=torch.int32, device="cuda"), table, 16, 512, 32)

    def test_graph_replay_reads_changed_inputs_and_block_mappings(self):
        tokens = torch.tensor([[17, 0, 31, -1, 2, 17]], device="cuda", dtype=torch.int32)
        table = torch.tensor([7, 2], device="cuda", dtype=torch.int32)
        out, counts = torch.empty_like(tokens), torch.empty(1, device="cuda", dtype=torch.int32)
        args = tokens, table, 16, 512, 32, out, counts
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.fused(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.fused(*args)
        expected, valid = torch.empty_like(out), torch.empty_like(counts)
        for changed in (False, True):
            if changed:
                tokens.fill_(-1)
                tokens[0, 0] = 16
                table.copy_(torch.tensor([3, 9], device="cuda", dtype=torch.int32))
            graph.replay()
            self.reference(tokens, table, 16, 512, 32, expected, valid)
            self.assertTrue(torch.equal(out, expected))
            self.assertTrue(torch.equal(counts, valid))


if __name__ == "__main__":
    unittest.main()
