"""The captured decode step's DSA glue kernels (engine/kernels/indexer.py) move the bytes and compute the
integers their torch references (engine/modules/sparse_indexer.py) do -- byte for byte, so the served and the
reference lane tables agree (45차, the C=4 question, third fold). Runs on a GPU, or on the CPU under
TRITON_INTERPRET=1 (the interpreter executes the same kernels)."""
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
class IndexerRowsKernelTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels import indexer as kernels
        from engine.modules import sparse_indexer as reference
        self.kernels, self.reference = kernels, reference
        self.g = torch.Generator(device=DEVICE).manual_seed(933)

    def table(self, rows, blocks, pages):
        return torch.randint(0, pages, (rows, blocks), device=DEVICE, dtype=torch.int32, generator=self.g)

    def test_row_lengths(self):
        contexts = torch.tensor([0, 3, 4095, 130000, 7], device=DEVICE)
        for t, kp in ((1, 4), (7, 4), (7, 8), (6, 4)):
            with self.subTest(tokens=t, pool=kp):
                got = self.kernels.row_lengths(contexts, t, kp)
                want = self.reference.row_lengths(contexts, t, kp)
                for a, b in zip(got, want):
                    self.assertEqual(a.dtype, torch.int32)
                    self.assertTrue(torch.equal(a, b))

    def test_latent_write_rows(self):
        rows, block, stride, offset = 3, 16, 37, 5
        table = self.table(rows, 64, 100)
        contexts = torch.tensor([0, 15, 1000], device=DEVICE)
        for t in (1, 7):
            for dtype, width in ((torch.float8_e4m3fn, 512), (torch.bfloat16, 64), (torch.float32, 32)):
                with self.subTest(tokens=t, dtype=dtype):
                    values = torch.randn(rows * t, width, device=DEVICE, generator=self.g).to(dtype)
                    latent = torch.randn(100 * stride + block + offset, width, device=DEVICE, generator=self.g).to(dtype)
                    want = latent.clone()
                    self.reference.latent_write_rows(values, want, table, block, stride, offset, contexts, t)
                    self.kernels.latent_write_rows(values, latent, table, block, stride, offset, contexts, t)
                    self.assertTrue(torch.equal(latent.view(torch.uint8), want.view(torch.uint8)))

    def test_gather_candidates(self):
        """Keys are 128-byte records with their fp32 scale in the record's tail, as the paged cache lays them out."""
        rows, per, stride, offset = 3, 4, 6, 1
        table = self.table(rows, 300, 500)
        records = 500 * stride + offset + per
        paged = torch.randint(0, 256, (records * 132,), device=DEVICE, dtype=torch.uint8, generator=self.g)
        keys = paged.as_strided((records, 128), (132, 1)).view(torch.float8_e4m3fn)
        base = paged.view(torch.float32)
        scales = base.as_strided((records,), (33,), 32)
        scales.copy_(torch.rand(records, device=DEVICE, generator=self.g))          # finite values: bytes compare as floats
        for n_cand in (1, 63, 64, 65, 1000):
            with self.subTest(n_cand=n_cand):
                got_k, got_s = self.kernels.gather_candidates(keys, scales, table, per, stride, offset, n_cand)
                want_k, want_s = self.reference.gather_candidates(keys, scales, table, per, stride, offset, n_cand)
                self.assertEqual((got_k.dtype, got_k.shape, got_s.shape), (torch.float8_e4m3fn, (rows, n_cand, 128), (rows, n_cand)))
                self.assertTrue(got_k.is_contiguous() and got_s.is_contiguous())
                self.assertTrue(torch.equal(got_k.view(torch.uint8), want_k.view(torch.uint8)))
                self.assertTrue(torch.equal(got_s, want_s))

    def test_pool_window(self):
        rows, w, kp = 5, 9, 4
        contexts = torch.tensor([0, 1, 2, 3, 100], device=DEVICE)
        for t in (1, 6, 7):
            for d in (8, 128):
                for strided in (False, True):
                    with self.subTest(tokens=t, d=d, strided=strided):
                        max_pools = (kp - 1 + t) // kp
                        tails = torch.randn(rows, w, 2, d, device=DEVICE, generator=self.g).to(torch.bfloat16)
                        wide = torch.randn(rows * t, 3 * d, device=DEVICE, generator=self.g).to(torch.bfloat16)
                        k = (wide[:, :d] if strided else wide[:, :d].contiguous()).view(rows, t, d)
                        gate = (wide[:, d:2 * d] if strided else wide[:, d:2 * d].contiguous()).view(rows, t, d)
                        got = self.kernels.pool_window(tails, k, gate, contexts, kp, max_pools)
                        want = self.reference.pool_window(tails, k, gate, contexts, kp, max_pools)
                        for a, b in zip(got, want):
                            self.assertEqual(a.shape, (rows * max_pools, kp, d))
                            self.assertTrue(torch.equal(a.view(torch.uint8), b.contiguous().view(torch.uint8)))

    def test_pool_addresses(self):
        rows, kp, per, stride, offset, cap = 6, 4, 192, 2994, 4, 2048
        table = self.table(rows, 12, 40)
        contexts = torch.tensor([0, 3, 4, 7, 4095, 8188], device=DEVICE)
        for t in (1, 7):
            with self.subTest(tokens=t):
                max_pools = (kp - 1 + t) // kp
                got = self.kernels.pool_addresses(contexts, table, per, stride, offset, kp, t, max_pools, cap)
                want = self.reference.pool_addresses(contexts, table, per, stride, offset, kp, t, max_pools, cap)
                for a, b in zip(got, want):
                    self.assertEqual(a.dtype, torch.int64)
                    self.assertTrue(torch.equal(a, b))

    @unittest.skipIf(INTERPRET, "graph replay needs a GPU")
    def test_graph_replay_reads_new_contexts_and_tables(self):
        """A captured launch of each kernel reads the contexts and block rows of the replay, not the capture."""
        rows, t, kp, per, stride, offset = 2, 7, 4, 4, 6, 1
        table = self.table(rows, 300, 500)
        contexts = torch.tensor([3, 700], device=DEVICE)
        records = 500 * stride + offset + per
        paged = torch.randint(0, 256, (records * 132,), device=DEVICE, dtype=torch.uint8, generator=self.g)
        keys = paged.as_strided((records, 128), (132, 1)).view(torch.float8_e4m3fn)
        scales = paged.view(torch.float32).as_strided((records,), (33,), 32)
        scales.copy_(torch.rand(records, device=DEVICE, generator=self.g))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run = lambda: (self.kernels.row_lengths(contexts, t, kp), self.kernels.gather_candidates(keys, scales, table, per, stride, offset, 100),
                           self.kernels.pool_addresses(contexts, table, per, stride, offset, kp, t, 2, 2048))
            run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        contexts.copy_(torch.tensor([40, 1201], device=DEVICE))
        table.copy_(self.table(rows, 300, 500))
        graph.replay()
        torch.cuda.synchronize()
        want = (self.reference.row_lengths(contexts, t, kp), self.reference.gather_candidates(keys, scales, table, per, stride, offset, 100),
                self.reference.pool_addresses(contexts, table, per, stride, offset, kp, t, 2, 2048))
        for got_pair, want_pair in zip(out, want):
            for a, b in zip(got_pair, want_pair):
                self.assertTrue(torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a,
                                            b.view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b))


if __name__ == "__main__":
    unittest.main()
