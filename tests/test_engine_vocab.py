"""Vocabulary-parallel greedy agrees with the full vocabulary, including ties."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None, "requires PyTorch")
class VocabTests(unittest.TestCase):
    def compare(self, logits, decodable=None):
        from engine.base.comm import LocalTP
        from engine.modules.vocab import argmax
        tp = LocalTP(4)
        width = logits.shape[-1] // 4
        before = logits.clone()
        packets = []

        def select(comm):
            original = comm.all_reduce_max
            def record(value):
                packets.append((tuple(value.shape), value.dtype))
                return original(value)
            comm.all_reduce_max = record
            comm.all_gather = lambda *a, **k: self.fail("greedy gathered the full vocabulary")
            return argmax(logits[:, comm.rank*width:(comm.rank+1)*width], comm, comm.rank*width, decodable)

        for result in tp.run(select):
            self.assertTrue(torch.equal(result, logits[:, :decodable].argmax(-1)))
        self.assertEqual(packets, [((logits.shape[0],), torch.int64)] * 4)
        torch.testing.assert_close(logits, before, rtol=0, atol=0, equal_nan=True)

    def test_random_strided_and_masked_shards(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            x = torch.randn(7, 96, generator=torch.Generator().manual_seed(71)).to(dtype)[:, ::2]
            for decodable in (None, 48, 29, 12, 3):
                with self.subTest(dtype=dtype, decodable=decodable):
                    self.compare(x, decodable)

    def test_ties_signed_zero_infinity_and_nan_match_argmax(self):
        x = torch.tensor([
            [2., 2., 0., 2., 2., 2., 1., 2.],
            [-0., -1., 0., -0., 0., -2., -0., 0.],
            [float("-inf")] * 8,
            [0., float("inf"), 0., float("inf"), -1., -1., -1., -1.],
            [-1., float("nan"), 0., float("nan"), -1., -1., -1., -1.],
            [-8., -7., -6., -5., -4., -3., -2., -1.],
        ])
        for decodable in (None, 5, 1):
            self.compare(x, decodable)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
    def test_device_selection_is_graph_safe_and_does_not_consume_rng(self):
        from engine.base.comm import Comm
        from engine.modules.vocab import argmax
        x = torch.empty(3, 32, device="cuda")
        state = torch.cuda.get_rng_state().clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            argmax(x, Comm(), 0, 29)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = argmax(x, Comm(), 0, 29)
        for offset in (0, 7, 28):
            x.fill_(-1); x[:, offset] = 2; x[:, 31] = 999
            graph.replay()
            self.assertEqual(result.tolist(), [offset]*3)
        self.assertTrue(torch.equal(state, torch.cuda.get_rng_state()))
        graph.reset()


if __name__ == "__main__":
    unittest.main()
