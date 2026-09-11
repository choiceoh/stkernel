"""Return-only one-warp pooling preserves FP8 bytes and FP32 power-of-two scales."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and importlib.util.find_spec("triton") is not None, "requires CUDA PyTorch and Triton")
class PoolCompressTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.kpool import compress_pool_keys, kpool_compress_and_write_cache
        self.compress = compress_pool_keys
        self.cache_compress = kpool_compress_and_write_cache

    def compare(self, keys, scores, ape):
        originals = tuple(x.clone() for x in (keys, scores, ape))
        dummy = torch.zeros((1, 64, 132), device="cuda", dtype=torch.uint8)
        expected = self.cache_compress(dummy, keys, scores, ape,
            torch.arange(keys.shape[0], device="cuda", dtype=torch.int64), keys.shape[1],
            return_compressed=True, write_cache=False)
        actual = self.compress(keys, scores, ape)
        self.assertEqual(actual[0].shape, (keys.shape[0], 128))
        self.assertEqual(actual[1].shape, (keys.shape[0], 1))
        self.assertTrue(torch.equal(expected[0].view(torch.uint8), actual[0].view(torch.uint8)))
        self.assertTrue(torch.equal(expected[1].view(-1), actual[1].view(-1)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(originals, (keys, scores, ape))))

    def test_exact_random_shapes_dtypes_and_scales(self):
        generator = torch.Generator(device="cuda").manual_seed(271)
        for pools in (0, 1, 2, 31, 32, 33, 64, 512):
            for size in (1, 4, 8):
                for magnitude in (1e-6, 1., 100.):
                    with self.subTest(pools=pools, size=size, magnitude=magnitude):
                        keys = (torch.randn(pools, size, 128, device="cuda", generator=generator) * magnitude).bfloat16()
                        scores = torch.randn_like(keys) * 10
                        ape = torch.randn(size, 128, device="cuda", generator=generator)
                        self.compare(keys, scores, ape)
                        self.compare(keys, scores.float(), ape)

    def test_strided_views_match_without_materializing_inputs(self):
        for stride in (1, 2, 3):
            keys = torch.randn(14, 8, 128 * stride, device="cuda", dtype=torch.bfloat16)[::2, ::2, ::stride]
            scores = torch.randn(14, 8, 128 * stride, device="cuda")[::2, ::2, ::stride]
            ape = torch.randn(8, 128 * stride, device="cuda")[::2, ::stride]
            self.compare(keys, scores, ape)

    def test_uniform_pool_matches_independent_quantization_reference(self):
        from engine.modules.sparse_indexer import fwht128_quant
        # Exactly representable pooled vectors avoid a separate softmax/FMA
        # tolerance question while checking both BF16 rounding boundaries.
        patterns = torch.stack((torch.zeros(128), torch.ones(128), -torch.ones(128),
                                torch.arange(128) % 2 * 2 - 1, torch.eye(128)[0])).cuda().bfloat16()
        keys = patterns[:, None].expand(-1, 4, -1)
        actual = self.compress(keys, torch.zeros_like(keys), torch.zeros(4, 128, device="cuda"))
        expected = fwht128_quant(patterns)
        self.assertTrue(torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8)))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_graph_replay_reads_changed_keys_scores_and_bias(self):
        keys = torch.randn(6, 4, 128, device="cuda", dtype=torch.bfloat16)
        scores, ape = torch.zeros_like(keys), torch.zeros(4, 128, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.compress(keys, scores, ape)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = self.compress(keys, scores, ape)
        for _ in range(3):
            keys.normal_()
            scores.normal_(std=10)
            ape.normal_()
            expected = self.compress(keys, scores, ape)
            graph.replay()
            self.assertTrue(torch.equal(result[0].view(torch.uint8), expected[0].view(torch.uint8)))
            self.assertTrue(torch.equal(result[1], expected[1]))


if __name__ == "__main__":
    unittest.main()
