"""The direct packet consumer must reproduce unpack then FP8 quantization."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and importlib.util.find_spec("triton"), "requires CUDA and Triton")
class PrefillConsumerTests(unittest.TestCase):
    @staticmethod
    def received(rows, seed=1):
        from engine.kernels.prefill_collectives import PrefillCollectives
        torch.manual_seed(seed)
        packets = []
        for rank in range(4):
            x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)*(2**rank)
            x[0].zero_()
            amplitudes = torch.exp2(torch.arange(32, device="cuda") % 16 - 8)
            x[-1].copy_(amplitudes.repeat_interleave(128))
            payload, _ = PrefillCollectives.pack(x, x.numel())
            packets.append(payload)
        return torch.cat(packets)

    @staticmethod
    def baseline(received, rows):
        from engine.kernels.prefill_collectives import BLOCK
        from engine.kernels.prefill_collectives.kernels import _unpack_gather
        from engine.kernels.dense.fp8 import quantize
        x = torch.empty((4*rows, 4096), device=received.device, dtype=torch.bfloat16)
        _unpack_gather[(x.numel()//BLOCK,)](
            received.view(torch.float8_e4m3fn), received.view(torch.float32), x,
            rows*4096, received.numel()//4, BLOCK=BLOCK)
        return quantize(x)

    def exact(self, left, right):
        for a, b in zip(left, right):
            self.assertEqual((a.shape, a.dtype), (b.shape, b.dtype))
            self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    def test_padded_packets_and_gemm_group_scales(self):
        from engine.kernels.prefill_collectives.consumer import quantize_gather
        for rows in (32, 35, 1024, 1728):
            with self.subTest(local_rows=rows):
                received = self.received(rows)
                before = received.clone()
                expected = self.baseline(received, rows)
                self.exact(quantize_gather(received, rows), expected)
                self.assertTrue(torch.equal(received, before))
                self.assertGreater(torch.unique(expected[1][-1]).numel(), 1)

    def test_graph_replay_reads_new_packet_values_and_scales(self):
        from engine.kernels.prefill_collectives.consumer import quantize_gather
        received = self.received(32)
        quantize_gather(received, 32)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = quantize_gather(received, 32)
        try:
            for seed in (2, 3):
                received.copy_(self.received(32, seed))
                expected = self.baseline(received, 32)
                graph.replay()
                self.exact(actual, expected)
        finally:
            graph.reset()


if __name__ == "__main__":
    unittest.main()
