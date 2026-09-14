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

    def test_real_rows_are_cropped_before_shared_quantization_output(self):
        from engine.kernels.prefill_collectives.consumer import quantize_gather
        received = self.received(2049)
        q, s = self.baseline(received, 2049)
        for rows in (8193, 8194, 8195, 8196):
            self.exact(quantize_gather(received, 2049, real_rows=rows), (q[:rows], s[:rows]))
        for invalid in (8192, 8197, True):
            with self.assertRaises(ValueError):
                quantize_gather(received, 2049, real_rows=invalid)

    def test_packet_router_preserves_logits_and_route_selection(self):
        from engine.kernels.prefill_collectives import BLOCK
        from engine.kernels.prefill_collectives.kernels import _unpack_gather
        from engine.kernels.prefill_router import router_logits, router_packet_logits
        from engine.kernels.glm_pointwise import route_weights
        from engine.modules.prefill_packets import PacketBatch, PacketGeometry
        torch.manual_seed(895)
        gate = (torch.randn(288, 4096, device='cuda') / 64).bfloat16()
        bias = torch.linspace(-.1, .1, 288, device='cuda')
        for rows in (8193, 8194, 8195, 9216, 32768):
            g = PacketGeometry(rows, (rows+3)//4)
            received = self.received(g.local_rows)
            x = torch.empty((g.padded_rows, 4096), device='cuda', dtype=torch.bfloat16)
            _unpack_gather[(x.numel()//BLOCK,)](
                received.view(torch.float8_e4m3fn), received.view(torch.float32), x,
                g.local_elements, g.stride, BLOCK=BLOCK)
            expected = router_logits(x[:rows], gate)
            actual = router_packet_logits(PacketBatch(received, g), gate)
            if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                from engine.kernels.prefill_router import _router_gemm
                import triton
                def difference(value):
                    unequal = value.view(torch.int32) != expected.view(torch.int32)
                    positions = unequal.nonzero()
                    return dict(changed=int(unequal.sum()),
                        first_positions=positions[:8].tolist(),
                        max_abs=float((value-expected).abs().max()),
                        finite=bool(torch.isfinite(value).all()))
                alternatives = {}
                for stages in (1, 2):
                    check = torch.empty_like(expected)
                    _router_gemm[(triton.cdiv(rows,64)*triton.cdiv(288,64),)](
                        received.view(torch.float8_e4m3fn), gate, check, rows,
                        BM=64, BN=64, BK=64, Scales=received.view(torch.float32),
                        LOCAL_ROWS=g.local_rows, PACKET_BYTES=g.stride, PACKETS=True,
                        num_warps=4, num_stages=stages, enable_fp_fusion=False)
                    alternatives[stages] = difference(check)
                self.fail(f'packet router rows={rows}: default={difference(actual)}, '
                          f'pipeline_diagnostics={alternatives}')
            self.exact((actual,), (expected,))
            self.exact(route_weights(actual, bias, 8, 2.5), route_weights(expected, bias, 8, 2.5))


if __name__ == "__main__":
    unittest.main()
