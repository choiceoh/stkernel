"""Exact storage and replay contracts for GB10 indexer launch geometry."""
import ast
import importlib.util
from pathlib import Path
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class QuantGeometryTests(unittest.TestCase):
    def test_qualified_ranges_and_unqualified_large_prefill(self):
        # Read the pure launch rule without importing CUDA/Triton on the CPU.
        source = Path(__file__).resolve().parents[1] / "engine/kernels/kpool.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_fwht_quant_config")
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        config = scope[node.name]
        for rows in (0, 1, 32, 192, 768, 1023, 1024):
            self.assertEqual(config(rows), (1, 1))
        for rows in (1025, 1536, 2048, 8192, 65535, 65536):
            self.assertEqual(config(rows), (8, 1))
        for rows in (65537, 131072, 1048576):
            self.assertEqual(config(rows), (32, 2))


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and importlib.util.find_spec("triton") is not None,
                     "requires CUDA PyTorch and Triton")
class QuantKernelTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.kpool import _fwht_quant_kernel, fwht128_quant_fp8
        self.kernel, self.quant = _fwht_quant_kernel, fwht128_quant_fp8

    def baseline(self, q):
        out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        scales = torch.empty(q.shape[0], 1, device=q.device, dtype=torch.float32)
        if q.shape[0]:
            self.kernel[((q.shape[0] + 31) // 32,)](
                q, out, scales, q.shape[0], BLOCK_R=32, num_warps=2)
        return out, scales

    def exact(self, actual, expected):
        self.assertEqual(actual[0].shape, expected[0].shape)
        self.assertEqual(actual[1].shape, expected[1].shape)
        self.assertTrue(torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8)))
        self.assertTrue(torch.equal(actual[1].view(torch.int32), expected[1].view(torch.int32)))

    def test_random_shape_boundaries_and_magnitudes(self):
        generator = torch.Generator(device="cuda").manual_seed(8128)
        for rows in (0, 1, 2, 7, 31, 32, 33, 191, 192, 193, 767, 768,
                     1023, 1024, 1025, 1536, 2048, 4096, 8192, 16384,
                     32768, 65535, 65536, 65537, 131072):
            for magnitude in (1e-20, 1e-6, 1., 100., 1e12):
                with self.subTest(rows=rows, magnitude=magnitude):
                    q = (torch.randn(rows, 128, generator=generator, device="cuda") * magnitude).bfloat16()
                    before = q.clone()
                    self.exact(self.quant(q), self.baseline(q))
                    self.assertTrue(torch.equal(q.view(torch.int16), before.view(torch.int16)))

    def test_finite_patterns_and_power_of_two_boundaries(self):
        from engine.modules.sparse_indexer import fwht128_quant
        patterns = torch.cat((torch.zeros(1, 128), torch.full((1, 128), -0.),
            torch.ones(1, 128), -torch.ones(1, 128),
            (torch.arange(128) % 2 * 2 - 1).view(1, 128), torch.eye(128)))
        patterns = patterns.to(device="cuda", dtype=torch.bfloat16)
        for exponent in (-120, -30, -10, 0, 10, 30):
            q = patterns * (2. ** exponent)
            self.exact(self.quant(q), self.baseline(q))
            self.exact(self.quant(q), fwht128_quant(q))
        # BF16 values around scale boundaries, with both signs and cancellation.
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        values = torch.where(torch.isfinite(values) & (values.abs() < 1e30), values, 0.)
        q = values.reshape(-1, 128).contiguous()
        self.exact(self.quant(q), self.baseline(q))

    def test_graph_replay_reads_changed_queries_across_cutovers(self):
        for rows in (32, 192, 768, 1024, 1025, 65536, 65537):
            q = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
            self.quant(q)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = self.quant(q)
            for iteration in range(4):
                if iteration == 0: q.zero_()
                else: q.normal_(std=10. ** (iteration - 2))
                graph.replay()
                self.exact(actual, self.baseline(q))

    def test_independent_streams_and_input_contract(self):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        queries = [torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16) for rows in (192, 1536)]
        expected = [self.baseline(q) for q in queries]
        for q in queries: self.quant(q)
        for stream in streams: stream.wait_stream(torch.cuda.current_stream())
        outputs = []
        for stream, q in zip(streams, queries):
            with torch.cuda.stream(stream): outputs.append(self.quant(q))
        torch.cuda.synchronize()
        for actual, ref in zip(outputs, expected): self.exact(actual, ref)
        for q in (torch.empty(2, 128, device="cuda"),
                  torch.empty(2, 127, device="cuda", dtype=torch.bfloat16),
                  torch.empty(4, 128, device="cuda", dtype=torch.bfloat16)[::2]):
            with self.assertRaises(AssertionError): self.quant(q)


if __name__ == "__main__": unittest.main()
