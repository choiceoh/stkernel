"""Default dispatch, arena accounting and capture contracts for head/FC."""
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch


class ServingContracts(unittest.TestCase):
    def test_fp8_default_reader_bypasses_deep_gemm_and_preserves_head_padding(self):
        from engine.kernels.dense import FP8Linear
        import triton  # load extension modules before the scoped sys.modules mock
        from engine.kernels.dense import fp8
        calls = []
        class Reader:
            def __call__(self, x, *, out=None, decode=False):
                calls.append((x, out, decode))
                return torch.full((x.shape[0], 128), 3, dtype=torch.bfloat16) if out is None else out.fill_(3)
            def project_quantized(self, q, scale, *, out=None):
                return self(q, out=out)
        layer = FP8Linear.__new__(FP8Linear)
        layer.rows, layer.cols = 127, 128
        layer.cublas, layer.observer, layer.executed = Reader(), None, False
        x = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
        with patch.dict('sys.modules', {'deep_gemm': NS()}):
            y = layer(x, decode=True)
            self.assertEqual(y.shape, (2, 4, 127))
            out = torch.empty(8, 128, dtype=torch.bfloat16)
            self.assertEqual(layer(x, out=out).data_ptr(), out.data_ptr())
            q = torch.zeros(8, 128, dtype=torch.float8_e4m3fn)
            self.assertEqual(layer.project_quantized(q, torch.ones(8, 1)).shape, (8, 127))
        self.assertTrue(calls[0][2])
        self.assertTrue(layer.executed)
        with self.assertRaisesRegex(RuntimeError, 'relocate'):
            layer.consume_weight(None)

    def test_fc_phase_reaches_the_shared_cublas_reader(self):
        from engine.kernels.dense import DenseLinear
        from engine.profiles.glm53.drafter import Drafter
        class Reader:
            cublas = object()
            def __call__(self, x, *, decode=False):
                self.decode = decode
                return x
        layer = DenseLinear.__new__(DenseLinear)
        layer.rows = layer.cols = 128
        layer.observer = None
        layer.decode_precision = 'fp8'
        layer.decode_fp8 = None
        layer.fp8, layer.executed = Reader(), 0
        drafter = Drafter.__new__(Drafter)
        drafter.dense = {'fc.weight': layer}
        x = torch.zeros(8, 128, dtype=torch.bfloat16)
        drafter.context_linear(x, decode=True, observe=False)
        self.assertTrue(layer.fp8.decode)
        drafter.context_linear(x, decode=False)
        self.assertFalse(layer.fp8.decode)

    def test_declared_memory_covers_shared_and_separate_decode_packs(self):
        from engine.profiles.glm53.cublas import resident_bytes
        from engine.profiles.glm53.draft_policy import DraftPolicy
        F = NS(vocab_local=38720, hidden=4096)
        D = NS(hidden=4096, target_layers=range(5))
        head = 38784*4096//32
        self.assertEqual(resident_bytes(F), head)
        self.assertEqual(resident_bytes(F, D, DraftPolicy('w4')), head+2621440)
        self.assertEqual(resident_bytes(F, D, DraftPolicy('fp8')), head+89128960+4096*5*384)
        self.assertEqual(resident_bytes(F, D, DraftPolicy('fp8', 'decode')), head+91750400+4096*5*384)

    def test_execution_proof_rejects_an_unexecuted_default_reader(self):
        from engine.profiles.glm53.cublas import execution_report
        def reader(split, executed):
            return NS(cublas=NS(split_decode=split, executed=set(executed), report=lambda: {'backend': 'cublaslt'}))
        net = NS(cublas_readers={'head': reader(False, ['direct']), 'fc_prefill': reader(True, ['direct'])})
        with self.assertRaisesRegex(RuntimeError, 'fc_prefill'):
            execution_report(net)
        net.cublas_readers['fc_prefill'].cublas.executed.add('split_decode')
        with self.assertRaisesRegex(RuntimeError, 'split_decode_norm'):
            execution_report(net)
        net.cublas_readers['fc_prefill'].cublas.executed.add('split_decode_norm')
        self.assertEqual(len(execution_report(net)), 2)
        net.cublas_readers['head'].cublas = None
        with self.assertRaisesRegex(RuntimeError, 'head'):
            execution_report(net)

    def test_fc_normalization_keeps_decode_phase_bias_and_observation(self):
        from engine.kernels.dense import DenseLinear
        from engine.profiles.glm53.drafter import Drafter
        class Reader:
            cublas = object()
            def __call__(self, x, *, decode=False, normalization=None):
                self.options = decode, normalization
                return x
        layer = DenseLinear.__new__(DenseLinear)
        layer.rows = layer.cols = 128
        observed = []
        layer.observer = lambda x, mask: observed.append(mask)
        layer.decode_precision, layer.executed = 'fp8', 0
        layer.fp8, layer.decode_fp8 = Reader(), Reader()
        d = Drafter.__new__(Drafter)
        d.dense, d.F = {'fc.weight': layer}, NS(rms_eps=1e-6)
        d.p = {'hidden_norm.weight': torch.ones(128, dtype=torch.bfloat16)}
        d.fc_bias = torch.ones(128)
        x, mask = torch.zeros(8, 128, dtype=torch.bfloat16), torch.ones(8, dtype=torch.bool)
        d.context_projected_norm(x, mask, decode=True)
        self.assertIs(observed[0], mask)
        self.assertTrue(layer.decode_fp8.options[0])
        self.assertIs(layer.decode_fp8.options[1][0], d.p['hidden_norm.weight'])
        self.assertIs(layer.decode_fp8.options[1][2], d.fc_bias)
        d.context_projected_norm(x, mask, decode=True, observe=False)
        self.assertEqual(len(observed), 1)
        layer.decode_fp8 = None
        d.context_projected_norm(x, decode=True, observe=False)
        self.assertIs(layer.fp8.options[1][2], d.fc_bias)


@unittest.skipUnless(os.environ.get('ST_TEST_CUBLASLT_GPU') == '1', 'owned GPU required')
class ServingGpu(unittest.TestCase):
    def test_private_graph_outputs_padding_alias_and_inference_mode(self):
        from engine.kernels.dense import FP8Linear
        from engine.kernels.dense.cublaslt_serving import weight_nbytes
        with torch.inference_mode():
            weight = (torch.ones(128, 640, device='cuda').to(torch.float8_e4m3fn), torch.ones(1, 5, device='cuda'))
            layer = FP8Linear(weight[0][:127], quantized=weight)
            storage = torch.empty(weight_nbytes(128, 640, split_decode=True), device='cuda', dtype=torch.uint8)
            layer.prepare_cublas(split_decode=True, storage=storage)
            sources = [torch.full((7, 640), value, dtype=torch.bfloat16, device='cuda') for value in (1, 2)]
            outputs = [torch.empty(7, 128, device='cuda', dtype=torch.bfloat16) for _ in sources]
            for x, y in zip(sources, outputs):
                layer(x, out=y, decode=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graphs = []
            try:
                for x, y in zip(sources, outputs):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        layer(x, out=y, decode=True)
                    graphs.append(graph)
                sources[0].fill_(3)
                sources[1].fill_(4)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                for graph in reversed(graphs):
                    graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(torch.cuda.memory_stats()['allocated_bytes.all.allocated'], before)
                for expected, y in zip((3*640, 4*640), outputs):
                    torch.testing.assert_close(y, torch.full_like(y, expected), rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    layer(sources[0], out=sources[0].view(-1)[:7*128].view(7,128), decode=True)
                from engine.kernels.dense.fp8 import quantize
                q, scales = quantize(sources[0])
                torch.testing.assert_close(layer.project_quantized(q, scales), outputs[0][:,:127])
                self.assertEqual(layer.cublas.report()['backend'], 'cublaslt')
                for bias in (None, torch.linspace(-2, 2, 128, device='cuda')):
                    from engine.kernels.common.norm_rope import norm
                    norm_weight = torch.ones(128, device='cuda', dtype=torch.bfloat16)
                    # This layer has a padded logical output; normalization must refuse it.
                    with self.assertRaisesRegex(ValueError, 'unpadded'):
                        layer(sources[0], decode=True, normalization=(norm_weight, 1e-6, bias))
                    layer.rows = 128
                    reference = norm(layer(sources[0], decode=True), norm_weight, 1e-6, bias=bias)
                    normalized = layer(sources[0], decode=True, normalization=(norm_weight, 1e-6, bias))
                    torch.testing.assert_close(normalized, reference, rtol=0, atol=0)
                    layer.rows = 127
                self.assertIn('split_decode_norm', layer.cublas.executed)
                self.assertTrue(all(p['preferred_hit'] for p in layer.cublas.report()['preparation'].values()))
            finally:
                for graph in graphs:
                    graph.reset()
