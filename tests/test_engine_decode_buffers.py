"""Destination ownership and padded-vocabulary boundaries; GPU equality/replay is a separate gate."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from engine.kernels.dense import FP8Linear
from engine.profiles.glm53.net import Glm53Net


def layer(device='cpu', rows=129):
    weight = torch.zeros(rows, 128, dtype=torch.bfloat16, device=device)
    padded = (rows + 127) // 128 * 128
    return FP8Linear(weight, quantized=(torch.zeros(padded, 128, dtype=torch.float8_e4m3fn, device=device),
                                      torch.ones(padded // 128, 1, device=device)))


class HeadDestinationTests(unittest.TestCase):
    def test_fp8_writes_owned_padded_buffer_and_retains_observation(self):
        from engine.kernels.dense import fp8  # import native Triton modules outside the sys.modules patch
        head = layer()
        head.observer = Mock()
        net = Glm53Net.__new__(Glm53Net)
        net.dense, net.vp = {'head': head}, head.rows
        gemm = Mock(side_effect=lambda a, w, out: out.copy_(torch.arange(out.numel()).reshape_as(out)))
        modules = {'deep_gemm': NS(fp8_gemm_nt=gemm), 'engine.kernels.deep_gemm': NS(_initialize=Mock())}
        with patch.dict('sys.modules', modules):
            for rows in (1, 8, 16, 24, 32):
                x = torch.randn(rows, 128, dtype=torch.bfloat16)
                q = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
                scale = torch.ones(rows, 1)
                storage = net.head_buffer(rows, x.device)
                with patch('engine.kernels.dense.fp8.quantize', return_value=(q, scale)):
                    logical = net.head_local(x, out=storage)
                self.assertIs(gemm.call_args.args[2], storage)
                self.assertEqual(logical.shape, (rows, 129))
                self.assertEqual(logical.data_ptr(), storage.data_ptr())
                if rows > 1:
                    self.assertEqual(logical.stride(0), 256)
                self.assertTrue(torch.equal(logical, storage[:, :129]))
                observed, rows_ok = head.observer.call_args.args
                self.assertEqual(observed.data_ptr(), x.data_ptr())
                self.assertTrue(torch.equal(observed, x))
                self.assertIsNone(rows_ok)
                self.assertTrue(head.executed)

    def test_invalid_destination_is_refused_before_gemm_or_initialization(self):
        head = layer()
        q = torch.zeros(8, 128, dtype=torch.float8_e4m3fn)
        scale = torch.ones(8, 1)
        init, gemm = Mock(), Mock()
        with patch.dict('sys.modules', {'deep_gemm': NS(fp8_gemm_nt=gemm),
                                       'engine.kernels.deep_gemm': NS(_initialize=init)}):
            for out in (torch.empty(8, 129, dtype=torch.bfloat16), torch.empty(8, 256),
                        torch.empty(256, 8, dtype=torch.bfloat16).T,
                        torch.empty(8 * 256 + 1, dtype=torch.bfloat16)[1:].view(8, 256),
                        torch.empty(8, 256, device='meta', dtype=torch.bfloat16)):
                with self.assertRaisesRegex(ValueError, 'FP8 output'):
                    head.project_quantized(q, scale, out=out)
            init.assert_not_called()
            gemm.assert_not_called()

    def test_reference_head_and_logical_vocabulary_exclude_padding(self):
        from engine.base.comm import Comm
        from engine.modules.vocab import argmax, topk
        net = Glm53Net.__new__(Glm53Net)
        net.dense, net.vp = {}, 129
        net.p = {'head': torch.randn(129, 128, dtype=torch.bfloat16)}
        x = torch.randn(8, 128, dtype=torch.bfloat16)
        expected = net.head_local(x)
        out = net.head_buffer(8, x.device)
        self.assertIs(net.head_local(x, out=out), out)
        self.assertTrue(torch.equal(expected, out))
        storage = torch.full((8, 256), float('inf'), dtype=torch.bfloat16)
        logits = storage[:, :129]
        logits.copy_(expected)
        self.assertTrue(torch.equal(argmax(logits, Comm(), 0), expected.argmax(-1)))
        a, b = topk(logits, Comm(), 0, 16), topk(expected, Comm(), 0, 16)
        self.assertTrue(all(torch.equal(i, j) for i, j in zip(a, b)))

    def test_target_capacities_share_the_same_view_and_destination(self):
        from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs
        head = layer()
        net = Glm53Net.__new__(Glm53Net)
        net.dense, net.vp = {'head': head}, head.rows
        net.F = NS(block=256, max_position=16384, spec_k=7)
        net.comm = NS()
        net.lanes = NS(graph_resources=())
        net.forward = lambda step, *args, **kwargs: torch.zeros(step.ids.numel(), 128, dtype=torch.bfloat16)
        written = []
        def write(h, *, out):
            written.append(out)
            out.fill_(len(written))
        net.head_local = write
        caches = NS(device='cpu', block_table=torch.empty(4, 64), reset=Mock(), slots=NS(owner=[-1]*5))
        class Scratch:
            def __init__(self, *args): pass
            def gather(self): self.block_table = None
        class EagerGraphs:
            def __init__(self, forward, make_inputs, shapes, **kwargs):
                self.outputs = {shape: forward(make_inputs(*shape)) for shape in shapes}
        empty = torch.empty
        def cpu_empty(*args, **kwargs):
            kwargs.pop('pin_memory', None)
            return empty(*args, **kwargs)
        with patch('engine.profiles.glm53.decode_graphs.GraphCaches', Scratch), \
                patch('engine.profiles.glm53.decode_graphs.DecodeGraphs', EagerGraphs), \
                patch('torch.empty', cpu_empty):
            target = Glm53DecodeGraphs(net, caches, tokens=8, max_seqs=4)
        self.assertGreater(len(target.capacities), 1)
        for shape, (_, _, logits) in target.graphs.outputs.items():
            self.assertIs(logits, target.logits[shape[:2]])
            self.assertEqual(logits.data_ptr(), target.head_outputs[shape[:2]].data_ptr())
            self.assertEqual(logits.stride(0), 256)
        self.assertEqual({id(x) for x in written}, {id(x) for x in target.head_outputs.values()})
        caches.reset.assert_called_once()


class DraftValueLayoutTests(unittest.TestCase):
    def test_wrapper_passes_packed_values_and_rejects_nonpacked_heads(self):
        from engine.kernels import draft_attention as attention
        launch = Mock()
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
                patch.object(attention, '_sms', return_value=48), \
                patch.object(attention, '_draft_head', return_value=128), \
                patch.object(attention, '_attend', kernel), patch.object(attention, '_combine', kernel):
            for n in (1, 2, 3, 4):
                q = torch.zeros(n, 8, 8, 128, dtype=torch.bfloat16)
                k = torch.zeros(n, 8, 2, 128, dtype=torch.bfloat16)
                packed = torch.zeros(n, 8, 1536, dtype=torch.bfloat16)
                v = packed[:, :, -256:].view_as(k)
                ring = torch.zeros(6, 2, 2, 127, 2, 128, dtype=torch.bfloat16)
                positions, slots = torch.arange(n), torch.arange(n) + 1
                launch.reset_mock()
                attention.attend_rows(q, k, v, ring, positions, slot=slots, layer=1)
                args = launch.call_args_list[0].args
                self.assertIs(args[2], v)
                self.assertEqual(args[11:13], (v.stride(0), v.stride(1)))
                invalid = torch.zeros(n, 8, 128, 2, dtype=torch.bfloat16).transpose(-1, -2)
                with self.assertRaisesRegex(ValueError, 'packed V heads'):
                    attention.attend_rows(q, k, invalid, ring, positions, slot=slots, layer=1)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class HeadDestinationCudaTests(unittest.TestCase):
    def test_fp8_equal_outputs_and_changed_input_graph_replay(self):
        for width in (129, 38720):
            head = FP8Linear(torch.randn(width, 128, device='cuda', dtype=torch.bfloat16) * .02)
            x = torch.randn(8, 128, device='cuda', dtype=torch.bfloat16)
            storage = torch.empty(8, (width + 127) // 128 * 128, device='cuda', dtype=torch.bfloat16)
            expected = head(x)
            actual = head(x, out=storage)
            self.assertTrue(torch.equal(actual, expected))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = head(x, out=storage)
            try:
                for _ in range(3):
                    x.normal_()
                    expected = head(x)
                    storage.fill_(float('nan'))
                    graph.replay()
                    self.assertEqual(actual.data_ptr(), storage.data_ptr())
                    self.assertTrue(torch.equal(actual, expected))
            finally:
                graph.reset()


if __name__ == '__main__':
    unittest.main()
