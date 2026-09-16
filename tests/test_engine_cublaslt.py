"""Prepared cuBLAS decisions, graph cleanup and the exact MX scale ABI."""
import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TORCH = importlib.util.find_spec('torch') is not None
TRITON = importlib.util.find_spec('triton') is not None
INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'


@unittest.skipUnless(TORCH, 'requires torch')
class PreparationTests(unittest.TestCase):
    def test_non_fleet_cards_cannot_enter_timing(self):
        import torch
        from engine.kernels.dense.cublaslt import Bank
        bank = Bank.__new__(Bank)
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
            for capability, sms in (((12, 0), 24), ((12, 1), 96)):
                bank.identity = dict(capability=capability, sms=sms)
                with self.assertRaisesRegex(RuntimeError, 'restricted to GB10'):
                    bank.prepare((8, 128, 128, 'bf16'), None, None, None)

    def test_both_paired_pipeline_samples_must_win(self):
        from engine.kernels.dense.cublaslt import select_winner
        # Fast means are insufficient if one side regresses or wins by <2%.
        self.assertIsNone(select_winner([(0, 0, 10., 5., 10.1, 10.)]).index)
        self.assertIsNone(select_winner([(0, 0, 10., 9.9, 9., 10.)]).index)
        self.assertIsNone(select_winner([]).index)
        choice = select_winner([(1, 8192, 10., 9., 9., 10.), (2, 4096, 10., 9., 9., 10.),
                                (3, 0, 10., 9.5, 9.5, 10.)])
        self.assertEqual((choice.index, choice.workspace), (2, 4096))
        for invalid in (float('inf'), float('nan'), -1., 0.):
            with self.assertRaises(ValueError):
                select_winner([(0, 0, invalid, 1., 1., 1.)])

    def test_workspace_generations_survive_growth_and_streams_do_not_alias(self):
        import torch
        from engine.kernels.dense.cublaslt import Bank, WORKSPACE_LIMIT
        bank = Bank.__new__(Bank)
        bank.device, bank.workspaces, bank.owners = 0, {}, []
        active = SimpleNamespace(cuda_stream=11)
        allocated = []
        def allocate(size, **kw):
            out = SimpleNamespace(numel=lambda: size, tag=len(allocated))
            allocated.append(out)
            return out
        with patch.object(torch.cuda, 'current_stream', return_value=active), patch.object(torch, 'empty', allocate):
            first = bank.workspace(257)
            self.assertEqual(first.numel(), 512)
            self.assertIs(first, bank.workspace(100))
            second = bank.workspace(1025)
            self.assertEqual(second.numel(), 2048)
            active.cuda_stream = 12
            third = bank.workspace(257)
            self.assertIsNot(first, third)
            self.assertEqual(bank.owners, [first, second, third])
            active.cuda_stream = 11
            self.assertIs(second, bank.workspace(1024))
            with self.assertRaises(ValueError):
                bank.workspace(WORKSPACE_LIMIT + 1)
            with self.assertRaises(ValueError):
                bank.workspace(-1)
            with self.assertRaises(ValueError):
                bank.workspace(True)

    def test_capture_cannot_prepare_an_algorithm(self):
        import torch
        from engine.kernels.dense.cublaslt import Bank, PreparedProjection
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'before capture'):
                Bank.__new__(Bank).prepare((8, 128, 128, 'bf16'), None, None, None)
            with self.assertRaisesRegex(RuntimeError, 'before capture'):
                PreparedProjection(None, 8, None, 'bf16')

    def test_graph_reset_runs_on_success_and_capture_failure_preserves_original_error(self):
        import torch
        from engine.kernels.dense.cublaslt import _measure
        calls = []
        graph = SimpleNamespace(capture_begin=lambda *a, **k: calls.append('begin'),
                                capture_end=lambda: calls.append('end'),
                                replay=lambda: calls.append('replay'), reset=lambda: calls.append('reset'))
        event = SimpleNamespace(record=lambda: None, synchronize=lambda: None, elapsed_time=lambda _: 8.)
        with patch.object(torch.cuda, 'CUDAGraph', return_value=graph), patch.object(torch.cuda, 'Event', return_value=event):
            self.assertEqual(_measure(lambda: calls.append('fn'), None), 2.)
            self.assertEqual(calls[-1], 'reset')
            self.assertEqual(calls.count('replay'), 5)
            original = ValueError('first failure')
            count = 0
            def broken():
                nonlocal count
                count += 1
                if count == 2:
                    raise original
            def bad_end():
                raise RuntimeError('secondary teardown')
            graph.capture_end = bad_end
            with self.assertRaises(ValueError) as caught:
                _measure(broken, None)
            self.assertIs(caught.exception, original)
            self.assertIn('secondary teardown', original.__notes__[0])
            self.assertEqual(calls[-1], 'reset')


@unittest.skipUnless(TORCH and TRITON, 'requires torch and triton')
class ProducerContracts(unittest.TestCase):
    def test_scale_size_is_metadata_padding_only(self):
        from engine.kernels.dense.mxfp8 import scale_bytes
        for rows in (1, 4, 7, 127, 128):
            self.assertEqual(scale_bytes(rows, 4096), 16384)
        self.assertEqual(scale_bytes(129, 4096), 32768)
        for rows, cols in ((True, 128), (0, 128), (1, 129), (1, 0)):
            with self.assertRaises(ValueError):
                scale_bytes(rows, cols)

    def test_preallocated_producer_buffers_cannot_alias(self):
        import torch
        from engine.kernels.dense.fp8 import require_disjoint
        arena = torch.empty(1024, dtype=torch.uint8)
        require_disjoint(arena[:128], arena[128:256], arena[512:])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            require_disjoint(arena[:128], arena[127:256])


@unittest.skipUnless(TORCH and TRITON, 'requires torch and triton')
class BoundExecutionTests(unittest.TestCase):
    def prepared(self):
        import torch
        from engine.kernels.dense.cublaslt import Choice, PreparedProjection
        p = PreparedProjection.__new__(PreparedProjection)
        p.rows, p.key = 8, (8, 128, 128, 'bf16')
        p.weight = torch.empty(128, 128, dtype=torch.float8_e4m3fn), torch.ones(1, 1)
        p.mx_weight = torch.empty(512, dtype=torch.uint8)
        p.choice = Choice(0, 1024, 'test')
        calls = []
        def bind(index, q, weight, scales, weight_scales, out, workspace):
            calls.append((q, scales, out, workspace))
            # This stub exercises Python ownership/dispatch, not GEMM numerics.
            return SimpleNamespace(run=lambda: out.fill_(len(calls)))
        p.owner = SimpleNamespace(plans={p.key[:3]: SimpleNamespace(bind=bind)})
        return p, calls

    @staticmethod
    def producer(mx, out):
        import torch
        if out is None:
            return torch.empty(8, 128, dtype=torch.float8_e4m3fn), torch.empty(512, dtype=torch.uint8)
        return out

    def test_bound_calls_allocate_nothing_and_independent_bindings_own_distinct_scratch(self):
        import torch
        p, calls = self.prepared()
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
            first, second = p.bind(self.producer), p.bind(self.producer)
        self.assertNotEqual(first.workspace.data_ptr(), second.workspace.data_ptr())
        self.assertNotEqual(first.buffers[0].data_ptr(), second.buffers[0].data_ptr())
        self.assertNotEqual(first.out.data_ptr(), second.out.data_ptr())
        with patch.object(torch, 'empty', side_effect=AssertionError('allocation after binding')):
            for _ in range(3):
                self.assertIs(first(), first.out)
                self.assertIs(second(), second.out)
        self.assertEqual(len(calls), 2)  # descriptors bind once, never per execution
        self.assertIs(calls[0][3], first.workspace)

    def test_replacing_bound_buffers_fails_before_matmul_and_capture_cannot_bind(self):
        import torch
        p, calls = self.prepared()
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'before capture'):
                p.bind(self.producer)
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
            bound = p.bind(self.producer)
        executed = []
        bound.matmul = lambda: executed.append(True)
        bound.producer = lambda mx, out: self.producer(mx, None)
        with self.assertRaisesRegex(RuntimeError, 'replaced'):
            bound()
        bound.producer = lambda mx, out: ()
        with self.assertRaisesRegex(RuntimeError, 'replaced'):
            bound()
        self.assertEqual(executed, [])



@unittest.skipUnless(TORCH and TRITON and INTERPRET, 'explicit no-GPU Triton interpreter check')
class InterpreterTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def assert_scale_layout(self, mx, scales, rows, k):
        import torch
        # Independent NVIDIA 128x4 scale tile oracle: 32 rows, 4 row
        # quarters, 4 contiguous K32 exponents; K tiles precede row tiles.
        expected = torch.full((int(mx.numel()),), 127, dtype=torch.uint8)
        groups = k // 128
        codes = ((scales.view(torch.int32) >> 23) & 255).to(torch.uint8)
        for row in range(rows):
            for group in range(groups):
                offset = (row // 128 * groups + group)*512 + (row % 32)*16 + (row % 128 // 32)*4
                expected[offset:offset+4] = codes[row, group]
        self.assertTrue(torch.equal(mx, expected))

    def test_quantizer_exact_bytes_odd_groups_padding_and_redzones(self):
        import torch
        import triton
        from engine.kernels.dense.fp8 import _quantize as baseline
        from engine.kernels.dense.mxfp8 import _quantize, scale_bytes, row_programs
        for rows, k in ((1, 128), (4, 384), (7, 4096), (8, 20480), (127, 128),
                        (128, 256), (129, 384), (257, 128)):
            groups = rows * (k//128)
            # Exactly representable FP8 values avoid the interpreter's known
            # halfway cast limitation. Native RTNE remains a GPU check.
            values = torch.arange(128).remainder(17).sub(8).float().repeat(groups, 1)
            values[:, -1] = 448
            powers = torch.exp2(torch.arange(groups).remainder(15).float()-8)
            x = (values*powers[:, None]).reshape(rows, k).bfloat16()
            x.view(-1, 128)[::7] = 0
            base_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            base_s = torch.empty(rows, k//128)
            baseline[(rows, triton.cdiv(k//128, 4))](x, base_q, base_s, k, k//128)
            q = torch.full((rows*k+128,), 93, dtype=torch.uint8)
            size = scale_bytes(rows, k)
            scales = torch.full((size+128,), 93, dtype=torch.uint8)
            _quantize[(row_programs(rows), k//128)](
                x, q[:rows*k].view(torch.float8_e4m3fn), scales.view(torch.int32), rows, k, k//128, rows >= 128)
            self.assertTrue(torch.equal(q[:rows*k].view_as(base_q), base_q.view(torch.uint8)))
            self.assertTrue((q[rows*k:] == 93).all())
            self.assertTrue((scales[size:] == 93).all())
            self.assert_scale_layout(scales[:size], base_s, rows, k)
        self.assertFalse(torch.cuda.is_initialized())

    def test_weight_scale_replication_covers_every_normal_exponent(self):
        import torch
        from engine.kernels.dense.mxfp8 import _pack_weights
        exponents = torch.arange(1, 255, dtype=torch.int32)
        scales = (exponents << 23).view(torch.float32)
        output = torch.full((254*512+128,), 93, dtype=torch.uint8)
        _pack_weights[(254,)](scales, output.view(torch.int32))
        self.assertTrue(torch.equal(output[:254*512].reshape(254, 512),
                                    exponents.to(torch.uint8)[:, None].expand(254, 512)))
        self.assertTrue((output[254*512:] == 93).all())

    def test_packet_conversion_preserves_rounding_rank_stride_and_tail(self):
        import torch
        import triton
        from engine.kernels.dense.mxfp8 import scale_bytes, row_programs
        from engine.kernels.prefill_collectives.consumer import _quantize_gather, _quantize_gather_mx
        k, block = 4096, 2048
        for local_rows, real_rows, extra in ((32, 128, 0), (33, 129, 128), (35, 138, 0)):
            local = local_rows*k
            stride = ((local + 4*(local//block) + 127)//128)*128 + extra
            received = torch.full((4*stride,), 93, dtype=torch.uint8)
            for rank in range(4):
                values = torch.arange(local).remainder(17).sub(8).float().to(torch.float8_e4m3fn)
                received[rank*stride:rank*stride+local].copy_(values.view(torch.uint8))
                scales = torch.exp2(torch.arange(local//block).remainder(7).float()-3+rank)
                received[rank*stride+local:rank*stride+local+4*scales.numel()].copy_(scales.view(torch.uint8))
            original = received.clone()
            q0 = torch.empty(real_rows, k, dtype=torch.float8_e4m3fn)
            s0 = torch.empty(real_rows, k//128)
            q1 = torch.empty_like(q0)
            size = scale_bytes(real_rows, k)
            s1 = torch.full((size+128,), 93, dtype=torch.uint8)
            _quantize_gather[(real_rows, k//512)](received.view(torch.float8_e4m3fn), received.view(torch.float32),
                                                q0, s0, local, stride, k, k//128, block)
            _quantize_gather_mx[(row_programs(real_rows), k//128)](
                received.view(torch.float8_e4m3fn), received.view(torch.float32), q1, s1.view(torch.int32),
                real_rows, local, stride, k, k//128, block, real_rows >= 128)
            self.assertTrue(torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)))
            self.assert_scale_layout(s1[:size], s0, real_rows, k)
            self.assertTrue((s1[size:] == 93).all())
            self.assertTrue(torch.equal(received, original))


if __name__ == '__main__':
    unittest.main()
