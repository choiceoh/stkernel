"""Prepared cuBLAS decisions, graph cleanup and the exact MX scale ABI."""
import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TORCH = importlib.util.find_spec('torch') is not None
TRITON = importlib.util.find_spec('triton') is not None
INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'


if TORCH and TRITON and INTERPRET:
    import triton
    import triton.language as tl
    from engine.kernels.dense.mxfp8 import _power2_scale

    @triton.jit
    def _scale_probe(X, S, I, Reference):
        index = tl.program_id(0)*256 + tl.arange(0, 256)
        amax = tl.maximum(tl.load(X + index).to(tl.float32), 1e-4)
        scale, inverse = _power2_scale(amax)
        tl.store(S + index, scale)
        tl.store(I + index, inverse)
        tl.store(Reference + index, tl.exp2(tl.ceil(tl.log2(amax / 448.))))


@unittest.skipUnless(TORCH, 'requires torch')
class PreparationTests(unittest.TestCase):
    def test_algorithm_alignment_uses_actual_storage_for_every_operand(self):
        from engine.kernels.dense.cublaslt import _aligned
        candidate = dict(alignment_a=256, alignment_b=128, alignment_c=64, alignment_d=256)
        tensor = lambda address: SimpleNamespace(data_ptr=lambda: address)
        aligned = tensor(4096)
        self.assertTrue(_aligned(candidate, aligned, aligned, aligned))
        for operand in range(3):
            args = [aligned, aligned, aligned]
            args[operand] = tensor(4096 + 16)
            self.assertFalse(_aligned(candidate, *args))
        # A sliced weight can still use the smaller-alignment algorithm.
        self.assertTrue(_aligned(dict.fromkeys(candidate, 16), tensor(4112), aligned, aligned))

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
        self.assertIsNone(select_winner([(0, 0, 4, 10., 5., 10.1, 10.)]).index)
        self.assertIsNone(select_winner([(0, 0, 4, 10., 9.9, 9., 10.)]).index)
        self.assertIsNone(select_winner([]).index)
        choice = select_winner([(1, 8192, 4, 10., 9., 9., 10.), (2, 4096, 4, 10., 9., 9., 10.),
                                (3, 0, 4, 10., 9.5, 9.5, 10.)])
        self.assertEqual((choice.index, choice.workspace), (2, 4096))
        for invalid in (float('inf'), float('nan'), -1., 0.):
            with self.assertRaises(ValueError):
                select_winner([(0, 0, 4, invalid, 1., 1., 1.)])

    def test_sm120_timing_requires_explicit_device_probe(self):
        from engine.kernels.dense.cublaslt import require_timing_target
        rtx = dict(capability=(12, 0), sms=20)
        with self.assertRaises(RuntimeError):
            require_timing_target(rtx)
        require_timing_target(rtx, 'sm120-probe')
        require_timing_target(dict(capability=(12, 1), sms=48))
        for wrong in (dict(capability=(12, 1), sms=48), dict(capability=(9, 0), sms=132)):
            with self.assertRaises(RuntimeError):
                require_timing_target(wrong, 'sm120-probe')

    def test_winner_keeps_the_producer_layout_of_the_complete_pipeline(self):
        from engine.kernels.dense.cublaslt import select_winner
        choice = select_winner([(7, 0, 4, 10., 9.5, 9.5, 10.),
                                (7, 0, 1, 10., 8.5, 8.5, 10.),
                                (8, 0, 2, 10., 7., 10.2, 10.)])
        self.assertEqual((choice.index, choice.producer_warps), (7, 1))
        for invalid in (0, 3, True):
            with self.assertRaisesRegex(ValueError, 'warp count'):
                select_winner([(7, 0, invalid, 10., 8., 8., 10.)])

    def test_custom_producer_cannot_silently_ignore_a_warp_choice(self):
        from engine.kernels.dense.cublaslt import _bind_producer
        with self.assertRaisesRegex(ValueError, 'custom producers'):
            _bind_producer(lambda mx, out: (), True, num_warps=1)

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
        from engine.kernels.dense.cublaslt import _measure, GRAPH_UNROLL
        calls = []
        graph = SimpleNamespace(capture_begin=lambda *a, **k: calls.append('begin'),
                                capture_end=lambda: calls.append('end'),
                                replay=lambda: calls.append('replay'), reset=lambda: calls.append('reset'))
        event = SimpleNamespace(record=lambda: None, synchronize=lambda: None, elapsed_time=lambda _: 8.)
        with patch.object(torch.cuda, 'CUDAGraph', return_value=graph), patch.object(torch.cuda, 'Event', return_value=event):
            self.assertEqual(_measure(lambda: calls.append('fn'), None), 2. / GRAPH_UNROLL)
            self.assertEqual(calls.count('fn'), GRAPH_UNROLL + 1)
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


@unittest.skipUnless(TORCH and os.environ.get('ST_TEST_CUBLASLT_GPU') == '1',
                     'requires explicit owned-GPU permission')
class NativePreparationTests(unittest.TestCase):
    def test_mx_query_uses_real_scales_and_binding_owns_execution_scales(self):
        import torch
        from engine.kernels.dense.cublaslt import _build, WORKSPACE_LIMIT
        native = _build()
        device = torch.cuda.current_device()
        context = native.Context(device, *torch.cuda.get_device_capability(device))
        scales = torch.full((512,), 127, device='cuda', dtype=torch.uint8)
        # A null scale pointer used to abort the heuristic query here.
        plan = native.Plan(context, 7, 128, 128, WORKSPACE_LIMIT, scales, scales)
        candidates = plan.candidates()
        self.assertTrue(candidates)
        with self.assertRaisesRegex(RuntimeError, 'query scale shape'):
            native.Plan(context, 7, 128, 128, WORKSPACE_LIMIT, scales[:256], scales)
        q = torch.ones(7, 128, device='cuda').to(torch.float8_e4m3fn)
        weight = torch.ones(128, 128, device='cuda').to(torch.float8_e4m3fn)
        out = torch.empty(7, 128, device='cuda', dtype=torch.bfloat16)
        candidate = candidates[0]
        scratch = torch.empty(candidate['workspace'], device='cuda', dtype=torch.uint8)
        a_scales, b_scales = scales.clone(), scales.clone()
        bound = plan.bind(candidate['index'], q, weight, a_scales, b_scales, out, scratch)
        scales.fill_(130)  # The query's scale values must not affect execution.
        bound.run()
        torch.testing.assert_close(out, torch.full_like(out, 128), rtol=0, atol=0)
        a_scales.fill_(128)
        bound.run()
        torch.testing.assert_close(out, torch.full_like(out, 256), rtol=0, atol=0)


    def test_batched_partials_use_each_scale_slice_and_reject_bad_buffers(self):
        import torch
        from engine.kernels.dense.cublaslt import _build, WORKSPACE_LIMIT
        native = _build()
        context = native.Context(0, *torch.cuda.get_device_capability())
        scales = torch.full((5, 512), 127, device='cuda', dtype=torch.uint8)
        plan = native.Plan(context, 8, 128, 128, WORKSPACE_LIMIT, scales, scales, 5, True)
        self.assertTrue(plan.candidates())
        q = torch.ones(5, 8, 128, device='cuda').to(torch.float8_e4m3fn)
        w = torch.ones(5, 128, 128, device='cuda').to(torch.float8_e4m3fn)
        out = torch.empty(5, 8, 128, device='cuda', dtype=torch.float32)
        c = plan.candidates()[0]
        scratch = torch.empty(c['workspace'], device='cuda', dtype=torch.uint8)
        actual = scales.clone()
        for part in range(5):
            actual[part].fill_(127+part)
        bound = plan.bind(c['index'], q, w, actual, scales, out, scratch)
        scales.fill_(128)
        bound.run()
        for part in range(5):
            torch.testing.assert_close(out[part], torch.full_like(out[part], 256*2**part), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, 'shape mismatch'):
            plan.bind(c['index'], q.reshape(8, 5, 128), w, actual, scales, out, scratch)
        with self.assertRaises(RuntimeError):
            plan.bind(c['index'], q, w, actual, scales, out.bfloat16(), scratch)
        with self.assertRaisesRegex(RuntimeError, 'FP32'):
            native.Plan(context, 8, 128, 128, WORKSPACE_LIMIT, scales, scales, 5, False)


    def test_split_bindings_have_private_storage_and_replay_changed_sources(self):
        import torch
        from engine.kernels.dense.cublaslt import _build, BF16Producer
        from engine.kernels.dense.cublaslt_split import PackedWeight, SplitPlan
        native = _build()
        owner = SimpleNamespace(native=native, context=native.Context(0, *torch.cuda.get_device_capability()))
        weight = (torch.ones(128, 640, device='cuda').to(torch.float8_e4m3fn),
                  torch.ones(1, 5, device='cuda'))
        packed = PackedWeight(weight, 5)
        source = torch.ones(8, 640, device='cuda', dtype=torch.bfloat16)
        other = torch.full_like(source, 2)
        plan = SplitPlan(owner, packed, 8, source)
        c = plan.native.candidates()[0]
        plan.index, plan.workspace_bytes = c['index'], c['workspace']
        first, second = plan.bind(BF16Producer(source)), plan.bind(BF16Producer(other))
        for name in ('q', 'scales', 'partials', 'out'):
            self.assertNotEqual(getattr(first, name).data_ptr(), getattr(second, name).data_ptr())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        for graph, execution in zip(graphs, (first, second)):
            with torch.cuda.graph(graph, stream=stream):
                execution()
        try:
            source.fill_(3)
            other.fill_(4)
            with patch.object(torch, 'empty', side_effect=AssertionError('allocation after binding')):
                first()
                second()
                for graph in graphs:
                    graph.replay()
            torch.testing.assert_close(first.out, torch.full_like(first.out, 640*3), rtol=0, atol=0)
            torch.testing.assert_close(second.out, torch.full_like(second.out, 640*4), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                plan.bind(BF16Producer(source), out=source.view(-1)[:8*128].view(8, 128))
            weight[1].mul_(2)
            with self.assertRaisesRegex(RuntimeError, 'modified'):
                plan.bind(BF16Producer(source))
        finally:
            for graph in graphs:
                graph.reset()
        with torch.inference_mode():
            inference_weight = tuple(t.clone() for t in weight)
            inference_pack = PackedWeight(inference_weight, 5)
            self.assertEqual(inference_pack.versions, (None, None))
            inference_plan = SplitPlan(owner, inference_pack, 8, source)
            c = inference_plan.native.candidates()[0]
            inference_plan.index, inference_plan.workspace_bytes = c['index'], c['workspace']
            execution = inference_plan.bind(BF16Producer(source))
            torch.testing.assert_close(execution(), torch.full_like(execution.out, 640*3*2), rtol=0, atol=0)


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
        bound.producer = lambda: self.producer(True, None)
        with self.assertRaisesRegex(RuntimeError, 'replaced'):
            bound()
        bound.producer = lambda: ()
        with self.assertRaisesRegex(RuntimeError, 'replaced'):
            bound()
        self.assertEqual(executed, [])

    def test_selected_warp_configuration_reaches_bound_execution(self):
        import torch
        from engine.kernels.dense.cublaslt import BF16Producer, Choice
        p, calls = self.prepared()
        p.choice = Choice(0, 1024, 'test', 1)
        source = torch.empty(8, 128, dtype=torch.bfloat16)
        outputs = self.producer(True, None)
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                patch('engine.kernels.dense.mxfp8.bind_quantize', return_value=lambda: outputs) as bind:
            execution = p.bind(BF16Producer(source))
            execution()
        bind.assert_called_once_with(source, out=None, num_warps=1)
        self.assertIs(execution.buffers, outputs)
        self.assertEqual(len(calls), 1)



@unittest.skipUnless(TORCH and TRITON and INTERPRET, 'explicit no-GPU Triton interpreter check')
class InterpreterTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_scale_and_inverse_for_every_bf16_magnitude(self):
        import torch
        # All 32,768 positive BF16 encodings, including zero, subnormals,
        # scale boundaries, infinity and NaNs. Negative magnitudes are the same.
        x = torch.arange(32768, dtype=torch.int16).view(torch.bfloat16)
        output, inverse, reference = (torch.empty(32768) for _ in range(3))
        _scale_probe[(128,)](x, output, inverse, reference)
        torch.testing.assert_close(output, reference, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(inverse, reference.reciprocal(), rtol=0, atol=0, equal_nan=True)

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
        from engine.kernels.dense.mxfp8 import _quantize, _quantize_bound, scale_bytes, row_programs
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
            # The bound path retains neutral padding from preparation. A
            # sentinel proves replay leaves those words untouched, including
            # tails of an otherwise coalesced full-row tile.
            scales[:size].fill_(127)
            padding = torch.ones(size, dtype=torch.bool)
            for row in range(rows):
                for group in range(k//128):
                    offset = (row//128*(k//128)+group)*512 + row%32*16 + row%128//32*4
                    padding[offset:offset+4] = False
            scales[:size][padding] = 91
            x.mul_(2)
            baseline[(rows, triton.cdiv(k//128, 4))](x, base_q, base_s, k, k//128)
            _quantize_bound[(row_programs(rows), k//128)](
                x, q[:rows*k].view(torch.float8_e4m3fn), scales.view(torch.int32), rows, k, True, num_warps=1)
            self.assertTrue(torch.equal(q[:rows*k].view_as(base_q), base_q.view(torch.uint8)))
            self.assertTrue((scales[:size][padding] == 91).all())
            self.assertTrue((scales[size:] == 93).all())
            scales[:size][padding] = 127
            self.assert_scale_layout(scales[:size], base_s, rows, k)
        self.assertFalse(torch.cuda.is_initialized())

    def test_split_producer_layout_redzones_and_fp32_reduction(self):
        import torch
        import triton
        from engine.kernels.dense.cublaslt_split import _quantize, _reduce
        from engine.kernels.dense.fp8 import _quantize as baseline
        from engine.kernels.dense.mxfp8 import scale_bytes
        for rows, k, parts in ((7, 640, 5), (8, 20480, 5), (16, 20480, 5)):
            groups = rows*(k//128)
            values = torch.arange(128).remainder(17).sub(8).float().repeat(groups, 1)
            values[:, -1] = 448
            x = (values * torch.exp2(torch.arange(groups).remainder(15).float()-8)[:, None]).reshape(rows, k).bfloat16()
            q0 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            s0 = torch.empty(rows, k//128)
            baseline[(rows, triton.cdiv(k//128, 4))](x, q0, s0, k, k//128)
            q = torch.full((rows*k+128,), 93, dtype=torch.uint8)
            size = scale_bytes(rows, k//parts)
            s = torch.full((parts*size+128,), 127, dtype=torch.uint8)
            s[parts*size:] = 93
            _quantize[(triton.cdiv(rows, 4), k//128)](x, q.view(torch.float8_e4m3fn), s.view(torch.int32),
                                                   rows, k, parts, num_warps=1)
            restored = q[:rows*k].reshape(parts, rows, k//parts).permute(1, 0, 2).reshape(rows, k)
            self.assertTrue(torch.equal(restored, q0.view(torch.uint8)))
            for part in range(parts):
                self.assert_scale_layout(s[part*size:(part+1)*size],
                                         s0[:, part*(k//parts//128):(part+1)*(k//parts//128)], rows, k//parts)
            self.assertTrue((q[rows*k:] == 93).all())
            self.assertTrue((s[parts*size:] == 93).all())
        partials = torch.tensor([1e6, 1, -1e6, 3, 5])[:, None].expand(5, 257).contiguous()
        out = torch.full((257+128,), 93, dtype=torch.bfloat16)
        _reduce[(2,)](partials, out, 257, 5)
        self.assertTrue((out[:257] == 9).all())
        self.assertTrue((out[257:] == 93).all())
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
        from engine.kernels.prefill_collectives.consumer import _quantize_gather, _quantize_gather_mx, _quantize_gather_mx_bound
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
            # Reuse exactly the initialized scale storage, without a padding
            # launch or repeated tail writes, for the packet producer too.
            s1[:size].fill_(127)
            _quantize_gather_mx_bound[(row_programs(real_rows), k//128)](
                received.view(torch.float8_e4m3fn), received.view(torch.float32), q1, s1.view(torch.int32),
                real_rows, local, stride, block, True, num_warps=1)
            self.assert_scale_layout(s1[:size], s0, real_rows, k)


if __name__ == '__main__':
    unittest.main()
