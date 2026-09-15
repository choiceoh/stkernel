"""Actual kernel addressing, eager routing and TP4 model handoff on CPU.

These checks do not qualify GPU MMA rounding, graph replay or serving speed.
"""
import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

from engine.modules.mla_absorb import mla_prefill_absorb_ref

ROOT = Path(__file__).resolve().parents[1]


class AbsorbRoutingTests(unittest.TestCase):
    def test_serving_default_on_with_experimental_rollback_and_neutral_bare_plan(self):
        from engine.base.config import ConfigError
        from engine.profiles.glm53.execution import ExecutionPlan
        from tests.test_engine_knobs import KnobDeclarationTests
        declared = KnobDeclarationTests()._declared
        self.assertFalse(ExecutionPlan().prefill_absorb_tiles)
        self.assertTrue(ExecutionPlan(prefill_absorb_tiles=True).active)
        self.assertIn('prefill_absorb_tiles=1', ExecutionPlan(prefill_absorb_tiles=True).label())
        with self.assertRaises(ValueError):
            ExecutionPlan(prefill_absorb_tiles=1)
        for production in (False, True):
            cfg = declared({}, production=production)
            self.assertEqual(cfg['prefill_absorb_tiles'], 1)
            self.assertEqual(cfg['prefill_dense_prefix'], 1)
        self.assertEqual(declared({'STK_prefill_absorb_tiles': '1'})['prefill_absorb_tiles'], 1)
        self.assertEqual(declared({'STK_prefill_absorb_tiles': '0'})['prefill_absorb_tiles'], 0)
        for value in ('0', '1'):
            with self.assertRaises(ConfigError):
                declared({'STK_prefill_absorb_tiles': value}, production=True)

    def test_only_bounded_single_segment_eager_prefill_uses_lane(self):
        from engine.profiles.glm53.net import Glm53Net
        for rows, captured, probe, segments, enabled in (
                (7, False, False, 1, True), (127, False, False, 1, True),
                (128, False, False, 1, True), (131, False, False, 1, True),
                (32256, False, False, 1, True), (32768, False, False, 1, True),
                (32769, False, False, 1, True), (131, True, False, 1, True),
                (131, False, True, 1, True), (131, False, False, 2, True),
                (131, False, False, 1, False)):
            lane = Mock(wraps=mla_prefill_absorb_ref)
            net = NS(prefill_absorb_tiles=enabled, probe=probe, lanes=NS(mla_absorb=lane),
                     prefill_absorb_tiles_executed=set())
            step = NS(captured=captured, segments=(None,)*segments)
            weight = torch.randn(2, 3, 5).bfloat16()
            eligible = enabled and not captured and not probe and segments == 1 and 128 <= rows <= 32768
            for transpose, inner in ((False, 3), (True, 5)):
                x = torch.randn(rows, 2, inner).bfloat16()
                actual = Glm53Net._mla_absorb(net, 3, x, weight, step, transpose=transpose)
                expected = mla_prefill_absorb_ref(x, weight, transpose=transpose)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if eligible:
                    self.assertTrue(actual.is_contiguous())
            self.assertEqual(lane.call_count, 2 if eligible else 0)
            self.assertEqual(net.prefill_absorb_tiles_executed,
                             {(3, 'query'), (3, 'output')} if eligible else set())
        net.lanes.mla_absorb = None
        net.prefill_absorb_tiles = True
        Glm53Net._mla_absorb(net, 3, x, weight, step, transpose=True)
        self.assertEqual(net.prefill_absorb_tiles_executed, set())

    def test_runtime_marker_requires_both_contractions_on_every_dsa_layer(self):
        from engine.profiles.glm53.boot import native_execution_report
        net = NS(layers=[0, 1, 2], dense={'a': NS(executed=3), 'head': NS(executed=True)},
                 mhc=NS(executed=set(range(5))), shared_mlp={1: NS(executed=True)},
                 shared_overlap=NS(executed=True), _router_layers={1}, _router_weights={1: object()}, _router_fp32={1},
                 F=NS(is_dsa=lambda L: L in (1, 2)), prefill_absorb_tiles=True,
                 prefill_transport=NS(executed={'fp8_all_gather', 'fp8_reduce_scatter'}, project_tiles=False))
        drafter = NS(dense={'fc.weight': NS(executed=3), 'q': NS(executed=1)})
        complete = {(L, side) for L in (1, 2) for side in ('query', 'output')}
        for omitted in complete:
            net.prefill_absorb_tiles_executed = complete - {omitted}
            with self.assertRaisesRegex(RuntimeError, 'both sides of every DSA layer'):
                native_execution_report(net, drafter)
        net.prefill_absorb_tiles_executed = complete
        self.assertEqual(native_execution_report(net, drafter)['prefill_absorb_tiles'], sorted(complete))


class AbsorbKernelTests(unittest.TestCase):
    def test_actual_wrapper_launch_geometry_and_fresh_output_without_a_device(self):
        # Execute the actual wrapper with only the launch/stream surface mocked.
        # This proves argument wiring and ownership, not a GPU launch.
        class CudaInput:
            is_cuda = True

            def __init__(self, tensor):
                self.tensor = tensor

            def __getattr__(self, name):
                return getattr(self.tensor, name)

        launch = Mock()
        grids = []

        class Kernel:
            def __getitem__(self, grid):
                grids.append(grid)
                return launch

        captured = [False]
        path = ROOT/'engine/kernels/mla/prefill_absorb.py'
        node = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                                  if isinstance(n, ast.FunctionDef) and n.name == 'mla_prefill_absorb'))
        scope = dict(torch=NS(bfloat16=torch.bfloat16, empty=torch.empty,
                             cuda=NS(is_current_stream_capturing=lambda: captured[0])),
                     triton=NS(cdiv=lambda a, b: (a+b-1)//b), _absorb=Kernel())
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        run = scope['mla_prefill_absorb']
        owner = torch.zeros(16, 512, 512).bfloat16()
        for transpose in (False, True):
            inner, outer = (512, 256) if transpose else (256, 512)
            x = CudaInput(torch.zeros(131, 16, inner).bfloat16())
            w = owner[:, 256:] if transpose else owner[:, :256]
            first = run(x, w, transpose=transpose)
            second = run(x, w, transpose=transpose)
            self.assertEqual(first.shape, (131, 16, outer))
            self.assertTrue(first.is_contiguous())
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
            self.assertEqual(grids[-1], (3, outer//64, 16))
            args = launch.call_args.args
            self.assertIs(args[0], x)
            self.assertIs(args[1], w)
            self.assertIs(args[2], second)
            self.assertEqual(args[3:], (131, 16, inner, outer, 512*512, 512, transpose, 64, 64, 32))
        captured[0] = True
        with self.assertRaisesRegex(ValueError, 'eager CUDA'):
            run(x, w, transpose=True)

    def test_actual_kernel_body_strides_offsets_tails_and_independent_einsums(self):
        torch.set_num_threads(1)

        class Pointer:
            def __init__(self, owner, offset=0):
                self.owner, self.offset = owner, offset

            def __add__(self, offset):
                return Pointer(self.owner, self.offset + offset)

        def load(pointer, mask, other=0):
            index, mask = torch.broadcast_tensors(pointer.offset, mask)
            data = pointer.owner.view(-1)
            self.assertTrue(bool(((index[mask] >= 0) & (index[mask] < data.numel())).all()))
            values = torch.full(index.shape, other, dtype=data.dtype)
            values[mask] = data[index[mask]]
            return values

        def store(pointer, values, mask):
            index, mask = torch.broadcast_tensors(pointer.offset, mask)
            data = pointer.owner.view(-1)
            self.assertTrue(bool(((index[mask] >= 0) & (index[mask] < data.numel())).all()))
            data[index[mask]] = values.expand(index.shape)[mask].to(data.dtype)

        program = [0, 0, 0]
        tl = NS(program_id=lambda d: program[d], arange=lambda lo, hi: torch.arange(lo, hi, dtype=torch.int32),
                int64=torch.int64, float32=torch.float32, bfloat16=torch.bfloat16,
                load=load, store=store, cdiv=lambda a, b: (a+b-1)//b,
                zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
                dot=lambda a, b, acc: a.float() @ b.float() + acc)
        path = ROOT/'engine/kernels/mla/prefill_absorb.py'
        node = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                                  if isinstance(n, ast.FunctionDef) and n.name == '_absorb'))
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        scope = dict(tl=tl)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        run = scope['_absorb']
        gen = torch.Generator().manual_seed(8931)
        for rows, heads, narrow, wide in ((1, 2, 33, 65), (7, 2, 33, 65),
                                          (8, 16, 256, 512), (16, 16, 256, 512),
                                          (24, 16, 256, 512), (32, 16, 256, 512),
                                          (131, 16, 256, 512), (132, 16, 256, 512)):
            owner = torch.randn(heads, 2*narrow, wide, generator=gen).bfloat16()
            before = owner.clone()
            for transpose in (False, True):
                inner, outer = (wide, narrow) if transpose else (narrow, wide)
                offset = narrow*wide if transpose else 0
                weight = owner[:, narrow:] if transpose else owner[:, :narrow]
                x = torch.randn(rows, heads, inner, generator=gen).bfloat16()
                x_before = x.clone()
                expected = torch.einsum('thc,hvc->thv' if transpose else 'thd,hdc->thc', x, weight)
                for bm, bn, bk in ((64, 64, 32), (32, 64, 64), (16, 64, 64)):
                    destination = torch.full((rows+2, heads, outer), -19., dtype=torch.bfloat16)
                    for h in range(heads):
                        for m in range((rows+bm-1)//bm):
                            for n in range((outer+bn-1)//bn):
                                program[:] = [m, n, h]
                                run(Pointer(x), Pointer(owner, offset), Pointer(destination, heads*outer),
                                    rows, heads, inner, outer, owner.stride(0), owner.stride(1),
                                    transpose, bm, bn, bk)
                    got = destination[1:-1]
                    self.assertTrue(got.is_contiguous())
                    self.assertEqual(got.reshape(rows, -1).data_ptr(), got.data_ptr())
                    self.assertTrue(bool((destination[0] == -19).all() & (destination[-1] == -19).all()))
                    error = float((got.float()-expected.float()).norm()/expected.float().norm())
                    self.assertLess(error, .0005, (rows, transpose, bm, bk, error))
                torch.testing.assert_close(x, x_before, rtol=0, atol=0)
            torch.testing.assert_close(owner, before, rtol=0, atol=0)

    def test_native_wrapper_rejects_unsupported_inputs_before_launch(self):
        from engine.kernels.mla.prefill_absorb import mla_prefill_absorb
        w = torch.zeros(16, 512, 512).bfloat16()
        for transpose in (False, True):
            x = torch.zeros(131, 16, 512 if transpose else 256).bfloat16()
            weight = w[:, 256:] if transpose else w[:, :256]
            with self.assertRaisesRegex(ValueError, 'eager CUDA'):
                mla_prefill_absorb(x, weight, transpose=transpose)
            for bad_x, bad_w in ((x.float(), weight), (x, weight.float()), (x[:127], weight),
                                  (x[:, :8], weight), (x.transpose(0, 1), weight),
                                  (x, weight.transpose(1, 2)), (x, weight[:, :, ::2]),
                                  (x, weight[:1].expand(16, -1, -1)),
                                  (torch.empty(32769, *x.shape[1:], dtype=x.dtype, device='meta'), weight)):
                with self.assertRaises(ValueError):
                    mla_prefill_absorb(bad_x, bad_w, transpose=transpose)
            with self.assertRaises(ValueError):
                mla_prefill_absorb(x, weight, transpose=1)

    def test_existing_einsums_copy_and_reference_outputs_own_token_major_storage(self):
        q = torch.randn(131, 16, 256).bfloat16()
        w = torch.randn(16, 512, 512).bfloat16()
        for transpose in (False, True):
            weight = w[:, 256:] if transpose else w[:, :256]
            x = torch.randn(131, 16, 512).bfloat16() if transpose else q
            legacy = torch.einsum('thc,hvc->thv' if transpose else 'thd,hdc->thc', x, weight)
            self.assertFalse(legacy.is_contiguous())
            self.assertNotEqual(legacy.reshape(131, -1).data_ptr(), legacy.data_ptr())
            actual = mla_prefill_absorb_ref(x, weight, transpose=transpose)
            torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
            self.assertTrue(actual.is_contiguous())
            self.assertEqual(actual.reshape(131, -1).data_ptr(), actual.data_ptr())
            self.assertNotEqual(actual.data_ptr(), x.data_ptr())


class AbsorbModelTests(unittest.TestCase):
    def test_tp4_reference_handoff_preserves_aux_cache_and_following_seven_token_decode(self):
        from engine.base.comm import LocalTP
        from engine.profiles.glm53.net import Step
        from engine.profiles.glm53.execution import prefill_layer_major
        from tests.test_engine_execution_plans import model
        from tests.test_engine_prefill_tiles import OraclePrefill
        torch.set_num_threads(1)
        for tiled, rows in ((False, 131), (False, 132), (True, 259)):
            def rank(comm):
                net, cache = model(('kda', 'dsa', 'kda'), comm=comm)
                net.prefill_transport = OraclePrefill(comm, False)
                net.F = replace(net.F, topk=128)
                net.prefill_dense_prefix = net.prefill_indexer_shards = True
                slot = cache.slots.take(1)
                cache.pool.reserve(1, rows+7)
                step = Step.prefill(torch.arange(rows)%net.vp, 0, 1, slot)
                follow = Step.prefill(torch.arange(7)%net.vp, rows, 1, slot)
                cache.prepare(step)
                initial, paged = cache.state.clone(), cache.paged.clone()

                def run():
                    if tiled:
                        return prefill_layer_major(net, step, cache, NS(tile_rows=128, prefill_tiles=4), [0, 2])
                    return net.forward(step, cache, aux_layers=[0, 2])

                expected = run()
                state, pages = cache.state.clone(), cache.paged.clone()
                cache.prepare(follow)
                next_hidden = net.forward(follow, cache)
                final, final_pages = cache.state.clone(), cache.paged.clone()
                cache.state.copy_(initial)
                cache.paged.copy_(paged)
                cache.prepare(step)
                net.prefill_absorb_tiles = True
                actual = run()
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(cache.state, state, rtol=0, atol=0)
                torch.testing.assert_close(cache.paged, pages, rtol=0, atol=0)
                self.assertEqual(net.prefill_absorb_tiles_executed, {(1, 'query'), (1, 'output')})
                net.prefill_absorb_tiles_executed.clear()
                cache.prepare(follow)
                torch.testing.assert_close(net.forward(follow, cache), next_hidden, rtol=0, atol=0)
                torch.testing.assert_close(cache.state, final, rtol=0, atol=0)
                torch.testing.assert_close(cache.paged, final_pages, rtol=0, atol=0)
                self.assertEqual(net.prefill_absorb_tiles_executed, set())
            LocalTP(4, timeout_s=30).run(rank)


if __name__ == '__main__':
    unittest.main()
