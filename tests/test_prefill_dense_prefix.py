"""Covered-token semantics, actual kernel-body arithmetic and model handoff."""
import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

from engine.modules.prefill_attention import covered_prefix, mla_dense_prefix_ref

ROOT = Path(__file__).resolve().parents[1]


class PrefixGeometryTests(unittest.TestCase):
    def test_coverage_matches_complete_pool_selection_including_partial_pool(self):
        for rows in (0, 1, 127, 128, 129, 131, 2000, 2051, 2052, 2672, 32256):
            for context in (0, 1, 3, 2048, 2051, 2052, 32256, 96768):
                for topk, pool in ((2048, 4), (128, 4), (8, 4), (7, 3)):
                    got = covered_prefix(rows, context, topk, pool)
                    expected = sum((context+r+1)//pool <= topk//pool for r in range(rows))
                    self.assertEqual(got, expected)
        for args in ((1, -1, 8, 4), (-1, 0, 8, 4), (1, 0, 0, 4), (1, 0, 8, 0), (True, 0, 8, 4)):
            with self.assertRaises(ValueError):
                covered_prefix(*args)

    def test_config_is_explicit_and_production_cannot_override(self):
        from tests.test_engine_knobs import KnobDeclarationTests
        from engine.base.config import ConfigError
        from engine.profiles.glm53.execution import ExecutionPlan
        declared = KnobDeclarationTests()._declared
        self.assertFalse(ExecutionPlan().prefill_dense_prefix)
        self.assertTrue(ExecutionPlan(prefill_dense_prefix=True).active)
        self.assertIn('prefill_dense_prefix=1', ExecutionPlan(prefill_dense_prefix=True).label())
        with self.assertRaises(ValueError):
            ExecutionPlan(prefill_dense_prefix=1)
        for production in (False, True):
            self.assertEqual(declared({}, production=production)['prefill_dense_prefix'], 0)
        cfg = declared({'STK_prefill_dense_prefix': '1', 'STK_prefill_indexer_shards': '1'})
        self.assertEqual((cfg['prefill_dense_prefix'], cfg['prefill_indexer_shards']), (1, 1))
        with self.assertRaises(ConfigError):
            declared({'STK_prefill_dense_prefix': '1'}, production=True)

    def test_actual_model_dispatch_splits_at_coverage_and_falls_back_outside_contract(self):
        from engine.profiles.glm53.net import Glm53Net, Step
        for rows, context in ((131, 0), (132, 0), (2672, 0), (131, 3), (131, 4), (7, 0), (32769, 0)):
            for enabled in (False, True):
                for captured in (False, True):
                    step = Step.prefill(torch.zeros(rows, dtype=torch.int64), context, 2, 1)
                    if captured:
                        step = NS(segments=step.segments, captured=True)
                    def fill(value):
                        def call(q, *args, out=None):
                            result = torch.empty_like(q) if out is None else out
                            return result.fill_(value)
                        return call
                    dense, sparse = Mock(side_effect=fill(2)), Mock(side_effect=fill(3))
                    net = NS(prefill_dense_prefix=enabled, probe=False, F=NS(topk=128, kpool=4, mla_scale=.1),
                             lanes=NS(mla_dense_prefix=dense, mla_sparse=sparse), prefill_dense_prefix_executed=set())
                    cache = NS(token_map=Mock(return_value=(None, 64, 128, 32)))
                    q = torch.zeros(rows, 2, 4).bfloat16()
                    result = Glm53Net._mla_context(net, 1, q, q, torch.zeros(rows, 131), torch.zeros(rows), step, cache)
                    prefix = covered_prefix(rows, context, 128, 4) if enabled and not captured and 128 <= rows <= 32768 else 0
                    if prefix < 128:
                        prefix = 0
                    self.assertTrue(bool((result[:prefix] == 2).all()))
                    self.assertTrue(bool((result[prefix:] == 3).all()))
                    self.assertEqual(dense.call_count, bool(prefix))
                    self.assertEqual(sparse.call_count, prefix != rows)
                    self.assertEqual(net.prefill_dense_prefix_executed, {1} if prefix else set())
                    if prefix:
                        cache.token_map.assert_called_once_with(1, 2)
                        self.assertEqual(dense.call_args.kwargs['out'].data_ptr(), result.data_ptr())
                    if prefix and prefix != rows:
                        self.assertEqual(sparse.call_args.kwargs['out'].data_ptr(), result[prefix:].data_ptr())

    def test_runtime_marker_requires_every_dsa_layer(self):
        from engine.profiles.glm53.boot import native_execution_report
        net = NS(layers=[0, 1], dense={'a': NS(executed=3), 'head': NS(executed=True)},
                 mhc=NS(executed={'a', 'b', 'c'}), shared_mlp={1: NS(executed=True)},
                 shared_overlap=NS(executed=True), _router_weights={1: None}, _router_tensorcore={1},
                 F=NS(is_dsa=lambda L: L == 1),
                 prefill_transport=NS(executed={'fp8_all_gather', 'fp8_reduce_scatter'}, project_tiles=False))
        drafter = NS(dense={'fc.weight': NS(executed=3), 'q': NS(executed=1)})
        net.prefill_dense_prefix = True
        net.prefill_dense_prefix_executed = set()
        with self.assertRaisesRegex(RuntimeError, 'dense prefix attention was not executed'):
            native_execution_report(net, drafter)
        net.prefill_dense_prefix_executed = {L for L in net.layers if net.F.is_dsa(L)}
        self.assertEqual(native_execution_report(net, drafter)['prefill_dense_prefix'],
                         sorted(net.prefill_dense_prefix_executed))


class PrefixKernelTests(unittest.TestCase):
    def test_actual_served_mla_skips_only_redundant_prefill_copy_and_writes_owned_output(self):
        source = ROOT/'engine/profiles/glm53/lanes.py'
        node = copy.deepcopy(next(n for n in ast.walk(ast.parse(source.read_text()))
                                  if isinstance(n, ast.FunctionDef) and n.name == 'mla'))
        capturing = [False]
        parts = []
        def decode(q, *args, out=None):
            result = (q.float()+1).bfloat16()
            if out is not None:
                out.copy_(result)
                result = out
            parts.append(result)
            return result
        concatenate = Mock(wraps=torch.cat)
        scope = dict(torch=NS(uint8=torch.uint8, cat=concatenate,
                              cuda=NS(is_current_stream_capturing=lambda: capturing[0])),
                     mk=NS(MLA_H=16, _ARMED={'mla': True}, maybe_arm=lambda: None, mla_decode=decode))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        lane = scope['mla']
        for rows, heads, captured in ((131, 16, False), (131, 64, False), (7, 16, False), (131, 16, True)):
            capturing[0] = captured
            q = torch.zeros(rows, heads, 32).bfloat16()
            parts.clear()
            concatenate.reset_mock()
            result = lane(q, torch.zeros(1, 32, dtype=torch.uint8), None, None, .1, 1.)
            self.assertTrue(bool((result == 1).all()))
            self.assertEqual(concatenate.call_count, int(heads != 16 or rows < 128 or captured))
            if heads == 16 and rows >= 128 and not captured:
                self.assertIs(result, parts[0])
                self.assertNotEqual(result.data_ptr(), q.data_ptr())
                again = lane(q, torch.zeros(1, 32, dtype=torch.uint8), None, None, .1, 1.)
                self.assertNotEqual(result.data_ptr(), again.data_ptr())
        capturing[0] = False
        q = torch.zeros(131, 16, 32).bfloat16()
        owner = torch.full((133, 16, 32), -7, dtype=torch.bfloat16)
        destination = owner[1:132]
        concatenate.reset_mock()
        result = lane(q, torch.zeros(1, 32, dtype=torch.uint8), None, None, .1, 1., out=destination)
        self.assertIs(result, destination)
        self.assertEqual(concatenate.call_count, 0)
        self.assertTrue(bool((owner[0] == -7).all() & (owner[-1] == -7).all()))
        for bad in (q.float(), q[:1], q.transpose(0, 1)):
            with self.assertRaises(ValueError):
                lane(q, torch.zeros(1, 32, dtype=torch.uint8), None, None, .1, 1., out=bad)

    def test_actual_kernel_body_masks_pages_and_matches_independent_attention(self):
        # Execute the real Triton body with checked CPU loads/stores. Dot uses
        # exact BF16 operands and FP32 accumulation; GPU MMA lowering is separate.
        torch.set_num_threads(1)
        class Pointer:
            def __init__(self, data, offset=0):
                self.data, self.offset = data.reshape(-1), offset
            def __add__(self, offset):
                return Pointer(self.data, self.offset + offset)
        def load(pointer, mask, other=0):
            index, mask = torch.broadcast_tensors(pointer.offset, mask)
            self.assertTrue(bool(((index[mask] >= 0) & (index[mask] < pointer.data.numel())).all()))
            values = torch.full(index.shape, other, dtype=torch.float32)
            values[mask] = pointer.data.float()[index[mask]]
            return values.to(pointer.data.dtype)
        def store(pointer, values, mask):
            index, mask = torch.broadcast_tensors(pointer.offset, mask)
            self.assertTrue(bool(((index[mask] >= 0) & (index[mask] < pointer.data.numel())).all()))
            pointer.data[index[mask]] = values.expand(index.shape)[mask].to(pointer.data.dtype)
        program = [0]
        tl = NS(program_id=lambda d: program[d], arange=torch.arange, int64=torch.int64,
                float32=torch.float32, bfloat16=torch.bfloat16, load=load, store=store,
                full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
                zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
                minimum=lambda a, b: min(a, b), maximum=lambda a, b: torch.maximum(torch.as_tensor(a), torch.as_tensor(b)),
                cdiv=lambda a, b: (a+b-1)//b, exp=torch.exp, where=torch.where,
                max=lambda x, dim: x.amax(dim), sum=lambda x, dim: x.sum(dim), trans=lambda x: x.T,
                dot=lambda a, b, c=0: a.float() @ b.float() + c)
        source = ROOT/'engine/kernels/mla/prefill_dense.py'
        node = copy.deepcopy(next(n for n in ast.parse(source.read_text()).body
                                  if isinstance(n, ast.FunctionDef) and n.name == '_dense_prefix'))
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        scope = dict(tl=tl)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        run = scope['_dense_prefix']
        generator = torch.Generator().manual_seed(8821)
        for rows, context, dim in ((1, 0, 512), (3, 3, 512), (5, 31, 512), (3, 2048, 512), (131, 0, 32)):
            for bm in (32, 64):
                for identity in (False, True):
                    heads, block, stride, offset = 16, 16, 48, 16
                    nblocks = (context+rows+block-1)//block
                    blocks = torch.randperm(nblocks, generator=generator).to(torch.int32)
                    latent = torch.randn(nblocks*stride, dim, generator=generator).to(torch.float8_e4m3fn)
                    q = (torch.randn(rows, heads, dim, generator=generator)*.5).bfloat16()
                    output = torch.full_like(q, float('nan'))
                    kv_scale, scale = (1.25 if identity else 1.), .07
                    for pid in range((rows*heads+bm-1)//bm):
                        program[0] = pid
                        run(Pointer(q), Pointer(latent), Pointer(blocks), Pointer(output), rows, context,
                            scale, kv_scale, block, stride, offset, identity, heads, dim, bm, 32)
                    # Independent full softmax over logical positions, without online tiles.
                    positions = torch.arange(context+rows)
                    physical = positions if identity else blocks[positions//block].long()*stride+offset+positions%block
                    kv = latent.float()[physical]*kv_scale
                    scores = torch.einsum('thd,kd->thk', q.float(), kv)*scale
                    scores.masked_fill_(positions[None, None, :] > (context+torch.arange(rows))[:, None, None], -float('inf'))
                    reference = torch.einsum('thk,kd->thd', scores.softmax(-1), kv)
                    self.assertTrue(bool(torch.isfinite(output).all()))
                    relative = float((output.float()-reference).norm()/reference.norm())
                    self.assertLess(relative, .006, (rows, context, dim, bm, identity, relative))

    def test_native_wrapper_rejects_wrong_dtype_geometry_page_table_and_cpu(self):
        try:
            from engine.kernels.mla.prefill_dense import mla_dense_prefix
        except ImportError:
            self.skipTest('Triton is an ST image dependency')
        q = torch.zeros(131, 16, 512).bfloat16()
        latent = torch.zeros(256, 512).to(torch.float8_e4m3fn)
        table = torch.tensor([0], dtype=torch.int32)
        args = (q, latent, table, 256, 256, 0, 0, .1, 1.)
        with self.assertRaisesRegex(ValueError, 'eager CUDA'):
            mla_dense_prefix(*args)
        for index, value in ((0, q.float()), (1, latent.bfloat16()), (2, table.long()),
                             (2, table[:0]), (6, 2051), (7, float('nan')), (8, 0.)):
            bad = list(args)
            bad[index] = value
            with self.assertRaises(ValueError):
                mla_dense_prefix(*bad)


class PrefixModelTests(unittest.TestCase):
    def test_four_rank_model_preserves_state_auxiliary_and_following_decode(self):
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
                net.prefill_dense_prefix = net.prefill_indexer_shards = True
                actual = run()
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(cache.state, state, rtol=0, atol=0)
                torch.testing.assert_close(cache.paged, pages, rtol=0, atol=0)
                cache.prepare(follow)
                torch.testing.assert_close(net.forward(follow, cache), next_hidden, rtol=0, atol=0)
                torch.testing.assert_close(cache.state, final, rtol=0, atol=0)
                torch.testing.assert_close(cache.paged, final_pages, rtol=0, atol=0)
                self.assertEqual(net.prefill_dense_prefix_executed, {1})
            LocalTP(4, timeout_s=30).run(rank)


if __name__ == '__main__':
    unittest.main()
