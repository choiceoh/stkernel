"""MoE finalization retains BF16 rounding and consumes borrowed state promptly."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

import torch


class MoeOutputContractTests(unittest.TestCase):
    def test_packet_api_consumes_one_exchange_and_blocks_interleaving(self):
        from engine.kernels.oneshot import OneShot
        owner = OneShot.__new__(OneShot)
        owner.pending, owner.packet_failed, owner.closed = None, False, False
        owner.hidden, owner.world = 4096, 4
        owner.eligible = lambda value: True
        acc = torch.empty(8, 4096)
        shared = torch.empty(8, 4096).bfloat16()
        descriptor, events = object(), []
        owner.ext = NS(healthy=lambda: True, moe_packets=lambda a, s: events.append((a, s)) or descriptor)
        with patch('torch.cuda.current_stream', return_value=NS(cuda_stream=19)):
            packet = owner.exchange_moe(acc, shared)
            self.assertEqual(len(events), 1)
            self.assertIs(events[0][0], acc)
            with self.assertRaisesRegex(RuntimeError, 'previous rank-packet'):
                owner.exchange_moe(acc, shared)
            self.assertIs(packet.consume(lambda source, desc: desc), descriptor)
            owner.assert_consumed()
            with self.assertRaises(ValueError):
                owner.exchange_moe(shared, shared)

    def test_default_preserves_four_rank_aux_and_recurrent_state(self):
        from dataclasses import replace
        from types import MethodType
        from engine.base.comm import LocalTP
        from engine.profiles.glm53.net import Step
        from engine.profiles.glm53.direct_mhc import decode_direct
        from tests.test_engine_direct_mhc import OracleTransport, oracle_consume
        from tests.test_engine_execution_plans import model
        torch.set_num_threads(1)
        def rank(comm):
            for count, tokens in ((c, t) for c in (1, 2, 3, 4) for t in (1, 8)):
                net, cache = model(('kda', 'dsa', 'kda'), comm=comm)
                net.F = replace(net.F, dense=())
                finalized = []
                def moe(self, layer, x, reduce=None, *, finalize=None):
                    acc = x.float() * (layer+1) * .01
                    shared = (x.float() * .03).bfloat16()
                    if finalize is not None:
                        finalized.append(layer)
                    return (finalize(acc, shared) if finalize else
                            (reduce or self.comm.all_reduce)(acc.bfloat16()+shared))
                net._moe = MethodType(moe, net)
                chunks = []
                for seq in range(count):
                    slot = cache.slots.take(seq)
                    cache.pool.reserve(seq, tokens)
                    chunks.append(((torch.arange(tokens)+seq) % net.vp, 0, seq, slot))
                step = Step.decode(chunks)
                cache.prepare(step)
                state, paged = cache.state.clone(), cache.paged.clone()
                expected = net.forward(step, cache, aux_layers=[0, 2])
                after, paged_after = cache.state.clone(), cache.paged.clone()
                cache.state.copy_(state); cache.paged.copy_(paged)
                transport = OracleTransport(comm)
                transport.exchange_moe = lambda a, s: transport.exchange(a.bfloat16()+s)
                comm.transport = transport
                try:
                    actual = decode_direct(net, step, cache, [0, 2], consumer=oracle_consume)
                finally:
                    comm.transport = None
                self.assertEqual(finalized, [0, 1, 2])  # tensor, packet, final tensor
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertTrue(torch.equal(cache.state, after))
                self.assertTrue(torch.equal(cache.paged, paged_after))
        with patch('engine.kernels.moe_output.combine', side_effect=lambda a, s: a.bfloat16()+s):
            LocalTP(4).run(rank)

    def test_explicit_finalizer_refuses_other_layouts_before_launch(self):
        from engine.kernels.moe_output import validate_finalizer
        args = dict(rows=8, experts=288, local_experts=288, hidden=4096,
                    intermediate=512, topk=8, quant_mode='nvfp4',
                    activation='swigluoai_uninterleave', limit=10., alpha=1., beta=0., tiled=True)
        for rows in (1, 8, 16, 24, 32):
            validate_finalizer(lambda value: value, **dict(args, rows=rows))
        for name, value in (('rows', 33), ('rows', 0), ('rows', True), ('experts', 8),
                            ('local_experts', 72), ('hidden', 2048), ('intermediate', 1024),
                            ('topk', 1), ('quant_mode', 'w4a16'), ('tiled', False),
                            ('activation', 'silu'), ('limit', 8.), ('alpha', 1.702), ('beta', 1.)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_finalizer(lambda value: value, **dict(args, **{name: value}))
        with self.assertRaises(ValueError):
            validate_finalizer(True, **args)

    def test_shared_stream_is_joined_before_finalize_and_after_failure(self):
        from engine.kernels.dense.shared_mlp import SharedOverlap
        events = []
        parent = NS(wait_stream=lambda stream: events.append('join'))
        side = NS(wait_stream=lambda stream: events.append('fork'))
        x = NS(device='cuda', shape=(16, 4096), record_stream=lambda stream: events.append('x_owned'))
        failing = NS(device='cuda', shape=(8, 4096), record_stream=x.record_stream)
        shared = NS(record_stream=lambda stream: events.append('shared_owned'))
        owner = SharedOverlap.__new__(SharedOverlap)
        owner.stream, owner.executed, owner.rows = side, False, set()
        def finish(acc, partial):
            self.assertIs(partial, shared)
            self.assertEqual(events[-2:], ['join', 'shared_owned'])
            events.append('finish')
            return 'packet'
        with patch('torch.cuda.current_stream', return_value=parent), patch('torch.cuda.stream', return_value=nullcontext()):
            actual = owner(lambda value: shared, x, lambda consume: consume('FP32'), finish=finish)
            self.assertEqual(actual, 'packet')
            self.assertTrue(owner.executed)
            self.assertEqual(owner.rows, {16})
            def fail(acc, partial):
                self.assertEqual(events[-2:], ['join', 'shared_owned'])
                raise RuntimeError('finalizer failed')
            with self.assertRaisesRegex(RuntimeError, 'finalizer failed'):
                owner(lambda value: shared, failing, lambda consume: consume('FP32'), finish=fail)
            with self.assertRaisesRegex(RuntimeError, 'did not consume'):
                owner(lambda value: shared, failing, lambda consume: None, finish=finish)
            self.assertEqual(events[-1], 'join')
            def twice(consume):
                consume('FP32')
                return consume('FP32')
            with self.assertRaisesRegex(RuntimeError, 'exactly once'):
                owner(lambda value: shared, failing, twice, finish=finish)
            # A join that failed is not a width the boot proof may count.
            self.assertEqual(owner.rows, {16})

    def test_wide_profile_consumes_before_next_expert_invocation(self):
        from engine.profiles.glm53.net import Glm53Net
        events = []
        acc = torch.randn(16, 8)
        x = torch.randn(16, 8).bfloat16()
        def expert(value, ids, weights, *, finalize):
            events.append('expert')
            result = finalize(acc)
            events.append('released')
            return result
        def linear(value, name):
            events.append(name)
            return torch.ones(16, 16 if name.endswith('sh_gate_up') else 8).bfloat16()
        net = NS(F=NS(spec_k=7, swiglu_limit=10.), p={}, shared_overlap=None,
                 route=lambda *args: (None, None), _experts={3: expert}, linear=linear,
                 _activation=lambda g, u, limit: g, comm=NS(all_reduce=lambda value: value))
        def finish(a, b):
            self.assertIs(a, acc)
            events.append('finish')
            return (a.bfloat16() + b)
        out = Glm53Net._moe(net, 3, x, finalize=finish)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(events, ['expert', 'L3.moe.sh_gate_up', 'L3.moe.sh_down', 'finish', 'released'])

    def test_shared_overlap_takes_c1_and_c2_verify_rows_and_leaves_wider_batches_serial(self):
        from itertools import product
        from engine.profiles.glm53.net import Glm53Net, shared_overlap_rows
        self.assertEqual([m for m in range(1, 33) if shared_overlap_rows(m, 7)], [1, 2, 3, 4, 5, 6, 7, 8, 16])
        self.assertEqual([m for m in range(1, 33) if shared_overlap_rows(m, 7, c2=False)], list(range(1, 9)))
        self.assertEqual([m for m in range(1, 65) if shared_overlap_rows(m, 15)], list(range(1, 17)) + [32])
        for rows, c2, finalized in product(range(1, 33), (True, False), (True, False)):
            events = []
            x = torch.randn(rows, 8).bfloat16()
            acc = torch.randn(rows, 8)
            def expert(value, ids, weights, *, finalize=None):
                events.append('expert')
                return acc.bfloat16() if finalize is None else finalize(acc)
            def overlap(shared, value, routed, *, finish=None):
                self.assertIs(value, x)
                events.append('overlap')
                if finish is None:
                    return routed() + shared(value)
                return routed(lambda partial: finish(partial, shared(value)))
            def linear(value, name):
                events.append(name)
                return torch.ones(rows, 16 if name.endswith('sh_gate_up') else 8).bfloat16()
            net = NS(F=NS(spec_k=7, swiglu_limit=10.), p={}, shared_overlap=overlap,
                     shared_mlp={3: lambda value: torch.ones(rows, 8).bfloat16()},
                     route=lambda *args: (None, None), _experts={3: expert}, linear=linear,
                     _activation=lambda g, u, limit: g, comm=NS(all_reduce=lambda value: value))
            kwargs = dict(finalize=lambda a, b: a.bfloat16() + b) if finalized else {}
            if not c2:
                kwargs['c2_overlap'] = False         # the probe's same-build control
            with self.subTest(rows=rows, c2=c2, finalized=finalized):
                out = Glm53Net._moe(net, 3, x, **kwargs)
                self.assertEqual(out.shape, x.shape)
                overlapped = rows <= 8 or (c2 and rows == 16)
                self.assertEqual(events, ['overlap', 'expert'] if overlapped else
                                 ['expert', 'L3.moe.sh_gate_up', 'L3.moe.sh_down'])


@unittest.skipUnless(torch.cuda.is_available(), 'requires admitted GB10')
class MoeOutputCudaTests(unittest.TestCase):
    @staticmethod
    def values(rows):
        # Halfway BF16 ties, cancellation, both signs, and subnormal outputs.
        seeds = torch.tensor([0., -0., 1., 1.+2**-8, 1.+3*2**-8, -1.-2**-8,
                              2**-133, -2**-133, 2**-126, 1e20, -1e20], device='cuda')
        acc = seeds.repeat((rows*4096+len(seeds)-1)//len(seeds))[:rows*4096].reshape(rows, 4096)
        shared = (-acc.roll(3, 1)).bfloat16()
        return acc, shared

    def test_both_rounding_boundaries_are_exact_with_changed_replay(self):
        from engine.kernels.moe_output import combine
        for rows in (1, 8, 16, 24, 32):
            acc, shared = self.values(rows)
            storage = torch.full((rows*4096+32,), 77., dtype=torch.bfloat16, device='cuda')
            out = storage[16:-16].view(rows, 4096)
            combine(acc, shared, out=out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                combine(acc, shared, out=out)
            try:
                for i in range(4):
                    if i:
                        acc.normal_(); shared.normal_()
                    expected = acc.bfloat16() + shared
                    out.fill_(float('nan'))
                    graph.replay()
                    self.assertTrue(torch.equal(out.view(torch.uint8), expected.view(torch.uint8)))
                    self.assertTrue(torch.all(storage[:16] == 77.) and torch.all(storage[-16:] == 77.))
                # The first BF16 boundary is observably necessary.
                acc.fill_(1.+2**-8); shared.fill_(-1.)
                graph.replay()
                self.assertEqual(out.count_nonzero().item(), 0)
                self.assertGreater((acc + shared.float()).bfloat16().count_nonzero().item(), 0)
                with self.assertRaises(ValueError):
                    combine(acc, shared, out=shared)
            finally:
                graph.reset()



if __name__ == '__main__':
    unittest.main()
