"""M2 uses the profile's existing dense readers, routing and quantizer binding."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from engine.kernels.dense import DenseLinear, W4Pack
from engine.modules.mixed_experts import ExpertInvocation
from engine.profiles.glm53 import lanes
from engine.profiles.glm53.mixed_shared import BoundMixedShared
from tests.test_glm53_modelopt_serving import model


class RecordingDense(DenseLinear):
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.decode_precision, self.observer = 'w4', None
        self.decode_input_rows, self.workspace, self.decode_fp8 = (), None, None
        self.packs = (W4Pack(torch.ones(1), torch.ones(1), torch.ones(1), rows, cols),)
        self.fp8 = SimpleNamespace(weight=(torch.ones(1), torch.ones(1)), observer=None)
        self.calls = []

    def __call__(self, x):
        self.calls.append(len(x))
        return torch.full((len(x), self.rows), 2., dtype=x.dtype)


class RecordingOverlap:
    def __init__(self):
        self.stream, self.calls = object(), []

    def __call__(self, fused, x, routed):
        self.calls.append((fused, len(x)))
        return routed() + 7.


def shared_net():
    up, down, overlap = RecordingDense(1024, 4096), RecordingDense(4096, 512), RecordingOverlap()
    return SimpleNamespace(dense={'L3.moe.sh_gate_up': up, 'L3.moe.sh_down': down},
        F=SimpleNamespace(spec_k=7, swiglu_limit=10.), shared_overlap=overlap,
        shared_mlp={3: SimpleNamespace(gate_up=up, down=down, limit=10.)},
        _activation=lambda g, v, limit: g*v)


class SharedReaderTests(unittest.TestCase):
    def test_c1_keeps_overlap_wide_decode_and_prefill_use_original_readers(self):
        net = shared_net(); shared = BoundMixedShared(net, 3)
        for rows, expected in ((8, 10.), (32, 5.)):
            x = torch.ones(rows, 4096, dtype=torch.bfloat16)
            self.assertTrue(bool((shared.decode(x, lambda: torch.full_like(x, 3.)) == expected).all()))
        self.assertEqual(len(net.shared_overlap.calls), 1)
        self.assertEqual(shared.up.calls, [32]); self.assertEqual(shared.down.calls, [32])
        self.assertTrue(bool((shared.prefill(torch.ones(129, 4096)) == 2.).all()))
        self.assertEqual(shared.up.calls, [32, 129])

    def test_reader_replacement_tensor_mutation_and_stream_replacement_refuse_execution(self):
        mutations = (
            lambda n: n.dense.__setitem__('L3.moe.sh_down', RecordingDense(4096, 512)),
            lambda n: n.dense['L3.moe.sh_gate_up'].packs[0].scale.add_(1),
            lambda n: setattr(n.dense['L3.moe.sh_down'], 'fp8', SimpleNamespace(weight=(torch.ones(1),), observer=None)),
            lambda n: setattr(n.shared_overlap, 'stream', object()),
            lambda n: setattr(n.shared_mlp[3], 'limit', 8.),
            lambda n: setattr(n.F, 'spec_k', 5),
            lambda n: setattr(n, '_activation', lambda g, v, limit: g),
        )
        for mutate in mutations:
            net = shared_net(); shared = BoundMixedShared(net, 3); mutate(net)
            with self.subTest(mutation=mutate), self.assertRaises(RuntimeError):
                shared.prefill(torch.ones(33, 4096))
            self.assertEqual(shared.up.calls, [])

    def test_native_inference_packs_are_pinned_and_ordinary_packs_track_versions(self):
        net = shared_net()
        with torch.inference_mode():
            tensor = torch.ones(1)
        up = net.dense['L3.moe.sh_gate_up']
        up.packs = (replace(up.packs[0], data=tensor),)
        shared = BoundMixedShared(net, 3); shared.validate()
        self.assertTrue(any(t is tensor for t in shared.resources))
        up.packs = (replace(up.packs[0], data=tensor.clone()),)
        with self.assertRaises(RuntimeError):
            shared.validate()

    def test_observed_or_unprepared_shared_reader_is_not_silently_replaced(self):
        for mode in ('observer', 'fp8_observer', 'no_prefill', 'no_dense'):
            net = shared_net(); up = net.dense['L3.moe.sh_gate_up']
            if mode == 'observer': up.observer = lambda *a: None
            if mode == 'fp8_observer': up.fp8.observer = lambda *a: None
            if mode == 'no_prefill': up.fp8 = None
            if mode == 'no_dense': net.dense.clear()
            with self.subTest(mode=mode), self.assertRaises((ValueError, RuntimeError)):
                BoundMixedShared(net, 3)

    def test_c1_without_overlap_uses_the_same_sequential_readers(self):
        net = shared_net(); net.shared_overlap = None
        shared = BoundMixedShared(net, 3)
        x = torch.ones(8, 4096)
        self.assertTrue(bool((shared.decode(x, lambda: torch.full_like(x, 3.)) == 5.).all()))
        self.assertEqual(shared.up.calls, [8])

    def test_prefill_fork_uses_existing_join_helper_and_does_not_add_partial_twice(self):
        net = shared_net()
        calls = []
        class JoinedOverlap:
            stream = object()
            def __call__(self, shared, x, routed, *, finish):
                calls.append('shared')
                partial = shared(x)
                def consume(value):
                    calls.append('joined')
                    return finish(value, partial)
                return routed(consume)
        net.shared_overlap = JoinedOverlap()
        bound = BoundMixedShared(net, 3)
        result = bound.prefill_during(torch.ones(129, 4096), lambda: calls.append('cold'))
        self.assertTrue(bool((result == 2.).all()))
        self.assertEqual(calls, ['shared', 'cold', 'joined'])
        self.assertEqual(bound.up.calls, [129])
        self.assertEqual(bound.down.calls, [129])
        net = shared_net(); net.shared_overlap = None
        bound = BoundMixedShared(net, 3)
        with self.assertRaisesRegex(ValueError, 'side stream'):
            bound.prefill_during(torch.ones(129, 4096), lambda: self.fail('unbound dispatch'))


class ProfileBindingTests(unittest.TestCase):
    def test_prepared_factory_uses_same_bound_weights_scales_and_real_route_method(self):
        factory = Mock(return_value=object())
        table = replace(lanes.reference(), moe_mixed_prepare=factory)
        net, views = model(3, table); net.bind(views)
        x, p = torch.ones(1, 128), torch.ones(33, 128)
        ids, routes = torch.tensor([[0, 1]], dtype=torch.int32), torch.tensor([[.25, .75]])
        net.route = Mock(return_value=(ids, routes))
        identity = ExpertInvocation(3, 1, 2, 3)
        shared = object()
        with patch('engine.profiles.glm53.mixed_shared.BoundMixedShared', return_value=shared), \
                patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
            self.assertIs(net.prepare_mixed_ffn(3, x, p, identity=identity), factory.return_value)
        self.assertEqual([call.args[0] for call in net.route.call_args_list], [3, 3])
        self.assertIs(net.route.call_args_list[0].args[1], x)
        self.assertIs(net.route.call_args_list[1].args[1], p)
        kw = factory.call_args.kwargs
        self.assertIs(kw['scales'], net._quant_scales[3])
        self.assertIs(kw['w13'], views['L3.moe.w13'])
        self.assertIs(kw['w2'], views['L3.moe.w2'])
        self.assertIs(kw['shared_execution'], shared)
        self.assertEqual(kw['identity'], identity)

    def test_local_prepare_failure_still_enters_admission_vote(self):
        net, views = model(3); net.bind(views)
        scheduler = SimpleNamespace(layer=3, admit=Mock(return_value='voted'))
        identity = ExpertInvocation(3, 1, 2, 3)
        result = net.submit_mixed_ffn(scheduler, None, None, identity=identity, request='a', slot=0)
        self.assertEqual(result, 'voted')
        self.assertIsNone(scheduler.admit.call_args.args[0])
        self.assertIsInstance(scheduler.admit.call_args.kwargs['preparation_error'], ValueError)
        with self.assertRaises(ValueError):
            net.mixed_layer_scheduler(3)


if __name__ == '__main__':
    unittest.main()
