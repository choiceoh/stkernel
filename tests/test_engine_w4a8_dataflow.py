"""Preserve the existing packed dense quantization while changing task execution."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from engine.base.arena import Arena
from engine.base.comm import Comm
from engine.kernels.dense import DenseLinear, W4Pack, _tile_pack
from engine.kernels.dense.packing import _mk_quant_x_ref, mk_w4_dequant
from engine.modules.speculative_tree import Tree
from engine.modules.w4a8_dataflow import W4A8Plan, W4A8PipelinePlan, W4A8Weights, PersistentW4A8, reference, pipeline_reference
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.lanes import reference as lanes_reference, swiglu_clamped
from engine.profiles.glm53.net import Glm53Net
from engine.profiles.glm53.tree_decode import Verification
from tests.test_engine_glm53 import tiny_facts


def pack(rows, cols, *, seed=49):
    gen = torch.Generator().manual_seed(seed)
    padded = (rows+127)//128*128
    codes = torch.randint(0, 16, (padded, cols), dtype=torch.uint8, generator=gen)
    scales = torch.randint(-40, 48, (padded, cols//16), dtype=torch.int8, generator=gen)
    shift = torch.randint(8, 13, (padded,), generator=gen).float()
    return replace(_tile_pack(codes, scales, shift, rows, 0, cols), calibrated=True)


def packed_weights(hidden=256, intermediate=384):
    return W4A8Weights(pack(2*intermediate, hidden), pack(hidden, intermediate, seed=18))


def linear_ref(x, p, *args, **kwargs):
    w = mk_w4_dequant(p.data, p.scale, p.rows, rgs=p.rowscale)
    return (_mk_quant_x_ref(x) @ w.T).bfloat16()


def operator(p):
    # Bind real packed planes to DenseLinear's real dispatch on a CPU host;
    # only its native GEMM leaf is replaced by the deployed numerical twin.
    op = object.__new__(DenseLinear)
    op.rows, op.cols, op.packs = p.rows, p.cols, (p,)
    op.decode_precision, op.observer, op.executed = "w4", None, 0
    op.workspace, op.decode_input_rows, op.bound_input_executed = None, (), set()
    return op


def w4a8_model(*, moe=False):
    f = replace(tiny_facts(), kinds=("kda", "dsa", "kda"), dense_inter=512, moe_inter=512, spec_k=7,
                dense=(0,) if moe else (0, 1, 2))
    net = Glm53Net(f, Comm(4, 0), lanes_reference())
    net.comm = Comm()
    gen = torch.Generator().manual_seed(948)
    values = {}
    for spec in net.specs():
        if spec.dtype == torch.uint8:
            value = torch.randint(0, 256, spec.shape, dtype=spec.dtype, generator=gen)
        elif spec.dtype == torch.float8_e4m3fn:
            value = torch.full(spec.shape, .125).to(spec.dtype)
        else:
            value = (torch.randn(spec.shape, generator=gen)*.04).to(spec.dtype)
        if "norm" in spec.name:
            value.fill_(1)
        values[spec.name] = value
    net.bind(values)
    for layer in net.layers:
        if f.is_moe(layer):
            continue
        w = packed_weights(f.hidden, f.dense_inter//4)
        for name, p in (("gate_up", w.gate_up), ("down", w.down)):
            key = f"L{layer}.mlp.{name}"
            net.dense[key] = operator(p)
            net.p[key] = None   # source arena has already been retired
    storage = layout(f, net.layers)
    cache = Glm53Caches(Arena(storage.nbytes(8, 2)+4096, device="cpu"), f, net.layers, 8, 2)
    return net, cache


class W4A8DataflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(372)

    def test_gptq_planes_fp8_activations_and_bf16_boundaries_match_existing_lane(self):
        w = packed_weights()
        before = tuple(t.clone() for p in (w.gate_up, w.down) for t in (p.data, p.scale, p.rowscale))
        for rows in (1, 4, 17, 32):
            x = torch.randn(rows, w.hidden).bfloat16()
            x[0, :128] = 0  # exact zero group must not become NaN
            g, u = linear_ref(x, w.gate_up).chunk(2, -1)
            expected = linear_ref(swiglu_clamped(g, u, 10.), w.down)
            plan = W4A8Plan(rows, w.hidden, w.intermediate)
            staged = pipeline_reference(W4A8PipelinePlan(rows, w.hidden, w.intermediate), x, w, 10.)
            for seed in (0, 8):
                actual, order = reference(plan, x, w, 10., seed=seed)
                torch.testing.assert_close(actual, expected, rtol=.008, atol=2e-5)
                torch.testing.assert_close(staged, actual, rtol=0, atol=0)
                self.assertEqual(sorted(order), list(range(len(plan.tasks))))
        for got, old in zip((t for p in (w.gate_up, w.down) for t in (p.data, p.scale, p.rowscale)), before):
            self.assertTrue(torch.equal(got, old))

    def test_pipeline_admission_fits_without_the_queued_partial_plane(self):
        queued, staged = W4A8Plan(16, 4096, 3072), W4A8PipelinePlan(16, 4096, 3072)
        self.assertEqual(staged.scratch_bytes, 249344)
        self.assertEqual(queued.scratch_bytes-staged.scratch_bytes, 6291456+240-67584)
        W4A8PipelinePlan(16, 4096, 3072, max_scratch_bytes=256 << 10)
        with self.assertRaisesRegex(ValueError, "scratch"):
            W4A8Plan(16, 4096, 3072, max_scratch_bytes=256 << 10)

    def test_borrow_existing_packs_and_refuse_incompatible_lane_or_observer(self):
        net, _ = w4a8_model()
        w = W4A8Weights.from_net(net, 0)
        op = net.dense["L0.mlp.gate_up"]
        self.assertIs(w.gate_up, op.packs[0])
        self.assertTrue(w.gate_up.calibrated)
        self.assertIsNone(net.p["L0.mlp.gate_up"])
        op.smooth = torch.ones(op.cols)*2
        self.assertIs(W4A8Weights.from_net(net, 0).gate_up, w.gate_up)
        for field, value in (("observer", lambda *_: None), ("decode_precision", "fp8"), ("packs", ())):
            old = getattr(op, field)
            setattr(op, field, value)
            with self.assertRaisesRegex(ValueError, "single-pack"):
                W4A8Weights.from_net(net, 0)
            setattr(op, field, old)
        net.dense_nvfp4 = True
        with self.assertRaisesRegex(ValueError, "W4A4"):
            W4A8Weights.from_net(net, 0)

    def test_full_target_tree_uses_packed_w4a8_with_retired_bf16_weights(self):
        net, cache = w4a8_model()
        slot = cache.slots.take(0)
        cache.pool.reserve(0, 16)
        tree = Tree((1, 2, 3, 4), (-1, 0, 0, 2))
        with patch("engine.kernels.dense.w4_gemm", side_effect=linear_ref):
            with Verification(net, cache, tree, seq=0, slot=slot, context=0) as run:
                expected = run.verify()
            binding = PersistentW4A8(backend="reference")
            with Verification(net, cache, tree, seq=0, slot=slot, context=0, persistent_mlp=binding) as run:
                actual = run.verify()
                torch.testing.assert_close(actual, expected, atol=.008, rtol=.008)
                self.assertTrue(torch.equal(net.head_tokens(actual), net.head_tokens(expected)))
                result = run.commit(budget=3)
                self.assertGreaterEqual(result["context"], 1)
            self.assertEqual(binding.executed, set(net.layers))
            self.assertTrue(all(op.executed & 1 for op in net.dense.values()))

    def test_owner_replacement_and_group_or_scratch_change_are_refused(self):
        net, _ = w4a8_model()
        binding = PersistentW4A8(backend="reference")
        binding.validate(net, 4)
        net.dense["L0.mlp.down"].packs = (pack(128, 128),)
        with self.assertRaisesRegex(RuntimeError, "owner changed"):
            binding(net, 0, torch.zeros(4, 128, dtype=torch.bfloat16))
        for kw in (dict(tile=64), dict(intermediate=127), dict(max_scratch_bytes=1)):
            args = dict(rows=16, hidden=4096, intermediate=3072)
            args.update(kw)
            with self.assertRaises(ValueError):
                W4A8Plan(**args)
        w = packed_weights()
        with self.assertRaisesRegex(ValueError, "signed scale"):
            W4A8Weights(replace(w.gate_up, scale=w.gate_up.scale.view(torch.uint8)), w.down)

    def test_routed_experts_stay_exact_and_observations_do_not_leak_into_recall(self):
        from engine.modules.speculative_tree import RouteTable
        net, cache = w4a8_model(moe=True)
        slot = cache.slots.take(0)
        cache.pool.reserve(0, 16)
        tree = Tree((1, 2, 3, 4), (-1, 0, 0, 2))
        table = RouteTable("mixed-w4a8", min_observations=1)
        binding = PersistentW4A8(backend="reference")
        before, paged = cache.state.clone(), cache.paged.clone()
        with patch("engine.kernels.dense.w4_gemm", side_effect=linear_ref):
            with Verification(net, cache, tree, seq=0, slot=slot, context=0) as run:
                expected = run.verify()
                routes = {L: r.clone() for L, r in run.routes.items()}
            for repeat in range(2):
                cache.state.copy_(before); cache.paged.copy_(paged)
                with Verification(net, cache, tree, seq=0, slot=slot, context=0, persistent_mlp=binding) as run:
                    actual = run.verify()
                    torch.testing.assert_close(actual, expected, atol=.008, rtol=.008)
                    self.assertEqual(set(run.routes), {1, 2})
                    self.assertTrue(all(torch.equal(routes[L], run.routes[L]) for L in routes))
                    result = run.commit(budget=1, predictor=table, runtime_id="mixed-w4a8")
                self.assertEqual(result["actual_route_uses"], 4*2*net.F.topk_experts)
                self.assertEqual(result["route_prediction_nodes"], repeat)
                self.assertEqual(result["route_prediction_recall"], 1. if repeat else None)
            self.assertEqual(binding.executed, {0})


if __name__ == "__main__":
    unittest.main()
