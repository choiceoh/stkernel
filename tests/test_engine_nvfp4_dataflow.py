"""Real NVFP4 byte/layout semantics in the persistent MLP, without a GPU queue."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

import torch

from engine.base.arena import Arena
from engine.base.comm import Comm
from engine.modules.expert_layout import TILE_MAJOR_ATTR, W13_K_IN_BYTES, W2_K_IN_BYTES
from engine.modules.nvfp4_dataflow import NVFP4Plan, NVFP4Weights, PersistentNVFP4, reference
from engine.modules.nvfp4_sf import swizzle_sf
from engine.modules.speculative_tree import Tree
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.lanes import reference as lanes_reference
from engine.profiles.glm53.modelopt_scales import ModelOptScales
from engine.profiles.glm53.net import Glm53Net
from engine.profiles.glm53.tree_decode import Verification, decode_once
from engine.profiles.glm53.weights import MODELOPT_WEIGHT_LAYOUT
from tests.test_engine_glm53 import tiny_facts


def packed_weights(hidden=512, intermediate=256, *, tiled=False, sf6=False):
    gen = torch.Generator().manual_seed(371)
    w13 = torch.randint(0, 256, (1, 2*intermediate, hidden//2), dtype=torch.uint8, generator=gen)
    w2 = torch.randint(0, 256, (1, hidden, intermediate//2), dtype=torch.uint8, generator=gen)
    s13 = swizzle_sf(torch.randint(40, 60, (2*intermediate, hidden//16), dtype=torch.uint8, generator=gen))[None]
    s2 = swizzle_sf(torch.randint(40, 60, (hidden, intermediate//16), dtype=torch.uint8, generator=gen))[None]
    scales = ModelOptScales.bind(*(torch.tensor([v]) for v in (.03, .7, .05, .4)), experts=1, device=torch.device("cpu"))
    if tiled:
        def tile(w, unit):
            e, rows, kb = w.shape
            out = w.reshape(e, rows, kb//unit, unit).permute(0, 2, 1, 3).contiguous().view_as(w)
            setattr(out, TILE_MAJOR_ATTR, "plain")
            return out
        w13, w2 = tile(w13, W13_K_IN_BYTES), tile(w2, W2_K_IN_BYTES)
    if sf6:
        # Existing independent byte oracle loads this source without importing
        # the CUDA-only b12x package entry point.
        from tests.test_moe_static_sf6_direct import sf6 as oracle
        def pack(raw, rows, k, kind):
            nr, nk = oracle.stage_shape(rows, k, kind)
            stages = oracle._stage_rows(raw, experts=1, rows=rows, k=k, kind=kind, first=0, last=nr*nk)
            return torch.tensor([list(oracle.pack_stage_bytes(bytes(s.tolist()))) for s in stages], dtype=torch.uint8)[None]
        s13, s2 = pack(s13, 2*intermediate, hidden, "fc1"), pack(s2, hidden, intermediate, "fc2")
    return NVFP4Weights(w13, w2, s13, s2, scales, W13_K_IN_BYTES if tiled else 0,
                        W2_K_IN_BYTES if tiled else 0, sf6)


def nvfp4_model():
    f = replace(tiny_facts(), kinds=("kda", "dsa", "kda"), dense_inter=512,
                spec_k=7, weight_layout=MODELOPT_WEIGHT_LAYOUT)
    net = Glm53Net(f, Comm(4, 0), lanes_reference())
    net.comm = Comm()
    gen = torch.Generator().manual_seed(948)
    values = {}
    for spec in net.specs():
        if spec.dtype == torch.uint8:
            value = torch.randint(0, 256, spec.shape, dtype=spec.dtype, generator=gen)
        elif spec.dtype == torch.float8_e4m3fn:
            value = torch.full(spec.shape, .125).to(spec.dtype)
        elif spec.name.endswith(("_alpha", "_scale")) and ".mlp." in spec.name:
            value = torch.full(spec.shape, .25, dtype=spec.dtype)
        else:
            value = (torch.randn(spec.shape, generator=gen)*.04).to(spec.dtype)
            if "norm" in spec.name:
                value.fill_(1)
        values[spec.name] = value
    net.bind(values)
    plan = layout(f, net.layers)
    cache = Glm53Caches(Arena(plan.nbytes(8, 2)+4096, device="cpu"), f, net.layers, 8, 2)
    return net, cache


class NVFP4DataflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_up_gate_scales_and_activation_quantization_match_existing_nvfp4_lane(self):
        w = packed_weights()
        x = torch.randn(4, w.hidden).bfloat16()
        plan = NVFP4Plan(4, w.hidden, w.intermediate)
        expected = lanes_reference().moe(x, torch.zeros(4, 1, dtype=torch.int32), torch.ones(4, 1),
            w.w13, w.sf13.view(torch.float8_e4m3fn), w.w2, w.sf2.view(torch.float8_e4m3fn), 10., scales=w.scales)
        for seed in (0, 1, 2):
            actual, order = reference(plan, x, w, 10., seed=seed)
            torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-5)
            self.assertEqual(sorted(order), list(range(len(plan.tasks))))

    def test_tiled_weights_and_lossless_sf6_match_raw_without_changing_global_scales(self):
        raw = packed_weights()
        tiled = packed_weights(tiled=True)
        packed = packed_weights(tiled=True, sf6=True)
        x = torch.randn(3, raw.hidden).bfloat16()
        plan = NVFP4Plan(3, raw.hidden, raw.intermediate)
        expected, _ = reference(plan, x, raw, 10.)
        for other in (tiled, packed):
            self.assertTrue(torch.equal(other.raw_scales_cpu().view(torch.uint8), raw.raw_scales_cpu().view(torch.uint8)))
            self.assertTrue(torch.equal(other.raw_scales_cpu(second=True).view(torch.uint8), raw.raw_scales_cpu(second=True).view(torch.uint8)))
            actual, _ = reference(plan, x, other, 10.)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_binding_borrows_weight_bytes_and_uses_retired_scales_owner(self):
        w = packed_weights(tiled=True, sf6=True)
        a, b = torch.empty(1, 512*32, dtype=torch.uint8), torch.empty(1, 512*16, dtype=torch.uint8)
        a._st_sf6_consumed = b._st_sf6_consumed = True
        net = NS(dense_nvfp4=True, F=NS(is_moe=lambda _: False),
                 p={"L0.mlp.w13": w.w13, "L0.mlp.w2": w.w2, "L0.mlp.w13_sf": a, "L0.mlp.w2_sf": b},
                 _quant_scales={0: w.scales}, _expert_views={0: NS(reform_scales=NS(enabled=True, fc1=w.sf13, fc2=w.sf2))})
        got = NVFP4Weights.from_net(net, 0)
        self.assertEqual(got.w13.data_ptr(), w.w13.data_ptr())
        self.assertEqual(got.w2.data_ptr(), w.w2.data_ptr())
        self.assertEqual(got.sf13.data_ptr(), w.sf13.data_ptr())
        self.assertTrue(got.sf6)
        net._expert_views.clear()
        with self.assertRaisesRegex(ValueError, "actual prepared SF6 owner"):
            NVFP4Weights.from_net(net, 0)

    def test_entire_glm_tree_executes_nvfp4_dense_binding_and_commits(self):
        net, cache = nvfp4_model()
        slot = cache.slots.take(0)
        cache.pool.reserve(0, 8)
        tree = Tree((1, 2, 3, 4), (-1, 0, 0, 2))
        with Verification(net, cache, tree, seq=0, slot=slot, context=0) as run:
            expected = run.verify()
        binding = PersistentNVFP4(backend="reference")
        with Verification(net, cache, tree, seq=0, slot=slot, context=0, persistent_mlp=binding) as run:
            actual = run.verify()
            torch.testing.assert_close(actual, expected, rtol=.008, atol=.008)
            self.assertEqual(binding.executed, set(net.layers))
            result = run.commit(budget=3)
            self.assertGreaterEqual(result["context"], 1)
        self.assertTrue(all(not net.F.is_moe(L) for L in binding.executed))
        self.assertTrue(all("gate_up" not in name for name in net.p))

    def test_quantization_layout_and_memory_limits_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "group-16"):
            NVFP4Plan(4, 127, 128)
        with self.assertRaisesRegex(ValueError, "scratch"):
            NVFP4Plan(32, 4096, 3072, max_scratch_bytes=1024)
        w = packed_weights()
        with self.assertRaisesRegex(ValueError, "scale planes"):
            replace(w, sf13=w.sf13[:, :-1].contiguous())
        self.assertEqual(NVFP4Plan(16, 4096, 3072).scratch_bytes,
                         16*(3072//2+3072//16)+48*16*4096*4+(4+48+64)*4+16*4096*2)

    def test_two_complete_dflash_cost_tree_nvfp4_steps_match_autoregressive_tokens(self):
        from unittest.mock import patch
        from engine.profiles.glm53.drafter import Drafter, DrafterFacts, ring_cells, specs
        from engine.profiles.glm53.net import Step
        net, cache = nvfp4_model()
        slot = cache.slots.take(0)
        cache.pool.reserve(0, 16)
        f = DrafterFacts(layers=1, hidden=net.F.hidden, heads=2, kv_heads=1, head_dim=4, inter=32,
                        rms_eps=1e-6, rope_theta=10000., window=8, block=4, mask_id=net.vp-1,
                        conv_taps=2, conv_group=4, sel_rank=4, sel_top_k=3, target_layers=(0, 2), k=3)
        d = Drafter(f, net, net.vp)
        gen = torch.Generator().manual_seed(98)
        d.p = {s.name: ((torch.ones(s.shape) if s.name.endswith("norm.weight") else
                        torch.randn(s.shape, generator=gen)*.04).to(s.dtype)) for s in specs(f)}
        field = torch.zeros(3, f.layers, 2, ring_cells(f), f.kv_heads, f.head_dim, dtype=torch.bfloat16)
        binding = PersistentNVFP4(backend="reference")
        anchor, context = 1, 0
        for _ in range(2):
            before, paged = cache.state.clone(), cache.paged.clone()
            with patch.object(d, "candidate_rows", wraps=d.candidate_rows) as draft_call:
                result = decode_once(d, cache, field, seq=0, slot=slot, anchor=anchor, context=context, budget=4,
                    runtime_id="tiny-nvfp4", bytes_per_expert=1024, fixed_node_bytes=1024, nodes=6,
                    persistent_mlp=binding)
                self.assertEqual(draft_call.call_count, 1)
            after, got_paged = cache.state.clone(), cache.paged.clone()
            cache.state.copy_(before); cache.paged.copy_(paged)
            token, expected = anchor, []
            for offset in range(len(result["tokens"])):
                step = Step.prefill(torch.tensor([token]), context+offset, 0, slot)
                cache.prepare(step)
                token = int(net.head_tokens(net.forward(step, cache), net.vp)[0])
                expected.append(token)
            self.assertEqual(result["tokens"], tuple(expected))
            self.assertEqual(result["features"].shape[1], 2*net.F.hidden)
            self.assertTrue(field[slot, :, :, :result["context"]].abs().sum() > 0)
            cache.state.copy_(after); cache.paged.copy_(got_paged)
            anchor, context = result["tokens"][-1], result["context"]


if __name__ == "__main__":
    unittest.main()
