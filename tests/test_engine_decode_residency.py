"""Router representation and all-layer draft writes retain their consumers' inputs."""
import unittest
from dataclasses import replace

import torch


class RouterResidencyTests(unittest.TestCase):
    def test_native_router_reads_one_budgeted_fp32_copy_on_each_rank(self):
        from types import SimpleNamespace
        from engine.base.arena import Arena
        from engine.profiles.glm53.net import Glm53Net
        from engine.profiles.glm53.specs import layer_specs, CK
        from tests.test_engine_glm53 import tiny_facts
        f = replace(tiny_facts(), dense=(0, 1), experts=32)
        spec = next(s for s in layer_specs(f, 2) if s.name == 'L2.moe.gate')
        source = {CK + 'layers.2.mlp.gate.weight': torch.randn(32, 128).bfloat16()}
        weight = next(iter(source.values()))
        self.assertEqual(spec.dtype, torch.bfloat16, 'existing rank files retain their binding contract')
        for rank in range(4):
            net = Glm53Net(f, SimpleNamespace(rank=rank, world_size=4), SimpleNamespace(rmsnorm=None, swiglu=None), [2])
            self.assertFalse(net.fused_decode_router)
            net.p = {spec.name: spec.build(source, rank, 4)}
            original = net.p[spec.name]
            self.assertEqual(net.router_nbytes(), weight.numel() * 4)
            arena = Arena(net.router_nbytes(), device='cpu')
            net.prepare_routers(arena)
            self.assertEqual(arena.remaining, 0)
            resident = net._router_weights[2]
            self.assertEqual(resident.dtype, torch.float32)
            self.assertEqual(resident.untyped_storage().data_ptr(), arena.buf.data_ptr())
            self.assertEqual(arena.regions[0].name, 'router/2')
            torch.testing.assert_close(resident, weight.float(), rtol=0, atol=0)
            self.assertEqual(net._router_layers, {2})
            self.assertIs(net.p[spec.name], original)
            torch.testing.assert_close(original, weight, rtol=0, atol=0)
            with self.assertRaises(RuntimeError):
                net.prepare_routers(arena)

    def test_router_memory_is_replicated_and_subtracted_from_each_ranks_budget(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from engine.profiles.glm53 import budget, facts
        from engine.profiles.glm53.net import Glm53Net
        from tests.test_engine_kernel_shape import GLM53_TEXT_CONFIG
        f = facts.architecture(GLM53_TEXT_CONFIG)
        net = SimpleNamespace(F=f, layers=range(f.layers))
        nbytes = Glm53Net.router_nbytes(net)
        self.assertEqual(nbytes, 189 * 2**20)
        with patch.object(budget.facts, 'load', return_value=f):
            args = dict(kv_gib=8.73, max_seqs=4, box_gib=121.6, drafter_dir=None)
            plain = budget.budget(**args)
            resident = budget.budget(**args, router_bytes=nbytes)
        line = next(line for line in resident.lines if line.name == 'resident FP32 routers')
        self.assertEqual(line.gib, nbytes / 2**30)
        self.assertAlmostEqual(plain.kv_gib - resident.kv_gib, nbytes / 2**30)


@unittest.skipUnless(torch.cuda.is_available(), 'native fused draft write requires CUDA')
class DraftWriteTests(unittest.TestCase):
    def test_all_layers_match_separate_native_kernels_with_wrap_ghosts_and_graph_replay(self):
        from engine.kernels.draft_observe import write_context
        from engine.kernels.draft_attention import write_draft_kv_rows
        from engine.kernels.common.norm_rope import norm_rope, warm
        for n, t, heads in ((1, 1, 1), (1, 7, 2), (4, 7, 2), (4, 7, 4)):
            layers, dim, window, field_heads = 5, 128, 32, heads + 1
            shape = (5, layers, 2, window, field_heads, dim)
            row_size = layers * 2 * window * field_heads * dim
            backing = torch.full((5, row_size + 128), -17., device='cuda', dtype=torch.bfloat16)
            plain = torch.empty(shape, device='cuda', dtype=torch.bfloat16)
            field = backing.as_strided(shape, (row_size + 128, *plain.stride()[1:]))
            expected = field.clone()
            context = torch.randn(n, t, layers, 2, heads, dim, device='cuda', dtype=torch.bfloat16)
            weights = torch.randn(layers, dim, device='cuda', dtype=torch.bfloat16)
            slots = torch.tensor([4, 2, 1, 0][:n], device='cuda', dtype=torch.int64)
            positions = torch.arange(n * t, device='cuda', dtype=torch.int64).reshape(n, t) + window - 3
            valid = torch.full((n,), t, device='cuda', dtype=torch.int64)
            warm(context.device, dim, 10000.)
            def fused():
                write_context(field, slots, positions, context, weights, valid, 1e-6, 10000.)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fused()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused()
            for iteration in range(3):
                backing.fill_(-17.)
                expected.fill_(-17.)
                context.normal_()
                weights.normal_()
                positions.add_(123457)
                valid.copy_((torch.arange(n, device='cuda') + iteration) % (t + 1))
                if n > 1:
                    valid[-1] = t
                graph.replay()
                for layer in range(layers):
                    key = norm_rope(context[:, :, layer, 0].reshape(n * t, heads, dim),
                                    weights[layer], 1e-6, positions.reshape(-1), 10000.).reshape(n, t, heads, dim)
                    write_draft_kv_rows(expected, slots, layer, positions, key, context[:, :, layer, 1], valid=valid)
                torch.testing.assert_close(field, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(backing[:, row_size:], torch.full_like(backing[:, row_size:], -17.)))


if __name__ == '__main__':
    unittest.main()
