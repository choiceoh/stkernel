"""Drafter precision follows supported rows; resident bytes follow live readers."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH, 'requires CPU torch')
class DrafterStorageTests(unittest.TestCase):
    def facts(self):
        from engine.profiles.glm53.drafter import DrafterFacts
        return DrafterFacts(layers=5, hidden=4096, heads=32, kv_heads=8, head_dim=128,
                            inter=12288, rms_eps=1e-6, rope_theta=10000., window=2048,
                            block=8, mask_id=1, conv_taps=2, conv_group=16, sel_rank=256,
                            sel_top_k=16, target_layers=(5, 14, 24, 33, 42), k=6)

    def test_precision_selection_covers_supported_rows_and_context_prefill(self):
        from engine.profiles.glm53.drafter_storage import needs_fp8, block_rows
        F = self.facts()
        for capacity in (1, 4):
            self.assertFalse(needs_fp8(F, capacity, 'layers.0.self_attn.qkv'))
            self.assertTrue(needs_fp8(F, capacity, 'fc.weight'))
        self.assertTrue(needs_fp8(F, 8, 'layers.0.mlp.down_proj.weight'))
        self.assertTrue(needs_fp8(F, None, 'layers.0.mlp.down_proj.weight'))
        self.assertEqual(block_rows(F, 4), 28)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                block_rows(F, invalid)

    def test_layout_removes_raw_projections_and_reserves_only_live_readers(self):
        from engine.profiles.glm53.drafter import specs
        from engine.profiles.glm53.drafter_storage import layout, nbytes
        F = self.facts()
        regions, size = layout(F, 4, 4)
        self.assertLess(size, sum(s.nbytes() for s in specs(F)) / 2)
        self.assertEqual(nbytes(F, 4, 1), size)
        self.assertGreater(nbytes(F, 4, 8), size)
        self.assertIn('source/candidate_selector.predecessor_codebook', regions)
        self.assertIn('source/layers.0.self_attn.k_norm.weight', regions)
        self.assertNotIn('source/layers.0.self_attn.k_proj.weight', regions)
        self.assertEqual(regions['context_norm'][1], F.layers * F.head_dim * 2)
        end = 0
        for start, count in regions.values():
            self.assertEqual(start % 256, 0)
            self.assertGreaterEqual(start, end)
            end = start + count
        self.assertLessEqual(end, size)
        with self.assertRaises(ValueError):
            layout(F, 3, 4)

    def test_fused_context_norm_is_owned_by_the_declared_compact_region(self):
        import torch
        from dataclasses import replace
        from engine.profiles.glm53.drafter_storage import compact, layout, retained_specs
        F = replace(self.facts(), layers=2, hidden=128, heads=4, kv_heads=4, inter=512,
                    sel_rank=16, sel_top_k=4)
        regions, size = layout(F, 4, 4)
        storage = torch.empty(size, dtype=torch.uint8)
        norm = torch.randn(F.layers, F.head_dim, dtype=torch.bfloat16)
        bias = torch.randn(F.hidden, dtype=torch.float32)
        d = SimpleNamespace(F=F, target=SimpleNamespace(comm=SimpleNamespace(world_size=4)),
            p={s.name: torch.zeros(s.shape, dtype=s.dtype) for s in retained_specs(F)}, dense={},
            context_kv=torch.zeros(F.layers * 2 * (F.kv_heads // 4) * F.head_dim, F.hidden,
                                   dtype=torch.bfloat16), context_norm=norm, fc_bias=bias)
        compact(d, SimpleNamespace(carve=lambda count, label: storage[:count]), 4)
        self.assertTrue(torch.equal(d.context_norm, norm))
        self.assertEqual(d.context_norm.data_ptr(), storage.data_ptr() + regions['context_norm'][0])
        self.assertEqual(regions['fc_bias'][1], F.hidden * 4)
        self.assertEqual(d.fc_bias.data_ptr(), storage.data_ptr() + regions['fc_bias'][0])
        self.assertTrue(torch.equal(d.fc_bias, bias))
        bias.zero_()
        self.assertFalse(torch.equal(d.fc_bias, bias))
        norm.zero_()
        self.assertFalse(torch.equal(d.context_norm, norm))

    def test_prepare_uses_the_declared_precision_and_refuses_oversized_blocks(self):
        import torch
        from engine.profiles.glm53.drafter import Drafter, specs
        F = self.facts()
        seen = {}
        def dense(weight, **kwargs):
            seen[kwargs['name']] = kwargs['prefill']
            return SimpleNamespace()
        d = Drafter(F, SimpleNamespace(comm=SimpleNamespace(world_size=4, rank=0)), 154880)
        d.bind({s.name: torch.empty(s.shape, dtype=s.dtype, device='meta') for s in specs(F)})
        with patch('engine.kernels.dense.DenseLinear', dense):
            d.prepare_fast(max_seqs=4)
        self.assertEqual(sum(seen.values()), 1)
        self.assertTrue(next(v for k, v in seen.items() if k.endswith('/model.fc')))
        with self.assertRaisesRegex(ValueError, 'above prepared capacity'):
            d.block_rows(None, None, None, None, None, 5, 7)
        self.assertEqual(d.max_block_rows, 28)

    def test_packed_reservation_fits_folded_and_unfolded_metadata(self):
        from engine.kernels.dense import packed_nbytes
        # Independent byte accounting for the widest FC, including five
        # separate row scales on an uncalibrated boot and padded output rows.
        for rows, cols in ((129, 128), (4096, 20480), (7168, 4096)):
            padded = (rows + 127) // 128 * 128
            for prefill in (False, True):
                for folded in (False, True):
                    widths = [cols] if folded else [min(4096, cols - k) for k in range(0, cols, 4096)]
                    sizes = [size for width in widths for size in
                             (padded * width // 2, padded * width // 16, padded * 4)]
                    if prefill:
                        sizes += [padded * cols, (padded // 128) * (cols // 128) * 4]
                    end = 0
                    for size in sizes:
                        end = (end + 255) // 256 * 256 + size
                    self.assertLessEqual(end, packed_nbytes(rows, cols, prefill=prefill))

    def test_combined_fp8_is_declared_and_both_packs_are_sent_to_compaction(self):
        import torch
        from engine.kernels.dense import DenseLinear, packed_nbytes
        from engine.profiles.glm53.draft_policy import DraftPolicy
        from engine.profiles.glm53.drafter_storage import nbytes
        F = self.facts()
        extra = nbytes(F, 4, 4, policy=DraftPolicy('fp8', 'decode', True)) - nbytes(F, 4, 4)
        fp8_bytes = 4096 * 20480 + (4096 // 128) * (20480 // 128) * 4
        w4_bytes = 4096 * 20480 // 2 + 4096 * 20480 // 16 + 5 * 4096 * 4
        self.assertEqual(extra, fp8_bytes - w4_bytes)
        self.assertEqual(nbytes(F, 4, 4) - nbytes(F, 4, 4, policy=DraftPolicy('fp8')), w4_bytes)
        layer = DenseLinear.__new__(DenseLinear)
        layer.packs = ()
        def fp8():
            return SimpleNamespace(weight=(torch.ones(128, 128, dtype=torch.uint8), torch.ones(1, 1)))
        layer.fp8, layer.decode_fp8 = fp8(), fp8()
        sources = tuple(t for p in layer.packs for t in (p.data, p.scale, p.rowscale)) + layer.fp8.weight + layer.decode_fp8.weight
        destinations = tuple(torch.zeros_like(t) for t in sources)
        storage = torch.empty(packed_nbytes(128, 128, decode_fp8=True, decode_w4=False), dtype=torch.uint8)
        # The allocator itself requires CUDA. This CPU contract checks the
        # complete handoff and rebinding without weakening that guard.
        with patch('engine.modules.packed_storage.consume', return_value=destinations) as consume:
            layer.consume_weight(storage)
        self.assertIs(consume.call_args.args[0], storage)
        self.assertEqual([id(t) for t in consume.call_args.args[1]], [id(t) for t in sources])
        rebound = tuple(t for p in layer.packs for t in (p.data, p.scale, p.rowscale)) + layer.fp8.weight + layer.decode_fp8.weight
        self.assertEqual([id(t) for t in rebound], [id(t) for t in destinations])

    def test_context_phase_survives_early_projection_and_short_commits(self):
        import torch
        from engine.profiles.glm53.drafter import Drafter
        calls = []
        class Projection:
            decode_fp8 = object()
            def __call__(self, x, mask=None, **kwargs):
                calls.append((mask, kwargs))
                return x
        d = Drafter(self.facts(), SimpleNamespace(comm=SimpleNamespace(world_size=4, rank=0)), 154880)
        d.dense = {'fc.weight': Projection()}
        x = torch.ones(7, 128, dtype=torch.bfloat16)
        mask = torch.arange(7) < 3
        d.context_linear(x)
        d.context_linear(x, mask, decode=True)
        d.context_linear(x, decode=True, observe=False)
        self.assertEqual([v for _, v in calls], [dict(decode=False, observe=True),
                         dict(decode=True, observe=True), dict(decode=True, observe=False)])
        self.assertIs(calls[1][0], mask)
        # Synchronous commit is still explicitly decode after calibration
        # has finished and there is no longer an observer attached.
        d.decode_calibration = True
        with patch.object(d, '_observe') as observe:
            d.observe_committed(None, torch.arange(3), x[:3])
        self.assertEqual(len(observe.call_args.args), 3)
        self.assertEqual(observe.call_args.kwargs, {'decode': True})


if __name__ == '__main__':
    unittest.main()
