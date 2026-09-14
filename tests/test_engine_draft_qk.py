"""Shared Q/K launch preserves the existing per-head operation and packed inputs."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from engine.kernels.common import norm_rope as kernel
from probes.engine_draft_qk import inputs


class PairTests(unittest.TestCase):
    def test_cpu_pair_matches_separate_norms_and_does_not_modify_packed_projection(self):
        for rows in (0, 1, 8, 16, 24, 32):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                packed, q, k, qw, kw, pos = inputs(rows, 'cpu', dtype)
                before = packed.clone()
                got = kernel.norm_rope_pair(q, k, qw, kw, 1e-6, pos, 10000.)
                want = (kernel.norm_rope(q, qw, 1e-6, pos, 10000.), kernel.norm_rope(k, kw, 1e-6, pos, 10000.))
                for a, b in zip(got, want):
                    self.assertTrue(torch.equal(a, b) and a.is_contiguous())
                self.assertTrue(torch.equal(packed, before))
                if rows:
                    self.assertEqual(len({got[0].data_ptr(), got[1].data_ptr(), q.data_ptr(), k.data_ptr()}), 4)

    def test_rejects_mismatched_geometry_storage_dtype_and_positions(self):
        _, q, k, qw, kw, pos = inputs(8, 'cpu', torch.bfloat16)
        invalid = ((q[1:], k, qw, kw, pos), (q, k[..., :64], qw, kw, pos),
                   (q[..., :6], k[..., :6], qw[:6], kw[:6], pos),
                   (q, k.float(), qw, kw, pos), (q, k, qw.float(), kw, pos),
                   (q.transpose(1, 2), k, qw, kw, pos), (q, k, qw, kw, pos.int()),
                   (q, k, qw, kw, pos[:, None]), (q, k, qw, kw, torch.arange(16)[::2]),
                   (q, k, qw.to('meta'), kw, pos))
        for args in invalid:
            with self.assertRaisesRegex(ValueError, 'paired norm_rope'):
                kernel.norm_rope_pair(*args[:4], 1e-6, args[4], 10000.)

    def test_native_wrapper_launches_one_head_grid_and_passes_original_strides(self):
        _, q, k, qw, kw, pos = inputs(8, 'cpu', torch.bfloat16)
        launch = Mock()
        native = Mock()
        native.__getitem__ = Mock(return_value=launch)
        with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
                patch.object(kernel, 'warm', return_value=torch.ones(64)), \
                patch.object(kernel, '_norm_rope_pair', native):
            oq, ok = kernel.norm_rope_pair(q, k, qw, kw, 1e-6, pos, 10000.)
        native.__getitem__.assert_called_once_with((8, 10))
        launch.assert_called_once()
        args = launch.call_args.args
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertEqual(args[8:14], (*q.stride()[:2], *k.stride()[:2], oq.stride(0), ok.stride(0)))
        self.assertTrue(oq.is_contiguous() and ok.is_contiguous())

    def test_actual_drafter_single_and_row_paths_share_qk_and_keep_v_in_place(self):
        from engine.profiles.glm53 import drafter as module
        from engine.kernels import draft_attention
        for rows, batched in ((8, False), (8, True), (32, True)):
            packed = torch.randn(rows, 12*128, dtype=torch.bfloat16)
            pos = torch.arange(rows)+31997
            net = module.Drafter.__new__(module.Drafter)
            net.F = NS(head_dim=128, rms_eps=1e-6, rope_theta=10000.)
            net.local_heads, net.local_kv_heads, net.fast_attention = 8, 2, True
            net.p = {f'layers.0.self_attn.{side}_norm.weight': torch.randn(128, dtype=torch.bfloat16)
                     for side in ('q', 'k')}
            net.linear = lambda x, name, *args: packed if name.endswith('qkv') else x
            net.target = NS(comm=NS(all_reduce=lambda x: x))
            def attention(q, k, v, *args, **kwargs):
                self.assertEqual(v.data_ptr(), packed[:, -256:].data_ptr())
                for got, part, side, heads in ((q, packed[:, :1024], 'q', 8),
                                                (k, packed[:, 1024:1280], 'k', 2)):
                    expected = kernel.norm_rope(part.view(rows, heads, 128), net.p[f'layers.0.self_attn.{side}_norm.weight'],
                                                1e-6, pos, 10000.)
                    self.assertTrue(torch.equal(got.reshape_as(expected), expected))
                return q
            pair = Mock(wraps=kernel.norm_rope_pair)
            with patch.object(module, 'norm_rope_pair', pair), \
                    patch.object(draft_attention, 'attend_rows', side_effect=attention), \
                    patch.object(draft_attention, 'draft_attention', side_effect=attention):
                if batched:
                    net._attn_rows(0, packed, pos, torch.arange(rows//8), torch.zeros(rows//8), None, rows//8, 8)
                else:
                    net._attn(0, packed, pos, torch.zeros(1), 31997)
            pair.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), 'requires the reserved GPU')
class ReplayTests(unittest.TestCase):
    def test_bf16_matches_separate_calls_on_changed_replay(self):
        from probes.engine_draft_qk import check
        check(lambda *args, **kw: None, timing=False)


if __name__ == '__main__':
    unittest.main()
