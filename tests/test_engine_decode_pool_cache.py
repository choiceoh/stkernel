"""Actual Triton address/copy kernels against independent torch references."""
import importlib.util
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
DEVICE = 'cpu' if INTERPRET else 'cuda'


@unittest.skipUnless(importlib.util.find_spec('triton') and (INTERPRET or torch.cuda.is_available()),
                     'requires Triton interpreter or the reserved GPU')
class CopyKernelTests(unittest.TestCase):
    def test_windows_and_updates_preserve_all_bytes_across_slots_pages_and_rollback(self):
        from engine.kernels import indexer as native
        from engine.modules import sparse_indexer as reference
        torch.manual_seed(920)
        per, stride, capacity = 192, 11 * 192, 32768
        for n in (1, 2, 3, 4):
            raw = [torch.empty(4 * n * stride, 136, dtype=torch.uint8, device=DEVICE) for _ in range(2)]
            tails = [torch.empty(9, 10, 2, 136, dtype=torch.bfloat16, device=DEVICE) for _ in range(2)]
            source = torch.randn(n, 8, 264, dtype=torch.bfloat16, device=DEVICE)
            k, gate = source[:, :, :128], source[:, :, 128:256]
            pk = torch.randint(0, 256, (n * 2, 128), dtype=torch.uint8, device=DEVICE)
            ps = torch.randn(n * 2, device=DEVICE)
            table = torch.empty(n, 352, dtype=torch.int32, device=DEVICE)[:, ::2]
            for phase, ctx in enumerate((31997, 131061, 31990, 0, 131064)):
                contexts = torch.tensor([max(ctx - i, 0) for i in range(n)], device=DEVICE)
                slots = (torch.tensor([8, 2, 5, 0], device=DEVICE)[:n] + phase) % 9
                table.copy_(4 * torch.arange(n, device=DEVICE)[:, None]
                            + (torch.arange(176, device=DEVICE)[None, :] + phase) % 4)
                layer = (0, 5, 10, 1, 9)[phase]
                tail_seed = torch.randn_like(tails[0])
                for arm, module in enumerate((reference, native)):
                    raw[arm].fill_(0x55)
                    tails[arm].copy_(tail_seed)
                    field = tails[arm][:, :, :, :128]
                    windows = module.pool_window(field if arm else field.index_select(0, slots),
                                                 k, gate, contexts, 4, 2, **(dict(slots=slots) if arm else {}))
                    if arm == 0:
                        expected = windows
                    else:
                        for got, want in zip(windows, expected):
                            self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))
                    module.update_pool_cache(pk, ps, raw[arm][:, :128], raw[arm][:, 128:132].view(torch.float32).flatten(),
                                             field, slots, contexts, k, gate, table, per, stride, layer * per, 4, capacity)
                self.assertTrue(torch.equal(raw[0], raw[1]), (n, ctx, layer))
                self.assertTrue(torch.equal(tails[0].view(torch.uint8), tails[1].view(torch.uint8)))

    def test_mapped_window_rejects_wrong_slot_layout(self):
        from engine.kernels.indexer import pool_window
        tails = torch.zeros(9, 10, 2, 128, device=DEVICE)
        k = torch.zeros(1, 8, 128, device=DEVICE)
        for slots in (torch.zeros(1, dtype=torch.int32, device=DEVICE), torch.zeros(2, dtype=torch.int64, device=DEVICE)):
            with self.assertRaisesRegex(ValueError, 'physical slots'):
                pool_window(tails, k, k, torch.zeros(1, dtype=torch.int64, device=DEVICE), 4, 2, slots=slots)


class WiringTests(unittest.TestCase):
    def test_indexer_reads_physical_tail_only_for_the_bound_decode(self):
        from engine.profiles.glm53.net import Glm53Net
        for n, enabled, probe in ((1, True, False), (4, True, False), (1, False, False), (1, True, True)):
            m = n * 8
            net = Glm53Net.__new__(Glm53Net)
            net.F = NS(kpool=4, idx_heads=1, idx_dim=128, topk=16, spec_k=7, idx_scale=1.)
            net.prefill_indexer_shards, net.probe = False, probe
            net.decode_dsa_rows = (8, 16, 24, 32) if enabled else ()
            net._mla_prefix = lambda *a: 0
            net._decode_pair = lambda *a: None
            net._indexer_head_gate = lambda *a: (torch.ones(m, 1), 1)
            net._select_rows = Mock()
            net.p = {'L3.idx.k_norm_w': torch.ones(128), 'L3.idx.k_norm_b': torch.zeros(128)}
            net.linear = lambda *a: torch.zeros(m, 128, dtype=torch.bfloat16)
            net.lanes = NS(layernorm=lambda x, *a: x, decode_rows=object(),
                           indexer_quant=lambda x: (x.to(torch.float8_e4m3fn), torch.ones(m)),
                           head_gate=lambda *a: torch.ones(m, 1))
            step = NS(captured=True, tokens=8, contexts=torch.zeros(n, dtype=torch.int64),
                      segments=tuple(NS(length=8) for _ in range(n)))
            field, gathered = torch.empty(9, 10, 2, 128), torch.empty(n, 10, 2, 128)
            caches = NS(tail_field=Mock(return_value=field), tails=Mock(return_value=gathered),
                        token_maps=object(), pool_maps=object(), pool_keys=lambda L: None, pool_scales=lambda L: None)
            with patch('engine.profiles.glm53.decode_graphs.complete_pools', return_value=32768) as complete:
                net._indexer(3, torch.empty(m, 4096), torch.empty(m, 1536), step, caches,
                             query=torch.zeros(m, 128, dtype=torch.bfloat16))
            mapped = enabled and not probe
            self.assertEqual(complete.call_args.kwargs['mapped'], mapped)
            self.assertIs(complete.call_args.args[4], field if mapped else gathered)
            self.assertEqual(caches.tails.call_count, int(not mapped))
            self.assertEqual(caches.tail_field.call_count, int(mapped))

    def test_mapped_completion_passes_arena_views_and_records_execution(self):
        from engine.profiles.glm53.decode_graphs import complete_pools
        window = (torch.empty(8, 4, 128), torch.empty(8, 4, 128))
        pk, ps = torch.empty(8, 128, dtype=torch.uint8), torch.empty(8)
        glue = NS(window=Mock(return_value=window), update=Mock(), addresses=Mock(), pools=Mock(), tails=Mock())
        net = NS(F=NS(kpool=4, idx_dim=128), lanes=NS(decode_rows=glue, kpool_compress=Mock(return_value=(pk, ps))),
                 p={'L3.idx.ape': object()}, decode_pools_executed=set())
        tails, k, gate = torch.empty(9, 10, 2, 128), torch.empty(4, 8, 128), torch.empty(4, 8, 128)
        contexts = torch.tensor([0, 31997, 131061, 32252])
        keys, scales, table = object(), object(), object()
        slots = torch.tensor([8, 2, 5, 0])
        caches = NS(slots=slots, pool_keys=lambda L: keys, pool_scales=lambda L: scales,
                    pool_maps=lambda L: (table, 192, 2112, 576), candidate_capacity=32768,
                    tails=Mock(side_effect=AssertionError('must not gather')), tail_field=Mock())
        self.assertEqual(complete_pools(net, 3, contexts, 8, tails, k, gate, caches, mapped=True), 32768)
        self.assertIs(glue.window.call_args.kwargs['slots'], slots)
        self.assertIs(glue.window.call_args.args[0], tails)
        self.assertIs(glue.update.call_args.args[4], tails)
        self.assertIs(glue.update.call_args.args[5], slots)
        self.assertIs(glue.update.call_args.args[9], table)
        self.assertEqual(net.decode_pools_executed, {(3, 32)})
        for old in (glue.addresses, glue.pools, glue.tails, caches.tails, caches.tail_field):
            old.assert_not_called()


if __name__ == '__main__':
    unittest.main()
