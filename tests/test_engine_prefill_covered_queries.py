"""Covered-query pruning preserves pool selection, writes and decode state."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from engine.modules.prefill_indexer import covered_pool_ids, project_query_rows


class CoveredQueriesTests(unittest.TestCase):
    def test_scored_shard_skips_discarded_fill_and_initializes_wire_padding(self):
        from engine.modules.prefill_indexer import QueryShard
        shard = QueryShard(131, 100, 3, 4, 4, 4)
        scored = torch.arange(shard.score_rows*4, dtype=torch.int32).reshape(-1, 4)%16
        packets = []

        def gather(packet, **kwargs):
            packets.append(packet.clone())
            return torch.cat((torch.zeros_like(packet),)*3 + (packet,))

        comm = NS(world_size=4, rank=3, all_gather=gather)
        complete = (100+torch.arange(131, dtype=torch.int32)+1)//4
        with patch('engine.modules.prefill_indexer.covered_pool_ids', side_effect=AssertionError('discarded initialization')):
            actual = shard.collect(scored, complete, comm)
        torch.testing.assert_close(actual[shard.begin:], scored, rtol=0, atol=0)
        self.assertTrue(bool((packets[0][-1] == -1).all()))

    def test_owned_pool_destination_and_short_query_padding(self):
        complete = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
        storage = torch.full((6, 4), -19, dtype=torch.int32)
        out = storage[1:-1]
        actual = covered_pool_ids(complete, 4, out=out)
        self.assertIs(actual, out)
        self.assertEqual(actual.tolist(), [[-1]*4, [0, -1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]])
        self.assertTrue(bool((storage[0] == -19).all() & (storage[-1] == -19).all()))
        for bad in (out.long(), out[:1], storage[:, :4:2], out.T):
            with self.assertRaises(ValueError):
                covered_pool_ids(complete, 4, out=bad)
        x = torch.arange(132*8).reshape(132, 8).bfloat16()
        for count in (1, 4, 31, 32, 33, 63, 64, 65):
            actual = project_query_rows(x, 132-count, 132)
            self.assertEqual(actual.shape, (max(64, count), 8))
            torch.testing.assert_close(actual[:count], x[-count:], rtol=0, atol=0)
            if count < 64:
                self.assertTrue(bool((actual[count:] == 0).all()))
            else:
                self.assertEqual(actual.data_ptr(), x[-count:].data_ptr())
        for begin, end in ((-1, 1), (4, 4), (0, 133)):
            with self.assertRaises(ValueError):
                project_query_rows(x, begin, end)

    def test_select_pools_writes_destination_in_single_and_multiple_passes(self):
        from engine.profiles.glm53.net import Glm53Net
        for rows in (1, 5, 13):
            x = torch.arange(rows*9).reshape(rows, 1, 9).float()
            lane = Mock(side_effect=lambda q, *args: q[:, 0].clone())
            net = NS(lanes=NS(indexer_logits=lane))
            lengths = torch.full((rows,), 9, dtype=torch.int32)
            storage = torch.full((rows+2, 4), -19, dtype=torch.int32)
            with patch('engine.profiles.glm53.net.SELECT_ROWS', 5):
                expected = Glm53Net._select_pools(net, x, None, None, None, lengths, 9, 4)
                actual = Glm53Net._select_pools(net, x, None, None, None, lengths, 9, 4, out=storage[1:-1])
            self.assertEqual(actual.data_ptr(), storage[1].data_ptr())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(bool((storage[0] == -19).all() & (storage[-1] == -19).all()))
            lane.reset_mock()
            with self.assertRaises(ValueError):
                Glm53Net._select_pools(net, x, None, None, None, lengths, 9, 4, out=storage.long())
            lane.assert_not_called()

    def test_indexer_avoids_only_covered_queries_and_preserves_every_cache_write(self):
        from engine.base.comm import LocalTP
        from engine.profiles.glm53.net import Step
        from tests.test_engine_execution_plans import model
        torch.set_num_threads(1)
        cases = ((128, 0), (129, 0), (130, 0), (131, 0), (132, 0), (133, 0),
                 (195, 0), (131, 3), (131, 4), (128, 131), (7, 0))
        for rows, context in cases:
            def rank(comm):
                net, cache = model(('kda', 'dsa', 'kda'), comm=comm)
                net.F = replace(net.F, topk=128)
                slot = cache.slots.take(1)
                cache.pool.reserve(1, context+rows)
                gen = torch.Generator().manual_seed(906)
                all_x = torch.randn(context+rows, net.F.hidden, generator=gen).bfloat16()
                all_q = torch.randn(context+rows, net.F.q_lora, generator=gen).bfloat16()
                if context:
                    pre = Step.prefill(torch.zeros(context, dtype=torch.int64), 0, 1, slot)
                    cache.prepare(pre)
                    net._indexer(1, all_x[:context], all_q[:context], pre, cache)
                step = Step.prefill(torch.zeros(rows, dtype=torch.int64), context, 1, slot)
                cache.prepare(step)
                state, pages = cache.state.clone(), cache.paged.clone()
                expected = net._indexer(1, all_x[context:], all_q[context:], step, cache)
                final, final_pages = cache.state.clone(), cache.paged.clone()
                cache.state.copy_(state)
                cache.paged.copy_(pages)
                net.prefill_dense_prefix = True
                linear = Mock(wraps=net.linear)
                logits = Mock(wraps=net.lanes.indexer_logits)
                net.linear = linear
                net.lanes = replace(net.lanes, indexer_logits=logits)
                actual = net._indexer(1, all_x[context:], all_q[context:], step, cache)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(cache.state, final, rtol=0, atol=0)
                torch.testing.assert_close(cache.paged, final_pages, rtol=0, atol=0)
                prefix = net._mla_prefix(rows, step)
                scored = rows-prefix
                projections = [len(call.args[0]) for call in linear.call_args_list if call.args[1].endswith('idx.wq_b')]
                self.assertEqual(projections, [max(64, scored)] if prefix and scored else ([] if prefix else [rows]))
                self.assertEqual(sum(len(c.args[0]) for c in logits.call_args_list), scored if (context+rows)//4 else 0)
                for suffix in ('idx.wk', 'idx.gate'):
                    self.assertEqual([len(c.args[0]) for c in linear.call_args_list if c.args[1].endswith(suffix)], [rows])
                self.assertEqual(net.prefill_covered_queries_executed, {1} if prefix else set())
            LocalTP(4, timeout_s=30).run(rank)

    def test_tp4_model_and_following_decode_match_with_independent_and_combined_lanes(self):
        from engine.base.comm import LocalTP
        from engine.profiles.glm53.net import Step
        from engine.profiles.glm53.execution import prefill_layer_major
        from tests.test_engine_execution_plans import model
        from tests.test_engine_prefill_tiles import OraclePrefill
        torch.set_num_threads(1)
        for shards, tiled, rows in ((False, False, 131), (False, False, 132),
                                    (False, True, 259), (True, False, 132), (True, True, 259)):
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
                net.prefill_dense_prefix = net.prefill_absorb_tiles = True
                net.prefill_indexer_shards = shards
                actual = run()
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(cache.state, state, rtol=0, atol=0)
                torch.testing.assert_close(cache.paged, pages, rtol=0, atol=0)
                self.assertEqual(net.prefill_covered_queries_executed, {1})
                net.prefill_covered_queries_executed.clear()
                cache.prepare(follow)
                torch.testing.assert_close(net.forward(follow, cache), next_hidden, rtol=0, atol=0)
                torch.testing.assert_close(cache.state, final, rtol=0, atol=0)
                torch.testing.assert_close(cache.paged, final_pages, rtol=0, atol=0)
                self.assertEqual(net.prefill_covered_queries_executed, set())
            LocalTP(4, timeout_s=30).run(rank)


if __name__ == '__main__':
    unittest.main()
