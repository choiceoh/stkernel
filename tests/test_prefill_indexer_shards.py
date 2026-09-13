"""Exact slot/cache checks for partitioned replicated indexer queries."""
from dataclasses import replace
import importlib.util
import unittest

from engine.modules.prefill_indexer import QueryShard

torch = None
if importlib.util.find_spec('torch'):
    import torch


class QueryOwnershipTests(unittest.TestCase):
    def test_every_required_query_has_one_owner_and_covered_queries_have_none(self):
        for rows in (128, 129, 130, 131, 2000, 2051, 2052, 2672, 32256):
            for context in (0, 1, 3, 2048, 2051, 32256, 96768):
                shards = [QueryShard(rows, context, rank, 4, 512, 4) for rank in range(4)]
                owned = [r for s in shards for r in range(s.begin, s.end)]
                scored = [r for s in shards for r in range(s.score_begin, s.end)]
                self.assertEqual(owned, list(range(rows)))
                self.assertEqual(scored, [r for r in range(rows) if (context+r+1)//4 > 512])
                self.assertLessEqual(max(s.end-s.begin for s in shards)-min(s.end-s.begin for s in shards), 3)

    def test_default_is_off_and_execution_plan_rejects_non_boolean(self):
        from engine.profiles.glm53.execution import ExecutionPlan
        self.assertFalse(ExecutionPlan().prefill_indexer_shards)
        self.assertTrue(ExecutionPlan(prefill_indexer_shards=True).active)
        with self.assertRaises(ValueError):
            ExecutionPlan(prefill_indexer_shards=1)


@unittest.skipUnless(torch is not None, 'requires CPU torch')
class IndexerStateTests(unittest.TestCase):
    def test_pool_id_wire_preserves_sign_boundary_sentinel_and_wide_fallback(self):
        from engine.base.comm import LocalTP
        from types import SimpleNamespace as NS
        for candidates, topk, bits in ((65535,4,16),(65536,4,32),(65535,3,32)):
            rows,context=131,candidates*4-131
            complete=(context+torch.arange(rows,dtype=torch.int32)+1)//4
            values=torch.tensor([0,32767,32768,65534,-1],dtype=torch.int32)
            expected=values[torch.arange(rows*topk).reshape(rows,topk)%len(values)]
            if candidates>65535: expected[0,0]=65535
            def rank(comm):
                shard=QueryShard(rows,context,comm.rank,4,topk,4)
                sizes=[]
                wrapped=NS(rank=comm.rank,world_size=4,all_gather=lambda packet,**kw:
                           sizes.append(packet.numel()*packet.element_size()) or comm.all_gather(packet,**kw))
                actual=shard.collect(expected[shard.begin:shard.end],complete,wrapped)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                self.assertEqual(shard.wire_bits,bits)
                self.assertEqual(sizes,[shard.capacity*topk*bits//8])
            LocalTP(4,timeout_s=10).run(rank)

    def test_fully_covered_selection_needs_no_collective(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        shard=QueryShard(2000,0,3,4,512,4)
        comm=NS(rank=3,world_size=4,all_gather=Mock(side_effect=AssertionError('unexpected collective')))
        complete=(torch.arange(2000,dtype=torch.int32)+1)//4
        result=shard.collect(None,complete,comm)
        self.assertEqual(result.shape,(2000,512))
        self.assertTrue(bool((result[0]==-1).all()))
        self.assertEqual(result[-1,:500].tolist(),list(range(500)))
        self.assertTrue(bool((result[-1,500:]==-1).all()))

    def test_full_model_regular_and_layer_major_preserve_hidden_state_and_next_decode(self):
        from types import SimpleNamespace as NS
        from engine.base.comm import LocalTP
        from engine.profiles.glm53.net import Step
        from engine.profiles.glm53.execution import prefill_layer_major
        from tests.test_engine_execution_plans import model
        from tests.test_engine_prefill_tiles import OraclePrefill
        torch.set_num_threads(1)
        for tiled, rows in ((False,131), (True,259)):
            def rank(comm):
                net,cache=model(('kda','dsa','kda'),comm=comm)
                net.prefill_transport=OraclePrefill(comm,False)
                net.F=replace(net.F,topk=128)
                slot=cache.slots.take(0);cache.pool.reserve(0,rows+7)
                step=Step.prefill(torch.arange(rows)%net.vp,0,0,slot)
                follow=Step.prefill(torch.arange(7)%net.vp,rows,0,slot)
                cache.prepare(step)
                initial,paged=cache.state.clone(),cache.paged.clone()
                def run():
                    if tiled: return prefill_layer_major(net,step,cache,NS(tile_rows=128,prefill_tiles=4),[0,2])
                    return net.forward(step,cache,aux_layers=[0,2])
                expected=run();state,pages=cache.state.clone(),cache.paged.clone()
                cache.prepare(follow);next_hidden=net.forward(follow,cache)
                final,final_pages=cache.state.clone(),cache.paged.clone()
                cache.state.copy_(initial);cache.paged.copy_(paged);cache.prepare(step)
                net.prefill_indexer_shards=True
                actual=run()
                for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(cache.state,state,rtol=0,atol=0)
                torch.testing.assert_close(cache.paged,pages,rtol=0,atol=0)
                cache.prepare(follow)
                torch.testing.assert_close(net.forward(follow,cache),next_hidden,rtol=0,atol=0)
                torch.testing.assert_close(cache.state,final,rtol=0,atol=0)
                torch.testing.assert_close(cache.paged,final_pages,rtol=0,atol=0)
                self.assertEqual(net.prefill_indexer_executed,{1})
            LocalTP(4,timeout_s=30).run(rank)

    def test_actual_indexer_matches_slots_valid_counts_pool_bytes_and_tail_state(self):
        from engine.base.comm import LocalTP
        from engine.profiles.glm53.net import Step
        from tests.test_engine_execution_plans import model
        torch.set_num_threads(1)
        for rows, context, topk in ((128,0,128), (129,0,128), (130,0,128), (131,0,128),
                                    (132,0,128), (129,3,8), (130,9,8), (131,64,8), (7,64,8)):
            with self.subTest(rows=rows, context=context, topk=topk):
                def rank(comm):
                    net,cache = model(('kda','dsa','kda'), comm=comm)
                    net.F = replace(net.F, topk=topk)
                    slot = cache.slots.take(1);cache.pool.reserve(1,context+rows)
                    gen = torch.Generator().manual_seed(791)
                    all_x = torch.randn(context+rows,net.F.hidden,generator=gen).bfloat16()
                    all_q = torch.randn(context+rows,net.F.q_lora,generator=gen).bfloat16()
                    if context:
                        pre = Step.prefill(torch.zeros(context,dtype=torch.int64),0,1,slot)
                        cache.prepare(pre);net._indexer(1,all_x[:context],all_q[:context],pre,cache)
                    step = Step.prefill(torch.zeros(rows,dtype=torch.int64),context,1,slot)
                    cache.prepare(step)
                    state,paged = cache.state.clone(),cache.paged.clone()
                    expected = net._indexer(1,all_x[context:],all_q[context:],step,cache)
                    after,pages = cache.state.clone(),cache.paged.clone()
                    cache.state.copy_(state);cache.paged.copy_(paged)
                    calls=[];linear=net.linear
                    def record(x,name):
                        calls.append((name,len(x)))
                        return linear(x,name)
                    net.linear=record;net.prefill_indexer_shards=True
                    actual=net._indexer(1,all_x[context:],all_q[context:],step,cache)
                    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)
                    torch.testing.assert_close(cache.state,after,rtol=0,atol=0)
                    torch.testing.assert_close(cache.paged,pages,rtol=0,atol=0)
                    qcalls=[count for name,count in calls if name.endswith('idx.wq_b')]
                    if rows>=128:
                        shard=QueryShard(rows,context,comm.rank,4,topk//4,4)
                        self.assertEqual(qcalls,[max(64,shard.score_rows)] if shard.score_rows else [])
                        self.assertEqual(net.prefill_indexer_executed,{1})
                    else:
                        self.assertEqual(qcalls,[rows]);self.assertEqual(net.prefill_indexer_executed,set())
                    for suffix in ('idx.wk','idx.gate'):
                        self.assertEqual([count for name,count in calls if name.endswith(suffix)],[rows])
                LocalTP(4,timeout_s=30).run(rank)

    def test_query_projection_padding_keeps_real_rows_and_never_enters_cache(self):
        x=torch.arange(132*8).reshape(132,8).bfloat16()
        shard=QueryShard(132,0,3,4,32,4)
        self.assertEqual(shard.score_rows,1)
        padded=shard.project_input(x)
        self.assertEqual(padded.shape,(64,8))
        torch.testing.assert_close(padded[:1],x[-1:],rtol=0,atol=0)
        self.assertTrue(bool((padded[1:]==0).all()))


if __name__=='__main__': unittest.main()
