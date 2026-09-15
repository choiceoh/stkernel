"""State publication must precede boundary snapshots and the next proposal."""
from types import SimpleNamespace as NS
import unittest

import torch

from engine.profiles.glm53.decode_graphs import GraphCaches, Glm53DecodeGraphs
from engine.profiles.glm53.execution import ExecutionPlan
from engine.profiles.glm53.pipeline import AsyncDecode
from tests import test_engine_pipeline as pipeline_fixtures


class DeferredStateContracts(unittest.TestCase):
    def test_host_sampling_materializes_only_clipped_counts_before_return(self):
        from engine.profiles.glm53.adapter import Glm53Engine
        stashed = []
        caches = NS(device=torch.device("cpu"), draft_ring=lambda slot: None,
                    stash_draft=lambda slot, position: stashed.append((slot, position)))
        drafter = NS(k=6, aux_layers=(), propose=lambda anchor, context, ring: list(range(6)))
        e = Glm53Engine(None, caches, NS(spec_k=6, block=768), drafter, eos_ids=(32,),
                       execution_plan=ExecutionPlan(deferred_kda=True))
        e.tokens, e.prompt_len, e.ctx = {1: [10], 2: [11]}, {1: 1, 2: 1}, {1: 100, 2: 767}
        e.limits = {1: (1, 0.), 2: (4, 0.)}
        e._moved = lambda: None
        e._rich = lambda seq: True
        e._prepare_masks = lambda *args: None
        e._gather = lambda values, **kwargs: values
        e._pick_rich = lambda *args: [(6, list(range(20, 27)), None), (1, [31, 32], None)]
        calls = []
        e.decode_graphs = NS(shape=lambda step: (2, 7, 32768), observations={},
                            run=lambda *args: (torch.zeros(14, 1), None, torch.zeros(14, 64)),
                            materialize=lambda shape, slots, contexts, counts: calls.append(
                                (slots.tolist(), contexts.tolist(), counts.tolist())))
        self.assertEqual(e.decode([1, 2], None, [3, 2]), [True, True])
        self.assertEqual(calls, [([3, 2], [100, 767], [1, 2])])
        self.assertEqual(e.ctx, {1: 101, 2: 769})
        self.assertEqual(stashed, [(2, 768)], "the row that crossed 768 put its drafter cells aside before observing")

    def test_commit_uses_before_context_and_real_slots_after_eos_clipping(self):
        e = pipeline_fixtures.BatchTransitionTests().engine()
        e.limits[1] = (1, 0.)
        e.ctx = {1: 15, 2: 31}
        events = []
        def materialize(shape, slots, contexts, counts):
            events.append(("materialize", slots.tolist(), contexts.tolist(), counts.tolist()))
        e.decode_graphs.materialize = materialize
        e.caches.stage_boundaries = lambda *args: events.append(("boundaries",))
        pipe = AsyncDecode(e)
        pending = pipe.launch([1, 2], [3, 2])
        self.assertEqual(events[:2], [("materialize", [3, 2], [15, 31], [1, 2]), ("boundaries",)])
        self.assertEqual(pipe.buf["slot"].tolist(), [0, 2])
        pending.resolve()
        self.assertEqual(e.ctx, {1: 16, 2: 33})

    def test_sparse_layer_ids_route_to_dense_factor_indices(self):
        received = []
        real = NS(F=NS(kpool=16, is_dsa=lambda layer: layer == 3), layout=None, layers=(0, 3, 4, 8))
        state = NS(verify=lambda *args: received.append(args) or "out")
        cache = GraphCaches(real, torch.tensor([2, 0]), torch.tensor([4, 1]), 128, state)
        context = torch.tensor([32767, 131071])
        self.assertEqual(cache.verify_kda(8, 1, 2, 3, 4, 5, 6, 7, context, -5.), "out")
        self.assertEqual(received[0][0], 2)
        self.assertIs(received[0][-3], cache.slots)
        self.assertIs(received[0][-2], context)
        with self.assertRaisesRegex(ValueError, "complete target batch"):
            cache.subset(0, 1)

    def test_capacity_buckets_share_one_factor_owner_by_rows_and_tokens(self):
        called = []
        target = object.__new__(Glm53DecodeGraphs)
        target.execution_plan = ExecutionPlan(deferred_kda=True)
        target.deferred_states = {(4, 7): NS(commit=lambda *args: called.append(args))}
        slots, contexts, counts = (torch.ones(4, dtype=torch.int64) for _ in range(3))
        for capacity in (32768, 131072):
            target.materialize((4, 7, capacity), slots, contexts, counts)
        self.assertEqual(len(called), 2)
        self.assertTrue(all(args[1] is contexts for args in called))

    def test_state_policy_is_explicit_and_rejects_split_target_ownership(self):
        self.assertFalse(ExecutionPlan().deferred_kda)
        self.assertTrue(ExecutionPlan(deferred_kda=True).active)
        with self.assertRaisesRegex(ValueError, "unsplit"):
            ExecutionPlan(overlap=True, deferred_kda=True)
        with self.assertRaisesRegex(ValueError, "booleans"):
            ExecutionPlan(deferred_kda=1)


if __name__ == "__main__":
    unittest.main()
