"""Compact cache integration: real eager model, clipped commit and durable state."""
from dataclasses import replace
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import torch

from engine.base.arena import Arena
from engine.base.comm import Comm
from engine.profiles.glm53.caches import (Glm53Caches, cache_capacity, layout, snapshot_layout,
                                         stage_bytes, state_format)
from engine.profiles.glm53.execution import ExecutionPlan
from engine.profiles.glm53.lanes import reference
from engine.profiles.glm53.net import Glm53Net, Step
from tests.test_engine_glm53 import tiny_facts


def cache_for(f, *, device="cpu", max_seqs=4, snapshots=4):
    layers = range(f.layers)
    size = (layout(f, layers).nbytes(32, max_seqs) + snapshots*snapshot_layout(f, layers)[0]
            + stage_bytes(f, layers, max_seqs) + 4096)
    return Glm53Caches(Arena(size, device=device), f, layers, 32, max_seqs, snapshots=snapshots, stage=True)


def pair():
    f = replace(tiny_facts(), block=128, spec_k=7, kinds=("kda", "dsa", "kda"))
    a, b = f, replace(f, kda_state_layout="committed_boundary")
    nets = [Glm53Net(v, Comm(4, 0), reference()) for v in (a, b)]
    gen = torch.Generator().manual_seed(895)
    weights = {s.name: (torch.randn(s.shape, generator=gen)*.04).to(s.dtype) for s in nets[0].specs()}
    for key, value in weights.items():
        if "norm" in key:
            value.fill_(1)
    for net in nets:
        net.p = weights
        net.comm = Comm()  # CPU rank arithmetic, no performance interpretation.
    return list(zip(nets, (cache_for(a), cache_for(b))))


class CompactLayoutTests(unittest.TestCase):
    def test_capacity_snapshot_abi_and_stage_alias_are_explicit(self):
        f = replace(tiny_facts(), layers=45, kinds=("kda",)*34 + ("dsa",)*11,
                    kda_heads=64, kda_dim=128, block=768, kv_lora=512, spec_k=7)
        compact = replace(f, kda_state_layout="committed_boundary")
        for n in (1, 4):
            self.assertEqual(cache_capacity(f, range(45), None, 7., n, 2.125),
                             cache_capacity(compact, range(45), None, 7., n, 2.125))
            self.assertEqual(snapshot_layout(f, range(45)), snapshot_layout(compact, range(45)))
            self.assertEqual(layout(f, range(45)).slot_bytes - layout(compact, range(45)).slot_bytes,
                             (204 << 20) - 256)  # one aligned metadata record
            self.assertEqual(stage_bytes(f, range(45), n) - stage_bytes(compact, range(45), n),
                             (n+1)*(34 << 20))
        c = cache_for(replace(tiny_facts(), kda_state_layout="committed_boundary"))
        for L in c.layers:
            if not c.F.is_dsa(L):
                self.assertIs(c._stage["rec", L], c._fields["rec_boundary", L])
                self.assertEqual(c.kda(L, 1)[1].shape[0], 1)
        self.assertEqual({f.name for f in c._stage_fields}, {"conv"})

    def test_invalid_layout_and_incompatible_plans_fail_before_execution(self):
        f = tiny_facts()
        for changes in ({"kda_state_layout": "two_ring"},
                        {"kda_state_layout": "committed_boundary", "kda_state_dtype": "fp16"}):
            with self.assertRaises(ValueError):
                layout(replace(f, **changes), range(f.layers))
        p = ExecutionPlan(compact_kda=True)
        self.assertTrue(p.deferred_kda)
        self.assertTrue(p.active)
        for kwargs in (dict(compact_kda=1), dict(compact_kda=True, overlap=True),
                       dict(compact_kda=True, prefill_tiles=2)):
            with self.assertRaises(ValueError):
                ExecutionPlan(**kwargs)
        self.assertEqual(len({state_format(f), state_format(replace(f, kda_state_dtype="fp16")),
                              state_format(replace(f, kda_state_layout="committed_boundary"))}), 3)

    def test_budget_returns_saved_bytes_without_buying_more_kv(self):
        from engine.profiles.glm53 import budget
        f = replace(tiny_facts(), kda_heads=64, kda_dim=128, spec_k=7)
        with tempfile.TemporaryDirectory() as root, patch.object(budget.facts, "load", return_value=f):
            options = dict(kv_gib=1., max_seqs=4, ckpt=root, box_gib=128., drafter_dir=None, snapshots=48)
            a = budget.budget(**options)
            b = budget.budget(**options, kda_state_layout="committed_boundary")
        self.assertEqual(a.paged_gib, b.paged_gib)
        self.assertEqual(a.kv_declared_gib, b.kv_declared_gib)
        self.assertGreater(b.kv_gib, a.kv_gib)


class CompactEagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def same(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def forward(self, model, step, **kwargs):
        net, cache = model
        for s in step.segments:
            if cache.slots.owner[s.slot] != s.seq:
                self.assertEqual(cache.slots.take(s.seq), s.slot)
            cache.pool.reserve(s.seq, s.ctx+s.length)
        cache.prepare(step)
        return net.forward(step, cache, **kwargs)

    def committed(self, baseline, compact, slot, position):
        for L in baseline.layers:
            if baseline.F.is_dsa(L):
                self.same(compact.tail(L, slot), baseline.tail(L, slot))
            else:
                ac, ar = baseline.kda(L, slot)
                bc, br = compact.kda(L, slot)
                self.same(ac, bc)
                self.same(br[0], ar[(position-1) % ar.shape[0]])
        self.assertEqual(int(compact._fields["rec_meta", -1][slot, 0]), position)

    def test_small_and_chunk_prefills_keep_outputs_marks_and_restore(self):
        a, b = pair()
        ctx = 0
        for length in (1, 7, 8, 9, 103):
            step = Step.prefill(torch.arange(length) % a[0].vp, ctx, 0, 1)
            self.same(self.forward(b, step), self.forward(a, step))
            ctx += length
            self.committed(a[1], b[1], 1, ctx)
        self.assertEqual(ctx, 128)
        for net, cache in (a, b):
            cache.checkpoint(1, 128, 0)
            cache.restore(2, 128, 0)
        self.committed(a[1], b[1], 2, 128)
        with self.assertRaises(ValueError):
            b[1].checkpoint(2, 256, 1)
        with self.assertRaises(ValueError):
            b[1].checkpoint_from_stage(2, 1, position=128)
        step = Step.prefill(torch.arange(256) % a[0].vp, 128, 0, 1, marks=((128, 1),))
        self.same(self.forward(b, step), self.forward(a, step))
        for key in a[1]._snap:
            self.same(b[1]._snap[key][1], a[1]._snap[key][1])

    def test_every_clipped_count_keeps_current_boundary_and_zero_noop(self):
        for count in range(9):
            with self.subTest(count=count):
                a, b = pair()
                prefill = Step.prefill(torch.arange(127) % a[0].vp, 0, 0, 1)
                for m in (a, b):
                    self.forward(m, prefill)
                step = Step.decode([(torch.arange(8) % a[0].vp, 127, 0, 1)])
                before = b[1].kda(0, 1)[1].clone()
                original = {L: b[1].kda(L, 1)[1].clone() for L in b[1].layers if not b[1].F.is_dsa(L)}
                expected = self.forward(a, step)
                self.same(self.forward(b, step, compact_commit=False), expected)
                self.same(b[1].kda(0, 1)[1], before)
                with self.assertRaises(RuntimeError):
                    b[1].reset_slot(1)
                with self.assertRaises(ValueError):
                    b[1].commit_compact([9])
                b[1].commit_compact([count])
                if count:
                    self.committed(a[1], b[1], 1, 127+count)
                else:
                    # An ordinary T-wide verification can overwrite the
                    # old committed cell; zero must retain the compact one.
                    for L, state in original.items():
                        self.same(b[1].kda(L, 1)[1], state)
                    self.assertEqual(b[1]._fields["rec_meta", -1][1].tolist(), [127, 0])
                if count:
                    a[1].stage_boundaries(torch.tensor([1]), torch.tensor([127]), torch.tensor([count]))
                    a[1].checkpoint_from_stage(1, 0)
                    b[1].checkpoint_from_stage(1, 0, position=128)
                    for key in a[1]._snap:
                        self.same(b[1]._snap[key][0], a[1]._snap[key][0])
                b[1].reset_slot(1)
                self.assertEqual(b[1]._fields["rec_meta", -1][1].tolist(), [0, 0])

    def test_ragged_c4_eager_commit_preserves_segment_and_slot_identity(self):
        a, b = pair()
        chunks = []
        for seq, (ctx, length) in enumerate(((127, 8), (124, 7), (128, 1), (121, 6))):
            step = Step.prefill(torch.arange(ctx) % a[0].vp, 0, seq, seq+1)
            for m in (a, b):
                self.forward(m, step)
            chunks.append((torch.arange(length) % a[0].vp, ctx, seq, seq+1))
        step = Step.decode(chunks)
        self.same(self.forward(b, step, compact_commit=False), self.forward(a, step))
        b[1].commit_compact([2, 4, 0, 6])
        for (ids, ctx, seq, slot), count in zip(chunks, (2, 4, 0, 6)):
            self.committed(a[1], b[1], slot, ctx+count)

    def test_host_sampling_passes_eos_and_length_clipped_counts_to_eager_commit(self):
        from engine.profiles.glm53.adapter import Glm53Engine
        calls = []
        caches = NS(compact=True, device=torch.device("cpu"), draft_ring=lambda slot: None,
                    commit_compact=lambda counts: calls.append(("commit", counts)))
        drafter = NS(k=7, aux_layers=(), propose=lambda *args: list(range(7)))
        e = Glm53Engine(NS(head_local=lambda x: x), caches, NS(spec_k=7, block=768), drafter,
                       eos_ids=(32,), execution_plan=ExecutionPlan(compact_kda=True))
        e.tokens, e.prompt_len, e.ctx = {1: [10], 2: [11]}, {1: 1, 2: 1}, {1: 100, 2: 767}
        e.limits = {1: (1, 0.), 2: (4, 0.)}
        e._moved = lambda: None
        e._rich = lambda seq: True
        e._prepare_masks = lambda *args: None
        e._gather = lambda values, **kwargs: values
        e._pick_rich = lambda *args: [(7, list(range(20, 28)), None), (1, [31, 32], None)]
        def forward(step, **options):
            calls.append(("forward", options))
            return torch.zeros(16, 64), None
        e._forward = forward
        self.assertEqual(e.decode([1, 2], None, [3, 2]), [True, True])
        self.assertEqual(calls, [("forward", {"compact_commit": False}), ("commit", [1, 2])])
        self.assertEqual(e.ctx, {1: 101, 2: 769})
        self.assertEqual(e.staged, {2: 768})


if __name__ == "__main__":
    unittest.main()
