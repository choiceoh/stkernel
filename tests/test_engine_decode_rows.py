"""A captured decode step folds its rows: which steps fold, and that the folded addressing is the per-row one.

The kernels themselves are pinned on the GPU (tests/test_engine_kda_ring.py, tests/test_engine_conv_ring.py,
tests/test_engine_state.py and tests/test_engine_pool_slots.py rows tests; probes/engine_decode_graph_check.py
end to end). What the CPU can pin is the decisions (`Glm53Net._ring_rows`, `_indexer_rows`), the gathered
block-table arithmetic (`GraphCaches.token_rows`, `candidate_rows`, `token_maps`) against the one-row
functions it replaces, and the folded indexer selection (`_select_rows`) against the segment loop's, which
on the CPU is the same torch ops row by row.
"""
import unittest
from types import SimpleNamespace as NS

import torch

from engine.base.constants import iota
from engine.profiles.glm53.caches import Glm53Caches
from engine.profiles.glm53.decode_graphs import GraphCaches
from engine.profiles.glm53.net import Glm53Net, Segment


def step(lengths, captured=True, contexts=True, tokens=None):
    segs, start = [], 0
    for i, n in enumerate(lengths):
        segs.append(Segment(i, i + 1, 10 * i, start, n)); start += n
    fields = dict(segments=tuple(segs), captured=captured)
    if contexts:
        fields["contexts"] = torch.tensor([10 * i for i in range(len(lengths))])
    if tokens is not None:
        fields["tokens"] = tokens
    return NS(**fields)


class RowFoldDecisionTests(unittest.TestCase):
    def net(self, rows=True):
        lanes = NS(kda_recurrent_ring_rows=object() if rows else None, conv_ring_rows=object() if rows else None)
        return NS(lanes=lanes)

    def test_a_captured_decode_step_folds_every_row(self):
        for lengths in ((7,), (7, 7), (1, 1, 1, 1), (7, 7, 7, 7)):
            self.assertEqual(Glm53Net._ring_rows(self.net(), step(lengths), 9, 7), len(lengths))

    def test_what_keeps_the_per_segment_loop(self):
        net = self.net()
        self.assertEqual(Glm53Net._ring_rows(net, step((7, 7), captured=False), 9, 7), 0)     # eager
        self.assertEqual(Glm53Net._ring_rows(net, step((7, 7), contexts=False), 9, 7), 0)     # no device contexts
        self.assertEqual(Glm53Net._ring_rows(net, step((2304,)), 9, 7), 0)                    # a prefill chunk
        self.assertEqual(Glm53Net._ring_rows(net, step((7, 6)), 9, 7), 0)                     # uneven rows
        self.assertEqual(Glm53Net._ring_rows(net, step((8, 8)), 9, 7), 0)                     # past the recurrent ring
        self.assertEqual(Glm53Net._ring_rows(net, step((9, 9)), 9, 9), 0)                     # past the conv ring's 8
        self.assertEqual(Glm53Net._ring_rows(self.net(rows=False), step((7, 7)), 9, 7), 0)   # a reference table


class IndexerRowsDecisionTests(unittest.TestCase):
    def net(self, glue=True):
        return NS(lanes=NS(decode_rows=object() if glue else None))

    def caches(self, graph=True):
        return NS(pool_maps=object(), token_maps=object()) if graph else NS()

    def test_a_captured_step_on_graph_caches_folds_every_row(self):
        for lengths in ((7,), (7, 7), (1, 1, 1, 1), (7, 7, 7, 7)):
            self.assertEqual(Glm53Net._indexer_rows(self.net(), step(lengths, tokens=lengths[0]), self.caches()), len(lengths))

    def test_what_keeps_the_per_segment_loop(self):
        fold, net = Glm53Net._indexer_rows, self.net()
        self.assertEqual(fold(net, step((7, 7), captured=False, tokens=7), self.caches()), 0)   # eager
        self.assertEqual(fold(net, step((7, 7), contexts=False, tokens=7), self.caches()), 0)   # no device contexts
        self.assertEqual(fold(net, step((7, 7), tokens=7), self.caches(graph=False)), 0)         # the eager caches
        self.assertEqual(fold(net, step((7, 6), tokens=7), self.caches()), 0)                    # uneven rows
        self.assertEqual(fold(net, step((7, 7)), self.caches()), 0)                              # no step width
        self.assertEqual(fold(self.net(glue=False), step((7, 7), tokens=7), self.caches()), 0)   # a table without the glue


def graph_caches(rows=4):
    torch.manual_seed(5)
    F = NS(block=768, kv_lora=512, kpool=4, idx_dim=128)
    layout = NS(block_bytes=768 * 512 + 2048, token_offsets={3: 0, 7: 768 * 512},
                pool_offsets={3: 768 * 512 + 1024, 7: 768 * 512 + 1536})
    real = NS(F=F, layout=layout, block_table=torch.randint(0, 40, (6, 12), dtype=torch.int32))
    caches = GraphCaches(real, torch.tensor([5, 2, 0, 3])[:rows], torch.tensor([1, 2, 3, 4])[:rows], 4096)
    caches.gather()
    return caches


class TokenRowsTests(unittest.TestCase):
    def test_token_rows_equals_token_slots_row_by_row(self):
        rows = 4
        caches = graph_caches(rows)
        positions = torch.tensor([[2000 + i for i in range(7)], [0, 1, 2, 3, 4, 5, 6],
                                  [767 + i for i in range(7)], [5000 + i for i in range(7)]])
        for layer in (3, 7):
            folded = caches.token_rows(layer, positions)
            self.assertEqual(folded.dtype, torch.int32)
            for i in range(rows):
                one = Glm53Caches.token_slots(caches, layer, i, positions[i])
                torch.testing.assert_close(folded[i], one, rtol=0, atol=0)

    def test_candidate_rows_is_pool_slots_over_the_capacity_row_by_row(self):
        caches = graph_caches()
        for layer in (3, 7):
            for n_cand in (1, 5, 1024):
                got = caches.candidate_rows(layer, n_cand)
                self.assertEqual((got.dtype, tuple(got.shape)), (torch.int64, (4, n_cand)))
                for i in range(4):
                    one = Glm53Caches.pool_slots(caches, layer, i, iota(n_cand, "cpu")).long()
                    torch.testing.assert_close(got[i], one, rtol=0, atol=0)

    def test_token_maps_is_token_map_row_by_row(self):
        caches = graph_caches()
        for layer in (3, 7):
            table, block, stride, offset = caches.token_maps(layer)
            self.assertEqual(table.shape, (4, 12))
            for i in range(4):
                row, b, s, o = caches.token_map(layer, i)
                self.assertTrue(torch.equal(table[i], row))
                self.assertEqual((block, stride, offset), (b, s, o))

    def test_pool_maps_is_pool_slots_arithmetic(self):
        caches = graph_caches()
        for layer in (3, 7):
            table, per, stride, offset = caches.pool_maps(layer)
            ids = torch.tensor([[0, 1, 191, 192, 1000]] * 4)
            want = caches.pool_rows(layer, ids)
            got = torch.gather(table, 1, (ids // per).long()) * stride + offset + ids % per
            self.assertTrue(torch.equal(got.to(want.dtype), want))


class GlueReferenceTests(unittest.TestCase):
    """The reference lane's glue (engine/modules/sparse_indexer.py) is the composition it replaced, piece by piece:
    the lengths, the latent write, the candidate gather and the pool addresses against `token_rows`,
    `candidate_rows`, `pool_rows` and the plain torch ops (the window is pinned by test_engine_graph_contracts)."""

    def setUp(self):
        from engine.modules import sparse_indexer as si
        self.si = si
        self.caches = graph_caches()
        self.contexts = torch.tensor([2000, 0, 767, 5000])

    def test_row_lengths(self):
        for t, kp in ((1, 4), (7, 4), (7, 8)):
            positions = self.contexts[:, None] + torch.arange(t)
            seq, ke = self.si.row_lengths(self.contexts, t, kp)
            want = (positions.reshape(-1) + 1).to(torch.int32)
            self.assertTrue(torch.equal(seq, want) and torch.equal(ke, want // kp))
            self.assertEqual((seq.dtype, ke.dtype), (torch.int32, torch.int32))

    def test_latent_write_rows(self):
        caches, t = self.caches, 7
        for layer in (3, 7):
            table, block, stride, offset = caches.token_maps(layer)
            values = torch.randn(4 * t, 512).to(torch.bfloat16)
            want = torch.zeros(40 * stride + 2 * block, 512, dtype=torch.bfloat16)
            got = want.clone()
            positions = self.contexts[:, None] + torch.arange(t)
            want[caches.token_rows(layer, positions).flatten().long()] = values
            self.si.latent_write_rows(values, got, table, block, stride, offset, self.contexts, t)
            self.assertTrue(torch.equal(got, want))

    def test_gather_candidates(self):
        caches = self.caches
        keys, scales = torch.randn(40 * 2994 + 4096, 16), torch.rand(40 * 2994 + 4096)
        for layer, n_cand in ((3, 1), (3, 700), (7, 1024)):
            cand = caches.candidate_rows(layer, n_cand)
            got_k, got_s = self.si.gather_candidates(keys, scales, *caches.pool_maps(layer), n_cand)
            self.assertTrue(torch.equal(got_k, keys[cand]) and torch.equal(got_s, scales[cand]))

    def test_head_gate(self):
        w, qs = torch.randn(28, 32), torch.rand(28, 32)
        got = self.si.head_gate(w, qs, 0.0625 ** 0.5)
        self.assertTrue(torch.equal(got, (w * qs * 0.0625 ** 0.5).contiguous()))

    def test_scatter_pools(self):
        n, pools, d = 3, 2, 8
        pk = torch.randint(0, 256, (n * pools, d), dtype=torch.uint8)
        ps = torch.rand(n * pools)
        slots = torch.tensor([[5, 9], [11, 40], [0, 1]])
        counts = torch.tensor([2, 1, 0])
        keys, scales = torch.zeros(50, d, dtype=torch.uint8), torch.zeros(50)
        want_k, want_s = keys.clone(), scales.clone()
        for i in range(n):
            for j in range(int(counts[i])):
                want_k[slots[i, j]] = pk[i * pools + j]
                want_s[slots[i, j]] = ps[i * pools + j]
        self.si.scatter_pools(pk, ps, keys, scales, slots, counts)
        self.assertTrue(torch.equal(keys, want_k) and torch.equal(scales, want_s))

    def test_write_tails(self):
        n, t, w, d = 3, 7, 9, 8
        field = torch.randn(5, w, 2, d).to(torch.bfloat16)
        want = field.clone()
        keys, gates = torch.randn(n, t, d).to(torch.bfloat16), torch.randn(n, t, d).to(torch.bfloat16)
        slots, contexts = torch.tensor([3, 1, 4]), torch.tensor([32768, 0, 7])
        for i in range(n):
            for j in range(t):
                want[slots[i], (contexts[i] + j) % w, 0] = keys[i, j]
                want[slots[i], (contexts[i] + j) % w, 1] = gates[i, j]
        self.si.write_tails(field, slots, contexts, keys, gates)
        self.assertTrue(torch.equal(field, want))

    def test_mask_horizon(self):
        from engine.modules.sparse_indexer import topk_positions
        logits = torch.randn(7, 40)
        ke = torch.tensor([0, 1, 5, 39, 40, 40, 12], dtype=torch.int32)
        want = logits.clone()
        topk_positions(want, 4, valid=ke, inplace=True)           # masks the caller's logits past `valid`
        got = self.si.mask_horizon(logits.clone(), ke)
        self.assertTrue(torch.equal(got, want))

    def test_pool_addresses(self):
        caches, kp = self.caches, 4
        for layer, t in ((3, 1), (3, 7), (7, 7)):
            max_pools = (kp - 1 + t) // kp
            counts, slots = self.si.pool_addresses(self.contexts, *caches.pool_maps(layer), kp, t, max_pools, caches.candidate_capacity)
            pids = (self.contexts[:, None] // kp + torch.arange(max_pools)).clamp_max(caches.candidate_capacity - 1)
            self.assertTrue(torch.equal(counts, (self.contexts % kp + t) // kp))
            self.assertTrue(torch.equal(slots, caches.pool_rows(layer, pids).long()))
            self.assertEqual((counts.dtype, slots.dtype), (torch.int64, torch.int64))


class SelectRowsTests(unittest.TestCase):
    """`_select_rows` against the segment loop's selection, row by row: the same lane calls on the same
    tensors, the same top-k, the same finalize -- on the CPU, literally the same ops."""

    KP, TOPK, T, NH, D, N_CAND = 4, 16, 7, 2, 8, 64

    def setUp(self):
        from engine.base import constants
        constants.forget()
        torch.manual_seed(11)
        rows, kp, d = 4, self.KP, self.D
        F = NS(block=16, kpool=kp, idx_dim=d, topk=self.TOPK, kv_lora=1)
        layout = NS(block_bytes=kp * (d + 4), pool_offsets={0: 0}, token_offsets={0: 0})   # one pool record per pool slot
        table = torch.randperm(64, dtype=torch.int32)[:rows * 16].view(rows, 16)         # 16 blocks x 4 pools = the capacity
        real = NS(F=F, layout=layout, block_table=table)
        self.caches = GraphCaches(real, torch.arange(rows), torch.arange(rows) + 1, self.N_CAND * kp)
        self.caches.gather()
        self.keys = torch.randn(64 * 4 + 4, d)
        self.scales = torch.rand(64 * 4 + 4)
        self.contexts = torch.tensor([0, 3, 100, 248])                                        # + T <= the capacity's 256 tokens
        self.q8 = torch.randn(rows * self.T, self.NH, d)
        self.w = torch.rand(rows * self.T, self.NH)
        self.calls = []

        def indexer_logits(q8, keys, scales, w, ke, ks=None):                                  # deterministic in its inputs
            self.calls.append((q8.shape[0], keys.shape[0], None if ks is None else ks.clone()))
            return (q8.float().sum((-1, -2))[:, None] * 0.001 + (keys.float().sum(-1) * scales)[None, :] * 0.37).sin()

        from engine.modules import sparse_indexer as si
        from engine.profiles.glm53.lanes import reference_decode_rows
        self.net = NS(F=F, lanes=NS(indexer_logits=indexer_logits, pool_slots=si.pool_slots, decode_rows=reference_decode_rows()))

    def tearDown(self):
        from engine.base import constants
        constants.forget()

    def loop(self):
        """The segment loop's selection (net._indexer), one row at a time."""
        from engine.modules.sparse_indexer import topk_positions, pool_slots
        rows, t, kp, k = 4, self.T, self.KP, self.TOPK // self.KP
        width = self.TOPK + kp - 1
        slots = torch.empty((rows * t, width), dtype=torch.int32)
        valid = torch.empty(rows * t, dtype=torch.int32)
        for r in range(rows):
            sl = slice(r * t, (r + 1) * t)
            seq_lens = (self.contexts[r] + iota(t, "cpu") + 1).to(torch.int32)
            cand = Glm53Caches.pool_slots(self.caches, 0, r, iota(self.N_CAND, "cpu")).long()
            ke = seq_lens // kp
            logits = self.net.lanes.indexer_logits(self.q8[sl], self.keys[cand], self.scales[cand], self.w[sl], ke)
            ids = topk_positions(logits[:, :self.N_CAND].float(), k, valid=ke, inplace=True)
            pool_slots(ids, seq_lens, kp, *self.caches.token_map(0, r), slots[sl], valid[sl])
        return slots, valid

    def test_the_fold_selects_what_the_loop_selects(self):
        rows, t, kp = 4, self.T, self.KP
        want = self.loop()
        self.calls.clear()
        slots = torch.full((rows * t, self.TOPK + kp - 1), -7, dtype=torch.int32)
        valid = torch.full((rows * t,), -7, dtype=torch.int32)
        Glm53Net._select_rows(self.net, 0, self.q8, self.w, self.keys, self.scales, self.N_CAND, self.contexts, t,
                              self.caches, slots, valid)
        self.assertTrue(torch.equal(slots, want[0]))
        self.assertTrue(torch.equal(valid, want[1]))
        # one logits call per row, over the row's own candidates, with the kept zeros as the keys' start
        self.assertEqual([c[:2] for c in self.calls], [(t, self.N_CAND)] * rows)
        for _, _, ks in self.calls:
            self.assertTrue(torch.equal(ks, torch.zeros(t, dtype=torch.int32)))

    def test_rows_short_of_the_selection_width_finalize_like_the_loop(self):
        """Contexts with fewer complete pools than the top-k width: the loop pads the misses with -1 before the
        finalize, the fold leaves them to it -- the slots and counts written are the same."""
        self.contexts = torch.tensor([0, 1, 5, 9])                                  # 0, 0, 1, 2 complete pools before the first query
        want = self.loop()
        slots = torch.full_like(want[0], -7)
        valid = torch.full_like(want[1], -7)
        Glm53Net._select_rows(self.net, 0, self.q8, self.w, self.keys, self.scales, self.N_CAND, self.contexts, self.T,
                              self.caches, slots, valid)
        self.assertTrue(torch.equal(slots, want[0]) and torch.equal(valid, want[1]))

    def test_a_capacity_below_the_selection_width_is_refused(self):
        with self.assertRaisesRegex(ValueError, "candidate capacity"):
            Glm53Net._select_rows(self.net, 0, self.q8, self.w, self.keys, self.scales, self.TOPK // self.KP - 1, self.contexts,
                                  self.T, self.caches, torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
