"""A host step the QSA budget covers attends every group it sees, unscored (engine/QWEN38_CARRY.md Q11, the covered half).

QSA keeps the `index_blocks` best complete groups a query sees (512 at the model's widths) and the open group's tail. A
query that sees no more complete groups than that keeps all of them, whatever they score: through position 2,050 the
scores and the top-k decide nothing. `Qwen38Net._qsa` ran them anyway -- one scoring launch over [rows, columns] fp32
logits and a selection a QSA layer, 12 layers and the MTP head's, on every prompt and first chunk that short.
`Qwen38Net._covered_blocks` answers for such a step from the positions alone: each row's groups ascending, -1 after,
built by the step's first QSA layer and read by the rest. The block attention sorts a row's blocks into that order in
its own program (carry Q6), so its bytes are the scored selection's.

Held here, on any box with torch: the reach is exact on both sides against the reference (modules/sparse_indexer
.qsa_select attends everything a covered query sees and drops a group one position later); the ids; the step rule
(the longest segment decides, a captured step is never covered, one build a step); and the layer's wiring (a covered
step never reaches the selection lane, any other step reaches it with the arguments it always had). On the served
kernels -- a GPU, or TRITON_INTERPRET=1 -- the scored selection of a covered step, sorted, is these ids byte for byte:

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_covered_blocks
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from tests import test_engine_qwen38_kernels as harness                  # the module, so its tests are not collected here
from tests.test_engine_qwen38_kernels import RUNS, RUNS_REASON, W, served_kernels, torch


def facts(ratio=4, budget=2048):
    return SimpleNamespace(idx_ratio=ratio, idx_budget=budget, index_blocks=budget // ratio)


def reach(F) -> int:
    """Positions a step may end at and stay covered: its last row sees (end // ratio) complete groups."""
    return (F.index_blocks + 1) * F.idx_ratio - 1


def host_step(*segments):
    """(ctx, length) a segment -> what `_covered_blocks` reads of a host step and its metadata."""
    positions = torch.cat([torch.arange(ctx, ctx + length, dtype=torch.int32) for ctx, length in segments])
    step = SimpleNamespace(segments=[SimpleNamespace(ctx=ctx, length=length) for ctx, length in segments])
    return step, SimpleNamespace(positions32=positions, covered_blocks=None)


def covered(F, step, meta):
    from engine.profiles.qwen38.net import Qwen38Net
    return Qwen38Net._covered_blocks(SimpleNamespace(F=F), step, meta)


@unittest.skipUnless(torch is not None, "requires torch")
class CoveredStepTests(unittest.TestCase):
    def test_the_reach_is_2051_positions_at_the_models_widths(self):
        F = facts()
        self.assertEqual(reach(F), 2051)
        self.assertIsNotNone(covered(F, *host_step((0, 2051))))
        self.assertIsNone(covered(F, *host_step((0, 2052))))               # its last row sees 513 groups
        self.assertIsNotNone(covered(F, *host_step((2050, 1))))            # a decode step at the edge
        self.assertIsNone(covered(F, *host_step((2051, 1))))

    def test_every_row_holds_the_groups_it_sees_ascending(self):
        F = facts(ratio=4, budget=12)
        step, meta = host_step((5, 9))                                     # positions 5..13: 1, 1, 2, 2, 2, 2, 3, 3, 3 groups
        ids = covered(F, step, meta)
        self.assertEqual((ids.dtype, tuple(ids.shape)), (torch.int32, (9, 3)))
        self.assertEqual(ids.tolist(), [[0, -1, -1]] * 2 + [[0, 1, -1]] * 4 + [[0, 1, 2]] * 3)

    def test_the_longest_segment_decides_and_rows_keep_their_own_positions(self):
        F = facts(ratio=4, budget=12)
        step, meta = host_step((0, 3), (11, 4), (7, 1))                    # ends 3, 15, 8: all within 15
        ids = covered(F, step, meta)
        self.assertEqual(ids.tolist(), [[-1, -1, -1]] * 3                  # positions 0..2: the first group closes at 3
                         + [[0, 1, 2]] * 4 + [[0, 1, -1]])                 # 11..14 see three groups, then 7 sees two
        self.assertIsNone(covered(F, *host_step((0, 3), (11, 5), (7, 1))))                     # one segment ends at 16

    def test_a_captured_step_is_never_covered(self):
        step = SimpleNamespace(captured=True)                              # no segments: its contexts are device values
        self.assertIsNone(covered(facts(), step, SimpleNamespace(positions32=None, covered_blocks=None)))

    def test_the_ids_are_built_once_a_step(self):
        from engine.modules import prefill_indexer
        F = facts(ratio=4, budget=12)
        step, meta = host_step((0, 9))
        with mock.patch.object(prefill_indexer, "covered_pool_ids", wraps=prefill_indexer.covered_pool_ids) as build:
            first = covered(F, step, meta)
            for _ in range(12):                                            # the other QSA layers and the MTP head's
                self.assertIs(covered(F, step, meta), first)
        self.assertEqual(build.call_count, 1)
        self.assertIs(meta.covered_blocks, first)


@unittest.skipUnless(torch is not None, "requires torch")
class ReferenceTests(unittest.TestCase):
    """The premise, against the oracle: inside the reach qsa_select attends every position the query sees -- the
    expansion of these ids and the open group's tail -- and one position past it a group is dropped."""

    def attended(self, F, position, seed):
        from engine.modules.rotary import rope_tables
        from engine.modules.sparse_indexer import qsa_select
        gen = torch.Generator().manual_seed(seed)
        D = 16
        raw = torch.randn(position + 1, D, generator=gen)
        cos, sin = rope_tables(torch.arange(position + 1), 8, 1e7)
        return set(qsa_select(torch.randn(4, D, generator=gen), raw, position, F.idx_ratio, F.index_blocks, cos, sin,
                              torch.randn(D, generator=gen) * 0.1, 1e-6).tolist())

    def expanded(self, F, position):
        ids = covered(F, *host_step((position, 1)))[0].tolist()
        groups = [b for b in ids if b >= 0]
        self.assertEqual(groups, sorted(groups))
        whole = (position + 1) // F.idx_ratio * F.idx_ratio
        return {b * F.idx_ratio + o for b in groups for o in range(F.idx_ratio)} | set(range(whole, position + 1))

    def test_inside_the_reach_the_reference_attends_what_the_ids_expand_to(self):
        F = facts(ratio=4, budget=12)
        for position in range(reach(F)):                                   # 0..14
            with self.subTest(position=position):
                self.assertEqual(self.expanded(F, position), self.attended(F, position, position))
                self.assertEqual(self.attended(F, position, position), set(range(position + 1)))

    def test_one_position_past_the_reach_a_score_decides(self):
        for F in (facts(ratio=4, budget=12), facts()):
            edge = reach(F)
            with self.subTest(budget=F.idx_budget):
                self.assertEqual(self.attended(F, edge - 1, 7), set(range(edge)))
                past = self.attended(F, edge, 7)
                self.assertEqual(len(past), F.idx_budget)                  # 513 groups seen, 512 kept, no tail
                self.assertLess(past, set(range(edge + 1)))
                self.assertIsNone(covered(F, *host_step((edge, 1))))


@unittest.skipUnless(torch is not None, "requires torch")
class LayerTests(unittest.TestCase):
    """`Qwen38Net._qsa` over recording lanes: what reaches the selection and the attention."""

    def layer(self, segments, F, attend_covered=None):
        from engine.profiles.qwen38.net import Qwen38Net
        step, meta = host_step(*segments)
        rows, calls = meta.positions32.numel(), []
        D, heads = 8, 2
        wide = SimpleNamespace(heads_local=heads, head_dim=D, kv_heads_local=1, idx_heads=4, idx_dim=D, rms_eps=1e-6,
                               rope_theta=1e7, rotary_dim=4, **vars(F))
        for name in ("positions", "rows_req", "page_table", "lengths", "starts", "slot_table", "kv_slots", "key_slots",
                     "ring_slots"):
            setattr(meta, name, name)                                      # opaque: the lanes here only pass them on
        scored = torch.full((rows, F.index_blocks), 7, dtype=torch.int32)

        def select(*args, group):
            calls.append(("select", args, group))
            return scored

        def attend(q, K, V, blocks, *args, gate, out=None, one_request=False):
            calls.append(("attend", blocks, args, one_request))
            return torch.zeros(q.shape[0], heads, D) if out is None else out

        lanes = SimpleNamespace(qsa_index_keys=lambda *a: None, qsa_select=select, qsa_attend=attend,
                                qsa_inputs=lambda *a: (torch.zeros(rows, heads, D), "iq"))
        if attend_covered is not None:
            lanes.qsa_attend_covered = attend_covered
        width = heads * 2 * D + D + D + wide.idx_heads * D + D
        net = SimpleNamespace(F=wide, lanes=lanes, p=mock.MagicMock(), comm=SimpleNamespace(all_reduce=lambda t: t),
                              linear=lambda x, name: torch.zeros(rows, width if name.endswith("in_proj") else 3))
        net._covered_blocks = lambda s, m: Qwen38Net._covered_blocks(net, s, m)
        net._sharded_blocks = lambda *a: Qwen38Net._sharded_blocks(net, *a)       # no query_shards declared: never splits
        net._score_runs = Qwen38Net._score_runs
        caches = SimpleNamespace(kv=lambda L: ("K", "V"), key_ring=lambda L: "ring", index_keys=lambda L: f"keys{L}")
        Qwen38Net._qsa(net, 3, torch.zeros(rows, 3), step, meta, caches)
        return meta, calls, scored

    def test_a_lane_table_with_the_covered_launch_attends_without_ids(self):
        """Carry Q10: the served table's covered launch takes the step -- no selection, no ids, no sparse launch --
        and only a step the budget covers."""
        from engine.profiles.qwen38.net import Qwen38Net
        seen = []

        def attend_covered(q, K, V, *args, gate, group, one_request, out=None):
            seen.append((q.shape[0], args, group, one_request))
            return torch.zeros(q.shape[0], 2, 8) if out is None else out

        F = facts(ratio=4, budget=12)
        meta, calls, _ = self.layer([(0, 9)], F, attend_covered)
        self.assertEqual(calls, [])
        self.assertIsNone(meta.covered_blocks)
        self.assertEqual(seen, [(9, (meta.positions32, "lengths", 4, 12, "page_table", "rows_req"), 4, True)])
        seen.clear()
        _, calls, _ = self.layer([(1, 3), (4, 5)], F, attend_covered)     # several segments, all covered: runs of one
        self.assertEqual([entry[2:] for entry in seen], [(1, False)])
        seen.clear()
        meta, calls, scored = self.layer([(0, 16)], F, attend_covered)    # one position past the reach
        # the 15 rows before the reach take the covered launch, the one past it the sparse launch over its ids
        self.assertEqual([entry[0] for entry in seen], [15])
        self.assertTrue(torch.equal(seen[0][1][0], meta.positions32[:15]))
        self.assertEqual([c[0] for c in calls], ["select", "attend"])
        self.assertTrue(torch.equal(calls[1][1], scored[15:]))
        self.assertTrue(torch.equal(calls[1][2][0], meta.positions32[15:]))
        self.assertTrue(Qwen38Net._covers(F, host_step((0, 15))[0]))
        self.assertFalse(Qwen38Net._covers(F, host_step((0, 16))[0]))
        self.assertFalse(Qwen38Net._covers(F, SimpleNamespace(captured=True)))

    def test_the_covered_rows_of_a_step(self):
        """`_covered_rows`: all of a covered step; a segment across the reach (15 positions at budget 12, ratio 4) up to
        it, from wherever it starts; nothing past it, of several uncovered segments, or of a captured step."""
        from engine.profiles.qwen38.net import Qwen38Net
        F = facts(ratio=4, budget=12)
        for segments, rows in ((((0, 9),), 9), (((1, 3), (4, 5)), 8), (((0, 16),), 15), (((10, 30),), 5),
                               (((15, 4),), 0), (((0, 3), (0, 20)), 0)):
            self.assertEqual(Qwen38Net._covered_rows(F, host_step(*segments)[0]), rows, segments)
        self.assertEqual(Qwen38Net._covered_rows(F, SimpleNamespace(captured=True)), 0)
        self.assertTrue(Qwen38Net._one_request(host_step((0, 16))[0]))
        self.assertFalse(Qwen38Net._one_request(host_step((1, 3), (4, 5))[0]))
        self.assertFalse(Qwen38Net._one_request(SimpleNamespace(captured=True)))

    def test_a_prefill_segment_attends_in_runs(self):
        """The sparse launch hears `one_request` for a host step of one segment -- its rows one request's consecutive
        positions, which the run launch needs -- and not for several segments."""
        _, calls, _ = self.layer([(0, 16)], facts(ratio=4, budget=12))
        self.assertEqual([(c[0], c[3]) for c in calls if c[0] == "attend"], [("attend", True)])
        _, calls, _ = self.layer([(0, 3), (0, 20)], facts(ratio=4, budget=12))
        self.assertEqual([(c[0], c[3]) for c in calls if c[0] == "attend"], [("attend", False)])

    def test_a_covered_step_never_reaches_the_selection(self):
        meta, calls, _ = self.layer([(0, 9)], facts(ratio=4, budget=12))
        self.assertEqual([c[0] for c in calls], ["attend"])
        self.assertIs(calls[0][1], meta.covered_blocks)
        self.assertEqual(calls[0][2], (meta.positions32, "lengths", 4, 12, "page_table", "rows_req"))

    def test_any_other_step_selects_with_the_arguments_it_always_had(self):
        meta, calls, scored = self.layer([(0, 16)], facts(ratio=4, budget=12))
        self.assertEqual([c[0] for c in calls], ["select", "attend"])
        self.assertEqual(calls[0][1], ("iq", "keys3", "page_table", "rows_req", meta.positions32, "lengths", 12, 4))
        self.assertEqual(calls[0][2], 4)                                   # one segment: runs of four rows (carry Q8)
        self.assertIs(calls[1][1], scored)
        self.assertIsNone(meta.covered_blocks)


def ascending(blocks):
    """Each row ascending with -1 last: the order the block attention reads a row's chosen blocks in (carry Q6)."""
    big = torch.iinfo(torch.int32).max
    ordered = torch.where(blocks < 0, torch.full_like(blocks, big), blocks).sort(dim=1).values
    return torch.where(ordered == big, torch.full_like(ordered, -1), ordered)


@unittest.skipUnless(RUNS, RUNS_REASON)
class ServedSelectionTests(unittest.TestCase):
    def test_the_scored_selection_of_a_covered_step_is_these_ids(self):
        """Three segments ending inside the reach, the last exactly at it: what qsa_select_paged_blocks scores and
        selects for them, read in the attention's order, against the unscored ids."""
        from engine.kernels import qsa
        F = facts(ratio=W.ratio, budget=W.budget)
        edge = reach(F)
        selection = harness.SelectionTests()
        selection.requests = lambda: ((0, 1, 0, 1), (1, 2, 3, 6), (2, 3, edge - 5, 5))   # (seq, slot, ctx, length)
        f = selection.fixture(1111)
        meta = f.meta
        with served_kernels():
            scored = qsa.qsa_select_paged_blocks(f.queries, f.key_cache, meta.page_table, meta.rows_req,
                                                 meta.positions32, meta.lengths, W.budget, W.ratio)
        step = SimpleNamespace(segments=[SimpleNamespace(ctx=ctx, length=length) for _, _, ctx, length in f.requests])
        ids = covered(F, step, meta)
        self.assertIsNotNone(ids)
        self.assertEqual((ids.dtype, tuple(ids.shape)), (scored.dtype, tuple(scored.shape)))
        self.assertTrue(torch.equal(ascending(scored), ids))
        self.assertEqual(int((ids[-1] >= 0).sum()), F.index_blocks)        # the last row sees the whole budget


if __name__ == "__main__":
    unittest.main()
