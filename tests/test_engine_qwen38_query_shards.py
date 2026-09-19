"""A long prefill step's index queries are scored a quarter a rank, and the selection is the unsplit step's
(engine/QWEN38_CARRY.md Q11, the rank half; GLM's #881).

Qwen3.8's indexer is replicated: every rank holds the same index queries and the same index keys, so four ranks scored
the same [rows, columns] logits and took the same top-k, four times over, in every QSA layer of every prefill chunk.
`Qwen38Net._sharded_blocks` has each rank score the rows it owns (engine/modules/prefill_indexer.QueryShard -- its
covered rows are their positions' ids and are not scored at all) and gathers the chosen ids once a layer.

What makes that a fold and not a change is a rule about the selectors, held first here. `qsa.select_blocks` takes the
radix select (ties to the lower block) for the rows it is handed where engine/kernels/prefill_topk admits them -- more
than 64 rows of a launch -- and torch.topk (ties as its candidates fall) elsewhere, a workspace's worth of rows a call.
Just past the budget's reach relu leaves many equal zeros at the budget's edge, where the two part. So a step is split
only where every call on both sides takes the radix select (`prefill_topk.admits_calls`, `qsa.shards_select_alike`
through `lanes.qsa_select_alike`); the ranks ask it of the same numbers and split together or not at all.

It is on unless a boot declines it (the operator's decision of 2026-09-18, the fleet unmeasured -- CHARTER D17):
fleet.py --no-query-shards, the launcher's ST_QUERY_SHARDS=0.

Held here, on any box with torch: the call arithmetic; that nothing is split where a boot declined it or the lane did
not say its calls select alike (and that an unsplit step reaches no collective); the default and its rollback's wiring; and, over LocalTP with a selection lane
whose answer is each row's own, that every row is scored by exactly one rank or by none, that the gathered ids are the
unsplit selection on every rank at both wire widths, and that a rank whose rows are all covered still meets the
gather. On the served kernels -- a GPU, or TRITON_INTERPRET=1 -- the split selection is the whole one's bytes:

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_query_shards
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from tests import test_engine_qwen38_kernels as harness                  # the module, so its tests are not collected here
from tests.test_engine_qwen38_kernels import INTERPRET, RUNS, RUNS_REASON, TRITON, W, served_kernels, torch

TP = 4


def facts(ratio=4, budget=12):
    return SimpleNamespace(idx_ratio=ratio, idx_budget=budget, index_blocks=budget // ratio)


def row_local_select(calls, rank):
    """A selection lane whose answer for a row is that row's position's alone -- as the served lane's is, whatever
    rows share its launch -- recording the positions each call was handed."""
    def select(iq, keys, table, rows_req, positions, lengths, budget, ratio, group=None):
        calls.append((rank, positions.tolist()))
        assert rank < 0 or group == 4                                      # a rank's call: one segment, runs of four (carry Q8)
        assert iq.shape[0] == rows_req.shape[0] == positions.shape[0]
        width = budget // ratio
        seen = ((positions + 1) // ratio).to(torch.int64)
        j = torch.arange(width, dtype=torch.int64)
        ids = (positions.to(torch.int64)[:, None] * 31 + j[None, :] * 17) % seen.clamp_min(1)[:, None]
        return torch.where(j[None, :] < seen[:, None], ids, torch.full_like(ids, -1)).to(torch.int32)
    return select


def prefill(rows, ctx):
    """What `_sharded_blocks` reads of a one-segment host step: the step, its metadata, the index queries and keys."""
    step = SimpleNamespace(segments=[SimpleNamespace(ctx=ctx, length=rows)])
    meta = SimpleNamespace(positions32=torch.arange(ctx, ctx + rows, dtype=torch.int32), lengths="lengths",
                           rows_req=torch.zeros(rows, dtype=torch.int32), page_table=torch.zeros(1, 8, dtype=torch.int32),
                           groups_seen=None)
    return step, meta, torch.zeros(rows, 1, 1), torch.zeros(1, 4, 1, 1)


def net(F, comm, select, *, declared=True, alike=lambda *a: True):
    lanes = SimpleNamespace(qsa_select=select, qsa_select_alike=alike)
    from engine.profiles.qwen38.net import Qwen38Net
    return SimpleNamespace(F=F, lanes=lanes, comm=comm, rank=comm.rank, query_shards=declared,
                           _score_runs=Qwen38Net._score_runs)


def sharded(stand_in, iq, step, meta, keys):
    from engine.profiles.qwen38.net import Qwen38Net
    return Qwen38Net._sharded_blocks(stand_in, iq, step, meta, keys)


@unittest.skipUnless(torch is not None, "requires torch")
class CallArithmeticTests(unittest.TestCase):
    def test_every_call_of_a_step_is_asked(self):
        from engine.kernels.prefill_topk import admits, admits_calls
        self.assertTrue(admits(65, 1024, 512) and admits(32768, 262144, 512))
        self.assertFalse(admits(64, 1024, 512) or admits(32769, 1024, 512) or admits(4096, 262145, 512)
                         or admits(4096, 1024, 3))
        for rows, per, want in ((4096, 32768, True),                       # one call
                                (1024, 512, True), (1100, 512, True),      # whole calls; and a remainder of 76
                                (1030, 512, False),                        # a remainder of 6 falls to torch.topk
                                (64, 32768, False), (40000, 65536, False), (0, 512, False)):
            with self.subTest(rows=rows, per=per):
                self.assertIs(admits_calls(rows, per, 1024, 512), want)

    def test_the_selection_asks_what_it_always_asked(self):
        """`select` reads the shape rule through `admits`: every refusal it made before it still makes."""
        from engine.kernels import prefill_topk
        valid = torch.zeros(100, dtype=torch.int32)
        for logits, k in ((torch.zeros(100, 8), 3), (torch.zeros(64, 8), 512), (torch.zeros(100), 512),
                          (torch.zeros(100, 8), 512)):                     # the last: a shape it takes, on the CPU
            self.assertIsNone(prefill_topk.select(logits, valid[:logits.shape[0]], k))

    @unittest.skipUnless(TRITON, "engine/kernels/qsa imports Triton")
    def test_the_lane_cuts_calls_by_its_logits_workspace(self):
        from engine.kernels import qsa
        columns = 1024
        with mock.patch.object(qsa, "_LOGITS_WORKSPACE_BYTES", 512 * columns * 4):       # 512 rows a scoring call
            self.assertTrue(qsa.shards_select_alike(4096, [1024, 1024, 1024, 1024], columns, 512))
            self.assertTrue(qsa.shards_select_alike(4096, [0, 989, 1024, 1024], columns, 512))     # a covered rank: none
            self.assertFalse(qsa.shards_select_alike(4096, [1024, 1024, 1024, 1030], columns, 512))  # 512 + 512 + 6
            self.assertFalse(qsa.shards_select_alike(1030, [258, 258, 257, 257], columns, 512))    # the whole's 6
            self.assertFalse(qsa.shards_select_alike(240, [60, 60, 60, 60], columns, 512))         # quarters of <= 64
            self.assertFalse(qsa.shards_select_alike(4096, [1024] * 4, columns, 3))                # never the radix


@unittest.skipUnless(torch is not None, "requires torch")
class DecisionTests(unittest.TestCase):
    """Where the step is not split, no rank scores a part and no rank reaches a collective."""

    def unsplit(self, rows=40, ctx=0, *, step=None, **kw):
        def gather(*a, **k):
            raise AssertionError("an unsplit step reached a collective")
        calls = []
        comm = SimpleNamespace(rank=1, world_size=TP, all_gather=gather)
        made, meta, iq, keys = prefill(rows, ctx)
        out = sharded(net(facts(), comm, row_local_select(calls, 1), **kw), iq, step or made, meta, keys)
        self.assertIsNone(out)
        self.assertEqual(calls, [])
        self.assertIsNone(meta.groups_seen)

    def test_a_boot_can_decline_it(self):
        self.unsplit(declared=False)
        stand_in = net(facts(), SimpleNamespace(rank=0, world_size=TP), None)
        del stand_in.query_shards                                          # a stand-in that never made the choice
        self.assertIsNone(sharded(stand_in, *[prefill(40, 0)[i] for i in (2, 0, 1, 3)]))

    def test_it_is_on_unless_a_boot_declines(self):
        """The operator's decision of 2026-09-18: the default, and the rollback from the launcher to the net."""
        import inspect
        from pathlib import Path
        from engine.profiles.qwen38.net import Qwen38Net
        root = Path(__file__).resolve().parents[1]
        self.assertIs(inspect.signature(Qwen38Net.__init__).parameters["query_shards"].default, True)
        fleet = (root / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        self.assertIn("prelude=None, query_shards: bool = True,", fleet)
        self.assertIn("hc_fp8=hc_fp8, query_shards=query_shards,", fleet)
        self.assertIn('ap.add_argument("--no-query-shards", action="store_true",', fleet)
        self.assertIn("query_shards=not a.no_query_shards,", fleet)
        launcher = (root / "launchers/start-st-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn('case "${ST_QUERY_SHARDS:-1}" in', launcher)
        self.assertIn('0) SHARDS_ARG="--no-query-shards" ;;', launcher)
        self.assertIn("$ONESHOT_ARG $SHARDS_ARG --port $PORT", launcher)

    def test_the_lane_says_its_calls_select_alike(self):
        asked = []
        self.unsplit(alike=lambda *a: asked.append(a) or False)
        self.assertEqual(asked, [(40, [0, 5, 10, 10], 8 * 4, 3, 4)])       # rows, each rank's scored rows, columns, top-k, runs
        self.unsplit(alike=None)                                           # a lane table that does not say

    def test_a_captured_step_several_segments_and_fewer_rows_than_ranks_are_whole(self):
        self.unsplit(step=SimpleNamespace(captured=True))
        self.unsplit(step=SimpleNamespace(segments=[SimpleNamespace(ctx=0, length=20)] * 2))
        self.unsplit(rows=3, ctx=100)

    def test_the_net_refuses_a_choice_that_is_not_a_boolean(self):
        from engine.profiles.qwen38.net import Qwen38Net
        with self.assertRaisesRegex(ValueError, "declared boolean"):
            Qwen38Net(SimpleNamespace(), SimpleNamespace(world_size=TP, rank=0), None, query_shards=1)


@unittest.skipUnless(torch is not None, "requires torch")
class GatherTests(unittest.TestCase):
    """Over LocalTP, a lane whose answer is each row's own: the split selection against the unsplit one."""

    def split(self, F, rows, ctx):
        from engine.base.comm import LocalTP
        from engine.modules.prefill_indexer import covered_pool_ids
        calls, sizes = [], []

        def rank(comm):
            wrapped = SimpleNamespace(rank=comm.rank, world_size=comm.world_size, all_gather=lambda packet, **kw:
                                      sizes.append(packet.numel() * packet.element_size()) or comm.all_gather(packet, **kw))
            step, meta, iq, keys = prefill(rows, ctx)
            return sharded(net(F, wrapped, row_local_select(calls, comm.rank)), iq, step, meta, keys)

        results = LocalTP(TP, timeout_s=20).run(rank)
        step, meta, iq, keys = prefill(rows, ctx)
        whole = row_local_select([], -1)(iq, keys, meta.page_table, meta.rows_req, meta.positions32, meta.lengths,
                                         F.idx_budget, F.idx_ratio)
        seen = (meta.positions32 + 1) // F.idx_ratio
        covered = seen <= F.index_blocks
        want = torch.where(covered[:, None], covered_pool_ids(seen, F.index_blocks), whole)
        return results, want, calls, covered, sizes

    def test_every_rank_holds_the_unsplit_selection(self):
        for F, rows, ctx in ((facts(budget=12), 40, 0),                    # rank 0 all covered, rank 1 half
                             (facts(budget=12), 37, 100),                  # nothing covered, a short last rank
                             (facts(budget=12), 16, 0),                    # one scored row in the step
                             (facts(budget=16), 40, 3), (facts(budget=16), 131, 9000)):   # an even width: 16-bit wire
            with self.subTest(budget=F.idx_budget, rows=rows, ctx=ctx):
                results, want, calls, covered, _ = self.split(F, rows, ctx)
                for got in results:
                    self.assertEqual((got.dtype, tuple(got.shape)), (torch.int32, (rows, F.index_blocks)))
                    self.assertTrue(torch.equal(got, want))
                scored = sorted(p for _rank, positions in calls for p in positions)
                self.assertEqual(scored, [ctx + r for r in range(rows) if not covered[r]])    # once each, covered never
                self.assertLessEqual(len(calls), TP)

    def test_a_rank_whose_rows_are_all_covered_still_meets_the_gather(self):
        results, want, calls, _, sizes = self.split(facts(budget=12), 40, 0)
        self.assertNotIn(0, [rank for rank, _ in calls])                   # rank 0 owns positions 0..9: nothing to score
        self.assertEqual(len(sizes), TP)
        self.assertTrue(all(torch.equal(got, want) for got in results))

    def test_the_wire_is_two_uint16_a_word_where_the_groups_fit(self):
        _, _, _, _, narrow = self.split(facts(budget=16), 40, 3)           # 10 rows a rank, 4 ids a row
        _, _, _, _, wide = self.split(facts(budget=12), 40, 0)             # an odd width cannot pair its ids
        self.assertEqual(set(narrow), {10 * 4 * 2})
        self.assertEqual(set(wide), {10 * 3 * 4})

    def test_the_groups_a_row_sees_are_counted_once_a_step(self):
        from engine.base.comm import LocalTP
        F = facts(budget=12)

        def rank(comm):
            step, meta, iq, keys = prefill(40, 0)
            stand_in = net(F, comm, row_local_select([], comm.rank))
            first = sharded(stand_in, iq, step, meta, keys)
            counted = meta.groups_seen
            for _ in range(12):                                            # the other QSA layers and the MTP head's
                self.assertTrue(torch.equal(sharded(stand_in, iq, step, meta, keys), first))
                self.assertIs(meta.groups_seen, counted)
            return counted.tolist()

        for counted in LocalTP(TP, timeout_s=20).run(rank):
            self.assertEqual(counted, [(p + 1) // 4 for p in range(40)])


@unittest.skipUnless(RUNS, RUNS_REASON)
class ServedSelectionTests(unittest.TestCase):
    def test_the_split_selection_is_the_whole_ones_bytes(self):
        """One prefill segment that starts inside the budget's reach and ends well past it, on the served kernels:
        each rank's gathered ids against one `qsa_select_paged_blocks` call over every row, both read in the
        attention's order (ascending, carry Q6): a covered row's scored selection is its ids in the selector's order,
        and the radix select's order within a row is its launch's, not the row's.
        Where the radix select builds -- a CUDA toolkit and the GB10 it is compiled for -- the step is wide enough to
        take it on both sides and the lane's own rule decides. Anywhere else (the interpreter, whose three-block budget
        never takes it; another GPU) the step stays under 64 rows and the rule is answered for it: torch.topk orders
        a row of this width by that row alone."""
        from engine.base.comm import LocalTP
        from engine.kernels import qsa
        from engine.profiles.qwen38.net import Qwen38Net
        F = facts(ratio=W.ratio, budget=W.budget)
        edge = (F.index_blocks + 1) * F.idx_ratio - 1
        small = INTERPRET or not radix_builds()
        rows, ctx = (40, edge - 5) if small else (400, edge - 11)
        selection = harness.SelectionTests()
        selection.requests = lambda: ((0, 1, ctx, rows),)                  # (seq, slot, ctx, length)
        f = selection.fixture(1212)
        meta = f.meta
        meta.groups_seen = None
        step = SimpleNamespace(segments=[SimpleNamespace(ctx=ctx, length=rows)])
        tp = LocalTP(TP, timeout_s=600)
        alike = (lambda *a: True) if small else qsa.shards_select_alike

        def rank(comm):
            lanes = SimpleNamespace(qsa_select=lambda *a, **k: tp.on_main(qsa.qsa_select_paged_blocks, *a, **k),
                                    qsa_select_alike=alike)
            stand_in = SimpleNamespace(F=F, lanes=lanes, comm=comm, rank=comm.rank, query_shards=True,
                                       _score_runs=Qwen38Net._score_runs)
            own = SimpleNamespace(positions32=meta.positions32, lengths=meta.lengths, rows_req=meta.rows_req,
                                  page_table=meta.page_table, groups_seen=None)
            return sharded(stand_in, f.queries, step, own, f.key_cache)

        with served_kernels():
            whole = qsa.qsa_select_paged_blocks(f.queries, f.key_cache, meta.page_table, meta.rows_req,
                                                meta.positions32, meta.lengths, W.budget, W.ratio)
            results = tp.run(rank)
        self.assertEqual(len(results), TP)
        for got in results:
            self.assertIsNotNone(got)                                      # the step was split
            self.assertTrue(torch.equal(ascending(got), ascending(whole)))
            if small:
                # torch.topk orders a row by that row alone, so past the covered rows even the order is the whole's.
                # The radix select's is not: its winners land as its bins fill, and the first GB10 run (2026-09-18,
                # measurements/qwen38_qsa_folds_20260918) found a row's ids in another order in a rank's 100-row launch
                # than in the step's 400 -- the same set, which is all the attention reads (it sorts them, carry Q6)
                self.assertTrue(torch.equal(got[5:], whole[5:]))


def radix_builds() -> bool:
    """engine/kernels/prefill_topk is a native build for the GB10 (sm_121a): it needs a CUDA toolkit and that device."""
    from torch.utils import cpp_extension
    return (torch.cuda.is_available() and cpp_extension.CUDA_HOME is not None
            and torch.cuda.get_device_capability() == (12, 1))


def ascending(blocks):
    """Each row ascending with -1 last: the order the block attention reads a row's chosen blocks in (carry Q6)."""
    big = torch.iinfo(torch.int32).max
    ordered = torch.where(blocks < 0, torch.full_like(blocks, big), blocks).sort(dim=1).values
    return torch.where(ordered == big, torch.full_like(ordered, -1), ordered)


if __name__ == "__main__":
    unittest.main()
