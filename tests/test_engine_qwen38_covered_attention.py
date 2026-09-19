"""A step the QSA budget covers attends densely, a run of rows a program, and the output is the sparse launch's bytes
(engine/QWEN38_CARRY.md Q10).

Every row of a covered step attends every position up to its own (carry Q11's covered half: its chosen blocks are all
the groups it sees). The sparse launch still treated it as a selection: each program loaded its row's 512 block ids,
sorted them, expanded them tile by tile into the positions 0, 1, 2, ... they always are, gathered K and V for its one
row, and walked all 2,051 columns of the budget however short the prompt. `_qsa_covered_paged_gqa_kernel` reads a
tile's positions off its columns, takes up to four consecutive rows of one request a program so a K/V tile is read
once for the run, and stops where the run's furthest row does. Each row steps its own softmax with the sparse kernel's
operations on the sparse launch's tiles and splits (`_split_profile`, shared), so nothing rounds differently.

Held on the served kernels -- a GPU, or TRITON_INTERPRET=1 -- byte for byte against
`qsa_sparse_paged_attention_blocks` over the covered ids: steps of several segments (runs of one) and prefill
segments through every split profile their row counts reach, runs of 1..4 with a short last run, one and two KV
heads, with and without the output gate (in the final store of a one-split launch, in the merge of a split one).

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_covered_attention
"""
import unittest
from unittest import mock

from tests.test_engine_qwen38_kernels import (DEVICE, INTERPRET, RUNS, RUNS_REASON, TRITON, W, Launches, bits,
                                              block_table, generator, host_meta, paged, randn, served_kernels, torch)


def reach(budget: int) -> int:
    return (budget // W.ratio + 1) * W.ratio - 1


def case(gen, requests, kv_heads):
    """A covered host step over paged BF16 K/V: its metadata, queries, caches and output gate."""
    page, D = W.block, W.head_dim
    pages = 8 * len(requests)
    wide = max(-(-(ctx + length) // page) for _, _, ctx, length in requests)
    table = block_table(gen, requests, wide, page, pages)
    meta = host_meta(requests, table, block=page, ratio=W.ratio)
    k_cache, _ = paged(gen, pages, page, kv_heads, D)
    v_cache, _ = paged(gen, pages, page, kv_heads, D)
    rows = meta.positions.numel()
    return meta, randn(gen, rows, W.heads, D), k_cache, v_cache, randn(gen, rows, W.heads, D)


def covered_ids(meta, budget):
    from engine.modules.prefill_indexer import covered_pool_ids
    return covered_pool_ids((meta.positions32 + 1) // W.ratio, budget // W.ratio)


@unittest.skipUnless(RUNS, RUNS_REASON)
class CoveredAttentionTests(unittest.TestCase):
    def assertSparseBytes(self, gen, requests, budget, kv_heads, groups):
        from engine.kernels import qsa
        meta, q, k_cache, v_cache, gate = case(gen, requests, kv_heads)
        self.assertLessEqual(max(ctx + length for _, _, ctx, length in requests), reach(budget))
        ids = covered_ids(meta, budget)
        splits = set()
        for gated in (None, gate):
            sparse = Launches(qsa._qsa_sparse_paged_gqa_splitk_kernel)
            with served_kernels(), mock.patch.object(qsa, "_qsa_sparse_paged_gqa_splitk_kernel", sparse):
                want = qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, ids, meta.positions32, meta.lengths,
                                                             W.ratio, budget, meta.page_table, meta.rows_req, gate=gated)
            for group in groups:
                dense = Launches(qsa._qsa_covered_paged_gqa_kernel)
                with served_kernels(), mock.patch.object(qsa, "_qsa_covered_paged_gqa_kernel", dense):
                    got = qsa.qsa_covered_paged_attention(q, k_cache, v_cache, meta.positions32, meta.lengths, W.ratio,
                                                          budget, meta.page_table, meta.rows_req, gate=gated, group=group)
                with self.subTest(rows=q.shape[0], budget=budget, kv_heads=kv_heads, group=group, gate=gated is not None):
                    rows, heads, split = sparse.grids[0]
                    self.assertEqual(dense.grids[0], (-(-rows // group), heads, split))     # the sparse launch's splits
                    self.assertTrue(torch.equal(got, want))
                    self.assertTrue(torch.equal(bits(got), bits(want)))
                    self.assertTrue(bool(got.float().abs().sum() > 0))
                splits.add(split)
        return splits

    def test_a_step_of_several_segments(self):
        """Runs of one: the rows are several requests'. A first position, a segment across a group boundary, one that
        ends exactly at the reach."""
        gen = generator(1010)
        for budget in ((12, 64) if INTERPRET else (12, W.budget)):
            edge = reach(budget)
            for kv_heads in (W.kv_heads, 2):
                self.assertSparseBytes(gen, ((0, 1, 0, 1), (1, 2, 3, 6), (2, 3, edge - 5, 5)), budget, kv_heads, (1,))

    def test_a_prefill_segment_through_every_split_profile(self):
        """Row counts on both sides of the profile's steps (2, 8 and 256 programs), the last run short."""
        gen = generator(1011)
        budget = 64 if INTERPRET else W.budget
        edge = reach(budget)
        splits = set()
        for rows in ((2, 3, 9, 33, edge) if INTERPRET else (2, 3, 9, 33, 257, 515, edge)):
            splits |= self.assertSparseBytes(gen, ((0, 1, edge - rows, rows),), budget, W.kv_heads, (1, 2, 3, 4))
        # several splits ran, and at the model's widths one too (the interpreter's 67 columns always split here; its
        # one-split launches are the 12-position budget's, in the other cases)
        self.assertTrue(max(splits) > 1 and (INTERPRET or 1 in splits), splits)

    def assertStackedBytes(self, gen, requests, budget, kv_heads, tiles=(None,)):
        """The stacked launch (one request's rows, forced from any row count) against the sparse launch over the
        covered ids, byte for byte; `tiles`: _STACK_OVERRIDE's geometries besides the rule's (None)."""
        from engine.kernels import qsa
        meta, q, k_cache, v_cache, gate = case(gen, requests, kv_heads)
        ids = covered_ids(meta, budget)
        for gated in (None, gate):
            with served_kernels():
                want = qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, ids, meta.positions32, meta.lengths,
                                                             W.ratio, budget, meta.page_table, meta.rows_req, gate=gated)
            for tile in tiles:
                stacked = Launches(qsa._qsa_covered_stacked_kernel)
                with served_kernels(), mock.patch.object(qsa, "_qsa_covered_stacked_kernel", stacked), \
                        mock.patch.object(qsa, "STACK_MIN_ROWS", 1), mock.patch.object(qsa, "_STACK_OVERRIDE", tile):
                    got = qsa.qsa_covered_paged_attention(q, k_cache, v_cache, meta.positions32, meta.lengths, W.ratio,
                                                          budget, meta.page_table, meta.rows_req, gate=gated,
                                                          one_request=True)
                run = (qsa.STACK_M if tile is None else tile[1]) // (W.heads // kv_heads)
                with self.subTest(rows=q.shape[0], kv_heads=kv_heads, tile=tile, gate=gated is not None):
                    self.assertEqual(stacked.grids[0], (-(-q.shape[0] // run), kv_heads))
                    # the sparse launch's tiles on the MMA: its bytes (a GB10 sums an element's K in the same order at
                    # any M). The interpreter's numpy matmul does not -- a stacked M of 64 against a row's 8 moves an
                    # element by a step -- and wider tiles round the softmax elsewhere: the oracle's band there
                    if not INTERPRET and (tile is None or tile[0] == 16):
                        self.assertTrue(torch.equal(bits(got), bits(want)))
                    else:
                        err = float((got.float() - want.float()).abs().max() / want.float().abs().max())
                        self.assertLess(err, 2.0 ** -6)
                    self.assertTrue(bool(got.float().abs().sum() > 0))

    def test_a_prefill_segment_stacked(self):
        """A prefill segment's rows stacked a run a program: from position 0 (most columns past every row of the early
        runs), one that ends at the reach, a short last run; one and two KV heads (runs of 10 and of 21 rows), and the
        probe hook's wider tiles and narrower stack."""
        gen = generator(1013)
        budget = 64 if INTERPRET else W.budget
        edge = reach(budget)
        for kv_heads in (W.kv_heads, 2):
            for rows in ((23, edge) if INTERPRET else (257, edge)):
                self.assertStackedBytes(gen, ((0, 1, edge - rows, rows),), budget, kv_heads)
        self.assertStackedBytes(gen, ((0, 1, 0, 37),), budget, W.kv_heads, tiles=(None, (32, 64, 8), (16, 32, 4)))

    def test_a_short_prompt_stops_at_its_own_end(self):
        """The whole prompt from position 0: most of the budget's columns are past every row."""
        gen = generator(1012)
        budget = 64 if INTERPRET else W.budget
        for rows in (1, 2, 7, 23):
            self.assertSparseBytes(gen, ((0, 1, 0, rows),), budget, W.kv_heads, (4,))


@unittest.skipUnless(torch is not None and TRITON, "engine/kernels/qsa imports Triton")
class WrapperTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
        patch.start()
        self.addCleanup(patch.stop)

    def test_it_refuses_what_it_cannot_run(self):
        from engine.kernels import qsa
        rows, D = 2, 16
        q = torch.zeros(rows, 4, D, dtype=torch.bfloat16)
        cache = torch.zeros(4, 8, 1, D, dtype=torch.bfloat16)
        table, req = torch.zeros(1, 4, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32)
        positions, lengths = torch.zeros(rows, dtype=torch.int32), torch.ones(1, dtype=torch.int32)
        run = lambda **kw: qsa.qsa_covered_paged_attention(**{**dict(
            q=q, k_cache=cache, v_cache=cache, query_positions=positions, sequence_lengths=lengths, compress_ratio=4,
            token_topk=12, block_table=table, token_to_req=req), **kw})
        for group in (0, 5, True):
            with self.assertRaisesRegex(ValueError, "groups 1..4 rows"):
                run(group=group)
        with self.assertRaisesRegex(ValueError, "one_request is a declared boolean"):
            run(one_request=1)
        with self.assertRaisesRegex(ValueError, "divisible by compression ratio"):
            run(token_topk=10)
        with self.assertRaisesRegex(ValueError, "metadata is int32"):
            run(query_positions=positions.long())
        with self.assertRaisesRegex(ValueError, "packed row metadata"):
            run(token_to_req=torch.zeros(1, dtype=torch.int32)[:, None].expand(1, rows).reshape(-1))
        with self.assertRaisesRegex(ValueError, "is BF16"):
            run(q=q.float())
        with self.assertRaisesRegex(ValueError, "output gate is BF16"):
            run(gate=q.float())

    def test_the_profile_is_the_gb10_s_table(self):
        """`_split_profile`, the one rule of both launches, step for step: 16-wide tiles at 4 warps, the splits by the
        programs (carry Q9, measurements/qwen38_qsa_geometry_20260919)."""
        from engine.kernels import qsa
        for rows, splits in ((1, 64), (2, 64), (3, 16), (8, 16), (9, 4), (32, 4), (256, 4), (257, 1), (4096, 1)):
            self.assertEqual(qsa._split_profile(rows, 1, 8, 2051), (16, 129, splits, 4), rows)
        self.assertEqual(qsa._split_profile(4, 2, 4, 2051), (16, 129, 16, 4))   # programs are rows x KV heads
        self.assertEqual(qsa._split_profile(5, 2, 4, 2051), (16, 129, 4, 4))
        self.assertEqual(qsa._split_profile(4, 1, 8, 15)[:3], (16, 1, 1))       # one tile cannot split
        self.assertEqual(qsa._split_profile(4, 1, 8, 67)[:3], (16, 5, 4))       # nor five into more than four


if __name__ == "__main__":
    unittest.main()
