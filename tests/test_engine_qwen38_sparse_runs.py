"""A prefill segment's sparse QSA rows attend in runs: RUN consecutive rows a program over the union of their chosen
blocks (engine/kernels/qsa `_qsa_sparse_runs_kernel`), each block read once for the rows that chose it.

The split launch gives every row a program of its own: it gathers the row's 512 chosen blocks and stacks the row's six
heads into an MMA of sixteen rows. Two rows' twelve heads fit the same MMA, and consecutive rows choose many of the same
blocks, so a program of two reads each shared block once at the cost the two rows' own tiles had. A row attends exactly
its own columns in ascending order, its open group last; its sums round at the union's tile boundaries, so it is held
to the split launch within the oracle's band, not byte for byte.

Held on the served kernels -- a GPU, or TRITON_INTERPRET=1 -- against `qsa_sparse_paged_attention_blocks`' split
launch: runs of one, two and four with a short last run, neighbours that choose alike and ones that do not, one and
two KV heads, with and without the output gate.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_sparse_runs
"""
import unittest
from unittest import mock

from tests.test_engine_qwen38_kernels import (DEVICE, INTERPRET, RUNS, RUNS_REASON, TRITON, W, Launches, block_table,
                                              generator, host_meta, paged, randn, served_kernels, torch)


def case(gen, ctx, rows, kv_heads, budget):
    page, D = W.block, W.head_dim
    requests = ((0, 1, ctx, rows),)
    pages = 8 + -(-(ctx + rows) // page)
    table = block_table(gen, requests, -(-(ctx + rows) // page), page, pages)
    meta = host_meta(requests, table, block=page, ratio=W.ratio)
    k_cache, _ = paged(gen, pages, page, kv_heads, D)
    v_cache, _ = paged(gen, pages, page, kv_heads, D)
    return meta, randn(gen, rows, W.heads, D), k_cache, v_cache, randn(gen, rows, W.heads, D)


def selections(gen, meta, budget, alike: float):
    """int32 [rows, budget // ratio]: each row's chosen complete groups in a random order, -1 after; a row keeps
    `alike` of the row before's choices that it sees (neighbours choose alike) and draws the rest."""
    blocks = budget // W.ratio
    positions = meta.positions32.cpu()
    out = torch.full((positions.numel(), blocks), -1, dtype=torch.int32)
    before = None
    for r, position in enumerate(positions.tolist()):
        sees = (position + 1) // W.ratio
        count = min(sees, blocks)
        kept = [] if before is None else [b for b in before if b < sees][:int(alike * count)]
        rest = [b for b in torch.randperm(sees, generator=gen).tolist() if b not in kept][:count - len(kept)]
        chosen = kept + rest
        out[r, :count] = torch.tensor(chosen, dtype=torch.int32)[torch.randperm(count, generator=gen)]
        before = chosen
    return out.to(DEVICE)


@unittest.skipUnless(RUNS, RUNS_REASON)
class SparseRunsTests(unittest.TestCase):
    def assertRunsHold(self, gen, ctx, rows, kv_heads, alike, runs=(1, 2, 4)):
        from engine.kernels import qsa
        budget = 64 if INTERPRET else W.budget
        meta, q, k_cache, v_cache, gate = case(gen, ctx, rows, kv_heads, budget)
        chosen = selections(gen, meta, budget, alike)
        for gated in (None, gate):
            with served_kernels():
                want = qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, chosen, meta.positions32,
                                                             meta.lengths, W.ratio, budget, meta.page_table,
                                                             meta.rows_req, gate=gated)
            for run in runs:
                launches = Launches(qsa._qsa_sparse_runs_kernel)
                with served_kernels(), mock.patch.object(qsa, "_qsa_sparse_runs_kernel", launches), \
                        mock.patch.object(qsa, "RUNS_MIN_ROWS", 1), mock.patch.object(qsa, "_RUNS_OVERRIDE", (run, 4)):
                    got = qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, chosen, meta.positions32,
                                                                meta.lengths, W.ratio, budget, meta.page_table,
                                                                meta.rows_req, gate=gated, one_request=True)
                with self.subTest(ctx=ctx, rows=rows, kv_heads=kv_heads, alike=alike, run=run, gate=gated is not None):
                    self.assertEqual(launches.grids, [(-(-rows // run), kv_heads)])
                    err = (got.float() - want.float()).abs()
                    self.assertLess(float(err.max() / want.float().abs().max()), 2.0 ** -6)
                    self.assertLess(float(err.square().mean().sqrt() / want.float().square().mean().sqrt()), 2.0 ** -7)
                    self.assertTrue(bool(got.float().abs().sum() > 0))

    def test_neighbours_that_choose_alike_and_ones_that_do_not(self):
        """Past the budget's reach every row chooses 16 of its groups (the interpreter's budget of 64); a row keeps
        nine tenths, half, or none of the row before's choices. A short last run (rows not a multiple of four)."""
        gen = generator(2020)
        for alike in (0.9, 0.5, 0.0):
            self.assertRunsHold(gen, 150, 23, W.kv_heads, alike)

    def test_rows_across_the_reach_and_two_kv_heads(self):
        """Rows that see fewer groups than the budget (all of them chosen) next to rows past the reach, whose open
        groups differ within a run; two KV heads (three heads each)."""
        gen = generator(2021)
        self.assertRunsHold(gen, 50, 30, 2, 0.8)

    def test_a_decode_row_does_not_take_the_run_launch(self):
        from engine.kernels import qsa
        budget = 64 if INTERPRET else W.budget
        gen = generator(2022)
        meta, q, k_cache, v_cache, _ = case(gen, 150, 3, W.kv_heads, budget)
        chosen = selections(gen, meta, budget, 0.5)
        launches = Launches(qsa._qsa_sparse_runs_kernel)
        with served_kernels(), mock.patch.object(qsa, "_qsa_sparse_runs_kernel", launches):
            qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, chosen, meta.positions32, meta.lengths, W.ratio,
                                                  budget, meta.page_table, meta.rows_req, one_request=True)
        self.assertEqual(launches.grids, [], "fewer rows than RUNS_MIN_ROWS keep the split launch")


@unittest.skipUnless(torch is not None and TRITON, "engine/kernels/qsa imports Triton")
class WrapperTests(unittest.TestCase):
    def test_the_hook_and_the_flag_are_checked(self):
        from engine.kernels import qsa
        with mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)):
            rows, D = 2, 16
            q = torch.zeros(rows, 4, D, dtype=torch.bfloat16)
            cache = torch.zeros(4, 8, 1, D, dtype=torch.bfloat16)
            table, req = torch.zeros(1, 4, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32)
            positions, lengths = torch.zeros(rows, dtype=torch.int32), torch.ones(1, dtype=torch.int32)
            ids = torch.zeros(rows, 3, dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, "one_request is a declared boolean"):
                qsa.qsa_sparse_paged_attention_blocks(q, cache, cache, ids, positions, lengths, 4, 12, table, req,
                                                      one_request=1)
        for bad in ((3, 4), (16, 4), (2,)):
            with mock.patch.object(qsa, "_RUNS_OVERRIDE", bad), self.assertRaisesRegex(ValueError, "_RUNS_OVERRIDE"):
                qsa._sparse_runs(None, None, None, None, None, None, None, None, (None, None, 4, 16), 6, 16)


if __name__ == "__main__":
    unittest.main()
