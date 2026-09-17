"""The QSA attention expands the chosen blocks inside its own tiles (engine/QWEN38_CARRY.md Q5).

Each QSA layer selected blocks, expanded them to positions in a launch of their own (engine/kernels/qsa
_expand_qsa_indices_kernel: int32 [rows, top-k + ratio - 1], the causal tail of the open group appended) and handed that
buffer to the sparse attention, which loads a tile of it at a time. `qsa_select_paged_blocks` keeps the blocks and
`qsa_sparse_paged_attention_blocks` computes each tile's positions with the expansion's own int32 arithmetic where it
loaded them (FROM_BLOCKS): the same positions on the same tiles and splits, so the attention's bytes are the expanded
path's -- 13 launches fewer a step (12 layers and the MTP head's) and no [rows, 2051] buffer at prefill. Since Q6 the
attention reads a row's blocks sorted ascending (-1 last) in its own program: its bytes are the expanded attention over
the sorted row, and every order of the same set gives them.

Both paths run on the served kernels -- on a GPU, or under TRITON_INTERPRET=1 with tests/test_engine_qwen38_kernels'
accommodations -- and are compared byte for byte; their agreement with the oracles is that file's.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_qsa_blocks
"""
from pathlib import Path
import unittest
from unittest import mock

from tests import test_engine_qwen38_kernels as harness                  # the module, so its tests are not collected here
from tests.test_engine_qwen38_kernels import (DEVICE, INTERPRET, RUNS, RUNS_REASON, W, Launches, block_table, generator,
                                              host_meta, paged, randn, served_kernels, torch)

ROOT = Path(__file__).resolve().parents[1]


def sorted_blocks(blocks):
    """Each row ascending with -1 last: the order the block attention reads the chosen blocks in."""
    big = torch.iinfo(torch.int32).max
    ordered = torch.where(blocks < 0, torch.full_like(blocks, big), blocks).sort(dim=1).values
    return torch.where(ordered == big, torch.full_like(ordered, -1), ordered)


@unittest.skipUnless(RUNS, RUNS_REASON)
class BlockSelectionTests(unittest.TestCase):
    def test_the_blocks_are_what_the_token_selection_expands(self):
        from engine.kernels import qsa
        f = harness.SelectionTests().fixture(71)
        meta = f.meta
        args = (f.queries, f.key_cache, meta.page_table, meta.rows_req, meta.positions32, meta.lengths, W.budget, W.ratio)
        columns = meta.page_table.shape[1] * f.key_cache.shape[1]
        with served_kernels():
            tokens = qsa.qsa_select_paged_tokens(*args)
            blocks = qsa.qsa_select_paged_blocks(*args)
            expanded = qsa.expand_qsa_block_indices_cuda(blocks, meta.positions32, meta.lengths, meta.rows_req, W.ratio,
                                                         W.budget)
            with mock.patch.object(qsa, "_LOGITS_WORKSPACE_BYTES", 2 * columns * 4):      # two rows a scoring chunk
                chunked = qsa.qsa_select_paged_blocks(*args)
        self.assertEqual((tuple(blocks.shape), blocks.dtype), ((meta.positions.numel(), W.budget // W.ratio), torch.int32))
        self.assertTrue(torch.equal(expanded, tokens))
        self.assertTrue(torch.equal(chunked, blocks))


@unittest.skipUnless(RUNS, RUNS_REASON)
class BlockAttentionTests(unittest.TestCase):
    """qsa_sparse_paged_attention_blocks against qsa_sparse_paged_attention over expand_qsa_block_indices_cuda's
    positions of the sorted blocks, byte for byte, in one split and in several; and the same bytes for every order of a
    row's blocks."""

    def case(self, gen, budget, kv_heads):
        ratio, page, D = W.ratio, W.block, W.head_dim
        blocks_wide = budget // ratio
        # (seq, slot, ctx, length): a first position (only its tail), fewer complete groups than the budget holds, the
        # budget exactly, more groups than it holds with a two-position tail, and a verify pair across a group boundary
        requests = ((0, 1, 0, 1), (1, 2, (blocks_wide // 2) * ratio + 1, 1), (2, 3, blocks_wide * ratio - 1, 1),
                    (3, 4, 3 * blocks_wide * ratio + 1, 1), (4, 5, 2 * blocks_wide * ratio + ratio - 1, 2))
        pages = 8 * len(requests)
        wide = max(-(-(ctx + length) // page) for _, _, ctx, length in requests)
        table = block_table(gen, requests, wide, page, pages)
        meta = host_meta(requests, table, block=page, ratio=ratio)
        k_cache, _ = paged(gen, pages, page, kv_heads, D)
        v_cache, _ = paged(gen, pages, page, kv_heads, D)
        rows = meta.positions.numel()
        q = randn(gen, rows, W.heads, D)
        # each row's chosen blocks as the selection leaves them: distinct visible blocks in any order, -1 past what it
        # found, and a row that found nothing
        blocks = torch.full((rows, blocks_wide), -1, dtype=torch.int32)
        for r, position in enumerate(meta.positions.tolist()):
            visible = (position + 1) // ratio
            found = 0 if r == rows - 1 else min(visible, blocks_wide)
            blocks[r, :found] = torch.randperm(max(visible, 1), generator=gen)[:found].to(torch.int32)
        return meta, q, k_cache, v_cache, blocks.to(DEVICE)

    def test_the_attention_over_blocks_is_the_attention_over_their_expansion(self):
        from engine.kernels import qsa
        gen = generator(72)
        splits = set()
        # a 12-position budget fits one 16-column tile (one split); the wider one splits its tiles
        for budget in ((12, 64) if INTERPRET else (12, W.budget)):
            for kv_heads in (W.kv_heads, 2):
                meta, q, k_cache, v_cache, blocks = self.case(gen, budget, kv_heads)
                attend = Launches(qsa._qsa_sparse_paged_gqa_splitk_kernel)
                with served_kernels(), mock.patch.object(qsa, "_qsa_sparse_paged_gqa_splitk_kernel", attend):
                    positions = qsa.expand_qsa_block_indices_cuda(sorted_blocks(blocks), meta.positions32, meta.lengths,
                                                                  meta.rows_req, W.ratio, budget)
                    want = qsa.qsa_sparse_paged_attention(q, k_cache, v_cache, positions, meta.page_table, meta.rows_req)
                    got = qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, blocks, meta.positions32,
                                                                meta.lengths, W.ratio, budget, meta.page_table,
                                                                meta.rows_req)
                expanded_grid, blocks_grid = attend.grids
                splits.add(blocks_grid[2])
                with self.subTest(budget=budget, kv_heads=kv_heads, splits=blocks_grid[2]):
                    self.assertEqual(blocks_grid, expanded_grid)                      # the same programs and splits
                    self.assertTrue(bool((positions >= 0).any()) and bool((positions < 0).any()))
                    self.assertTrue(torch.equal(got, want))
        self.assertTrue(1 in splits and max(splits) > 1, splits)                     # one split and several both ran

    def test_every_order_of_a_row_s_blocks_attends_alike(self):
        """The selectors leave a row's set in their own orders (torch.topk by value, prefill_topk and st_dsa_select by
        their bins): reversed, rotated or shuffled with its -1 entries anywhere, the output is the same bytes."""
        from engine.kernels import qsa
        gen = generator(73)
        for budget in ((12, 64) if INTERPRET else (12, W.budget)):
            meta, q, k_cache, v_cache, blocks = self.case(gen, budget, W.kv_heads)
            args = lambda b: (q, k_cache, v_cache, b, meta.positions32, meta.lengths, W.ratio, budget, meta.page_table,
                              meta.rows_req)
            shuffled = torch.stack([row[torch.randperm(row.numel(), generator=gen).to(row.device)] for row in blocks])
            orders = {"sorted": sorted_blocks(blocks), "reversed": blocks.flip(1), "rotated": blocks.roll(1, dims=1),
                      "shuffled": shuffled}
            with served_kernels():
                want = qsa.qsa_sparse_paged_attention_blocks(*args(blocks))
                got = {name: qsa.qsa_sparse_paged_attention_blocks(*args(order.contiguous()))
                       for name, order in orders.items()}
            for name, out in got.items():
                with self.subTest(budget=budget, order=name):
                    self.assertTrue(torch.equal(out, want))

    def test_the_blocks_entry_refuses_what_the_expansion_refuses(self):
        from engine.kernels import qsa
        rows, D = 2, W.head_dim
        q = torch.zeros(rows, W.heads, D, dtype=torch.bfloat16)
        cache = torch.zeros(4, W.block, 1, D, dtype=torch.bfloat16)
        table, req = torch.zeros(1, 4, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32)
        positions, lengths = torch.zeros(rows, dtype=torch.int32), torch.ones(1, dtype=torch.int32)
        good = torch.zeros(rows, W.budget // W.ratio, dtype=torch.int32)
        call = lambda **kw: qsa.qsa_sparse_paged_attention_blocks(
            q, cache, cache, kw.get("blocks", good), kw.get("positions", positions), kw.get("lengths", lengths), W.ratio,
            kw.get("topk", W.budget), table, req)
        with self.assertRaisesRegex(ValueError, "divisible"):
            call(topk=W.budget + 1)
        with self.assertRaisesRegex(ValueError, "compressed top-k"):
            call(blocks=torch.zeros(rows, W.budget // W.ratio + 1, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "int32"):
            call(positions=positions.long())
        with self.assertRaisesRegex(ValueError, "packed row metadata"):
            call(positions=torch.zeros(1, dtype=torch.int32).expand(rows))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):                         # the launch contract, past the checks
            call()


class ServedLaneTests(unittest.TestCase):
    def test_the_layer_attends_the_blocks(self):
        lanes = (ROOT / "engine/profiles/qwen38/lanes.py").read_text()
        self.assertIn("qsa.qsa_select_paged_blocks, qsa.qsa_sparse_paged_attention_blocks", lanes)
        net = (ROOT / "engine/profiles/qwen38/net.py").read_text()
        self.assertIn("attended = lanes.qsa_attend(q, K, V, blocks, meta.positions32, meta.lengths, F.idx_ratio,", net)


if __name__ == "__main__":
    unittest.main()
