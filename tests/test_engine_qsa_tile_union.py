"""The QSA tile-union prefill attention (engine/kernels/qsa_tile_union.py; sm_121a intake U12, vLLM PR 55430) against
the engine's split-K QSA launch on the same inputs.

Two consecutive rows of one segment form a tile; the kernel walks the union of their chosen blocks, gathers each
block once, and masks each row to its own blocks inside the online softmax. Every row still attends exactly what the
split-K launch (`qsa.qsa_sparse_paged_attention_blocks`) attends for it -- its chosen blocks and its open group's
causal tail -- but the softmax steps through them in other chunks (eight blocks and then the tails, against sixteen
columns), so the probabilities round to BF16 against other running maxima: the two are held to each other within the
served sparse attention's oracle band (tests/test_engine_qwen38_kernels.SparseAttentionTests': two BF16 steps at the
largest element, one in rms), not byte for byte. The largest differences seen are printed at the end of the run.

The steps: a single segment from a sequence's first position and one deep in its context, several segments of odd
lengths with a one-row segment among them, rows that keep fewer blocks than they could with -1 among their ids and rows
that keep none; one and two KV heads, with and without the output gate. The selections are a prefill's: a segment-wide
score plus a little of each row's own, so neighbours share most of their blocks.

On a GPU the widths are Qwen3.8's per-rank cell (6 query heads over one KV head of 256, pages of 768, the 2,048-token
budget) and every step is one `admits` takes -- at least 1,024 rows, 64 a segment on average -- since `attention`
refuses any other (D3); the single-GPU lane runs this class (probes/engine_qwen38_cells.GLUE_CASES). Under
TRITON_INTERPRET=1 the widths shrink (tests/test_engine_qwen38_kernels' W; a 64-token budget so a tile's union takes
several steps) and so do the steps, to a few tens of rows, with the tile's two row gates lowered for them
(`interpreter_gates`: where the launch pays is not what it computes); every launch is also held there to one store an
address (the GB10 race of qsa._norm_rope_partial, tests/test_engine_qwen38_store_once). `admits` at its own gates, the
refusals and the provenance record run on any CPU.

    wsl: TRITON_INTERPRET=1 ~/.cache/stk-engine-cpu/bin/python -m unittest tests.test_engine_qsa_tile_union -v
"""
import contextlib
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.test_engine_qwen38_kernels import (BF16_STEP, DEVICE, INTERPRET, RUNS, RUNS_REASON, TRITON, W, Launches,
                                              block_table, generator, host_meta, paged, randn, served_kernels, torch)

ROOT = Path(__file__).resolve().parents[1]
BAND = (2 * BF16_STEP, BF16_STEP)                  # SparseAttentionTests' band: (largest / largest, rms / rms)
PR_SOURCE_SHA256 = "9a843bf24775220b91e65d0df00b3b75b4a78e86fcbf43ae5056353f48f52a59"  # jschmied/vllm c5d7eba3
# a budget of several union steps under the interpreter (16 blocks a row, up to 32 a tile, 8 a step); the model's on a
# GPU
BUDGET = 64 if INTERPRET else W.budget
# how deep a "deep" segment starts: past the budget, so every row chooses among more blocks than it keeps
DEEP = 150 if INTERPRET else 12000
OBSERVED = dict(abs=0.0, largest=0.0, rms=0.0, rows=0)


def sized(on_a_gpu, interpreted):
    """A step's rows: what `admits` takes at its own gates on a GPU, a few tens under the interpreter."""
    return interpreted if INTERPRET else on_a_gpu


@contextlib.contextmanager
def interpreter_gates():
    """Under TRITON_INTERPRET=1, `TILE`'s two row gates at zero and nothing else of the tile moved: the gates say where
    the launch pays, not what it computes, and a 1,025-row step took the interpreter 46 s a comparison (35 s of it the
    split-K launch's). On a GPU: untouched -- every step there is one `admits` takes as it stands."""
    if not INTERPRET:
        yield
        return
    from engine.kernels import qsa_tile_union
    lowered = dataclasses.replace(qsa_tile_union.TILE, min_rows=0, min_rows_per_request=0)
    with mock.patch.object(qsa_tile_union, "TILE", lowered):
        yield


def tearDownModule():
    if OBSERVED["rows"]:
        print(f"\ntile-union vs split-K over {OBSERVED['rows']} rows: max |diff| {OBSERVED['abs']:.3g}, largest error / "
              f"largest value {OBSERVED['largest']:.3g}, error rms / rms {OBSERVED['rms']:.3g} (band {BAND[0]:.3g}, "
              f"{BAND[1]:.3g})", file=sys.stderr)


def chosen(gen, meta, block_topk: int, ratio: int, *, noise: float = 0.05, fewer=(), none=()):
    """int32 [rows, block_topk]: each row's distinct blocks among those it sees ((position + 1) // ratio), the best
    min(seen, block_topk) of a score its segment shares plus `noise` of its own -- neighbours share most of their
    blocks, as a prefill's rows do -- at shuffled columns with -1 in the rest; rows in `fewer` keep half of theirs,
    rows in `none` nothing."""
    positions, owners = meta.positions32.cpu(), meta.rows_req.cpu()
    seen = (positions + 1) // ratio
    base = torch.rand(int(owners.max()) + 1, int(seen.max()) + 1, generator=gen)
    out = torch.full((positions.numel(), block_topk), -1, dtype=torch.int32)
    for row in range(positions.numel()):
        visible = int(seen[row])
        keep = 0 if row in none else min(visible, block_topk)
        if keep:
            scores = base[owners[row], :visible] + noise * torch.rand(visible, generator=gen)
            ids = scores.topk(keep).indices.to(torch.int32)
            if row in fewer:
                ids = ids[torch.randperm(keep, generator=gen)[:max(1, keep // 2)]]
            out[row, torch.randperm(block_topk, generator=gen)[:ids.numel()]] = ids
    return out.to(DEVICE)


def case(gen, requests, kv_heads, **selection):
    """A prefill step of `requests` (seq, slot, ctx, length) over paged BF16 K/V at the served strides: its metadata
    (Qwen38Net.step_meta's), queries, caches, output gate and chosen blocks."""
    page, D = W.block, W.head_dim
    need = [-(-(ctx + length) // page) for _, _, ctx, length in requests]
    pages = sum(need) + 3
    table = block_table(gen, requests, max(need), page, pages)
    meta = host_meta(requests, table, block=page, ratio=W.ratio)
    k_cache, _ = paged(gen, pages, page, kv_heads, D)
    v_cache, _ = paged(gen, pages, page, kv_heads, D)
    rows = meta.positions.numel()
    q, gate = randn(gen, rows, W.heads, D, scale=2.0), randn(gen, rows, W.heads, D)
    return meta, q, k_cache, v_cache, gate, chosen(gen, meta, BUDGET // W.ratio, W.ratio, **selection)


def split_k(meta, q, k_cache, v_cache, blocks, gate=None):
    from engine.kernels import qsa
    return qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, blocks, meta.positions32, meta.lengths, W.ratio,
                                                 BUDGET, meta.page_table, meta.rows_req, gate=gate)


def tile_union(meta, q, k_cache, v_cache, blocks, gate=None, **kw):
    from engine.kernels import qsa_tile_union
    return qsa_tile_union.attention(q, k_cache, v_cache, blocks, meta.positions32, meta.starts, W.ratio, BUDGET,
                                    meta.page_table, meta.rows_req, gate=gate, **kw)


@unittest.skipUnless(RUNS, RUNS_REASON)
class TileUnionTests(unittest.TestCase):
    """`attention` against the split-K launch over the same step, within the served band."""

    def assertLikeSplitK(self, requests, *, kv_heads_set=(W.kv_heads, 2), seed=0, **selection):
        from engine.kernels import qsa, qsa_tile_union
        from engine.kernels.gated_residual import drift
        gen = generator(seed)
        for kv_heads in kv_heads_set:
            meta, q, k_cache, v_cache, gate, blocks = case(gen, requests, kv_heads, **selection)
            rows = q.shape[0]
            for gated in (None, gate):
                sparse = Launches(qsa._qsa_sparse_paged_gqa_splitk_kernel)
                with served_kernels(), interpreter_gates(), \
                        mock.patch.object(qsa, "_qsa_sparse_paged_gqa_splitk_kernel", sparse):
                    self.assertTrue(qsa_tile_union.admits(rows, len(requests), compress_ratio=W.ratio,
                                                          token_topk=BUDGET, page_size=W.block,
                                                          table_width=meta.page_table.shape[1],
                                                          cache_pages=k_cache.shape[0]))
                    want = split_k(meta, q, k_cache, v_cache, blocks, gated)
                    got = tile_union(meta, q, k_cache, v_cache, blocks, gated)
                with self.subTest(requests=len(requests), rows=rows, kv_heads=kv_heads, gate=gated is not None):
                    (grid,) = sparse.grids
                    # an admitted step is past 256 programs: the split-K launch is one split, its gate in the store
                    self.assertEqual(grid, (rows, kv_heads, grid[2] if INTERPRET else 1))
                    self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
                    largest, rms = drift(got, want)
                    difference = float((got.float() - want.float()).abs().max())
                    OBSERVED.update(abs=max(OBSERVED["abs"], difference), largest=max(OBSERVED["largest"], largest),
                                    rms=max(OBSERVED["rms"], rms), rows=OBSERVED["rows"] + rows)
                    self.assertLessEqual(largest, BAND[0], f"largest error {largest:.3g} of the largest value")
                    self.assertLessEqual(rms, BAND[1], f"error rms {rms:.3g} of the values' rms")
                    # a row with nothing to attend is zero in both, exactly; every other row attends something
                    empty = ~want.flatten(1).any(dim=1)
                    self.assertTrue(torch.equal(empty, ~got.flatten(1).any(dim=1)))
                    self.assertTrue(bool(got[~empty].abs().sum() > 0))
        return meta, blocks

    def test_a_segment_from_a_sequence_s_first_position(self):
        """Its first rows see no complete group (only their tails), then fewer groups than the budget holds, then
        more; an odd count of rows, so the last tile holds one."""
        meta, _ = self.assertLikeSplitK(((0, 1, 0, sized(2053, 71)),), seed=1)
        self.assertEqual(int(meta.positions32[0]), 0)
        self.assertGreater(int(meta.positions32[-1] + 1) // W.ratio, BUDGET // W.ratio)

    def test_a_segment_deep_in_its_context(self):
        """Every row sees more groups than the budget keeps, and chooses among them."""
        meta, blocks = self.assertLikeSplitK(((2, 3, DEEP + 1, sized(1025, 33)),), seed=2)
        self.assertTrue(bool(((meta.positions32 + 1) // W.ratio > BUDGET // W.ratio).all()))
        self.assertTrue(bool((blocks >= 0).all()))

    def test_several_segments_of_odd_lengths_and_a_one_row_segment(self):
        """Five segments: from position 0, a one-row segment at position 0 (its tail only), and three part-way into
        their contexts; tiles never straddle two of them."""
        lengths = sized((513, 1, 300, 257, 41), (13, 1, 10, 7, 5))
        requests = ((0, 1, 0, lengths[0]), (3, 2, 0, 1), (1, 4, 37, lengths[2]), (4, 5, DEEP + 2, lengths[3]),
                    (2, 3, 5, lengths[4]))
        meta, _ = self.assertLikeSplitK(requests, seed=3)
        self.assertEqual(meta.starts.tolist(), [0, *torch.tensor(lengths).cumsum(0).tolist()])

    def test_rows_that_keep_fewer_blocks_or_none(self):
        """The selection's -1 among a row's ids (a row keeping half of what it could) and rows keeping nothing: those
        attend their tails alone, or nothing at all when their position closes a group."""
        first, second = sized((700, 400), (24, 16))
        requests = ((0, 1, DEEP, first), (1, 2, 3, second))
        rows = first + second
        none = {5, 6, 7, 8, first + 1, first + 2, first + 3, first + 4, rows - 1}
        meta, blocks = self.assertLikeSplitK(requests, seed=4, fewer=set(range(0, rows, 3)) - none, none=none)
        tails = ((meta.positions32 + 1) % W.ratio).cpu()
        self.assertEqual({int(tails[r]) for r in none}, set(range(W.ratio)))      # a tail of every length, 0 included
        self.assertFalse(bool((blocks[sorted(none)] >= 0).any()))
        kept = (blocks >= 0).sum(dim=1).cpu()
        seen = ((meta.positions32 + 1) // W.ratio).clamp(max=BUDGET // W.ratio).cpu()
        self.assertTrue(bool((kept[3:first:3] < seen[3:first:3]).all()))         # fewer than they could

    def test_a_step_s_shared_layout_is_the_same_launch(self):
        """The layout is the step's alone (`tiles`): every QSA layer of a step may share one."""
        from engine.kernels import qsa_tile_union
        gen = generator(5)
        first, second = sized((600, 451), (21, 15))
        requests = ((0, 1, 9, first), (1, 2, DEEP, second))
        meta, q, k_cache, v_cache, gate, blocks = case(gen, requests, W.kv_heads)
        rows = q.shape[0]
        layout = qsa_tile_union.tiles(meta.starts, rows, len(requests))
        tile_row0, tile_request, count = layout
        self.assertEqual(count, -(-rows // qsa_tile_union.TILE.rows) + len(requests))
        used = tile_row0 >= 0
        self.assertEqual(tile_row0[used].tolist(), list(range(0, first, 2)) + list(range(first, rows, 2)))
        self.assertEqual(tile_request[used].tolist(), [0] * -(-first // 2) + [1] * -(-second // 2))
        with served_kernels(), interpreter_gates():
            alone = tile_union(meta, q, k_cache, v_cache, blocks, gate)
            shared = tile_union(meta, q, k_cache, v_cache, blocks, gate, layout=layout)
        self.assertTrue(torch.equal(shared, alone))

    @unittest.skipIf(INTERPRET, "a 1,024-row step at the model's widths: the GB10 lane runs it")
    def test_the_boot_qualification_passes(self):
        """What Qwen3.8's boot runs before it serves the launch (lanes.qualify)."""
        from engine.kernels import qsa_tile_union
        held = qsa_tile_union.qualify(DEVICE, heads=W.heads, head_dim=W.head_dim, ratio=W.ratio, budget=W.budget,
                                      page_size=W.block)
        self.assertLessEqual(held["largest"], qsa_tile_union.BAND[0])
        self.assertLessEqual(held["rms"], qsa_tile_union.BAND[1])

    def test_the_band_has_power(self):
        """One block fewer in each row moves the output far past the band: the comparison above would see a tile
        that lost or leaked a block."""
        from engine.kernels.gated_residual import drift
        gen = generator(6)
        meta, q, k_cache, v_cache, _, blocks = case(gen, ((0, 1, DEEP, sized(1024, 40)),), W.kv_heads)
        dropped = blocks.clone()
        dropped[:, 0] = -1                                  # every row keeps all it could: column 0 holds a block
        self.assertTrue(bool((blocks[:, 0] >= 0).all()))
        with served_kernels(), interpreter_gates():
            want = split_k(meta, q, k_cache, v_cache, blocks)
            lost = tile_union(meta, q, k_cache, v_cache, dropped)
        self.assertGreater(drift(lost, want)[0], 4 * BAND[0])


@unittest.skipUnless(RUNS and INTERPRET, "counts the interpreter's stores: requires TRITON_INTERPRET=1 with Triton")
class StoreOnceTests(unittest.TestCase):
    def test_every_launch_stores_each_address_once_and_every_output(self):
        from tests.test_engine_qwen38_store_once import launches, twice
        gen = generator(7)
        requests = ((0, 1, 0, 13), (3, 2, 0, 1), (1, 4, DEEP, 21))
        for kv_heads in (W.kv_heads, 2):
            meta, q, k_cache, v_cache, gate, blocks = case(gen, requests, kv_heads, fewer={4, 9}, none={20})
            with served_kernels(), interpreter_gates(), launches() as seen:
                out = tile_union(meta, q, k_cache, v_cache, blocks, gate)
            with self.subTest(kv_heads=kv_heads):
                self.assertEqual([name for name, _ in seen], ["_qsa_tile_union_pack_kernel",
                                                              "_qsa_tile_union_build_kernel",
                                                              "_qsa_tile_union_attn_kernel"])
                self.assertEqual({name: twice(addresses) for name, addresses in seen if twice(addresses)}, {})
                self.assertEqual(len(seen[2][1]), out.numel())               # and nothing it owes is left unwritten


@unittest.skipUnless(torch is not None and TRITON, "engine/kernels/qsa_tile_union imports Triton")
class EligibilityTests(unittest.TestCase):
    """`admits` from host integers alone, and `attention` refusing what it does not take before any launch."""

    SHAPE = dict(compress_ratio=4, token_topk=2048, page_size=768, table_width=342, cache_pages=4000)   # Qwen3.8

    def test_the_tile_is_the_sm121_one(self):
        from engine.kernels import qsa_tile_union as tu
        self.assertEqual((tu.TILE.rows, tu.TILE.blocks, tu.TILE.warps, tu.TILE.min_rows, tu.TILE.min_rows_per_request),
                         (2, 8, 4, 1024, 64))
        for bad in (dict(rows=3), dict(blocks=12), dict(blocks=64), dict(warps=3), dict(min_rows=-1)):
            with self.subTest(**bad), self.assertRaises(ValueError):
                tu.Tile(**{**dict(rows=2, blocks=8, warps=4, min_rows=1024), **bad})

    def test_admits_a_prefill_step_by_its_shape(self):
        from engine.kernels import qsa_tile_union as tu
        admits = lambda rows, requests, **kw: tu.admits(rows, requests, **{**self.SHAPE, **kw})
        self.assertTrue(admits(4096, 1))
        self.assertTrue(admits(1024, 1))
        self.assertTrue(admits(1024, 16))                                   # 64 rows a segment
        self.assertFalse(admits(1023, 1))                                   # the split-K launch is faster
        self.assertFalse(admits(1120, 140))                                 # 8 rows a segment: fragmented
        self.assertFalse(admits(8, 4))                                      # a decode step
        self.assertFalse(admits(4096, 0))
        self.assertFalse(admits(4096, 1, compress_ratio=1))                 # nothing to union
        self.assertFalse(admits(4096, 1, compress_ratio=3, token_topk=1536))    # not a power of two
        self.assertFalse(admits(4096, 1, token_topk=2050))
        self.assertFalse(admits(4096, 1, token_topk=0))
        self.assertFalse(admits(4096, 1, page_size=770))                    # a block would straddle a page
        self.assertFalse(admits(4096, 1, table_width=1 << 20))              # block ids reach the sentinel
        self.assertFalse(admits(4096, 1, cache_pages=(1 << 31) // 768 + 1))  # a physical token past int32
        self.assertTrue(admits(4096, 1, token_topk=1280))                    # a budget that is not a power of two

    def test_attention_refuses_before_any_launch(self):
        from engine.kernels import qsa_tile_union as tu
        rows, D = 1024, 16
        q = torch.zeros(rows, 4, D, dtype=torch.bfloat16)
        cache = torch.zeros(8, 16, 1, D, dtype=torch.bfloat16)
        table, req = torch.zeros(1, 8, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32)
        positions = torch.arange(rows, dtype=torch.int32)
        starts = torch.tensor([0, rows], dtype=torch.int32)
        blocks = torch.full((rows, 3), -1, dtype=torch.int32)
        call = lambda **kw: tu.attention(**{**dict(
            q=q, k_cache=cache, v_cache=cache, block_indices=blocks, query_positions=positions, starts=starts,
            compress_ratio=4, token_topk=12, block_table=table, token_to_req=req), **kw})
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            call()
        refusals = (
            (dict(token_topk=10), "divisible by compression ratio"),
            (dict(q=q[:1000], block_indices=blocks[:1000], query_positions=positions[:1000], token_to_req=req[:1000]),
             "does not take this step"),
            (dict(starts=torch.tensor([0, 512, rows], dtype=torch.int32)), "a row offset a segment"),
            (dict(query_positions=positions.long()), "int32"),
            (dict(block_indices=blocks[:, :2]), "compressed top-k"),
            (dict(token_to_req=torch.zeros(1, dtype=torch.int32).expand(rows)), "packed row metadata"),
            (dict(q=q.float()), "BF16"),
            (dict(gate=q.float()), "output gate is BF16"),
            (dict(compress_ratio=3, block_indices=torch.full((rows, 4), -1, dtype=torch.int32)),
             "does not take this step"),
            (dict(layout=(torch.zeros(3, dtype=torch.int32), torch.zeros(3, dtype=torch.int32), 3)), "tile layout"),
        )
        kernels = {name: Unlaunched() for name in ("_qsa_tile_union_pack_kernel", "_qsa_tile_union_build_kernel",
                                                   "_qsa_tile_union_attn_kernel")}
        with mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)), \
                mock.patch.multiple(tu, **kernels):
            for kw, message in refusals:
                with self.subTest(refused=message, args=sorted(kw)), self.assertRaisesRegex(ValueError, message):
                    call(**kw)
        self.assertEqual({name: kernel.grids for name, kernel in kernels.items() if kernel.grids}, {})


class Unlaunched:
    """A stand-in for a jitted kernel that records a launch and runs nothing."""

    def __init__(self):
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(tuple(grid))
        return lambda *args, **meta: None


class ProvenanceTests(unittest.TestCase):
    def test_the_vendored_file_is_recorded(self):
        record = json.loads((ROOT / "engine/kernels/SOURCES.json").read_text())["files"]["qsa_tile_union.py"]
        self.assertEqual(record["sha256"], PR_SOURCE_SHA256)
        self.assertEqual(record["local_sha256"],
                         hashlib.sha256((ROOT / "engine/kernels/qsa_tile_union.py").read_bytes()).hexdigest())
        self.assertTrue(record["local_modifications"])
        self.assertIn("vllm-project/vllm#55430", record["source"])
        notices = (ROOT / "engine/kernels/THIRD_PARTY_NOTICES.md").read_text()
        self.assertIn("qsa_tile_union.py", notices)


class ServedTests(unittest.TestCase):
    """On by the operator's decision of 2026-09-19 (fleet unmeasured): the lane table binds it, the net takes it for a
    prefill step `admits` takes, and a boot can decline it from the launcher to the net."""

    def net(self, **kw):
        from engine.profiles.qwen38.net import Qwen38Net
        stand_in = Qwen38Net.__new__(Qwen38Net)
        stand_in.F = SimpleNamespace(idx_ratio=W.ratio, idx_budget=W.budget)
        stand_in.lanes = SimpleNamespace(qsa_attend_union=kw.pop("lane", object()))
        stand_in.tile_union = kw.pop("tile_union", True)
        return stand_in

    def asks(self, stand_in, rows=1024, segments=1, captured=False):
        meta = SimpleNamespace(page_table=torch.zeros(segments, 40, dtype=torch.int32))
        K = torch.zeros(64, W.block, 1, 1)
        return stand_in._tile_union(SimpleNamespace(captured=captured), meta, K, rows)

    def test_a_prefill_step_admits_takes_is_on_the_union(self):
        self.assertTrue(self.asks(self.net()))

    def test_every_other_step_stays_on_the_split_k_launch(self):
        self.assertFalse(self.asks(self.net(), rows=1023))                         # below the tile's row gate
        self.assertFalse(self.asks(self.net(), rows=1024, segments=17))           # under 64 rows a segment
        self.assertFalse(self.asks(self.net(), captured=True))
        self.assertFalse(self.asks(self.net(tile_union=False)))                   # a boot declined it
        self.assertFalse(self.asks(self.net(lane=None)))                          # a lane table without it

    def test_it_is_on_unless_a_boot_declines(self):
        import inspect
        from engine.profiles.qwen38 import lanes
        from engine.profiles.qwen38.net import Qwen38Net
        self.assertIs(inspect.signature(Qwen38Net.__init__).parameters["tile_union"].default, True)
        self.assertIs(inspect.signature(lanes.qualify).parameters["tile_union"].default, True)
        self.assertIn("qsa_attend_union=on_main(qsa_tile_union.attention)",
                      (ROOT / "engine/profiles/qwen38/lanes.py").read_text(encoding="utf-8"))
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--no-tile-union", action="store_true",', fleet)
        self.assertIn("tile_union=not a.no_tile_union", fleet)
        self.assertIn("qualify(torch.device(\"cuda\"), F, tile_union=not a.no_tile_union)", fleet)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn('case "${ST_QSA_TILE_UNION:-1}" in', launcher)
        self.assertIn('0) UNION_ARG="--no-tile-union" ;;', launcher)
        self.assertIn("$CALIB_ARG $UNION_ARG'", launcher)
        defaults = (ROOT / "engine/SERVING_DEFAULTS.md").read_text(encoding="utf-8")
        self.assertIn("`ST_QSA_TILE_UNION=1`", defaults)

    def test_the_net_refuses_a_choice_that_is_not_a_boolean(self):
        from engine.profiles.qwen38.net import Qwen38Net
        with self.assertRaisesRegex(ValueError, "declared boolean"):
            Qwen38Net(SimpleNamespace(), SimpleNamespace(world_size=4, rank=0), None, tile_union=1)


if __name__ == "__main__":
    unittest.main()
