"""probes/engine_qwen38_qsa_geometry.py on the CPU (engine/QWEN38_CARRY.md Q9, Q13): the case table is Qwen3.8's per-rank
QSA cell and the captured ladder's buckets, the geometry hooks are inert until set, validated, come back unset and reach
the launches they name (stand-in kernel objects record the launch and run nothing), the two K/V layouts hold the same
values at the block's and the records' strides, the grids and verdicts' arithmetic, and -- under TRITON_INTERPRET=1 --
the probe's gates end to end on the real kernels at a small cell.

    docker exec -w <repo> stk-test python3 -m unittest tests.test_probe_qwen38_qsa_geometry
    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_probe_qwen38_qsa_geometry
"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
KERNELS = torch is not None and TRITON


def probe():
    import probes.engine_qwen38_qsa_geometry as module
    return module


@contextmanager
def cuda_view():
    """The launchers' CUDA-only argument checks see CPU tensors as the device's; nothing launches on them here."""
    with patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)):
        yield


class Recorder:
    """A stand-in for a jitted kernel: `kernel[grid](*args, **meta)` records (grid, meta) and runs nothing."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **meta: self.launches.append((tuple(grid), meta))


@unittest.skipUnless(torch is not None, "requires torch")
class CaseTable(unittest.TestCase):
    def test_the_probe_imports_without_a_gpu_or_the_kernel_package(self):
        code = ("import sys, probes.engine_qwen38_qsa_geometry as p; "
                "print(callable(p.run), 'engine.kernels' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["True", "False"])

    def test_the_case_table_is_qwen38_s_per_rank_cell(self):
        from engine.profiles.qwen38 import caches, net
        from probes.engine_qwen38_cells import facts
        p, F = probe(), facts()
        self.assertEqual((p.HEADS, p.KV_HEADS, p.HEAD_DIM, p.ROTARY), (F.heads_local, F.kv_heads_local, F.head_dim,
                                                                       F.rotary_dim))
        self.assertEqual((p.IDX_HEADS, p.IDX_DIM, p.RATIO, p.BUDGET), (F.idx_heads, F.idx_dim, F.idx_ratio, F.idx_budget))
        self.assertEqual((p.BLOCK, p.KEY_RING, p.THETA, p.EPS), (F.block, caches.QSA_KEY_RING, F.rope_theta, F.rms_eps))
        self.assertEqual((p.FIRST_BUCKET, p.MAX_POSITION), (net.FIRST_BUCKET, F.max_position))
        cell = p.QWEN38
        self.assertEqual((cell.width, cell.index_blocks, cell.key_page), (2051, F.index_blocks, F.block // F.idx_ratio))
        self.assertEqual([r * t for r, t in p.DECODE_STEPS], [1, 2, 4, 8, 16, 32])    # a draft row; K=1 rows 1..8, K=3 rows 8
        self.assertTrue(all(t - 1 in (0, 1, 3) for _, t in p.DECODE_STEPS + p.SCORE_STEPS + p.INPUT_STEPS))
        from engine.kernels import qsa
        for rows, bucket in p.SCORE_PREFILL_SHAPES:                                    # one scoring call's rows
            columns = p.bucket_pages(cell, bucket) * cell.key_page
            self.assertLessEqual(rows, qsa._rows_a_scoring_call(columns, 4))
        self.assertEqual(p.COVERED_ROWS // cell.ratio, cell.index_blocks)              # the longest covered prompt
        self.assertGreater(p.ATTEND_CONTEXT // cell.ratio, 4 * cell.index_blocks)      # a real choice among the blocks

    def test_the_buckets_are_the_captured_ladder_s_rungs(self):
        from engine.profiles.qwen38.decode_graphs import bucket_ladder
        p = probe()
        cell = p.QWEN38
        ladder = bucket_ladder(cell.block, 10 ** 6, p.MAX_POSITION, 2)
        self.assertEqual([p.bucket_pages(cell, b) for b in (4096, 8192, 16384, 32768, 65536, 131072)], ladder[:6])
        self.assertTrue(set(p.bucket_pages(cell, b) for b in p.SCORE_BUCKETS) <= set(ladder) | {ladder[-1] - 1})
        # the 131K bucket's columns sit just past the decode selection's cut today: what the select arm is for
        from engine.kernels import qsa_select
        self.assertEqual(p.bucket_pages(cell, 131072) * cell.key_page, 32832)
        self.assertGreater(32832, qsa_select.WIDEST)

    def test_the_pages_turn_under_a_static_table(self):
        p = probe()
        cell = p.Cell(heads=6, kv_heads=1, head_dim=32, rotary=8, idx_heads=4, idx_dim=16, ratio=4, budget=48, block=16)
        step, pages = p.step_of(cell, 3, 2, 70, torch.device("cpu"), sets=3)      # contexts 70, 75, 80: six blocks
        self.assertEqual((step.rows, len(step.tables), pages), (6, 3, 3 * 3 * 6))
        self.assertEqual(step.meta.positions.tolist(), [70, 71, 75, 76, 80, 81])
        self.assertEqual(step.meta.lengths.tolist(), [72, 77, 82])
        held = step.meta.page_table
        used = [set(t[t >= 0].tolist()) for t in step.tables]
        self.assertTrue(all(len(u) == 18 for u in used) and not (used[0] & used[1] or used[1] & used[2]
                                                                 or used[0] & used[2]))
        step.turn(1)
        self.assertIs(step.meta.page_table, held)                                       # a captured graph's tensor
        self.assertTrue(torch.equal(held, step.tables[1]))
        wide, _ = p.step_of(cell, 1, 2, 70, torch.device("cpu"), sets=1, table_blocks=p.bucket_pages(cell, 4096))
        self.assertEqual(wide.meta.page_table.shape, (1, 256))                          # the bucket's whole width

    def test_a_prefill_selection_shares_most_of_a_tile_s_blocks(self):
        """The tile-union lane's selections: each row's distinct blocks among those it sees, as many as the budget
        keeps; the prefill-like one shares more of a tile's blocks than independent subsets do."""
        p = probe()
        cell = p.Cell(heads=6, kv_heads=1, head_dim=32, rotary=8, idx_heads=4, idx_dim=16, ratio=4, budget=48, block=16)
        cpu = torch.device("cpu")
        step, _ = p.step_of(cell, 1, 64, 300, cpu, sets=1)
        seen = (step.meta.positions32 + 1) // cell.ratio
        shapes = {}
        for selection, noise in p.TILE_UNION_SELECTIONS.items():
            generator = torch.Generator().manual_seed(15)
            blocks = (p.chosen_blocks(cell, step, generator, cpu) if noise is None
                      else p.overlapping_blocks(cell, step, generator, cpu, noise))
            for row in range(step.rows):
                ids = blocks[row][blocks[row] >= 0]
                with self.subTest(selection=selection, row=row):
                    self.assertEqual(len(set(ids.tolist())), min(int(seen[row]), cell.index_blocks))
                    self.assertTrue(bool((ids < seen[row]).all()))
            shapes[selection] = p.neighbours(blocks, step.meta.starts, 2)
        self.assertGreater(shapes["prefill"]["jaccard"], shapes["independent"]["jaccard"])
        self.assertLess(shapes["prefill"]["union_blocks"], shapes["independent"]["union_blocks"])

    def test_the_two_layouts_hold_the_same_values_at_their_strides(self):
        p = probe()
        cell = p.Cell(heads=6, kv_heads=1, head_dim=32, rotary=8, idx_heads=4, idx_dim=16, ratio=4, budget=48, block=16)
        cpu = torch.device("cpu")
        (bk, bv), (rk, rv) = (p.kv_pair(torch.Generator().manual_seed(5), 3, cell, cpu, layout) for layout in p.LAYOUTS)
        self.assertTrue(torch.equal(bk, rk) and torch.equal(bv, rv) and not torch.equal(bk, bv))
        row = cell.kv_heads * cell.head_dim
        # today's block (caches.Qwen38Caches): the K rows, then the V rows; a page's stride is both
        self.assertEqual((bk.stride(), bv.stride()), ((2 * 16 * row, row, row, 1),) * 2)
        self.assertEqual((bv.data_ptr() - bk.data_ptr()) // 2, 16 * row)
        # one region of records: a position's key, then its value
        self.assertEqual((rk.stride(), rv.stride()), ((2 * 16 * row, 2 * row, row, 1),) * 2)
        self.assertEqual((rv.data_ptr() - rk.data_ptr()) // 2, row)
        with self.assertRaisesRegex(ValueError, "layout"):
            p.kv_pair(torch.Generator().manual_seed(5), 3, cell, cpu, "rows")
        cases = p.layout_cases(cell, 2, 2, 150, cpu, 2)
        self.assertEqual(tuple(cases), p.LAYOUTS)
        self.assertTrue(torch.equal(cases["block"].gate, cases["records"].gate))
        self.assertEqual([r * t for r, t in p.RECORD_STEPS], [2, 8, 32])

    def test_the_served_block_is_the_block_layout(self):
        """caches.Qwen38Caches puts a layer's V rows a whole block of K rows after its K: what "block" stands for."""
        source = (ROOT / "engine" / "profiles" / "qwen38" / "caches.py").read_text()
        self.assertIn("bf.storage_offset() + (offset + F.block * kv_row) // 2)", source)
        self.assertIn("paged += 2 * F.block * kv_row", source)

    def test_the_rule_s_profiles_are_the_record_s(self):
        """What the sweep calls today's rule is the table its own record set (qsa._split_profile): every tier is
        reached by a step of the ladder or an eager arm, so a later run re-judges each against the grid."""
        p = probe()
        self.assertEqual([p.rule_profile(p.QWEN38, r * t) for r, t in p.DECODE_STEPS],
                         [(16, 64, 4), (16, 64, 4), (16, 16, 4), (16, 16, 4), (16, 4, 4), (16, 4, 4)])
        self.assertEqual([p.rule_profile(p.QWEN38, rows) for rows in p.MID_ROWS],
                         [(16, 4, 4), (16, 4, 4), (16, 1, 4), (16, 1, 4)])
        self.assertEqual(p.rule_profile(p.QWEN38, p.PREFILL_ROWS), (16, 1, 4))
        self.assertEqual(p.rule_profile(p.QWEN38, p.COVERED_ROWS), (16, 1, 4))

    def test_a_split_grid_clips_to_what_a_width_s_tiles_can_use(self):
        p = probe()
        cell = p.QWEN38                                                     # 2,051 columns: 129, 65 and 33 tiles
        grid = p.split_grid(cell, (16, 32, 64), (1, 4, 16, 64), (2,), rules=[(16, 64, 4), (64, 8, 2), (16, 64, 2)])
        self.assertEqual(grid, [(16, 1, 2), (16, 4, 2), (16, 16, 2), (16, 64, 2), (32, 1, 2), (32, 4, 2), (32, 16, 2),
                                (32, 64, 2), (64, 1, 2), (64, 4, 2), (64, 16, 2), (64, 32, 2), (16, 64, 4), (64, 8, 2)])
        self.assertEqual(len(set(grid)), len(grid))
        full = p.split_grid(cell, p.ATTEND_TILES, p.ATTEND_SPLITS, p.ATTEND_WARPS, [p.WIDE_TILE])
        self.assertEqual(len(full), (7 + 7 + 6) * 3 + 1)                  # 64-wide tiles: 33 of them, 32 splits at most
        self.assertTrue(all(n <= 64 for n, _, _ in full[:-1]) and full[-1][0] == 128)

    def test_bf16_steps_count_adjacent_values_across_zero(self):
        p = probe()
        bits = [0, 1, 0x8001 - 65536, -32768, 0x3F80, 0x3F81, 0xBF80 - 65536]    # +0, the next value, its mirror, -0, 1, ...
        tiny = torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)
        zero, up, down, minus_zero, one, above_one, minus_one = tiny
        self.assertEqual(p.bf16_steps(torch.stack([zero, up, one]), torch.stack([minus_zero, down, above_one])), (2, 2))
        self.assertEqual(p.bf16_steps(one[None], one[None]), (0, 0))
        self.assertEqual(p.bf16_steps(torch.stack([one, zero]), torch.stack([above_one, zero])), (1, 1))
        self.assertEqual(p.bf16_steps(one[None], minus_one[None])[0], 2 * 0x3F80)
        self.assertEqual(p.bf16_steps(tiny[:0], tiny[:0]), (0, 0))

    def test_a_verdict_names_the_fastest_passing_geometry(self):
        p = probe()
        rule, fast, inexact = (16, 64, 4), (64, 1, 2), (32, 4, 2)
        timings = {rule: dict(us=50.0), fast: dict(us=40.0), inexact: dict(us=30.0)}
        got = p.verdict(rule, {rule: True, fast: True, inexact: False}, timings, "us")
        self.assertEqual(got, dict(rule=[16, 64, 4], passing=2, failing=[[32, 4, 2]], fastest=[64, 1, 2],
                                   fastest_us=40.0, rule_us=50.0, fastest_over_rule=0.8, fastest_launched=[32, 4, 2],
                                   fastest_launched_us=30.0))
        tie = p.verdict(rule, {rule: True, fast: True}, {rule: dict(us=40.0), fast: dict(us=40.0)}, "us")
        self.assertEqual((tie["fastest"], tie["fastest_over_rule"]), ([16, 64, 4], 1.0))       # a tie keeps the rule
        self.assertNotIn("fastest_launched", tie)
        self.assertEqual(p.verdict(rule, {rule: False}, {}, "us"), dict(rule=[16, 64, 4], passing=0,
                                                                        failing=[[16, 64, 4]]))


@unittest.skipUnless(KERNELS, "requires torch and triton")
class HookTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels import qsa, qsa_select
        self.qsa, self.select = qsa, qsa_select

    def unset(self):
        p = probe()
        return ([getattr(self.qsa, name) for name in p.HOOKS] + [getattr(self.select, name) for name in p.SELECT_HOOKS])

    def test_the_hooks_start_unset_and_the_rules_stand(self):
        self.assertEqual(self.unset(), [None] * 5)
        self.assertEqual([self.qsa._score_profile(rows) for rows in (1, 32, 33, 4096)],
                         [(64, 1, 2), (64, 1, 2), (128, 32, 4), (128, 32, 4)])       # the record's prefill geometry
        self.assertEqual(self.qsa._input_warps(), 4)
        self.assertEqual(self.qsa._split_profile(2, 1, 8, 2051), (16, 129, 64, 4))

    def test_forced_takes_validates_and_restores(self):
        p, qsa = probe(), self.qsa
        with p.forced(_SPLIT_PROFILE_OVERRIDE=(64, 4, 8), _SCORE_PROFILE_OVERRIDE=(128, 4, 1),
                      _INPUT_WARPS_OVERRIDE=(2,)):
            self.assertEqual(qsa._split_profile(2, 1, 8, 2051), (64, 33, 4, 8))
            self.assertEqual(qsa._split_profile(4096, 1, 8, 2051), (64, 33, 4, 8))      # whatever the rows
            self.assertEqual(qsa._split_profile(2, 1, 8, 15), (64, 1, 1, 8))            # one tile cannot split
            self.assertEqual(qsa._score_profile(2), (128, 4, 1))
            self.assertEqual(qsa._input_warps(), 2)
        self.assertEqual(self.unset(), [None] * 5)
        with p.forced(self.select, _WIDEST_OVERRIDE=0, _WARPS_OVERRIDE=(16,)):
            self.assertEqual((self.select._WIDEST_OVERRIDE, self.select._WARPS_OVERRIDE), (0, (16,)))
        self.assertEqual(self.unset(), [None] * 5)
        bad_profiles = ((8, 4, 4), (48, 4, 4), (16, 3, 4), (16, 4, 0), (16, 4), [16, 4, 4], (16.0, 4, 4), (True, 4, 4))
        for bad in bad_profiles:
            with self.subTest(split=bad), p.forced(_SPLIT_PROFILE_OVERRIDE=bad), \
                    self.assertRaisesRegex(ValueError, "_SPLIT_PROFILE_OVERRIDE"):
                qsa._split_profile(2, 1, 8, 2051)
            with self.subTest(score=bad), p.forced(_SCORE_PROFILE_OVERRIDE=bad), \
                    self.assertRaisesRegex(ValueError, "_SCORE_PROFILE_OVERRIDE"):
                qsa._score_profile(2)
        for bad in (4, (3,), (0,), (4, 4), (4.0,)):
            with self.subTest(warps=bad), p.forced(_INPUT_WARPS_OVERRIDE=bad), \
                    self.assertRaisesRegex(ValueError, "_INPUT_WARPS_OVERRIDE"):
                qsa._input_warps()
        with self.assertRaises(RuntimeError):
            with p.forced(_SPLIT_PROFILE_OVERRIDE=(64, 4, 8)):
                raise RuntimeError("the block dies")
        self.assertEqual(self.unset(), [None] * 5)
        for names in (dict(_NOT_A_HOOK_OVERRIDE=1), dict(_LOGITS_WORKSPACE_BYTES=1)):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "not a probe hook"):
                with p.forced(**names):
                    pass

    def launch(self, case, gated=False):
        with cuda_view():
            return case.launch(gated) if hasattr(case, "blocks") else case.launch()

    def small(self):
        p = probe()
        return p, p.Cell(heads=6, kv_heads=1, head_dim=32, rotary=8, idx_heads=4, idx_dim=16, ratio=4, budget=48,
                         block=16, ring=8)

    def test_a_forced_split_profile_reaches_both_attention_launches(self):
        p, cell = self.small()
        generator = torch.Generator().manual_seed(3)
        cpu = torch.device("cpu")
        for covered, kernel in ((False, "_qsa_sparse_paged_gqa_splitk_kernel"), (True, "_qsa_covered_paged_gqa_kernel")):
            case = p.attention_case(cell, 1, 4, 0 if covered else 90, cpu, generator, covered=covered)
            for arm, want in (((16, 4, 8), (16, 4, 4, 8)), ((32, 1, 2), (32, 2, 1, 2)), (None, (16, 4, 4, 4))):
                split, merge = Recorder(), Recorder()
                with self.subTest(covered=covered, arm=arm), patch.object(self.qsa, kernel, split), \
                        patch.object(self.qsa, "_qsa_merge_splitk_kernel", merge), p.forced(_SPLIT_PROFILE_OVERRIDE=arm):
                    self.launch(case)
                    (grid, meta), = split.launches
                    self.assertEqual((meta["BLOCK_N"], meta["NUM_TILES"], meta["NUM_SPLITS"], meta["num_warps"]), want)
                    self.assertEqual(grid[2], want[2])
                    self.assertEqual(len(merge.launches), int(want[2] > 1))

    def test_a_forced_score_profile_reaches_the_row_and_the_run_kernel(self):
        p, cell = self.small()
        generator = torch.Generator().manual_seed(4)
        case = p.scoring_case(cell, 2, 2, 4096, torch.device("cpu"), generator)
        columns = case.step.meta.page_table.shape[1] * cell.key_page
        for group, kernel in ((1, "_qsa_mqa_paged_kernel"), (2, "_qsa_mqa_paged_group_kernel")):
            case.group = group
            for arm, want in (((128, 4, 1), (128, 4, 1)), (None, (64, 1, 2))):
                launched = Recorder()
                with self.subTest(group=group, arm=arm), patch.object(self.qsa, kernel, launched), \
                        p.forced(_SCORE_PROFILE_OVERRIDE=arm):
                    self.launch(case)
                    (grid, meta), = launched.launches
                    self.assertEqual((meta["BLOCK_N"], meta["TILES_PER_PROG"], meta["num_warps"]), want)
                    self.assertEqual(grid, (4 // group, -(-columns // (want[0] * want[1]))))

    def test_the_forced_warps_reach_the_input_launches_and_the_selection(self):
        p, _ = self.small()
        source = (ROOT / "engine" / "kernels" / "qsa.py").read_text()
        self.assertEqual(source.count("num_warps=_input_warps()"), 2)                  # qsa_index_keys and qsa_inputs
        logits = torch.randn(2, 100)
        visible = torch.tensor([100, 40], dtype=torch.int32)
        out = torch.empty(2, 8, dtype=torch.int32)
        for arm, want in (((16,), 16), (None, 4)):
            launched = Recorder()
            with self.subTest(arm=arm), patch.object(self.select, "_select_rows", launched), cuda_view(), \
                    p.forced(self.select, _WARPS_OVERRIDE=arm):
                self.select.select(logits, visible, 8, out)
                self.assertEqual(launched.launches[0][1]["num_warps"], want)
        for bad in (8, (3,), (8, 8)):
            with self.subTest(bad=bad), cuda_view(), p.forced(self.select, _WARPS_OVERRIDE=bad), \
                    self.assertRaisesRegex(ValueError, "_WARPS_OVERRIDE"):
                self.select.select(logits, visible, 8, out)
        with cuda_view():
            self.assertTrue(self.select.admits(logits, 8))
            with p.forced(self.select, _WIDEST_OVERRIDE=0):
                self.assertFalse(self.select.admits(logits, 8))                         # the torch form's turn
            wide = torch.empty(1, self.select.WIDEST + 1)
            self.assertFalse(self.select.admits(wide, 8))
            with p.forced(self.select, _WIDEST_OVERRIDE=1 << 30):
                self.assertTrue(self.select.admits(wide, 8))


@unittest.skipUnless(KERNELS and INTERPRET, "requires TRITON_INTERPRET=1 with Triton")
class GateTests(unittest.TestCase):
    """The probe's gates on the real kernels, interpreted, at a cell small enough for it: four 16-wide tiles."""

    def setUp(self):
        from tests.test_engine_qwen38_kernels import served_kernels
        self.p = probe()
        self.cell = self.p.Cell(heads=6, kv_heads=1, head_dim=32, rotary=8, idx_heads=4, idx_dim=16, ratio=4,
                                budget=48, block=16, ring=8)
        self.served = served_kernels
        self.cpu = torch.device("cpu")

    def test_the_attention_gate_holds_every_split_of_the_sparse_launch(self):
        p, cell = self.p, self.cell
        case = p.attention_case(cell, 2, 2, 150, self.cpu, torch.Generator().manual_seed(11), sets=2)
        rule = (16, 4, 4)
        with self.served():
            self.assertEqual(p.rule_profile(cell, 4), rule)
            rows = p.attention_gate(case, [(16, 1, 4), (16, 2, 2), (32, 2, 1), rule], rule)
        self.assertEqual(set(rows), {(16, 1, 4), (16, 2, 2), (32, 2, 1), rule})
        for arm, row in rows.items():
            with self.subTest(arm=arm):
                self.assertTrue(row["passed"], row)
                self.assertEqual((row["gated_steps"], row["gated_differ"]), (0, 0))     # the interpreter's exp is numpy's
                self.assertNotIn("alike", row)
        self.assertEqual(rows[rule]["from_rule"], [0.0, 0.0])
        self.assertTrue(all(0 <= x <= 2 ** -6 for row in rows.values() for x in row["from_rule"]))

    def test_the_covered_launch_holds_the_sparse_launch_s_bytes_at_a_forced_profile(self):
        p, cell = self.p, self.cell
        case = p.attention_case(cell, 1, 40, 0, self.cpu, torch.Generator().manual_seed(12), covered=True, sets=1)
        self.assertEqual((case.blocks, case.group), (None, 4))
        rule = p.rule_profile(cell, 40)
        with self.served():
            rows = p.attention_gate(case, [(16, 1, 4), (32, 2, 2)], rule, sample=torch.tensor([0, 3, 17, 39]))
        for arm, row in rows.items():
            with self.subTest(arm=arm):
                self.assertTrue(row["passed"] and row["alike"] and row["gated_steps"] <= 1, row)

    def test_the_records_layout_is_the_same_launch_and_the_same_bytes(self):
        p, cell = self.p, self.cell
        cases = p.layout_cases(cell, 2, 2, 150, self.cpu, 2)
        with self.served():
            block, records = (cases[layout].launch() for layout in p.LAYOUTS)
        self.assertTrue(torch.equal(block, records))
        self.assertTrue(bool(block.any()))

    def test_the_tile_union_gate_holds_its_band_against_the_split_k_launch(self):
        import dataclasses
        from engine.kernels import qsa_tile_union
        p, cell = self.p, self.cell
        # the interpreter's rows: the tile's two row gates lowered, nothing else of it (tests/test_engine_qsa_tile_union)
        lowered = dataclasses.replace(qsa_tile_union.TILE, min_rows=0, min_rows_per_request=0)
        for selection in p.TILE_UNION_SELECTIONS:
            case = p.tile_union_case(cell, 24, 150, self.cpu, torch.Generator().manual_seed(14), selection)
            with self.served(), patch.object(qsa_tile_union, "TILE", lowered):
                row = p.tile_union_gate(case)
            with self.subTest(selection=selection):
                self.assertTrue(row["passed"], row)
                self.assertLessEqual(row["max_abs"], 2 ** -5)

    def test_the_scoring_gate_asks_for_the_rule_s_bytes(self):
        p, cell = self.p, self.cell
        case = p.scoring_case(cell, 2, 2, 4096, self.cpu, torch.Generator().manual_seed(13), sets=2)
        rule = (64, 1, 2)
        with self.served():
            rows = p.scoring_gate(case, [(16, 4, 1), (128, 1, 2), rule], rule)
        for arm, row in rows.items():
            with self.subTest(arm=arm):
                self.assertTrue(row["passed"] and row["launched"], row)
                self.assertEqual(row["largest_difference"], 0.0)


class LaneRoutingTests(unittest.TestCase):
    def test_kernel_check_routes_the_lane_to_the_probe(self):
        text = (ROOT / "probes" / "engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'qwen38_qsa_geometry'", text)
        self.assertIn("from probes.engine_qwen38_qsa_geometry import run as qwen38_qsa_geometry", text)
        self.assertIn("args.lanes == 'qwen38_tile_union'", text)
        self.assertIn("from probes.engine_qwen38_qsa_geometry import run_tile_union as qwen38_tile_union", text)

    def test_the_queue_admits_the_probe(self):
        self.assertIn("'probes/engine_qwen38_qsa_geometry.py'", (ROOT / "bench" / "fleet_onepass.py").read_text())


if __name__ == "__main__":
    unittest.main()
