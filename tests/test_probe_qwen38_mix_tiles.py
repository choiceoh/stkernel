"""gated_residual.mix_mean's tile axis and probes/engine_qwen38_mix_tiles.py (engine/QWEN38_CARRY.md H3): the rule is the
GB10's (256-wide tiles at 4 warps up to 32 rows, one block a row above), the hook is inert until set, validated and
reaches the launch's grid, every tile width is the one-block launch's bytes (CUDA, or TRITON_INTERPRET=1), and the
probe's table is the model's.

    docker exec -w <repo> stk-test python3 -m unittest tests.test_probe_qwen38_mix_tiles
    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_probe_qwen38_mix_tiles
"""
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
RUNS = KERNELS and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET or torch is None or not torch.cuda.is_available() else "cuda"


def probe():
    import probes.engine_qwen38_mix_tiles as module
    return module


class Recorder:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **meta: self.launches.append((tuple(grid), meta))


@unittest.skipUnless(torch is not None, "requires torch")
class TableTests(unittest.TestCase):
    def test_the_probe_imports_without_a_gpu_or_the_kernel_package(self):
        code = "import sys, probes.engine_qwen38_mix_tiles as p; print(callable(p.run), 'engine.kernels' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["True", "False"])

    def test_the_table_is_the_model_s(self):
        from probes.engine_qwen38_cells import facts
        p, F = probe(), facts()
        self.assertEqual((p.HIDDEN, p.HC), (F.hidden, F.hc))
        self.assertTrue(all(tile >= 16 and not tile & (tile - 1) for tile in p.TILES))
        self.assertIn(512, p.TILES)                                         # 2,560 = 5 x 512: no masked lane
        self.assertEqual(p.HIDDEN % 512, 0)
        self.assertTrue(min(p.CAPTURED_ROWS) <= 16 < sorted(p.CAPTURED_ROWS)[1])   # under --hc-fp8, and past the fold
        self.assertEqual(p.site_count(4), p.SITES)
        self.assertEqual(p.site_count(4096), 2)                             # 189 MiB a site
        self.assertGreaterEqual(p.MEMORY_CAP_GIB * 2 ** 30, 2 * p.SITE_BYTES)

    def test_kernel_check_routes_the_lane_and_the_queue_admits_it(self):
        text = (ROOT / "probes" / "engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'qwen38_mix_tiles'", text)
        self.assertIn("from probes.engine_qwen38_mix_tiles import run as qwen38_mix_tiles", text)
        self.assertIn("'probes/engine_qwen38_mix_tiles.py'", (ROOT / "bench" / "fleet_onepass.py").read_text())


@unittest.skipUnless(KERNELS, "requires torch and triton")
class HookTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels import gated_residual
        self.hcr = gated_residual

    def test_the_rule_is_the_gb10_s(self):
        """Up to MIX_TILE_ROWS rows 256-wide tiles at 4 warps (never wider than the row), above them one block a row as
        before (measurements/qwen38_mix_tiles_20260919)."""
        hcr = self.hcr
        self.assertIsNone(hcr._MIX_TILE_OVERRIDE)
        self.assertEqual(hcr.MIX_TILE_ROWS, 32)
        self.assertEqual([hcr._mix_tile(2560, rows) for rows in (1, 4, 17, 32, 33, 64, 4096)],
                         [(256, 4)] * 4 + [(4096, 8)] * 3)
        self.assertEqual([hcr._mix_tile(hid, 4) for hid in (16, 100, 1024)], [(16, 4), (128, 4), (256, 4)])
        self.assertEqual([hcr._mix_tile(hid, 64) for hid in (16, 1024, 4096)], [(16, 4), (1024, 4), (4096, 8)])
        p = probe()
        self.assertEqual((p.rule_tile(), p.rule_tile(2560, 64)), ((256, 4), (4096, 8)))
        self.assertIn((256, 4), p.arms(2560, 4))
        self.assertIn((4096, 8), p.arms(2560, 64))

    def test_forced_takes_validates_and_restores(self):
        p, hcr = probe(), self.hcr
        with p.forced((512, 2)):
            self.assertEqual((hcr._mix_tile(2560, 4), hcr._mix_tile(2560, 4096)), ((512, 2), (512, 2)))
        self.assertIsNone(hcr._MIX_TILE_OVERRIDE)
        for bad in ((8, 4), (500, 4), (512, 3), (512,), [512, 4], (512.0, 4), (True, 4)):
            with self.subTest(bad=bad), p.forced(bad), self.assertRaisesRegex(ValueError, "_MIX_TILE_OVERRIDE"):
                hcr._mix_tile(2560, 4)
        with self.assertRaises(RuntimeError):
            with p.forced((512, 2)):
                raise RuntimeError("the block dies")
        self.assertIsNone(hcr._MIX_TILE_OVERRIDE)

    def test_the_tile_reaches_mix_s_launch(self):
        p, hcr = probe(), self.hcr
        hid, hc, rank = 2560, 4, 8                                          # past the folded rows: mix launches mix_mean
        down, up = torch.zeros(rank + hc, hc * hid, dtype=torch.bfloat16), torch.zeros(hc * hid, rank,
                                                                                         dtype=torch.bfloat16)
        for arm, rows, grid, block, warps in (((512, 2), 20, (20, 5), 512, 2), (None, 20, (20, 10), 256, 4),
                                              (None, 40, (40, 1), 4096, 8)):
            normed = torch.zeros(rows, hc * hid, dtype=torch.bfloat16)
            mean, gates = Recorder(), Recorder()
            with self.subTest(arm=arm, rows=rows), patch.object(hcr, "_mix_mean", mean), patch.object(hcr, "_gates", gates), \
                    patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)), \
                    patch.object(hcr, "folds", lambda *a: False), p.forced(arm):
                hcr.mix(normed, down, up, hc)
                (launched, meta), = mean.launches
                self.assertEqual((launched, meta["BD"], meta["num_warps"]), (grid, block, warps))


@unittest.skipUnless(RUNS, "requires CUDA and Triton, or TRITON_INTERPRET=1 with Triton")
class BytesTests(unittest.TestCase):
    def test_every_tile_is_the_one_block_launch_s_bytes(self):
        p = probe()
        hidden, hc = (80, 4) if INTERPRET else (2560, 4)                    # 80 = 5 x 16: whole tiles, and a masked one
        gen = torch.Generator().manual_seed(7)
        for rows in (1, 3, 19):
            inputs = p.site_inputs(rows, hidden, hc, torch.device(DEVICE), gen, sites=2)
            grid = [(16, 1), (32, 2), (64, 4), (128, 4)] if INTERPRET else p.arms(hidden, rows)
            rule = p.rule_tile(hidden, rows)
            gates = p.gate(inputs, hidden, hc, grid + [rule])
            with self.subTest(rows=rows):
                self.assertTrue(all(row["exact"] for row in gates.values()), gates)
                self.assertTrue(bool(inputs[0][2].any()))

    def test_a_verdict_names_the_fastest_exact_tile(self):
        p = probe()
        rule, fast, inexact = (4096, 8), (512, 4), (256, 1)
        gates = {rule: dict(exact=True), fast: dict(exact=True), inexact: dict(exact=False)}
        timed = {rule: dict(median_us=10.0), fast: dict(median_us=7.0)}
        self.assertEqual(p.verdict(rule, gates, timed), dict(rule=[4096, 8], rule_us=10.0, fastest=[512, 4],
                                                             fastest_us=7.0, fastest_over_rule=0.7, inexact=[[256, 1]]))


if __name__ == "__main__":
    unittest.main()
