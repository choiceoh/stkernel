"""The served MoE lane pads a captured launch that is above the micro kernel's cap and below one 16-row tile of the
static kernel (engine/profiles/qwen38/lanes.static_pad): on 2026-09-18 the static kernel faulted at 10 and 12 rows and
passed at 14..32, and 10, 12 and 14 pass through the 16-row launch (measurements/qwen38_moe_static_20260918)."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@unittest.skipUnless(torch is not None, "requires torch (the lanes module imports it)")
class StaticPadTests(unittest.TestCase):
    def test_only_the_rows_between_the_micro_cap_and_one_tile_are_padded(self):
        from engine.profiles.qwen38.lanes import STATIC_TILE_ROWS, static_pad
        self.assertEqual(STATIC_TILE_ROWS, 16)
        self.assertEqual([static_pad(rows) for rows in range(1, 9)], [0] * 8)          # micro
        self.assertEqual({rows: static_pad(rows) for rows in range(9, 16)},
                         {9: 7, 10: 6, 11: 5, 12: 4, 13: 3, 14: 2, 15: 1})            # the eight-row ladder's 10, 12, 14
        self.assertEqual([static_pad(rows) for rows in (16, 20, 24, 28, 32, 64)], [0] * 6)

    def test_the_cap_is_the_dispatchers(self):
        from engine.profiles.qwen38.lanes import static_pad
        self.assertEqual(static_pad(10, micro_cap=12), 0)
        self.assertEqual(static_pad(13, micro_cap=12), 3)

    def test_the_launch_gains_zero_rows_at_weight_zero(self):
        from engine.profiles.qwen38.lanes import pad_static_launch
        x = torch.randn(12, 8)
        ids = torch.arange(12 * 3, dtype=torch.int32).reshape(12, 3)
        w = torch.rand(12, 3)
        px, pids, pw, rows = pad_static_launch(x, ids, w, 5)
        self.assertEqual((rows, px.shape[0], pids.shape[0], pw.shape[0]), (12, 16, 16, 16))
        self.assertTrue(torch.equal(px[:12], x) and torch.equal(pids[:12], ids) and torch.equal(pw[:12], w))
        self.assertEqual((int(px[12:].abs().sum()), float(pw[12:].abs().sum())), (0, 0.0))
        self.assertTrue(bool((pids[12:] == 5).all()))
        self.assertEqual(pids.dtype, torch.int32)
        same = pad_static_launch(x[:8], ids[:8], w[:8], 5)             # the micro kernel's rows: untouched
        self.assertEqual((same[0].shape[0], same[3]), (8, 8))
        self.assertTrue(torch.equal(same[1], ids[:8]))

    def test_both_captured_paths_pad_and_route_local_names_the_padded_launch(self):
        """The served step's MoE is route_local's path (`local=True`): the pad on the global routes' path alone left
        a K=3 capture of three rows (12) on the faulting static kernel (2026-09-19, probes/engine_qwen38_step)."""
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "engine/profiles/qwen38/lanes.py").read_text(encoding="utf-8")
        served = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "served")
        inner = {n.name: ast.unparse(n) for n in ast.walk(served) if isinstance(n, ast.FunctionDef)}
        moe = inner["moe"]
        local = moe[moe.index("if local:"):moe.index("if not compact:")]
        self.assertIn("pad_static_launch(x, ids, weights, 0", local)
        self.assertLess(local.index("pad_static_launch("), local.index("dispatch("))
        self.assertIn("pad_static_launch(x, ids, weights, first_expert", moe[moe.index("if not compact:"):])
        self.assertIn("static_pad(rows, md._MICRO_MAX_TOKENS)", inner["route_local"])


if __name__ == "__main__":
    unittest.main()
