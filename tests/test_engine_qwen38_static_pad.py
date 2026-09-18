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


if __name__ == "__main__":
    unittest.main()
