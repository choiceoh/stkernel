"""The experimental split keeps token order and never launches unsupported m=1."""
import unittest

from probes.engine_qwen38_moe_chunks import chunks


class ChunkTests(unittest.TestCase):
    def test_serving_split_is_bounded_and_default_off(self):
        from engine.profiles.qwen38.lanes import moe_decode_ranges
        for rows in range(1, 34):
            self.assertEqual(moe_decode_ranges(rows), ((0, rows),))
            ranges = moe_decode_ranges(rows, True)
            if 8 < rows <= 16:
                self.assertEqual(ranges, chunks(rows, 8))
            else:
                self.assertEqual(ranges, ((0, rows),))

    def test_all_captured_widths_cover_each_token_once(self):
        for rows in range(2, 33):
            for limit in (4, 8):
                with self.subTest(rows=rows, limit=limit):
                    ranges = chunks(rows, limit)
                    self.assertEqual([i for a, b in ranges for i in range(a, b)], list(range(rows)))
                    self.assertTrue(all(2 <= b - a <= limit for a, b in ranges))

    def test_nine_tokens_do_not_leave_a_single_token_launch(self):
        self.assertEqual(chunks(9, 8), ((0, 5), (5, 9)))
        self.assertEqual(chunks(16, 8), ((0, 8), (8, 16)))

    def test_invalid_shapes_fail_before_capture(self):
        for rows, limit in ((0, 8), (1, 8), (3, 2), (16, 1), (16, 9)):
            with self.subTest(rows=rows, limit=limit), self.assertRaises(ValueError):
                chunks(rows, limit)


if __name__ == "__main__":
    unittest.main()
