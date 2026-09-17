"""The draw probe's host half: the addressing it checks and the bit-for-bit rule it applies.

The device half needs CUDA and runs on the probe's lane; what can be held on a CPU is
that the host words are addressed the way the engine addresses them, that the rule is
bit-for-bit rather than approximate, and that no two (purpose, position) words collide.
"""
import importlib.util
from pathlib import Path
import struct
import unittest

from engine.base import draws

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("engine_draw_contract", ROOT / "probes/engine_draw_contract.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class HostWordTests(unittest.TestCase):
    def test_the_host_words_are_addressed_by_row_purpose_and_position(self):
        nonces, generations, k = [0, 5], [12, 13], 3
        words = probe.host_words(7, nonces, generations, k)
        self.assertEqual(len(words), 2 * len(probe.PURPOSES) * (k + 1))
        for row, (nonce, generation) in enumerate(zip(nonces, generations)):
            key = draws.row_key(7, nonce, generation)
            for purpose in probe.PURPOSES:
                for position in range(k + 1):
                    self.assertEqual(words[(row, purpose, position)], draws.uniform(key, purpose, position))

    def test_two_rows_with_the_same_counters_still_draw_different_words(self):
        """A row's key carries its nonce: identical generations are not identical draws."""
        words = probe.host_words(7, [0, 1], [12, 12], 2)
        self.assertNotEqual(words[(0, draws.VERIFY, 0)], words[(1, draws.VERIFY, 0)])

    def test_the_five_purposes_share_no_word_at_any_position(self):
        words = {draws.word(purpose, position) for purpose in probe.PURPOSES for position in range(8)}
        self.assertEqual(len(words), len(probe.PURPOSES) * 8)


class BitForBitTests(unittest.TestCase):
    def test_a_last_bit_difference_fails(self):
        want = {("a",): 0.5}
        one_ulp_up = struct.unpack("f", struct.pack("I", struct.unpack("I", struct.pack("f", 0.5))[0] + 1))[0]
        with self.assertRaises(AssertionError):
            with self.assertLogs(level="CRITICAL"):                     # keep the print out of the report
                if not probe.compare("bit", {("a",): one_ulp_up}, want):
                    raise AssertionError("a one-ulp difference was accepted")
        self.assertTrue(probe.compare("bit", {("a",): 0.5}, want))

    def test_a_missing_key_is_not_a_pass(self):
        with self.assertRaises(KeyError):
            probe.compare("bit", {}, {("a",): 0.5})


if __name__ == "__main__":
    unittest.main()
