"""Configured launch metadata remains separate from EP execution evidence."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import proof


class OnepassParallelismTests(unittest.TestCase):
    def test_ep_marker_requires_the_actual_full_token_launch_line(self):
        knob = "VLLM_GLM53_EP_PREFILL_LOCAL"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "head.log"
            log.write_text("[prefill-sp] sequence-parallel prefill armed\n")
            self.assertFalse(proof.check([knob], str(log))["proof"][knob])
            log.write_text("[ep-prefill-local] LAUNCHED full-token E72/I2048/top8 T=8192\n")
            self.assertTrue(proof.check([knob], str(log))["proof"][knob])


if __name__ == "__main__":
    unittest.main()
