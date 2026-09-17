"""The margin tool decides where two arms diverge and how tight that choice was.

The engine is deterministic (a repeat of one arm is bit-identical over 90 prefill
stages), so a text difference between arms is caused by whatever changed between
them. Whether it *flips* the answer is a margin question, and these tests hold the
tool's arithmetic: the margin, the chosen probability, which tensor of a capture is
the logit row, and which shared rows flipped with how much room.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("incident_logit_margin", ROOT / "tools/incident_logit_margin.py")
margin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(margin)


def capture(directory, admission, generation, logits, extra=True):
    """A capture in the shape the engine writes: the logit row plus smaller side tensors."""
    path = Path(directory) / f"rank0-admit{admission}-gen{generation}-ctx0.pt"
    payload = {"logits": logits}
    if extra:
        payload["small"] = torch.zeros(4)                      # always narrower than the row
    torch.save(payload, path)
    return path


def logits_row(*values):
    padding = [0.0] * (8 - len(values))
    return torch.tensor([list(values) + padding])


class MarginTests(unittest.TestCase):
    def test_the_margin_is_the_gap_to_the_runner_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            logits = logits_row(0.0, 3.0, 1.5, -2.0)
            path = capture(tmp, 1, 0, logits)
            row = margin.read_capture(path)
        self.assertEqual((row["top1"], row["top2"]), (1, 2))
        self.assertAlmostEqual(row["margin"], 1.5, places=6)
        self.assertAlmostEqual(row["chosen_probability"], float(torch.softmax(logits, dim=-1)[0, 1]), places=6)
        self.assertEqual(row["width"], 8)

    def test_the_logit_row_is_the_largest_tensor_not_the_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank0-admit2-gen7-ctx0.pt"
            torch.save({"small": torch.zeros(16), "logits": torch.arange(1000.0)}, path)
            row = margin.read_capture(path)
        self.assertEqual((row["admission"], row["generation"], row["width"]), (2, 7, 1000))

    def test_a_file_without_the_name_pattern_or_a_readable_row_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.save({"logits": torch.zeros(2000)}, Path(tmp) / "not-a-capture.pt")
            torch.save("not a tensor", Path(tmp) / "rank0-admit3-gen1-ctx0.pt")
            self.assertEqual(margin.profile(Path(tmp)), {})

    def test_a_flip_is_reported_with_its_tightest_margin(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            capture(left_dir, 1, 0, logits_row(5.0, 4.5))                        # top1 = 0, margin 0.5
            capture(right_dir, 1, 0, logits_row(4.6, 5.0))                       # top1 = 1, margin 0.4
            capture(left_dir, 1, 1, logits_row(9.0, 0.0))                        # same top1 both sides
            capture(right_dir, 1, 1, logits_row(8.0, 0.0))
            found = margin.flips(margin.profile(Path(left_dir)), margin.profile(Path(right_dir)))
        self.assertEqual(len(found), 1)
        row = found[0]
        self.assertEqual((row["admission"], row["generation"]), (1, 0))
        self.assertAlmostEqual(row["tightest_margin"], 0.4, places=6)
        self.assertTrue(row["tight"])

    def test_a_flip_at_a_wide_margin_is_not_a_knife_edge(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            capture(left_dir, 2, 0, logits_row(20.0, 0.0))                       # margin 20
            capture(right_dir, 2, 0, logits_row(0.0, 20.0))
            found = margin.flips(margin.profile(Path(left_dir)), margin.profile(Path(right_dir)))
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]["tight"])


if __name__ == "__main__":
    unittest.main()
