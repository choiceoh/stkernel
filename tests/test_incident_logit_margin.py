"""The margin tool decides where two arms diverge and how tight that choice was.

The engine is deterministic (a repeat of one arm is bit-identical over 90 prefill
stages), so a text difference between arms is caused by whatever changed between
them. Whether it *flips* the answer is a margin question, and these tests hold the
tool's arithmetic: the margin, the chosen probability, which tensor of a capture is
the logit row, and which shared rows flipped with how much room.
"""
import importlib.util
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("incident_logit_margin", ROOT / "tools/incident_logit_margin.py")
margin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(margin)


def capture(directory, admission, generation, logits, extra=True, prefix=None, uniform=None):
    """A capture in the shape the engine writes: the logit row plus the harness's side fields."""
    path = Path(directory) / f"rank0-admit{admission}-gen{generation}-ctx0.pt"
    payload = {"logits": logits}
    if extra:
        payload["small"] = torch.zeros(4)                      # always narrower than the row
    if prefix is not None:
        payload["prefix_sha256"] = prefix
    if uniform is not None:
        payload["uniform"] = uniform
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
            found, skipped = margin.flips(margin.profile(Path(left_dir)), margin.profile(Path(right_dir)))
        self.assertEqual(len(found), 1)
        row = found[0]
        self.assertEqual((row["admission"], row["generation"]), (1, 0))
        self.assertAlmostEqual(row["tightest_margin"], 0.4, places=6)
        self.assertTrue(row["tight"])

    def test_a_flip_at_a_wide_margin_is_not_a_knife_edge(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            capture(left_dir, 2, 0, logits_row(20.0, 0.0))                       # margin 20
            capture(right_dir, 2, 0, logits_row(0.0, 20.0))
            found, skipped = margin.flips(margin.profile(Path(left_dir)), margin.profile(Path(right_dir)))
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]["tight"])


class PrefixAndDrawTests(unittest.TestCase):
    """Logits from different prefixes differ by construction; a moved draw is not a moved distribution."""

    def test_cli_counts_compared_rows_even_when_top1_does_not_flip(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            capture(left_dir, 1, 0, logits_row(5., 4.), prefix='aa' * 32)
            capture(right_dir, 1, 0, logits_row(5., 3.), prefix='aa' * 32)
            capture(left_dir, 1, 1, logits_row(5., 4.), prefix='bb' * 32)
            capture(right_dir, 1, 1, logits_row(4., 5.), prefix='cc' * 32)
            output = io.StringIO()
            with patch('sys.argv', ['margin', '--compare', left_dir, right_dir]), redirect_stdout(output):
                self.assertEqual(margin.main(), 0)
        self.assertIn('shared rows 2; compared 1; top-1 flips 0; skipped for differing prefixes 1', output.getvalue())

    def test_rows_whose_prefixes_differ_are_never_compared(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            capture(left_dir, 1, 0, logits_row(5.0, 4.5), prefix="aa" * 32)
            capture(right_dir, 1, 0, logits_row(4.6, 5.0), prefix="bb" * 32)
            found, skipped = margin.flips(margin.profile(Path(left_dir)), margin.profile(Path(right_dir)))
        self.assertEqual((len(found), len(skipped)), (0, 1))
        self.assertEqual(skipped[0], (1, 0))

    def test_a_matching_prefix_is_compared_and_the_fields_are_read(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            capture(left_dir, 1, 0, logits_row(5.0, 4.5), prefix="cc" * 32, uniform=0.25)
            capture(right_dir, 1, 0, logits_row(4.6, 5.0), prefix="cc" * 32, uniform=0.25)
            left, right = margin.profile(Path(left_dir)), margin.profile(Path(right_dir))
            found, skipped = margin.flips(left, right)
        self.assertEqual((len(found), len(skipped)), (1, 0))
        self.assertEqual(left[(1, 0)]["prefix"], "cc" * 32)
        self.assertEqual(left[(1, 0)]["uniform"], 0.25)
        self.assertTrue(found[0]["same_uniform"])

    def test_a_row_whose_uniforms_differ_is_flagged_as_a_draw(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            capture(left_dir, 1, 0, logits_row(5.0, 4.5), prefix="dd" * 32, uniform=0.25)
            capture(right_dir, 1, 0, logits_row(4.6, 5.0), prefix="dd" * 32, uniform=0.75)
            found, _ = margin.flips(margin.profile(Path(left_dir)), margin.profile(Path(right_dir)))
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]["same_uniform"])

    def test_mixed_prefixes_can_be_forced_but_only_on_request(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            capture(left_dir, 1, 0, logits_row(5.0, 4.5), prefix="aa" * 32)
            capture(right_dir, 1, 0, logits_row(4.6, 5.0), prefix="bb" * 32)
            left, right = margin.profile(Path(left_dir)), margin.profile(Path(right_dir))
            found, skipped = margin.flips(left, right, prefixes=False)
        self.assertEqual((len(found), len(skipped)), (1, 0))


class DrawAddressTests(unittest.TestCase):
    """The draws are stateless, so the address is recoverable -- and a silent mismatch is findable."""

    def test_seeded_capture_uses_actual_row_key_not_its_admission_number(self):
        from engine.base.draws import RICH, row_key, uniform as draw_uniform

        key = row_key(7, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = capture(tmp, 9, 0, logits_row(1., 0.))
            payload = torch.load(path, weights_only=True)
            payload.update(seed=7, admission=9, row_key=key, uniform=draw_uniform(key, RICH, 0))
            torch.save(payload, path)
            row = margin.read_capture(path)
        self.assertIsNone(row['nonce'])
        self.assertEqual(row['row_key'], key)
        self.assertEqual(margin.draw_address(row), (RICH, 0))

    def test_admission_alone_cannot_substitute_for_an_unknown_draw_nonce(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = capture(tmp, 9, 0, logits_row(1., 0.))
            payload = torch.load(path, weights_only=True)
            payload.update(seed=7, admission=9, uniform=.5)
            torch.save(payload, path)
            row = margin.read_capture(path)
        self.assertIsNone(margin.draw_address(row))

    def test_the_last_position_of_every_purpose_can_be_attributed(self):
        from engine.base.draws import DRAFT, PICK, VERIFY, FRESH, RICH, row_key, uniform as draw_uniform

        key = row_key(7, 0, 12)
        for purpose in (DRAFT, PICK, VERIFY, FRESH, RICH):
            row = dict(seed=None, nonce=None, row_key=key, generation=12,
                       uniform=draw_uniform(key, purpose, 7))
            self.assertEqual(margin.draw_address(row), (purpose, 7))

    def test_a_captured_uniform_is_named_by_the_address_it_was_drawn_from(self):
        from engine.base.draws import VERIFY, row_key, uniform as draw_uniform

        seed, nonce, generation = 7, 0, 42
        value = draw_uniform(row_key(seed, nonce, generation), VERIFY, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = capture(tmp, 1, generation, logits_row(1.0, 0.0), prefix="ee" * 32)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload.update(seed=seed, nonce=nonce, uniform=value)
            torch.save(payload, path)
            row = margin.read_capture(path)
        self.assertEqual(row["seed"], seed)
        self.assertEqual(row["nonce"], nonce)
        self.assertEqual(margin.draw_address(row), (VERIFY, 3))

    def test_a_rich_pick_is_found_too_not_only_the_step_block(self):
        from engine.base.draws import RICH, row_key, uniform as draw_uniform

        seed, nonce, generation = 7, 0, 11
        value = draw_uniform(row_key(seed, nonce, generation), RICH, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = capture(tmp, 1, generation, logits_row(1.0, 0.0), prefix="22" * 32)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload.update(seed=seed, nonce=nonce, uniform=value)
            torch.save(payload, path)
            row = margin.read_capture(path)
        self.assertEqual(margin.draw_address(row), (RICH, 2))

    def test_a_uniform_from_nowhere_is_reported_as_unmatched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = capture(tmp, 1, 0, logits_row(1.0, 0.0), prefix="ff" * 32)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload.update(seed=7, nonce=0, uniform=0.4242424242)
            torch.save(payload, path)
            row = margin.read_capture(path)
        self.assertEqual(margin.draw_address(row), (None, None))

    def test_without_a_seed_or_nonce_there_is_nothing_to_recompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = capture(tmp, 1, 0, logits_row(1.0, 0.0), prefix="11" * 32, uniform=0.5)
            row = margin.read_capture(path)
        self.assertIsNone(margin.draw_address(row))


if __name__ == "__main__":
    unittest.main()
