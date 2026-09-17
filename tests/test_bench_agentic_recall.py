"""The agentic-recall gate decides without a judge, so its decision is testable here.

`bench/agentic-recall.py` exists because the glyph counters scored the incident's
derailed answers clean. These tests hold its two halves: the case it builds
(deterministic, one fact in the oldest third, one decoy each in the newest) and
the verdict (the exact fact stated, no decoy stated).
"""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("agentic_recall", ROOT / "bench/agentic-recall.py")
recall = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recall)


class CaseBuilderTests(unittest.TestCase):
    def test_the_same_seed_builds_the_same_case(self):
        first, second = recall.build(7, 4000), recall.build(7, 4000)
        self.assertEqual(first, second)

    def test_the_fact_sits_once_in_the_oldest_third_and_every_decoy_after_it(self):
        case = recall.build(7, 20000)
        body = case["body"]
        self.assertEqual(body.count(case["code"]), 1)
        self.assertLess(body.find(case["code"]), len(body) // 3)
        for code, date in zip(case["decoys"], case["decoy_dates"]):
            with self.subTest(decoy=code):
                self.assertEqual(body.count(code), 1)
                self.assertGreater(body.find(code), body.find(case["code"]))
                self.assertEqual(body.count(date), 1)

    def test_the_context_keeps_the_shape_the_product_sends(self):
        body = recall.build(7, 20000)["body"]
        for marker in ("source=session ref=", "cl:main#", "[ctx] [assistant]", "**[assistant]**",
                       "<tool_call>", "</tool_call>"):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.case = recall.build(7, 4000)

    def test_the_exact_fact_passes(self):
        answer = f"예산 코드는 {self.case['code']}, 확정일은 {self.case['date']}입니다."
        self.assertEqual(recall.grade(self.case, answer), dict(recalled=True, code=True, date=True, decoys_stated=[]))

    def test_stating_a_decoy_instead_of_the_fact_fails_and_names_it(self):
        decoy, decoy_date = self.case["decoys"][0], self.case["decoy_dates"][0]
        verdict = recall.grade(self.case, f"예산 코드는 {decoy}, 확정일은 {decoy_date}입니다.")
        self.assertFalse(verdict["recalled"])
        self.assertEqual(verdict["decoys_stated"], sorted([decoy, decoy_date]))

    def test_half_the_fact_is_not_the_fact(self):
        self.assertFalse(recall.grade(self.case, f"코드는 {self.case['code']}입니다.")["recalled"])

    def test_the_glyph_scan_still_counts_what_the_other_gate_counts(self):
        scan = recall.glyph_scan("하ㄹ수 있다 \ufffd 漢 \u0416 \u0e01")
        self.assertEqual((scan["replacement"], scan["welded_jamo"], scan["han"], scan["cyrillic"], scan["thai"]),
                         (1, 1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
