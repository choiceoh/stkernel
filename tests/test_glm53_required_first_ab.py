"""CPU-only checks for paired quality evidence; no serving request is sent."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import glm53_required_first_ab as ab


class PairedEvidenceTests(unittest.TestCase):
    def test_balanced_pairs_differ_only_in_order_flag(self):
        selected = ab.cases()
        self.assertEqual(len(selected), 30)
        rows = list(ab.schedule(selected))
        for i in range(0, len(rows), 2):
            pair, case, enabled = rows[i]
            self.assertEqual(enabled, pair % 2 == 1)
            before = deepcopy(case)
            a = ab.request_body(case, "model", pair, False, 4096)
            b = ab.request_body(case, "model", pair, True, 4096)
            self.assertFalse(a.pop("glm53_required_first"))
            self.assertTrue(b.pop("glm53_required_first"))
            self.assertEqual(a, b)
            self.assertEqual(case, before)
        for effort in ("low", "high", "max"):
            for stream in (False, True):
                subset = [case for case in selected if case["choice"] == "auto" and
                          case["effort"] == effort and case["stream"] == stream]
                self.assertEqual(len(subset), 4)

    def message(self, arguments):
        return {"tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "collect_preferences", "arguments": arguments}}]}

    def test_omission_is_separate_from_truncation_and_schema_error(self):
        case = ab.cases()[0]
        self.assertEqual(ab.classify(self.message('{"prompt":"choose"}'), "tool_calls", case),
                         (["missing_required"], ["/tag"]))
        self.assertEqual(ab.classify(self.message('{"prompt":'), "length", case),
                         (["invalid_json", "length"], []))
        errors, _ = ab.classify(self.message('{"prompt":3,"tag":"topic"}'), "stop", case)
        self.assertEqual(errors, ["schema_invalid"])
        self.assertEqual(ab.classify(self.message('{"prompt":"choose","tag":"topic"}'), "tool_calls", case), ([], []))

    def test_nested_required_and_wrong_tool_envelope(self):
        case = ab.cases()[24]
        message = {"tool_calls": [{"function": {"name": "wrong", "arguments": json.dumps({
            "tag": "engineering", "title": "developer", "location": {"notes": "test"},
            "responsibilities": ["code"] * 10})}}]}
        errors, paths = ab.classify(message, "tool_calls", case)
        self.assertEqual(errors, ["missing_required", "tool_envelope", "wrong_function"])
        self.assertEqual(paths, ["/location/city", "/location/remote"])

    def test_array_item_omission_keeps_nested_path(self):
        case = ab.cases()[1]
        self.assertEqual(ab.classify(self.message('{"questions":[{"prompt":"choose"}]}'),
                         "tool_calls", case), (["missing_required"], ["/questions/0/tag"]))

    def test_total_request_deadline_is_removed_after_transport_failure(self):
        with patch.object(ab, "_send", side_effect=TimeoutError), patch.object(ab.signal, "signal", return_value="previous") as handler, patch.object(ab.signal, "setitimer") as timer:
            with self.assertRaises(TimeoutError):
                ab.send("unused", {}, 5)
            self.assertEqual(timer.call_args_list[0].args, (ab.signal.ITIMER_REAL, 5))
            self.assertEqual(timer.call_args_list[-1].args, (ab.signal.ITIMER_REAL, 0))
            self.assertEqual(handler.call_args_list[-1].args, (ab.signal.SIGALRM, "previous"))

    def records(self):
        return [{"pair": pair, "required_first": enabled, "choice": case["choice"],
                 "errors": ["missing_required"] if pair < 6 and not enabled else [], "seconds": 1.0}
                for pair, case, enabled in ab.schedule(ab.cases())]

    def test_promotion_requires_matched_benefit_and_complete_stable_guardrails(self):
        rows = self.records()
        self.assertTrue(ab.summarize(rows, 30, True)["promotion_review_supported"])
        self.assertFalse(ab.summarize(rows, 30, False)["promotion_review_supported"])
        self.assertFalse(ab.summarize(rows[:-1], 30, True)["promotion_review_supported"])
        tied = deepcopy(rows)
        for row in tied:
            row["errors"] = []
        self.assertFalse(ab.summarize(tied, 30, True)["promotion_review_supported"])
        next(r for r in rows if r["pair"] == 29 and r["required_first"])["errors"] = ["length"]
        self.assertFalse(ab.summarize(rows, 30, True)["promotion_review_supported"])

    def test_regressions_sample_size_duplicates_and_latency_block_promotion(self):
        rows = self.records()
        with self.assertRaises(ValueError):
            ab.summarize(rows + [rows[0]], 30, True)
        self.assertFalse(ab.summarize(rows[:12], 6, True)["promotion_review_supported"])
        slow = deepcopy(rows)
        for row in slow:
            if row["required_first"]:
                row["seconds"] = 1.21
        self.assertFalse(ab.summarize(slow, 30, True)["promotion_review_supported"])
        next(r for r in rows if r["pair"] == 7 and r["required_first"])["errors"] = ["invalid_json"]
        self.assertFalse(ab.summarize(rows, 30, True)["promotion_review_supported"])


if __name__ == "__main__":
    unittest.main()
