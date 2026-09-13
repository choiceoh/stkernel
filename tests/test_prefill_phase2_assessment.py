"""Actual token accounting and fail-closed incomplete campaign evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'measurements/st_prefill_phase2_20260913/assess.py'
spec = importlib.util.spec_from_file_location('phase2_assessment', path)
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)


class AssessmentTests(unittest.TestCase):
    def test_first_request_uses_its_own_tokens_not_the_context_summary(self):
        first = assessment.request_row(dict(ctx=2000, prompt_tokens=2121, ttft_s=.644))
        last = assessment.request_row(dict(ctx=2000, prompt_tokens=2128, ttft_s=.644))
        self.assertFalse(first['target_met'])
        self.assertTrue(last['target_met'])
        self.assertAlmostEqual(first['prefill_tok_s'], 2121/.644)

    def test_unknown_or_invalid_usage_cannot_pass(self):
        for tokens, ttft in ((None,1), (0,1), (True,1), (2128,0), (2128,float('nan')), (2128,float('inf'))):
            with self.subTest(tokens=tokens,ttft=ttft), self.assertRaises(ValueError):
                assessment.request_row(dict(ctx=2000,prompt_tokens=tokens,ttft_s=ttft))

    def test_missing_runs_are_not_a_goal_success(self):
        with tempfile.TemporaryDirectory() as directory:
            result = assessment.assess(Path(directory), 'a'*40)
        self.assertFalse(result['consumer_target_passed'])
        self.assertEqual(len(result['issues']),2)
