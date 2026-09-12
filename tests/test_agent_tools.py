"""The tools an agent uses to judge its own work (tools/check.py, tools/mutate.py).

They decide whether a change is safe to merge, so their reading of a test run has to be pinned like anything else:
a tool that calls an import error a test failure sends the next agent chasing a bug that is not there.
"""
from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check, mutate = load("check"), load("mutate")

PASSED = "....\n----\nRan 4 tests in 0.1s\n\nOK\n"
SKIPPED = "..ss\n----\nRan 4 tests in 0.1s\n\nOK (skipped=2)\n"
BROKE = "F...\n----\nRan 4 tests in 0.1s\n\nFAILED (failures=1)\n"
NEVER = ("E\n====\nERROR: test_engine_x (unittest.loader._FailedTest.test_engine_x)\n"
         "ImportError: Failed to import test module: test_engine_x\n----\nRan 1 test in 0.0s\n\nFAILED (errors=1)\n")


class VerdictTests(unittest.TestCase):
    def test_a_pass_is_a_pass(self):
        v = check.judge("m", PASSED, 0)
        self.assertEqual((v.state, v.tests, v.skipped), ("ok", 4, 0))

    def test_a_skip_is_reported_rather_than_swallowed(self):
        """145 of the engine suite's tests skip. That is not the same as 145 guarantees."""
        v = check.judge("m", SKIPPED, 0)
        self.assertEqual((v.state, v.skipped), ("ok", 2))

    def test_a_failure_is_a_failure(self):
        v = check.judge("m", BROKE, 1)
        self.assertEqual(v.state, "FAILED")
        self.assertIn("1 failures", v.detail)

    def test_a_module_that_never_imported_is_not_a_failing_test(self):
        """unittest calls this FAILED (errors=1). It is the difference between broken code and a missing wheel, and
        the whole suite's verdict used to turn on it: `PYTHONPATH=tests` decided two files either way."""
        v = check.judge("m", NEVER, 1)
        self.assertEqual(v.state, "CANNOT RUN")
        self.assertIn("ImportError", v.detail)

    def test_output_with_no_result_line_at_all_cannot_be_read_as_a_pass(self):
        self.assertEqual(check.judge("m", "Segmentation fault\n", -11).state, "CANNOT RUN")

    def test_only_a_real_failure_sets_the_exit_code(self):
        states = [check.judge("m", text, 0).state for text in (PASSED, SKIPPED, NEVER)]
        self.assertNotIn("FAILED", states)                 # CI stays quiet about a wheel it could not install


class MutationTests(unittest.TestCase):
    def source(self):
        return ("def f(a):\n"                              # 1
                '    """doc."""\n'                         # 2
                "    n = 0\n"                              # 3
                "    if a > 1:\n"                          # 4
                "        n += 1\n"                         # 5
                "    return n\n")                          # 6

    def test_a_statement_is_dropped_and_a_branch_is_forced_both_ways(self):
        got = mutate.mutants(self.source(), {3, 4, 5})
        self.assertEqual({(at, what) for at, _, what in got},
                         {(3, "dropped"), (5, "dropped"), (4, "if -> True"), (4, "if -> False")})

    def test_structure_is_not_behaviour(self):
        """def, return and a docstring are not guards: mutating them proves nothing and costs a test run each."""
        self.assertEqual(mutate.mutants(self.source(), {1, 2, 6}), [])

    def test_only_the_lines_you_added(self):
        self.assertEqual([at for at, _, _ in mutate.mutants(self.source(), {5})], [5])

    def test_every_mutant_is_still_a_program(self):
        import ast
        rows = self.source().splitlines(keepends=True)
        for at, replacement, _ in mutate.mutants(self.source(), {3, 4, 5}):
            broken = list(rows)
            broken[at - 1] = replacement + "\n"
            ast.parse("".join(broken))                     # would raise if a mutation left a dangling block


if __name__ == "__main__":
    unittest.main()
