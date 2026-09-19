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


check, mutate, push = load("check"), load("mutate"), load("push_check")

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

    def test_a_failure_says_which_test_and_why_because_ci_has_no_terminal_to_go_back_to(self):
        # The tool's first run on GitHub reported `FAILED  6 errors` and nothing else about two
        # files, and the answer -- a missing triton wheel -- took a harness that faked the runner.
        out = ("E\n"
               "======================================================================\n"
               "ERROR: test_a_cached_segment (tests.test_engine_graph_contracts.DeviceStepTests)\n"
               "----------------------------------------------------------------------\n"
               "Traceback (most recent call last):\n"
               '  File "x.py", line 3, in test_a_cached_segment\n'
               "    from engine.profiles.glm53.decode_graphs import DeviceStep\n"
               "ModuleNotFoundError: No module named 'triton'\n"
               "\n----\nRan 33 tests in 0.1s\n\nFAILED (errors=6, skipped=3)\n")
        v = check.judge("m", out, 1)
        self.assertEqual(v.state, "FAILED")
        self.assertIn("6 errors", v.detail)
        self.assertIn("test_a_cached_segment", v.detail)
        self.assertIn("No module named 'triton'", v.detail)

    def test_a_failure_with_no_exception_line_still_names_the_test(self):
        out = ("F\n====\nFAIL: test_budget (tests.x.Y)\n----\n"
               "AssertionError\n----\nRan 1 test in 0.0s\n\nFAILED (failures=1)\n")
        self.assertIn("test_budget", check.judge("m", out, 1).detail)

    def test_the_reason_comes_from_the_first_case_not_a_later_one(self):
        out = ("EE\n====\nERROR: test_first (tests.x.Y)\n----\nValueError: the first one\n"
               "\n====\nERROR: test_second (tests.x.Y)\n----\nKeyError: a later one\n"
               "\n----\nRan 2 tests in 0.0s\n\nFAILED (errors=2)\n")
        detail = check.judge("m", out, 1).detail
        self.assertIn("test_first: ValueError: the first one", detail)
        self.assertNotIn("KeyError", detail)

    def test_a_pass_carries_no_failure_note(self):
        self.assertEqual(check.judge("m", PASSED, 0).detail, "")

    def test_output_with_no_result_line_at_all_cannot_be_read_as_a_pass(self):
        self.assertEqual(check.judge("m", "Segmentation fault\n", -11).state, "CANNOT RUN")

    def test_only_a_real_failure_sets_the_exit_code(self):
        states = [check.judge("m", text, 0).state for text in (PASSED, SKIPPED, NEVER)]
        self.assertNotIn("FAILED", states)                 # CI stays quiet about a wheel it could not install


class ShardTests(unittest.TestCase):
    """CI runs the verdict as parts on separate runners that never talk to each other. A file in no part is a test
    that silently stopped running; a file in two is a verdict counted twice. Both would read as a green check."""

    MODULES = [f"tests.test_engine_{i:03d}" for i in range(37)]

    def test_the_parts_cover_every_module_exactly_once(self):
        weight = check.weights(self.MODULES, {m: float(i % 7 + 1) for i, m in enumerate(self.MODULES)})
        for n in range(1, 9):
            parts = [check.part(self.MODULES, weight, k, n) for k in range(1, n + 1)]
            flat = [m for p in parts for m in p]
            self.assertEqual(sorted(flat), self.MODULES, f"{n} parts")

    def test_the_cut_depends_only_on_the_list_and_the_table(self):
        """Each runner computes its own part; the order it listed the files in must not move the cut."""
        weight = check.weights(self.MODULES, {self.MODULES[3]: 50.0, self.MODULES[9]: 40.0})
        shuffled = self.MODULES[::-1]
        self.assertEqual([check.part(self.MODULES, weight, k, 4) for k in (1, 2, 3, 4)],
                         [check.part(shuffled, weight, k, 4) for k in (1, 2, 3, 4)])

    def test_the_long_files_are_spread_and_start_first(self):
        table = {m: 1.0 for m in self.MODULES}
        table.update({self.MODULES[1]: 60.0, self.MODULES[2]: 50.0, self.MODULES[5]: 40.0})
        weight = check.weights(self.MODULES, table)
        parts = [check.part(self.MODULES, weight, k, 3) for k in (1, 2, 3)]
        self.assertEqual(sorted(p[0] for p in parts), sorted([self.MODULES[1], self.MODULES[2], self.MODULES[5]]))
        loads = [sum(weight[m] for m in p) for p in parts]
        self.assertLessEqual(max(loads) - min(loads), 1.0)

    def test_a_file_the_table_has_not_seen_counts_as_its_median(self):
        weight = check.weights(["a", "b", "c", "new"], {"a": 1.0, "b": 3.0, "c": 50.0})
        self.assertEqual(weight["new"], 3.0)
        self.assertEqual(check.weights(["x"], {})["x"], 1.0)

    def test_k_of_n_is_read_strictly(self):
        self.assertEqual(check.shard("2/4"), (2, 4))
        for text in ("0/4", "5/4", "x/4", "4", "2/0", "-1/4"):
            with self.assertRaises(Exception, msg=text):
                check.shard(text)

    def test_every_file_gets_one_thread_unless_the_caller_chose(self):
        """A thread per core made the engine suite 2.8x slower (check.py's docstring has the numbers)."""
        import os
        import subprocess
        from unittest import mock
        seen = []
        done = subprocess.CompletedProcess([], 0, PASSED, "")
        with mock.patch.object(check.subprocess, "run", side_effect=lambda *a, **k: seen.append(k["env"]) or done):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OMP_NUM_THREADS", None)
                v = check.run("tests.m", False, 5)
                os.environ["OMP_NUM_THREADS"] = "3"
                check.run("tests.m", False, 5)
        self.assertEqual((v.state, v.tests), ("ok", 4))
        self.assertEqual([env["OMP_NUM_THREADS"] for env in seen], ["1", "3"])
        self.assertEqual(seen[0]["CUDA_VISIBLE_DEVICES"], "")

    def test_the_table_keeps_what_ran_and_forgets_what_is_gone(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "durations.json"
            path.write_text(json.dumps({"tests.test_agent_tools": 9.0, "tests.test_no_such_file": 5.0}))
            check.record(path, [check.Verdict("tests.test_agent_tools", "ok", 4, seconds=1.25),
                                check.Verdict("tests.test_engine_prefix", "CANNOT RUN", seconds=0.1),
                                check.Verdict("tests.test_engine_serve[2/3]", "ok", 9, seconds=4.0)])
            self.assertEqual(json.loads(path.read_text()), {"tests.test_agent_tools": 1.2})


class SliceTests(unittest.TestCase):
    """A file longer than a slot's share runs as slices. Every test of it has to run in exactly one slice, and the
    slices together have to read the way the file would have: a slice that dropped a test, or ran one twice, or turned
    an import error into a pass, would each still show a green check."""

    FILE = '''
import os, unittest
def mark(test):
    with open(os.environ["SLICE_LOG"], "a") as log:
        log.write(test.id() + "\\n")
class A(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ready = True
    def test_1(self): mark(self); self.assertTrue(self.ready)
    def test_2(self): mark(self)
    def test_3(self): mark(self)
class B(unittest.TestCase):
    def test_1(self): mark(self)
    def test_2(self): mark(self); self.assertEqual(os.environ.get("BREAK"), None)
'''

    def slices(self, k, body=None, env=None):
        import os
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "test_sliced.py").write_text(body or self.FILE)
            log = pathlib.Path(tmp) / "log"
            log.touch()
            out = []
            for j in range(k):
                p = subprocess.run([sys.executable, "-c", check.SLICE, "test_sliced", str(j), str(k)], cwd=tmp,
                                   env={**os.environ, "SLICE_LOG": str(log), **(env or {})},
                                   capture_output=True, text=True, timeout=60)
                name = "tests.x" if k == 1 else f"tests.x[{j + 1}/{k}]"      # what run() calls it
                out.append((("tests.x", j, k), check.judge(name, p.stdout + p.stderr, p.returncode)))
            return out, log.read_text().split()

    def test_every_test_runs_in_exactly_one_slice(self):
        for k in (1, 2, 3, 5, 7):                       # 7 slices of 5 tests: two of them run nothing
            done, ran = self.slices(k)
            self.assertEqual(sorted(ran), sorted(f"test_sliced.{c}.test_{i}" for c, n in (("A", 3), ("B", 2))
                                                 for i in range(1, n + 1)), f"{k} slices")
            merged, = check.gather(done)
            self.assertEqual((merged.module, merged.state, merged.tests), ("tests.x", "ok", 5), f"{k} slices")

    def test_a_failure_in_one_slice_fails_the_file(self):
        done, _ = self.slices(2, env={"BREAK": "1"})
        merged, = check.gather(done)
        self.assertEqual((merged.state, merged.tests), ("FAILED", 5))
        self.assertIn("test_2", merged.detail)

    def test_an_import_error_is_still_one_that_cannot_run(self):
        done, ran = self.slices(3, body="import no_such_module_anywhere\n" + self.FILE)
        merged, = check.gather(done)
        self.assertEqual(ran, [])
        self.assertEqual(merged.state, "CANNOT RUN")
        self.assertIn("Failed to import test module", merged.detail)

    def test_slices_on_another_runner_are_named_not_hidden(self):
        ok = check.Verdict("tests.x[1/3]", "ok", 4, seconds=2.0)
        merged, = check.gather([(("tests.x", 0, 3), ok), (("tests.x", 2, 3), check.Verdict("tests.x[3/3]", "ok", 1))])
        self.assertEqual((merged.module, merged.tests), ("tests.x[1,3/3]", 5))

    def test_only_a_file_longer_than_a_slot_share_is_sliced(self):
        weight = {"tests.long": 90.0, "tests.mid": 20.0, **{f"tests.s{i}": 1.0 for i in range(70)}}
        pieces = check.units(list(weight), weight, 2, 4)       # share = 180 s / 8 slots = 22.5 s
        self.assertEqual(sorted(u for u in pieces if u[0] == "tests.long"), [("tests.long", j, 4) for j in range(4)])
        self.assertEqual(pieces[("tests.long", 0, 4)], 22.5)
        self.assertIn(("tests.mid", 0, 1), pieces)
        self.assertEqual(len(pieces), 72 + 3)
        self.assertEqual(len(check.units(list(weight), weight, 1, 4)), 73)   # one runner: share 45 s, cut in two


class PushCheckTests(unittest.TestCase):
    """A branch whose pull request is already merged still accepts pushes; the commit just never reaches main.
    That happened twice on 2026-09-12, the second time with a note in memory saying not to, which is how a thing
    that should be a check ends up being a story. These are the cases."""

    MERGED = [{"number": 694, "state": "MERGED", "mergedAt": "2026-09-12T09:29:20Z"}]
    OPEN = [{"number": 696, "state": "OPEN"}]

    def test_a_merged_pull_request_stops_the_push(self):
        code, lines = push.verdict(self.MERGED, ahead=1, behind=5)
        self.assertEqual(code, 1)
        self.assertIn("already merged", lines[0])
        self.assertIn("#694", lines[0])
        self.assertTrue(any("cherry-pick" in l for l in lines), "it has to say what to do instead")

    def test_a_reopened_branch_with_an_open_request_is_fine(self):
        """The same head can carry a second request. Merged history does not condemn it."""
        code, lines = push.verdict(self.MERGED + self.OPEN, ahead=1, behind=0)
        self.assertEqual(code, 0)
        self.assertIn("#696", lines[0])

    def test_nothing_to_push_is_also_a_stop(self):
        code, lines = push.verdict([], ahead=0, behind=3)
        self.assertEqual(code, 1)
        self.assertIn("nothing to push", lines[0])

    def test_a_branch_behind_the_base_is_told_so_but_not_blocked(self):
        code, lines = push.verdict(self.OPEN, ahead=2, behind=7)
        self.assertEqual(code, 0)
        self.assertTrue(any("rebase" in l for l in lines))

    def test_no_request_yet_says_so(self):
        code, lines = push.verdict([], ahead=1, behind=0)
        self.assertEqual(code, 0)
        self.assertIn("gh pr create", lines[0])


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
