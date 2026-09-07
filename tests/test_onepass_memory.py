import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

spec = importlib.util.spec_from_file_location("guard", Path(__file__).resolve().parents[1] / "bench/onepass_memory.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class OnepassMemoryTests(unittest.TestCase):
    def test_invalid_memory_counters(self):
        for text in ("", "MemTotal: 10 kB", "MemTotal: 10 MB\nMemAvailable: 5 MB",
                     "MemTotal: 10 kB\nMemAvailable: 11 kB", "MemTotal: 10 kB\nMemAvailable: -1 kB"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                guard.parse_memory(text)

    def test_refuses_before_spawning_for_any_low_or_missing_host(self):
        for state in ({"available_kib": 9}, {"error": "SSH unavailable"}):
            def spawn(command):
                self.fail("must not start client")
            out = io.StringIO()
            rc = guard.run_guarded([], lambda: {"head": {"available_kib": 20}, "worker": state}, out, 10, spawn=spawn)
            self.assertEqual(rc, 3)
            self.assertTrue(json.loads(out.getvalue())["issues"])

    def test_child_exit_code_is_preserved(self):
        rc = guard.run_guarded([sys.executable, "-c", "raise SystemExit(7)"],
            lambda: {"head": {"available_kib": 10}}, io.StringIO(), 10, interval=.02)
        self.assertEqual(rc, 7)

    def test_memory_drop_cancels_only_the_owned_client(self):
        children = []
        def spawn(command):
            child = subprocess.Popen(command)
            children.append(child)
            return child
        samples = iter([{"head": {"available_kib": 20}}, {"head": {"available_kib": 9}}])
        report = io.StringIO()
        rc = guard.run_guarded([sys.executable, "-c", "import time; time.sleep(10)"],
            lambda: next(samples), report, 10, interval=.02, spawn=spawn)
        self.assertEqual(rc, 3)
        self.assertIsNotNone(children[0].poll())
        self.assertEqual(len(report.getvalue().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
