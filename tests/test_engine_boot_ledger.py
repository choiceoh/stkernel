"""The boot table is the whole boot, on every rank, and it can be subtracted from another boot's.

Three things were outside it. The time before the recorder existed -- python, torch, the kernel modules,
the lease, the facts -- which was 15.6 s of a measured 90.2 s boot and 9.17 s of an 87 s one, and both
times somebody recovered it from container timestamps by hand. The sum, which no reader ever had. And
ranks 1-3, whose tables existed only as an unprinted object: `wait for weight preparation` says rank 0
waited 9.69 s for the slowest peer and cannot say which peer, or why.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.base import instruments

ROOT = Path(__file__).resolve().parents[1]


class MarkTests(unittest.TestCase):
    def test_a_marked_phase_is_a_row_like_any_other_in_call_order(self):
        rec = instruments.Recorder("rank0", memory_sampling=False)
        rec.mark("front", 9.17)
        with rec.phase("comm"):
            pass
        names = [child.name for child in rec.root.children]
        self.assertEqual(names, ["front", "comm"])
        self.assertEqual(rec.root.children[0].seconds, 9.17)
        self.assertEqual(rec.root.children[0].calls, 1)

    def test_a_marked_phase_carries_its_counters(self):
        rec = instruments.Recorder(memory_sampling=False)
        rec.mark("front", 1.0, source="proc")
        self.assertIn("source=proc", rec.table())


class TotalTests(unittest.TestCase):
    def table(self):
        rec = instruments.Recorder("rank0", memory_sampling=False)
        rec.mark("front", 9.0)
        with rec.phase("load"):
            with rec.phase("load drafter"):
                pass
        with rec.phase("capture decode"):
            pass
        for span in rec.root.children:
            span.seconds = {"front": 9.0, "load": 5.0, "capture decode": 50.0}[span.name]
        rec.root.children[1].children[0].seconds = 3.0            # nested: inside `load`, not beside it
        return rec.table()

    def test_the_last_line_is_the_sum_of_the_top_level(self):
        last = self.table().splitlines()[-1]
        self.assertTrue(last.startswith("total"), last)
        self.assertEqual(float(last.split()[1]), 64.0)            # 9 + 5 + 50; the nested 3 is inside the 5

    def test_the_nested_row_is_still_printed_under_its_parent(self):
        lines = [line for line in self.table().splitlines() if "load drafter" in line]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("  "), "a child is indented under its phase")


class ProcessStartTests(unittest.TestCase):
    """The one number a module-level `perf_counter` cannot have: the time before the module."""

    # /proc/self/stat: fields 1 and 2 are the pid and the comm in parens, and the rest follow it.
    # Field 22 is starttime, in clock ticks since the machine booted.
    FIELDS = [str(n) for n in range(3, 53)]
    FIELDS[22 - 3] = "6000"
    STAT = "42 (python3) " + " ".join(FIELDS)

    def read(self, text):
        return type("Proc", (), {"read_text": lambda _self: text})()

    def test_it_reads_the_process_start_out_of_proc(self):
        files = {"/proc/self/stat": self.STAT, "/proc/stat": "cpu 1 2 3\nbtime 1000000\nprocesses 7\n"}
        with patch.object(instruments, "Path", lambda p: self.read(files[p])), \
             patch.object(instruments.os, "sysconf", create=True, return_value=100), \
             patch.object(instruments.time, "time", return_value=1000069.17):
            # started 6000 ticks (60 s) after the machine booted at 1000000; it is now 1000069.17
            self.assertAlmostEqual(instruments.process_seconds(), 9.17, places=2)

    def test_a_box_without_proc_says_none_rather_than_guessing(self):
        with patch.object(instruments, "Path", side_effect=OSError("no /proc")):
            self.assertIsNone(instruments.process_seconds())
        files = {"/proc/self/stat": self.STAT, "/proc/stat": "cpu 1 2 3\n"}     # no btime line
        with patch.object(instruments, "Path", lambda p: self.read(files[p])), \
             patch.object(instruments.os, "sysconf", create=True, return_value=100):
            self.assertIsNone(instruments.process_seconds())

    def test_a_clock_that_disagrees_never_reports_a_negative_front(self):
        files = {"/proc/self/stat": self.STAT, "/proc/stat": "btime 1000000\n"}
        with patch.object(instruments, "Path", lambda p: self.read(files[p])), \
             patch.object(instruments.os, "sysconf", create=True, return_value=100), \
             patch.object(instruments.time, "time", return_value=1000000.0):
            self.assertEqual(instruments.process_seconds(), 0.0)

    def test_on_this_box_it_is_none_or_a_plausible_age(self):
        seconds = instruments.process_seconds()
        if seconds is not None:
            self.assertGreaterEqual(seconds, 0.0)
            self.assertLess(seconds, 86400.0)


class DumpTests(unittest.TestCase):
    def test_a_dumped_table_carries_the_rows_and_their_counters(self):
        rec = instruments.Recorder("rank3", memory_sampling=False)
        rec.mark("front", 9.17)
        with rec.phase("load"):
            rec.gauge("direct", 1)
            rec.count("bytes", 44 << 30)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "boot-rank3.json"
            rec.dump(path)
            out = json.loads(path.read_text())
        self.assertEqual(out["root"]["name"], "rank3")
        rows = {child["name"]: child for child in out["root"]["children"]}
        self.assertEqual(rows["front"]["seconds"], 9.17)
        self.assertEqual(rows["load"]["counters"], {"direct": 1, "bytes": 44 << 30})


class WiringTests(unittest.TestCase):
    """Where the three go in the boot -- and that the loader now says which path it read by."""

    def setUp(self):
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text(encoding="utf-8")
        self.fleet = source[source.index("def fleet(a)"):]
        self.loader = (ROOT / "engine/base/loader.py").read_text(encoding="utf-8")

    def test_the_front_row_is_marked_before_the_first_phase_and_names_its_imports(self):
        self.assertLess(self.fleet.index('rec.mark("front", front, import_s=round(_IMPORT_SECONDS, 3))'),
                        self.fleet.index('with rec.phase("comm")'))
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text(encoding="utf-8")
        began = source.index("_IMPORTS_BEGAN = ")
        ended = source.index("_IMPORT_SECONDS = time.perf_counter()")
        self.assertLess(began, source.index(chr(10) + "import torch"), "the stamp opens above torch")
        self.assertLess(source.index("from engine.profiles.glm53 import natives"), ended)

    def test_every_rank_writes_its_table_after_its_last_phase(self):
        dump = self.fleet.index('rec.dump(Path(a.dump_dir) / f"boot-rank{comm.rank}.json")')
        self.assertGreater(dump, self.fleet.index('with rec.phase("release warmup cache")'))
        self.assertNotIn("comm.rank == 0", self.fleet[self.fleet.rindex("\n", 0, dump):dump])

    def test_the_load_row_says_which_path_it_read_by_and_where_the_phase_went(self):
        for gauge in ('recorder.gauge("direct", int(self.direct))', 'recorder.gauge("wait_s"',
                      'recorder.gauge("copy_s"', 'recorder.gauge("read_bytes"'):
            self.assertIn(gauge, self.loader)
        # the wait is measured around the read's result, not around the whole iteration
        body = self.loader[self.loader.index("blocked = time.perf_counter()"):]
        self.assertLess(body.index("host = pending.result()"), body.index("waited_s +="))


if __name__ == "__main__":
    unittest.main()
