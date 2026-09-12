#!/usr/bin/env python3
"""One command, one verdict: does this tree's test suite pass, and what could not be judged?

Written because the verdict used to depend on things nobody could read off the repository. Two files
(test_engine_prefix, test_engine_fleet_lease) passed or failed according to whether `PYTHONPATH=tests` happened to be
set, which appeared in no file; and "FAILED" covered both a real broken guard and a module that could not be imported
at all. An agent that cannot tell those apart will either chase a bug that is not there or ship one that is.

So this reports four states, not two:

    ok            every test ran and passed
    ok (skipped)  it passed, but some tests opted out -- a dependency is missing, not a guarantee
    FAILED        a test ran and did not pass. This is the only state that sets the exit code
    CANNOT RUN    the module could not even be imported here

Usage
    python3 tools/check.py                       # every tests/test_engine_*.py, on the CPU
    python3 tools/check.py --pattern 'test_*'    # a different slice
    python3 tools/check.py --list                # every file, not just the interesting ones
    python3 tools/check.py --gpu                 # leave CUDA_VISIBLE_DEVICES alone (see below)

`--gpu` is off by default on purpose: production GLM-5.3 *is* this engine, and a test that takes the GPU beside it
can get the server killed by earlyoom. Run with `--gpu` only when you know the box is yours.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
DONE = re.compile(r"^(OK|FAILED)(?: \((.*)\))?$", re.M)


class Verdict:
    __slots__ = ("module", "state", "tests", "skipped", "detail")

    def __init__(self, module, state, tests=0, skipped=0, detail=""):
        self.module, self.state, self.tests, self.skipped, self.detail = module, state, tests, skipped, detail

    @property
    def line(self) -> str:
        note = f"  {self.detail}" if self.detail else ""
        skip = f", {self.skipped} skipped" if self.skipped else ""
        return f"  {self.state:<11} {self.module:<44} {self.tests} tests{skip}{note}"


def judge(module: str, out: str, code: int) -> Verdict:
    ran = RAN.search(out)
    tests = int(ran.group(1)) if ran else 0
    done = DONE.search(out)
    counts = dict(p.split("=", 1) for p in (done.group(2) or "").split(", ") if "=" in p) if done else {}
    skipped = int(counts.get("skipped", 0))
    if "unittest.loader._FailedTest" in out:                 # the module never imported: this is not a test result
        why = next((l.strip() for l in out.splitlines() if "Error:" in l), "import failed")
        return Verdict(module, "CANNOT RUN", tests, skipped, why)
    if done is None:
        tail = out.strip().splitlines()[-1:] or [f"no result line, exit {code}"]
        return Verdict(module, "CANNOT RUN", tests, skipped, tail[0][:90])
    if done.group(1) == "FAILED":
        what = ", ".join(f"{v} {k}" for k, v in counts.items() if k != "skipped")
        return Verdict(module, "FAILED", tests, skipped, what)
    return Verdict(module, "ok", tests, skipped)


def run(module: str, gpu: bool, timeout: int) -> Verdict:
    env = dict(os.environ)
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH", "")]))
    try:
        p = subprocess.run([sys.executable, "-m", "unittest", module], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Verdict(module, "CANNOT RUN", detail=f"no verdict in {timeout}s")
    return judge(module, p.stdout + p.stderr, p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="test_engine_*", help="test file stem glob (default: test_engine_*)")
    ap.add_argument("--jobs", type=int, default=4, help="files at a time (default 4: this box also serves)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds one file may take")
    ap.add_argument("--gpu", action="store_true", help="do not hide the GPU from the tests")
    ap.add_argument("--list", action="store_true", help="print every file, not only the ones worth reading")
    a = ap.parse_args()

    files = sorted((ROOT / "tests").glob(a.pattern + ".py"))
    if not files:
        print(f"no tests match {a.pattern!r} under {ROOT / 'tests'}")
        return 2
    modules = [f"tests.{f.stem}" for f in files]
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        verdicts = list(pool.map(lambda m: run(m, a.gpu, a.timeout), modules))

    bad = [v for v in verdicts if v.state != "ok"]
    for v in (verdicts if a.list else bad):
        print(v.line)
    tests = sum(v.tests for v in verdicts)
    skipped = sum(v.skipped for v in verdicts)
    failed = [v for v in verdicts if v.state == "FAILED"]
    stuck = [v for v in verdicts if v.state == "CANNOT RUN"]
    if bad and not a.list:
        print()
    print(f"  {len(files)} files, {tests} tests: {len(verdicts) - len(bad)} ok, {len(failed)} failed, "
          f"{len(stuck)} cannot run" + (f", {skipped} skipped" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
